# vLLM + BaM MDS 调用流程

## 1. 文档目标

本文档记录当前 `xhk/bam-kvstore-vllm-connector-MDS-async-prefetch` 分支中，
一个 sequence 从进入 vLLM、分配 KV cache、被 swap out、重新 swap in，直到继续
attention/decode 的完整调用链。

同时说明：

- 当前 MDS 为什么仍然是同步接口；
- 哪些底层接口已经具备异步能力；
- vLLM 中哪些现有控制流可以复用；
- 后续异步 prefetch 应按什么顺序实现。

## 2. 当前结论

当前存在两类异步基础，但没有可直接使用的异步 swap API：

```text
BaMDirectKVIO:
  已有 submit / poll / finish

vLLM Engine:
  已有 DeferredModelExecution
  可以保留当前 scheduler output，并在下一轮重试

MDSClient / BaMMDSConnector / CacheEngine:
  仍然是同步 submit-and-wait
```

因此后续不需要从零设计，但需要把底层 request handle 向上暴露，并接入 vLLM
已有的 defer/retry 控制流。

## 3. MDS KV cache 初始化

评测入口创建 `LLM`，然后调用 `generate()`：

```text
evaluation/v0_swap_trace_eval.py:283
```

初始化链路：

```text
LLM 初始化
  -> Worker 初始化模型
  -> CacheEngine 计算 GPU KV cache shape/stride
  -> BaMMDSConnector 向 daemon 请求 allocation
  -> daemon cudaMalloc
  -> daemon 对原始 VA 做 BaM P2P map
  -> daemon CUDA IPC export
  -> vLLM CUDA IPC import
  -> imported pointer 包装为 non-owning PyTorch KV tensor
  -> 模型 warmup / CUDA graph
  -> CacheEngine.initialize_bam_direct_kv_store()
  -> connector.start()
  -> daemon 进入 SERVICE_READY
```

MDS allocation 接入点：

```text
vllm/worker/cache_engine.py:200
```

warmup 后启动 MDS service：

```text
vllm/worker/cache_engine.py:109
```

MDS 模式下 GPU KV allocation 的 owner 是 daemon。vLLM 只持有 imported CUDA
mapping 和 PyTorch tensor view。Tensor 使用 no-op deleter，不能释放 daemon-owned
allocation。

## 4. 请求进入 vLLM

`LLM.generate()` 首先把 prompt 转换为 request，然后调用：

```text
LLM._validate_and_add_requests()
  -> LLM._add_request()
  -> LLMEngine.add_request()
```

代码位置：

```text
vllm/entrypoints/llm.py:465
vllm/entrypoints/llm.py:1314
vllm/entrypoints/llm.py:1363
```

每个 request 获得一个 `request_id`。同步离线 API 随后进入：

```python
while self.llm_engine.has_unfinished_requests():
    step_outputs = self.llm_engine.step()
```

代码位置：

```text
vllm/entrypoints/llm.py:1404
```

## 5. 一个 seq 的 KV block

以一个 2048-token sequence 为例，当前 vLLM block size 为 16：

```text
2048 / 16 = 128 个 physical KV blocks
```

每个 physical block 在当前 Qwen2.5-7B 模型中包含：

```text
28 attention layers
每层 K/V 两个 fragments
28 × 2 = 56 fragments
每个 fragment = 16 KiB
每个 logical KV block = 896 KiB
```

GPU 端保持 vLLM 原始 layer-major KV allocation；SSD 端采用 block-major：

```text
storage block N
  layer 0 K
  layer 0 V
  layer 1 K
  layer 1 V
  ...
  layer 27 K
  layer 27 V
```

## 6. LLMEngine.step()

`LLMEngine.step()` 负责一次完整 engine iteration：

```text
1. scheduler 选择下一批 seq
2. scheduler 生成 blocks_to_swap_in/out/copy
3. model executor 执行 Worker
4. model runner 执行 forward/logits/sample
5. 更新 seq 状态并返回 token
```

代码位置：

```text
vllm/engine/llm_engine.py:1298
```

scheduler 调用：

```text
vllm/engine/llm_engine.py:1378
```

`SchedulerOutputs` 被封装为：

```python
ExecuteModelRequest(
    blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
    blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
    blocks_to_copy=scheduler_outputs.blocks_to_copy,
)
```

代码位置：

```text
vllm/engine/llm_engine.py:1421
```

## 7. seq 正常 prefill/decode

正常情况下，seq 状态经历：

```text
WAITING
  -> RUNNING
  -> prefill
  -> decode step 1
  -> decode step 2
  -> ...
  -> FINISHED
```

prefill 时每个 attention layer 把 prompt 对应的 K/V 写入该 seq 的 GPU physical
blocks。decode 时，每一步读取历史 KV，并把新 token 的 K/V 追加到 cache。

## 8. GPU KV 不足时的 swap_out

当 GPU KV blocks 不足时，scheduler 会抢占某个运行中的 seq group：

```text
vllm/core/scheduler.py:1744
```

当前实验强制：

```text
preemption_mode=swap
```

因此 scheduler 执行：

```text
Scheduler._preempt_by_swap()
  -> Scheduler._swap_out()
  -> block_manager.swap_out(seq_group)
```

代码位置：

```text
vllm/core/scheduler.py:1810
vllm/core/scheduler.py:1834
```

`block_manager.swap_out()` 生成 mapping：

```text
(gpu_physical_block_id, storage_block_id)
```

然后 seq 状态变为：

```text
RUNNING -> SWAPPED
```

对应代码：

```text
vllm/core/scheduler.py:1854
```

这些 mappings 被写入：

```text
SchedulerOutputs.blocks_to_swap_out
```

## 9. Worker 执行 cache 操作

Worker 把 scheduler mapping 转为 `[N, 2]` 的 CPU tensor。

当前 `Worker.execute_worker()` 固定按以下顺序执行：

```text
swap_in
  -> swap_out
  -> GPU block copy
  -> return
```

代码位置：

```text
vllm/worker/worker.py:418
```

`execute_worker()` 返回后，上层才调用：

```python
self.model_runner.execute_model(...)
```

代码位置：

```text
vllm/worker/worker_base.py:387
vllm/worker/worker_base.py:429
```

因此在当前同步语义下，attention 开始前，当前 step 的所有 swap 已完成。

## 10. CacheEngine 路由到 MDS

swap_out 路由：

```text
CacheEngine.swap_out()
  -> bam_mds_connector.swap_out(src_to_dst)
```

代码位置：

```text
vllm/worker/cache_engine.py:262
```

swap_in 路由：

```text
CacheEngine.swap_in()
  -> bam_mds_connector.swap_in(src_to_dst)
```

代码位置：

```text
vllm/worker/cache_engine.py:241
```

## 11. Connector 解释 mapping

代码位置：

```text
vllm/bam/mds/connector.py:123
```

swap_out mapping：

```text
source      = GPU physical block
destination = SSD storage block
```

swap_in mapping：

```text
source      = SSD storage block
destination = GPU physical block
```

connector 会把 scheduler mapping 拆成多个 batch。当前 request capacity 为 1024
descriptors，每个 vLLM block 需要 56 descriptors，因此：

```text
max_blocks_per_batch = floor(1024 / 56) = 18
```

约 130 blocks 的 logical swap 会拆成约 8 个 batch。

当前每个 batch 直接调用：

```python
self.client.wait(self.client.submit(payload, operation="write"))
self.client.wait(self.client.submit(payload, operation="read"))
```

## 12. GranuleKV Client 统一请求接口

代码位置：

```text
GranuleKV/gids_module/granulekv/client.py
```

`submit(payload, operation)` 与同步 `wait()` 便利包装实际执行：

```text
生成 request_id / generation
  -> 原子写 batch_request.json
  -> control slot = SUBMITTED
  -> wait_for_states(DONE/ERROR)
  -> 校验 request_id / generation
  -> control slot = SERVICE_READY
  -> 返回
```

因此当前接口虽然叫 `submit_*`，但实际语义是：

```text
submit_and_wait
```

不是只提交后立即返回 handle。

## 13. daemon 执行 I/O

daemon 长期运行以下状态机：

```text
SERVICE_READY
  -> SUBMITTED
  -> IN_FLIGHT
  -> DONE / ERROR
  -> SERVICE_READY
```

代码位置：

```text
BaM_IOStack/gids_module/bam_mds/service.py:145
```

daemon 读取 request payload，校验 `request_id/generation`，然后调用 executor。

## 14. Executor 展开 descriptors

代码位置：

```text
BaM_IOStack/gids_module/bam_mds/executor.py:95
```

executor 对每个 block 展开：

```text
for layer in 0..27:
  for K/V:
    operation
    SSD byte offset
    region id
    GPU region byte offset
    length = 16 KiB
```

随后调用底层异步风格接口：

```python
handle = direct_io.submit(**descriptors)
while handle.generation not in {
    item.generation for item in direct_io.progress()
}:
    pass
direct_io.finish(handle)
```

NVMe DMA 直接访问 daemon-owned、vLLM 已导入的 GPU allocation，不经过 CPU KV
payload，也没有 daemon/client 间 GPU copy。

## 15. seq 重新 swap_in

scheduler 会优先检查 `swapped` 队列：

```text
vllm/core/scheduler.py:822
```

如果 GPU block 和调度 budget 足够：

```text
Scheduler._swap_in()
  -> block_manager.swap_in(seq_group)
  -> 生成 (storage_block_id, new_gpu_block_id)
  -> seq.status = RUNNING
```

代码位置：

```text
vllm/core/scheduler.py:1817
```

MDS read 完成后，`CacheEngine.swap_in()` 才返回。随后 model runner 执行 attention，
因此 attention 读取到的 blocks 已经完成 NVMe DMA。

## 16. 当前哪些部分已经异步

### 16.1 BaM native 数据面

底层已经提供：

```text
submit -> native handle
poll(handle)
finish(handle)
```

### 16.2 daemon 与 vLLM 是不同执行域

daemon 与 vLLM 是不同 CUDA/MPS client。请求发布后，daemon 可以独立执行 BaM
submit 和 resident CQ poll。

### 16.3 当前缺失的部分

MDSClient 在发布后立即等待 DONE，因此没有把 native handle 生命周期暴露给
connector、CacheEngine 或 scheduler。

## 17. vLLM 已有 DeferredModelExecution

定义位置：

```text
vllm/worker/model_runner_base.py:284
```

语义：

```text
connector/runtime 已经启动 in-flight retrieve
但当前 batch 还不能安全 forward
engine 保留当前调度结果
下一 engine iteration 重试同一批输入
```

同步 `LLMEngine` 捕获位置：

```text
vllm/engine/llm_engine.py:1442
```

捕获后执行：

```text
_skip_scheduling_next_step = True
缓存 scheduler_outputs / seq_group_metadata
本轮返回空输出
下一轮不重新 schedule，继续执行同一个 batch
```

异步 `AsyncLLMEngine` 也有相同路径：

```text
vllm/engine/async_llm_engine.py:357
```

当前已有 KV transfer connector 使用 `KVReceiveStatus.DEFERRED` 触发该控制流：

```text
vllm/worker/model_runner.py:1879
```

## 18. 第一阶段异步接口

首先只拆接口，不改变 CacheEngine 同步行为。

目标 MDSClient API：

```python
handle = client.submit(payload, operation="read")
status = client.status(handle)
if status.ready:
    client.complete(handle)
```

稳定 handle 至少包含：

```text
request_id
generation
operation
```

同步 wrapper 继续保留：

```python
def swap_in(mapping):
    handle = submit_swap_in(mapping)
    wait(handle)
```

这样现有 CacheEngine、scheduler、输出一致性测试和性能基线都不改变。

## 19. logical request 拆批应下沉到 daemon

当前 connector 的行为：

```text
submit batch 1 -> wait
submit batch 2 -> wait
...
submit batch 8 -> wait
```

第一阶段建议改为：

```text
vLLM 一次提交完整 logical mapping
  -> daemon 内部按 18 blocks 拆成多个 BaM batches
  -> daemon 依次执行 submit/poll/finish
  -> 所有 batch 完成后发布一个 logical DONE
```

这样现有单槽协议可以先支持一个完整 in-flight logical request，不需要立刻实现
多槽 ring。

## 20. 接入 DeferredModelExecution

Worker 第一次看到一个 swap mapping：

```text
没有 live handle
  -> submit logical request
  -> 保存 handle
  -> raise DeferredModelExecution
```

下一 engine iteration 会收到相同 scheduler output：

```text
找到 live handle
  -> poll(handle)
  -> 未完成：再次 defer
  -> 已完成：finish(handle)
  -> 删除 handle
  -> 继续 copy / model forward
```

handle 查找必须至少使用：

```text
virtual_engine
operation
mapping fingerprint
generation
```

否则同一份 scheduler output 每次重试都会重复提交 I/O。

## 21. 第一阶段的能力边界

第一阶段可以做到：

```text
vLLM engine thread 不阻塞在 wait_for_states()
daemon 独立推进 NVMe I/O 和 resident CQ poll
同一 batch 在后续 engine iteration 恢复
```

但 `DeferredModelExecution` 当前是整批 defer：

```text
seq A 的 KV 正在 LOADING
同一 model batch 中的 seq B/C 也不会 forward
```

因此它还不是最终的 I/O/attention 重叠。

## 22. 真正的异步 prefetch

要实现：

```text
seq A 正在 swap_in
seq B/C 使用 READY KV blocks 继续 attention
```

需要 scheduler 或独立 block state table 管理：

```text
LOADING
  NVMe read 正在进行，attention 禁止读取

READY
  KV 已完成，可以进入 model batch

EVICTING
  NVMe write 正在进行，GPU block 禁止复用

FREE
  可以重新分配

ERROR
```

状态必须绑定 generation，防止旧 completion 把已复用 block 错误标记为 READY。

最终目标链路：

```text
当前 step:
  attention 使用 READY sequences

后台 MDS daemon:
  为未来 step 执行 NVMe prefetch + resident CQ poll

未来 step:
  wait_required_blocks()
  -> READY 后把对应 seq 放入 model batch
```

## 23. 推荐实施阶段

### Phase 5A：接口拆分

```text
submit / poll / wait / finish
同步 wrapper 保持不变
输出一致性与性能回归
```

### Phase 5B：整批 defer

```text
logical mapping 一次提交
daemon 内部拆批
Worker 保存 handle
DeferredModelExecution 跨 iteration poll
```

### Phase 5C：block-aware prefetch

```text
LOADING/READY/EVICTING
只调度 READY seq
I/O 与其他 seq attention 重叠
```

### Phase 5D：多请求并发

```text
request ring
多个 in-flight prefetch
completion 按 request_id/generation 回收
```

第一阶段不应直接引入多槽 ring 或 scheduler 大改。先保证 handle 生命周期、整批
defer 和同步 fallback 全部正确，再进入 block-aware prefetch。
