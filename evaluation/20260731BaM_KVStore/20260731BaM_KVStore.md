# BaM Direct KVStore 当前实现、问题与推进方案

日期：2026-08-01

## 1. 文档定位

本文记录 `vllm-bam` direct KVStore 主线的当前有效状态。内容按以下顺序组织：

1. 已经实现并验证的功能。
2. 真实 Qwen2.5 preemption 负载遇到的问题。
3. 已排除的假设和已经收束掉的临时逻辑。
4. 与旧 BaM one-copy 的准确对比。
5. AGIO 的源码级执行模型及其对当前问题的直接启示。
6. 第一优先级：解决 resident I/O service 与 attention 的执行冲突。

本文不再把早期设计建议、已删除的诊断接口和当前实现混写。Git 历史和旧稳定
分支继续保存 LMCache/BaM one-copy 及早期实验实现。

## 2. 目标架构与不变量

Direct KVStore 只优化掉 BaM payload cache 和中间搬运。除数据直接落入 vLLM KV
cache 外，submit、CQ completion、ready 和 service lifetime 应与已经跑通的 BaM
one-copy 语义一致。

目标读取链路：

```text
CPU/vLLM scheduler 整理需要恢复的 physical KV blocks
  -> CPU 异步发布 GPU-visible descriptor
  -> GPU submit kernel 构造 PRP 并提交 NVMe read
  -> resident GPU service 持续轮询 NVMe CQ
  -> NVMe DMA 直接写入 vLLM paged KV cache
  -> GPU completion path 发布 request DONE
  -> CPU 只读取 DONE
  -> CPU 推进 model forward / attention
```

目标写入链路：

```text
vLLM 产生 KV block
  -> CPU 异步发布 source block descriptor
  -> GPU submit NVMe write
  -> NVMe DMA 直接从 vLLM paged KV cache 写入 SSD
  -> resident GPU service 收取 CQ completion
  -> GPU 发布 request DONE
  -> CPU 只读取 DONE 并完成本次 swap_out
```

必须保持的不变量：

- CPU 不读取、不 dequeue、不推进 NVMe CQ。
- CPU 不搬运 KV payload。
- request ready/DONE 只能由 GPU completion path 发布。
- resident service 在正常 batch 完成时不停止、不暂停、不重置。
- `close()`/进程退出是正常路径唯一的 service stop 边界。
- host `batch_active/active_count/generation` 只能用于 host bookkeeping，不能控制
  resident GPU loop。
- 不修改、不破坏旧 LMCache BaM one-copy 路径。

## 3. 数据布局与请求粒度

当前逻辑对象是 vLLM physical token block。Qwen2.5-7B、FP16、block size 16、
28 个 attention layer、4 个 KV head、head size 128 时：

```text
1 layer K fragment = 16 KiB
1 layer V fragment = 16 KiB
1 logical block    = 28 layers * 2 fragments
                   = 56 fragments
                   = 917,504 bytes
```

vLLM 每层 K/V allocation 不是跨层连续区域，因此一个 logical block 展开为 56
个 NVMe fragment request。PRP 直接指向原始 vLLM KV allocation，不经过 BaM
page cache、staging pages 或 mover scatter。

真实 130-block swap 的规模：

```text
token slots       = 130 * 16 = 2,080
fragment requests = 130 * 56 = 7,280
payload           = 130 * 917,504 = 119,275,520 bytes
```

Native request table 容量为 1,024 fragments，上层一次最多提交 18 个 logical
blocks，因此 130 blocks 被顺序拆成 8 个 direct batch。这不是异常放大，也没有
按 token 扫描整个请求集合。

## 4. 当前代码边界

### 4.1 BaM_IOStack

核心文件：

```text
BaM_IOStack/gids_module/direct_kv_io.cu
BaM_IOStack/gids_module/include/direct_kv_io.h
BaM_IOStack/gids_module/bam_direct_kv_io.py
BaM_IOStack/gids_module/gids_nvme.cu
```

`BaMDirectKVIO` 当前职责：

- 初始化 controller、request table、region table、PRP pool 和 completion lookup。
- 注册 vLLM 已有 CUDA allocation，并保存 DMA IO address table。
- 在 vLLM 当前 CUDA stream 上发布 request descriptor。
- GPU submit kernel 构造 PRP、分配 CID、登记 `(queue,cid) -> request` lookup、
  写 SQ 并敲 doorbell。
- 独立 `cudaStreamNonBlocking` service stream 启动 `1 CTA x 128 threads` 的
  persistent CQ service。
- GPU CQ service 发布 request DONE。
- CPU control stream 只 D2H snapshot request table 并读取状态。
- `close()` 写 `running=0` 并等待 resident kernel 退出。

当前 service loop：

```text
128 threads 静态分片 CQ
  -> 每条 CQ 每轮最多处理 64 个 completion
  -> CTA barrier
  -> heartbeat++
  -> system fence
  -> CTA barrier
  -> 下一轮
```

### 4.2 vllm-bam

核心文件：

```text
vllm/bam/direct_block_store.py
vllm/cache_engine.py
vllm/worker/worker.py
vllm/worker/worker_base.py
vllm/worker/model_runner.py
evaluation/v0_swap_trace_eval.py
evaluation/20260731BaM_KVStore/run_direct_vllm_preemption_smoke.sh
```

V0 CacheEngine 调用链：

```text
Scheduler blocks_to_swap_out / blocks_to_swap_in
  -> Worker
  -> CacheEngine
  -> BaMVLLMDirectKVStore
  -> BaMDirectBlockStore 展开 block -> layer/K/V fragments
  -> BaMDirectKVIO submit/poll/finish
```

Direct backend 只由 `VLLM_BAM_DIRECT_KVSTORE_ENABLE=1` 显式打开。普通 V0 CPU
swap、LMCache、GDS 和旧 BaM page-cache 路径默认行为不变。

## 5. 已完成的正确性验证

### 5.1 Synthetic direct-block roundtrip

已经验证：

- 1 block / 56 fragments roundtrip，`exact_equal=1`。
- 4 blocks / 224 fragments / 3,670,016 bytes roundtrip，`exact_equal=1`。
- 随机、非连续 physical block。
- NVMe DMA 直接读写最终 vLLM layout allocation。
- BaM payload cache、staging 和 refill 均为 0。

4-block 小批量 write/read 约 1.26 GiB/s，只用于正确性验证，不作为正式性能
baseline。

### 5.2 真实 CacheEngine smoke

Qwen2.5-7B、FP16、XFormers layout 已验证：

```text
cache_engine_layout=(2, 8, 8192)
stride=(65536, 8192, 1)
cpu_kv_payload_cache_layers=0
bam_page_cache=0 staging=0 refill=0
exact_equal=1
[DIRECT_KV_CACHE_ENGINE] PASS
```

### 5.3 stop-after-batch 诊断闭环

历史诊断配置曾在每个 direct batch 完成后停止 service。它完成了真实闭环：

```text
swap_out/write_done=31
swap_in/read_done=31
Run summary
elapsed=98.356s
```

该结果只证明：

```text
数据布局、SSD extent、write、read、swap_out、swap_in 和 attention 数据路径可用；
当前 resident service 存在时会干扰后续 forward。
```

它不证明正确架构必须每批停服。相关 direct-only 环境变量、Python/native API 和
正常路径调用已经删除，不得恢复为最终方案。

## 6. 当前真实负载故障

真实 workload：

```text
model=Qwen2.5-7B-Instruct
num_prompts=8
prompt_len=2048
max_tokens=128
best_of=4
max_num_seqs=8
num_gpu_blocks_override=260
preemption_mode=swap
enforce_eager=true
```

最新 resident 路径日志：

```text
evaluation/logs/direct_kvstore_preemption/20260801_111257/console.log
```

稳定停点：

```text
Scheduler swap_out mappings=130
BAM_DIRECT_KVSTORE op=write phase=done blocks=130 elapsed_ms=57.931
CacheEngine.swap_out returned
Worker.execute phase=done
WorkerBase phase=after_execute_worker
ModelRunner.execute_model phase=enter decode tokens=1
ModelRunner.execute_model phase=before_forward decode tokens=1
<no after_forward>
```

挂起时连续采样：

```text
GPU utilization = 100%
GPU memory utilization = 0%
唯一 CUDA 进程 = 本次 Qwen2.5 测试
```

当前结论：

- 首批 GPU submit 成功。
- resident service 成功收取全部 CQ completion。
- 130-block write 已发布 DONE。
- CPU 已读取 DONE，`CacheEngine.swap_out` 和 `Worker.execute` 已返回。
- scheduler 已经推进到下一个 decode forward。
- 停滞位于 `before_forward` 之后的 GPU 执行，不是 swap_out 仍在等待 SSD。

## 7. 已排除和已收束的错误方向

### 7.1 CPU 提前推进 attention

高层顺序已经核对：

```text
submit
  -> GPU CQ 发布 request DONE
  -> CPU D2H 读取 request table
  -> finish 验证 DONE
  -> Worker.execute 返回
  -> ModelRunner/attention
```

曾增加 GPU service generation ACK，要求 CPU 等 resident CTA 在全 DONE 后越过
barrier。结果仍停在同一个 `before_forward`，因此 ACK 已删除。CPU 不是在 request
DONE 之前启动 attention。

### 7.2 CPU 控制 resident service

已删除以下错误设计：

```text
DirectKVControlDevice.active_requests
direct_kv_publish_active_count_kernel()
finish_batch() 清 active_requests
service loop 以 active_requests 作为 CQ gate
```

`finish_batch()` 现在只验证 request DONE 并清理 host bookkeeping。正常 completion
不会向 GPU resident control plane 写任何开关。

### 7.3 每批停止 service

已删除 direct-only：

```text
stop_service_if_idle()
VLLM_BAM_DIRECT_KVSTORE_STOP_SERVICE_AFTER_BATCH
finish 后的 idle-stop 调用
```

`close()` 是唯一正常 stop 边界。

### 7.4 CQ completion payload 的额外读取

direct 曾在 dequeue 后读取 CQ payload status。旧 single-CTA one-copy 的稳定顺序是：

```text
peek -> dequeue -> put_cid -> lookup -> finalize
```

额外 payload 读取已经删除，避免读取可能已经复用的 CQ entry。

## 8. 与旧 BaM one-copy 的准确对比

两者共同点：

```text
1 CTA x 128 threads
每个线程静态拥有一组 CQ
每条 CQ 每轮最多 64 completions
cudaStreamNonBlocking service stream
GPU 持续推进 CQ/request state
CPU 只读取 request state
service 正常只在 shutdown 时退出
```

编译后资源占用：

| kernel | registers/thread | stack | shared |
|---|---:|---:|---:|
| direct CQ service | 24 | 0 | 0 |
| one-copy single-CTA service | 146-160 | 184 B | 4-16 B |

因此 direct service 不是因为单 CTA 寄存器/shared memory 过重而占满整卡。

主要差异在一轮 service loop 的内容：

```text
one-copy:
  poll CQ
  -> 128 threads 分片扫描默认 1024 runtime slots
  -> thread0 再扫描 1024 slots 的 retire/placement/control
  -> heartbeat + system fence

direct:
  poll CQ
  -> heartbeat + system fence
```

两者都是“每轮一次 heartbeat/fence”，但 direct 空闲轮非常短，wall-clock 频率
可能高得多。heartbeat 只用于诊断，不参与 request ready；completion path 上用于
发布 DMA 数据和 DONE 的 system fence 则是正确性要求，不能降频。

heartbeat 高频是次要候选，不是当前第一根因。旧 one-copy 能跑通说明 V100 上
resident I/O 与 attention 并存并非原则上不可能，仍应优先对齐它的启动环境和
GPU 调度条件。

## 9. AGIO 源码级分析

### 9.1 来源与本地代码

论文：

```text
Asynchrony and GPUs: Bridging this Dichotomy for I/O with AGIO
ASPLOS 2026
DOI: 10.1145/3779212.3790130
```

Artifact：

```text
Concept DOI: 10.5281/zenodo.18333270
Version DOI: 10.5281/zenodo.18333271
version: v1
license: CC BY 4.0
archive MD5: bfe4c3eadb6ed6effffa409d274fda74
local path: /home/xhk/llm-inference/AGIO
```

核心代码：

```text
AGIO/g_src/main.cu          # 切分 SM resource、创建 Green Context
AGIO/g_src/threads.cuh      # runtime/application context 与 stream
AGIO/g_io/runtime.cuh       # persistent runtime、软件 SQ/WQ consumer
AGIO/g_io/submit.cuh        # application 侧异步发布 IOCB
AGIO/g_io/comm.cuh          # 软件 request/completion queue
AGIO/g_io/lqueue.cuh        # lock-free slotted ring queue
AGIO/g_io/iocmd.cuh         # BaM NVMe submit + blocking cq_poll
AGIO/g_src/options.toml     # runtime/application SM 配额
```

### 9.2 AGIO 的异步含义

AGIO 不是让发起 I/O 的 application thread 在同一个调用内 submit 后同步 poll。
它把 initiation 与 completion 在时间和线程空间上解耦：

```text
application GPU thread
  -> g_submit_async()
  -> 向 GPU software submission queue 写 IOCB
  -> 立即继续其它 application work

runtime GPU thread
  -> poll software submission queue
  -> process_io()
  -> get_cid / build PRP / sq_enqueue
  -> cq_poll / cq_dequeue / put_cid
  -> notify_io() 写 software completion queue/stage

application GPU thread
  -> g_wait_any()/g_check_cid()
  -> 消费已完成 request
```

`AGIO/g_io/submit.cuh` 先写 IOCB payload，最后以 release store 发布
`ENTRY_VALID`。`AGIO/g_io/runtime.cuh` 以 acquire/atomic 状态领取 entry，完成后
发布 `ENTRY_PROCESSED` 和 completion notification。

### 9.3 AGIO 仍然使用 blocking NVMe CQ poll

`AGIO/g_io/iocmd.cuh::nvm_cmd_process()` 的实际顺序是：

```text
get_cid
  -> nvme_build_prp
  -> sq_enqueue
  -> cq_poll(cid)
  -> cq_dequeue
  -> put_cid
```

`cq_poll()` 本身仍循环等待目标 CID，只在循环中使用指数增长到 256 cycles 的
`__nanosleep()`。AGIO 的“异步”不是 NVMe poll thread 不阻塞，而是阻塞发生在
专用 runtime threads 上，application threads 不被阻塞。

这点对当前问题非常关键：只给 direct CQ loop 添加 sleep 不是 AGIO 的核心方案。

### 9.4 AGIO 如何解决 resident runtime 与 compute 冲突

AGIO 明确不依赖两个普通 CUDA streams 的公平调度。它使用 Green Context 做
硬件级 SM specialization：

```text
cuDeviceGetDevResource(CU_DEV_RESOURCE_TYPE_SM)
  -> cuDevSmResourceSplitByCount()
  -> runtime SM resource descriptor
  -> application SM resource descriptor
  -> cuGreenCtxCreate(runtime context)
  -> cuGreenCtxCreate(application context)
  -> 分别创建 CU_STREAM_NON_BLOCKING stream
```

默认配置：

```text
green_group_size = 4 SMs
green_rt_n_groups = 8
runtime partition = 32 SMs
n_rt_warps = 4 per block
```

论文使用 A100 的 108 SM，将 32 SM 固定给 persistent runtime，其余给 application。
runtime kernel 可以持续 poll 软件队列和 NVMe CQ，但不会占用 application 的 SM
集合。application 与 runtime 通过 GPU HBM 中的双向软件队列通信。

因此 AGIO 解决常驻冲突的主因果链是：

```text
不是：降低 heartbeat -> 希望 attention 获得机会

而是：
  persistent blocking I/O runtime
  + hardware SM partition
  + application/runtime 单向软件队列
  -> compute 与 I/O 在不同执行资源上并存
```

### 9.5 AGIO 生命周期

AGIO runtime 同样是 resident service：

```text
runtime init
  -> persistent runtime running
  -> 多次 software SQ request/completion
  -> shutdown 设置 queue qstate=0
  -> runtime 退出
```

它不会在每个 I/O 完成后停止 runtime。这进一步确认 direct 的最终方案不应恢复
per-batch stop。

## 10. AGIO 对当前 direct KVStore 的可借鉴边界

### 10.1 直接借鉴

1. **service lifetime 与 request lifetime 分离**。
2. **submit producer、NVMe CQ owner、completion producer 的单向 ownership**。
3. **request payload 先写，最后以 release 语义发布 valid/ready**。
4. **blocking poll 必须运行在不会阻塞 application 的执行资源上**。
5. **所有 runtime allocation、DMA mapping 和 queue allocation 在 service 启动前完成**。
6. **正常 batch 完成只 retire request，不停止 resident runtime**。

### 10.2 不直接照搬

- 不引入 AGIO 的完整 megakernel、allocator 和第二套 NVMe controller。
- 不替换当前已验证的 BaM SQ/CQ、CID 和 DMA mapping。
- 不在 direct 主线再叠加一套复杂的 AGIO software SQ/CQ，除非后续要让 resident
  runtime 同时接管 GPU submit；当前问题首先是执行资源冲突。
- 不引入 CPU completion thread、CPU CQ poll 或 per-batch service stop。
- 不直接复制 Green Context 代码到当前 V100 环境。

## 11. 第一优先级：解决常驻线程冲突

### 11.1 当前平台约束

AGIO 验证环境：

```text
GPU=A100
driver=570 open
CUDA=12.8
```

当前环境：

```text
GPU=Tesla V100S 32GB
compute capability=7.0
driver=535.230.02
CUDA toolkit=12.2
```

当前 `/usr/local/cuda-12.2/include/cuda.h` 没有 `CUgreenCtx`、
`cuDevSmResourceSplitByCount` 等 API。因此当前不能编译 AGIO 的硬件 SM partition
代码，也不能假设升级 header 后 V100/driver 组合一定支持。

第一步不是立即把 AGIO Green Context 移植进 BaM，而是先验证普通 stream 路径
是否因为 hardware work queue 映射产生 head-of-line blocking。旧 one-copy 已在同一
V100 上跑通，说明应先寻找当前 direct 与旧路径的调度差异。

### 11.2 第一步 A：确认 CUDA hardware queue 映射

必须从 root wrapper 的真实进程环境确认：

```text
CUDA_DEVICE_MAX_CONNECTIONS
CUDA_LAUNCH_BLOCKING
CUDA_VISIBLE_DEVICES
```

特别检查 `CUDA_DEVICE_MAX_CONNECTIONS=1`。如果 service stream 与 compute stream
被映射到同一个 hardware work queue，先入队且永不结束的 persistent kernel 可能
让后续 attention kernel 排在其后。

验证方式必须是 direct-only wrapper 固定环境，不修改系统全局环境：

```text
CUDA_DEVICE_MAX_CONNECTIONS=8
CUDA_LAUNCH_BLOCKING=0
```

只运行一次相同 Qwen2.5 workload。判断标准：

```text
resident service 不停止
write DONE 后出现 after_forward
随后出现 swap_in/read DONE 和 Run summary
```

如果该测试通过，根因是 hardware queue 映射，而不是 CQ ownership 或 ready
协议。最终只需要把 direct 运行环境固定为多个 connection，并增加启动时诊断。

### 11.3 第一步 B：降低 service stream 调度优先级

如果多个 hardware connections 仍复现，下一项最小代码实验是把 service stream
创建为最低优先级：

```cpp
int least_priority = 0;
int greatest_priority = 0;
cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
cudaStreamCreateWithPriority(
    &service_stream, cudaStreamNonBlocking, least_priority);
```

PyTorch compute stream 保持默认优先级。该改动不停止 resident service、不改变
request/CQ 状态机，只向 CUDA 调度器声明：有新 block 可调度时优先 compute。

限制：stream priority 不能抢占已经运行的 thread block，因此它不是 AGIO Green
Context 的等价替代。它只适合验证当前阻塞是否来自待调度 work 的优先级/queue
选择。

### 11.4 第一步 C：对齐旧 one-copy 启动条件

若 A/B 都失败，逐项核对：

- service 启动前是否已完成所有 PyTorch/XFormers workspace 和 lazy kernel 初始化。
- service stream 是否与旧 one-copy 使用相同 CUDA context。
- 首次 submit event 是否只建立 `submit -> service start` 单向依赖。
- compute stream 是否意外等待 service stream 的终止事件，而不是 request DONE。
- `before_forward` 内首次 tokens=1 路径是否触发 `cudaMalloc/cudaFree` 或
  device-wide synchronization。

观测只加 layer/attention 窄 trace，不改变数据面：

```text
layer0 enter
qkv projection done
attention enter
attention exit
layer0 exit
```

### 11.5 第一步 D：硬件资源隔离的决策边界

如果确认普通 stream 在当前 V100 上无法可靠容纳 resident poll，真正对齐 AGIO
需要硬件执行资源隔离，而不是 per-batch stop。候选路线：

1. 在支持 Green Context 的 GPU/driver/CUDA 组合上建立 I/O control domain。
2. 评估 V100 可用的 MPS execution affinity/active-thread partition，但这可能需要
   独立 context/process 和 CUDA IPC，会显著扩大架构改动。
3. 在做平台迁移前，以旧 one-copy 的可运行版本继续作为 V100 correctness
   reference。

未确认平台 API 前，不实现 MPS/多 context，不把问题复杂化。

### 11.6 heartbeat 的处理顺序

heartbeat 降频放在资源/queue 验证之后。若需要实验，只降低诊断 heartbeat：

```text
每 1024 个空闲 service loops：
  heartbeat++
  system fence
```

以下 completion 发布顺序永远不变：

```text
CQ completion
  -> publish SSD DMA writes
  -> system fence
  -> request.state = DONE
```

## 12. 后续测试顺序

已完成验证：

1. `CUDA_DEVICE_MAX_CONNECTIONS=8` 仍复现第一次 `swap_out/write DONE` 后
   卡在 `before_forward`。
2. 最低优先级 service stream 仍复现同一停点。
3. batch DONE 后退役空转 direct CQ service，可跑出
   `after_forward -> swap_in/read DONE -> Run summary`。

当前正式策略：

```text
VLLM_BAM_DIRECT_SERVICE_LIFETIME=io_active   # 默认
```

语义：

```text
IO batch 活跃期间保持 GPU persistent CQ service
batch DONE 且 direct active_count=0 后停止空转 service
下一次 submit 自动重新启动 service
```

显式设置：

```text
VLLM_BAM_DIRECT_SERVICE_LIFETIME=resident
```

才保留“service 跨 forward 永久常驻”的旧实验行为。当前 V100 路径不把
`resident` 作为默认。

每次记录：

```text
write phase=done count
read phase=done count
before_forward / after_forward count
GPU utilization / memory utilization
Run summary
console log path
```

完整成功标准：

```text
swap_out
  -> IO 活跃期 resident service 完成 write
  -> idle-stop 退役空转 service
  -> attention after_forward
  -> swap_in
  -> read DONE
  -> Run summary
```

## 13. 构建、权限与会话约束

- 不读取旧 Codex JSONL。
- 只用窄范围 `rg`、`sed`、`tail` 检查日志和代码。
- 测试脚本一次只运行一个。
- 权限不足立即停止，并给出 sudoers/wrapper 命令。
- root wrapper 启动的进程必须由授权 wrapper 或 root 精确终止。
- 不改 LMCache SSD、GDS 和旧 BaM one-copy baseline。
- direct 正常路径采用 `io_active` 生命周期；保留后台 poll 的 IO 活跃期常驻语义，
  但不在 V100 上让空转 service 跨 PyTorch forward。
- 新 direct 路径注释继续使用 `【BaM KVStore 直通调用链】` 标记。

## 14. 当前阶段结论

Direct KVStore 的真实 SSD 数据面已经成立：GPU submit、NVMe DMA 直接读写 vLLM
KV cache、GPU CQ completion 和 CPU 只读 DONE 均已验证。当前 V100 可运行方案是
`io_active` 生命周期：后台 persistent poll 在 IO 活跃期保留，batch 空闲后退役，
避免空转 service 与真实 Qwen2.5 attention 争用执行资源。

AGIO 给出的最重要结论不是“给 poll 加 sleep”，而是：

```text
blocking/persistent I/O runtime 必须拥有与 application compute 隔离的执行资源；
application 和 runtime 通过 GPU-visible request/completion state 通信；
runtime 正常只在 shutdown 时退出。
```

当前 V100/CUDA 12.2 不能直接使用 AGIO 的 Green Context 实现。已验证普通 stream
路径下 hardware work queue connections 和 service stream priority 不足以保证
resident service 跨 forward 共存；因此 direct 默认对齐旧 one-copy 的 idle-stop
生命周期。未来若迁移到支持 Green Context/MPS partition 的平台，再重新评估
`resident` 生命周期。
