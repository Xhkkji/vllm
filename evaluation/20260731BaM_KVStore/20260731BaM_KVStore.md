# BaM Direct KVStore 当前实现、问题与推进方案

日期：2026-08-01

## 1. 文档定位

本文记录 `vllm-bam` direct KVStore 主线的当前有效状态。内容按以下顺序组织：

1. 已经实现并验证的功能。
2. 真实 Qwen2.5 preemption 负载遇到的问题。
3. 已排除的假设和已经收束掉的临时逻辑。
4. 与旧 BaM one-copy 的准确对比。
5. AGIO 的源码级执行模型及其对当前问题的直接启示。
6. Hyperion 的异步 IO、CUDA IPC 与 MPS 参考。
7. 当前 V100 保底策略，以及独立 BaM daemon 的后续验证边界。

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
6. **长期隔离模式下正常 batch 完成只 retire request，不停止 resident runtime**；
   当前 V100 因缺少可靠资源隔离，仍采用 `io_active` 的 idle-stop 保底策略。

### 10.2 不直接照搬

- 不引入 AGIO 的完整 megakernel、allocator 和第二套 NVMe controller。
- 不替换当前已验证的 BaM SQ/CQ、CID 和 DMA mapping。
- 不在 direct 主线再叠加一套复杂的 AGIO software SQ/CQ，除非后续要让 resident
  runtime 同时接管 GPU submit；当前问题首先是执行资源冲突。
- 不引入 CPU completion thread、CPU CQ poll 或 per-batch service stop。
- 不直接复制 Green Context 代码到当前 V100 环境。

### 10.3 Hyperion 的相关参考

本地参考实现：

```text
/home/xhk/hyperion/Hyperion/README.md
/home/xhk/hyperion/Hyperion/IOStack/iostack.cuh
/home/xhk/hyperion/Hyperion/sampling_server/src/engine/ipc_service.cu
/home/xhk/hyperion/Hyperion/training_backend/ipc_cuda_kernel.cu
```

Hyperion 的关键点不是把 completion 做成永久 resident kernel，而是把提交和完成
分开：

```text
io_submission(micro-batch 1)
  -> application compute
io_submission(micro-batch 2)
  -> application compute
io_submission(micro-batch 3)
  -> io_completion()
```

其 IOStack 配置使用少量 submission blocks 和有限的 completion blocks；
`io_completion()` 在消费数据前处理此前累计的请求，completion kernel 完成后退出。
这与当前 `io_active` 的“IO 活跃期保留 service、空闲后退役”方向一致，说明细粒度
预取不要求所有请求完成后才能发起 attention，真正需要的是 request/block 级 ready
和分阶段 completion，而不是全局 batch barrier。

Hyperion 还实现了跨进程 GPU buffer 共享：server 通过
`cudaIpcGetMemHandle()` 发布 GPU allocation，consumer 通过
`cudaIpcOpenMemHandle()` 打开；共享内存只保存 handle，命名 semaphore 和多槽
pipeline 负责 buffer 的写入、消费和复用生命周期。这是独立 BaM daemon 方案可参考
的控制面和数据面骨架。

Hyperion README 还给 training backend 设置了：

```text
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=80
```

这说明“独立 CUDA client + MPS 执行资源限制”在同类 GPU 直达异步 IO 系统中有实际
使用依据。但这不是当前 direct daemon 已经验证成功的证据，仍需单独验证 BaM direct
映射和 completion 语义。

### 10.4 独立 BaM daemon + MPS 的判断

该方案有希望从根上缓解当前 V100 问题，因果链应当是：

```text
独立 BaM daemon CUDA context
  + MPS 限制 daemon 的 resident poll 执行资源
  + vLLM 使用另一 MPS client 执行 attention
  -> poll 不再占满 application 可用执行资源
  -> attention 能够获得稳定调度
```

必须区分三个条件：

1. 仅拆成独立进程不保证隔离；没有 MPS 或其他资源限制时，两个 CUDA client 仍可能
   竞争同一 GPU 执行资源。
2. MPS active-thread percentage 不是 Green Context 那样的固定 SM 编号分区，但在
   Volta 上有机会限制 resident kernel 的并发资源占用，值得优先做 PoC。
3. CUDA IPC 只解决 GPU allocation 共享，不自动解决当前 BaM 使用的 GPU 地址、DMA
   mapping、completion event 和 buffer 生命周期问题。daemon 打开 IPC handle 后，
   必须确认其 CUDA context 中的地址能够被 BaM direct IO 路径正确使用。

因此长期目标不是简单地把当前 service thread 搬到另一个进程，而是：

```text
vLLM
  -> 通过共享内存/Unix socket 发布 block IO descriptor
  -> 通过 CUDA IPC 共享固定 KV buffer
  -> 按 block/request 等待 completion

BaM daemon
  -> 打开 CUDA IPC buffer
  -> 保留后台 GPU poll
  -> 写入 completion ring 或 IPC event
```

这个架构仍然支持连续请求和细粒度预取。attention 只等待当前依赖的 KV blocks，
不等待 daemon 清空全部请求队列。

### 10.5 是否必须拆出整个 BaM

不一定要把整个 BaM_IOStack 拆成独立 daemon，但必须区分 CPU thread、CUDA stream
和 CUDA 执行资源域：

```text
当前同一进程、同一 CUDA primary context
  service stream  -> persistent CQ poll
  PyTorch streams -> attention compute
```

MPS active-thread percentage 的资源配额作用于 CUDA client/context，不作用于单个
CPU thread 或单个 CUDA stream。因此当前结构即使开启 MPS，也不能在同一 context
内得到 `service stream=10%`、`attention streams=90%` 的隔离。已经失败的
hardware connections 和 stream priority 实验也说明普通 stream 调度参数不能提供
所需边界。

当前有三条可选路线：

| 路线 | 进程边界 | GPU poll 形态 | 隔离性质 | 当前状态 |
|---|---|---|---|---|
| `io_active` | 同进程/同 context | IO 活跃期 resident，空闲后停止 | 生命周期避让 | 已验证可运行 |
| finite poll slice | 同进程/同 context | host service 常驻，GPU poll kernel 周期退出并重启 | 软件调度点 | 尚未验证 |
| 独立 service execution domain | 独立 context，优先采用独立进程 | GPU poll 可永久 resident | MPS 执行资源限制 | synthetic/PyTorch 调度层已验证 |

finite poll slice 不要求所有请求完成后才能执行 attention。后台 host service 可以
持续接收请求，每个 GPU poll kernel 只运行固定轮数或固定 cycles，退出后立即按活跃
请求重新 launch；request/block ready 后即可唤醒对应消费者。它保留的是逻辑常驻
service，而不是单个永不退出的 GPU kernel。优点是无需跨 context 共享 KV allocation，
代价是增加 kernel relaunch 开销，并且没有硬件级隔离保证。

同一进程创建两个 CUDA context 后，理论上也可以使用 Volta MPS 做 context 级资源
限制，因此“独立进程”不是硬性条件。但 PyTorch KV allocation 属于 primary context，
第二个 service context 仍需解决跨 context memory、request/completion queue、GPU
地址和 DMA mapping；还会引入 CUDA context 切换和 PyTorch runtime 交互风险。这些
问题与独立 daemon 的 CUDA IPC 数据面复杂度接近，但生命周期更隐蔽。

因此准确判断是：

```text
不是必须拆出整个 BaM；
严格永久 resident poll 必须拆出独立 CUDA 执行资源域。
```

若采用 daemon，只需要拆出 persistent CQ service、其 CUDA context 和必要的
request/completion 数据面。vLLM 的 KV block table、请求调度、引用计数和策略层仍
保留在当前进程，不需要把整个 BaM_IOStack 或 KVStore 生命周期都迁出去。

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

当前已完成的普通 stream 验证如下：

- `CUDA_DEVICE_MAX_CONNECTIONS=8`、`CUDA_LAUNCH_BLOCKING=0` 仍在第一次
  `before_forward` 停止。
- service stream 设置最低优先级仍在同一位置停止。
- batch DONE 后停止空转 service，能够继续到 `after_forward`、`swap_in/read DONE`
  和 `Run summary`。

因此目前证据支持：问题不只是 connection 数量或 stream priority，而是 V100 上
resident poll 与 PyTorch attention 的执行资源/调度共存不可靠。旧 one-copy 能跑通，
说明应继续对比其“IO 活跃期 service、空闲阶段停止”的生命周期条件。

### 11.2 已收束：普通 stream 调度参数

`CUDA_DEVICE_MAX_CONNECTIONS=8` 和最低优先级 service stream 均未恢复
`after_forward`。它们可以保留为诊断或兼容配置，但不能作为 resident service 与
attention 隔离方案。

### 11.3 已收束：生命周期条件

停止 batch 完成后的空转 service 可以恢复完整 Qwen2.5 direct KVStore 链路。因此
当前 V100 的可交付策略是：IO 活跃期间保留后台 poll，IO batch 空闲后退役 service，
下一次 submit 再启动。这个策略不破坏 IO 活跃期的后台常驻语义，也不要求所有未来
请求排队到当前请求完全结束后才能计算。

### 11.4 对齐旧 one-copy 的保留检查

以下检查仍作为 direct 与旧 one-copy 的链路对照项保留，不再作为普通 stream 调度
参数实验的前置条件：

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

### 11.5 中长期：独立执行资源域与 MPS PoC

如果目标是恢复真正的 resident service，同时允许 attention 持续推进，优先级应为：

1. 用两个独立 CUDA client 做最小 persistent-poll/compute 共存 PoC。
2. 在 MPS 下给 poll client 设置较小配额、给 compute client 设置主要配额，观察
   compute 是否稳定推进。
3. 再验证固定 GPU buffer 的 CUDA IPC：一个进程导出 allocation，另一个进程打开并
   完成一次 BaM direct IO。
4. 最后才把 request ring、completion ring、KV block 引用计数接入 vLLM。

前两步的调度 PoC 已于 2026-08-01 完成。测试没有修改 BaM、vLLM 或 direct 逻辑，
只使用 `/tmp/bam_mps_isolation_probe.cu` 构造独立 synthetic poll/compute client。
环境：

```text
GPU=Tesla V100S 32GB, 80 SM
driver=535.230.02
CUDA=12.2
MPS control daemon=user xhk
GPU compute mode=EXCLUSIVE_PROCESS
```

MPS 配额在 V100 上确实生效：

```text
poll client:    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=10 -> 可见 8 SM
compute client: CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=90 -> 可见 72 SM
```

关键结果：

| 场景 | poll 配置 | compute 结果 |
|---|---:|---:|
| synthetic compute baseline | 无 poll | `494.823 ms` |
| synthetic full-resource 对照 | `80 blocks x 1024 threads`, poll 100% | `450.113 ms`，未等待 poll 退出 |
| synthetic 10/90 配额 | `8 blocks x 1024 threads`, poll 10% | `416.099 ms`，未等待 poll 退出 |
| PyTorch GEMM baseline | 无 poll，compute 90% | `0.572 s` |
| PyTorch + 重负载 poll | `8 blocks x 1024 threads`, poll 10% | `5.145 s`，能完成但干扰明显 |
| PyTorch + direct 同规模 poll | `1 block x 128 threads`, poll 10% | `0.565 s`，先于 poll 完成 |

当前 direct persistent CQ service 的实际 launch 是 `1 block x 128 threads`。在与其
相同规模的独立 poll client 持续运行时，PyTorch 50 次 `4096 x 4096` FP32 GEMM
用时与 baseline 基本一致。这提供了直接证据：

```text
独立 CUDA client
  + MPS 10% poll / 90% compute
  + 受控的单 CTA persistent poll
  -> PyTorch compute 能持续获得调度
  -> 未复现同进程普通 stream 的永久 before_forward 阻塞
```

同时，重负载 poll 的 `5.145 s` 结果说明 MPS active-thread percentage 不是 Green
Context 那样的完全无干扰固定 SM 分区；daemon 的 block/thread 数仍必须严格受控。
当前 `1 x 128` 配置满足这一条件，但后续扩展 queue/service 并行度时必须重新测量。

调度层已经验证，数据面仍未验证。下一步不是修改当前 direct 主线，而是验证固定 GPU
buffer 的 CUDA IPC，以及 daemon context 中的 BaM GPU 地址/DMA mapping：

1. 一个进程导出 CUDA allocation，另一个进程打开并完成 GPU buffer roundtrip。
2. daemon 对该 IPC buffer 完成一次 BaM direct IO。
3. 验证 completion event、buffer 生命周期和数据正确性。
4. 最后才把 request ring、completion ring、KV block 引用计数接入 vLLM。

在当前 CUDA 12.2 header 缺少 Green Context API 的前提下，不直接移植 AGIO Green
Context。MPS 的调度隔离能力已验证，但不能据此宣称独立 BaM daemon 的 CUDA IPC
和 NVMe DMA 数据面已经成立。

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

中长期 daemon+MPS 验证不进入当前 direct workload 的默认测试路径，单独按以下
顺序进行：

```text
独立 persistent poll 进程 + compute 进程  [完成]
  -> MPS 10/90 配额对比                     [完成]
  -> direct 同规模 poll + PyTorch compute   [完成]
  -> CUDA IPC 固定 GPU buffer roundtrip     [待验证]
  -> BaM direct IO roundtrip
  -> request/block completion pipeline
```

调度阶段的成功标准已经满足：PyTorch compute 在 direct 同规模 resident poll 持续
运行时稳定完成，耗时与 baseline 基本一致。完整 daemon 方案仍需 IPC buffer 数据、
BaM DMA mapping 和 completion 顺序正确；不能只以 MPS 下 compute 能运行为完整成功。

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

### 13.1 长上下文排查纪律

- 不恢复旧会话上下文，不读取 `~/.codex/sessions/*.jsonl`，不依赖旧 rollout JSONL
  或超大历史；只以当前交接文档、本文和当前链路的必要证据恢复状态。
- 每轮只执行一个目标明确的读取命令。执行前说明该命令要回答的问题，优先使用窄
  范围 `sed -n`、精确 `rg --max-count`、`head` 或 `tail`。
- 禁止无边界的全仓库 `rg`/`find`，禁止读取长日志全文，禁止为“完整性”展开与当前
  调度、completion、buffer 生命周期无关的实现。
- 一旦证据足够，立即停止读取并记录：事实、判断、未证实假设和下一步；不重复读取
  已确认内容，也不把早期假设继续当作当前结论。
- 读取日志时只保留关键事件窗口：`write/read DONE`、`before_forward`、
  `after_forward`、`Run summary`、错误和退出状态，并记录日志路径与测试环境。
- 用户要求“只回答问题、只总结思路”时不修改代码；需要修改或运行测试时，先明确
  影响范围，再执行最小必要动作。

### 13.2 权限、进程与系统状态

- 测试脚本一次只运行一个。
- 需要 `sudo`、加载/卸载内核模块、修改系统配置、启动/停止系统级服务或执行其他
  破坏性操作时立即停止，只给出用户手动执行的命令和权限配置步骤。
- 只有用户明确授权的测试进程清理可以执行；清理时必须按授权 wrapper 或精确 PID
  操作，不能使用模糊的全局 kill，并在结束后确认没有残留测试进程。
- 权限不足立即停止，并给出最小 sudoers/wrapper 命令；不通过扩大权限范围绕过问题。
- 不改 LMCache SSD、GDS 和旧 BaM one-copy baseline。
- direct 正常路径采用 `io_active` 生命周期；保留后台 poll 的 IO 活跃期常驻语义，
  但不在 V100 上让空转 service 跨 PyTorch forward。
- 新 direct 路径注释继续使用 `【BaM KVStore 直通调用链】` 标记。

## 14. 当前阶段结论

Direct KVStore 的真实 SSD 数据面已经成立：GPU submit、NVMe DMA 直接读写 vLLM
KV cache、GPU CQ completion 和 CPU 只读 DONE 均已验证。当前 V100 可运行方案是
`io_active` 生命周期：后台 persistent poll 在 IO 活跃期保留，batch 空闲后退役，
避免空转 service 与真实 Qwen2.5 attention 争用执行资源。

当前判断分成三层：

```text
短期 V100 保底：io_active
  IO 活跃期保留后台 poll
  batch 空闲后停止空转 service
  保证 Qwen2.5 direct KVStore 链路能跑通

同进程备选：finite poll slice
  host service 逻辑常驻
  GPU poll kernel 周期退出并重启
  为 attention 建立明确调度点
  尚未验证性能和稳定性

严格 resident 目标：独立 CUDA execution domain
  优先实现为最小 BaM service daemon + MPS
  daemon 保留永久后台 GPU poll
  MPS 限制 daemon 执行资源
  CUDA IPC 共享固定 KV buffer 和 completion 状态
  vLLM 按 block/request 消费 ready 数据
```

AGIO 给出的最重要结论不是“给 poll 加 sleep”，而是：

```text
blocking/persistent I/O runtime 必须拥有与 application compute 隔离的执行资源；
application 和 runtime 通过 GPU-visible request/completion state 通信；
runtime 正常只在 shutdown 时退出。
```

Hyperion 进一步说明，独立进程方案需要 CUDA IPC buffer、多槽 pipeline 和明确的
producer/consumer 生命周期；它的 completion 也是阶段性启动，而不是无边界地占用
GPU。当前 V100/CUDA 12.2 不能直接使用 AGIO 的 Green Context 实现，普通 stream
connections 和最低优先级也不足以保证 resident service 跨 forward 共存。因此
direct 默认采用 `io_active`，而 `resident` 只保留为实验模式。

独立 daemon + MPS 的调度层已在当前 V100 上完成最小验证：10% poll client 实际只
看到 8 SM，90% PyTorch client 看到 72 SM；与 direct 相同的 `1 x 128` persistent
poll 持续运行时，PyTorch GEMM 为 `0.565 s`，与 `0.572 s` baseline 基本一致。
因此该方向比同一进程内调整 thread 或 stream 更接近当前故障根因，并且已经证明有
能力解除计算永久阻塞。

这不表示必须拆出整个 BaM_IOStack。真正必须拆开的是 GPU 执行资源域；只拆
persistent CQ service 是首选边界。若不要求单个 GPU poll kernel 永久 resident，
也可以先验证同 context 的 finite poll slice，以较小改动换取软件调度点。

尚未验证的是独立 BaM daemon 的数据面：CUDA IPC allocation、daemon context 中的
GPU 地址、NVMe DMA mapping、跨进程 completion 和 KV block 生命周期。只有最小
CUDA IPC/BaM roundtrip 通过后，才能把该方向提升为 direct `resident` 的可替代实现。
