# GPU-initiated BaM 实现思路

日期：2026-06-25

最近整理：2026-06-30

本文记录 `vllm-bam` 中 BaM 接入 LMCache/vLLM KVCache 的当前状态、已经跑通的路径，以及后续参考 AGIO / Tutti / TARDIS 推进 GPU-initiated asynchronous I/O 的主线。

这版文档只保留当前仍有工程价值的路线。早期过渡性方案、已经证明不是主线的临时路径，只在“归档/不再推进”中保留结论，避免后续实现时被旧路线干扰。

建议阅读顺序：

```text
1. 先看“1. 当前结论”和“1.2 当前最新口径（2026-06-30）”
2. 再看“3. KVCache 数据组织”和“4. 当前调用流程”
3. 再看“12.7 当前性能结论”和“12.7.1 当前 direct placement 的实际数据通路”
4. 最后看“13. 下一阶段改进路线图（2026-06-30）”
```

## 1. 当前结论

当前主线已经从“把 KVCache 当成通用 feature 读写”推进到“KVCache 专用 fast path + executor 分层”。

已经跑通或验证过的路径：

```text
LMCache 原生 SSD baseline
LMCache-style GDS replay baseline
BaM sync 读写路径
BaM page-level prefetch/refill 路径
BaM KV fast path replay
BaM KV fast path batch replay
真实 vLLM + LMCache + BaM prefer-load + KV fast path
```

当前真实系统分工：

```text
CPU 控制面:
  vLLM scheduler
  LMCache prefix/chunk lookup
  chunk metadata 管理
  batch request table 构造
  当前仍调用 submit / poll / consume
  Triton refill kernel launch

GPU/BaM 数据面:
  BaM page read
  BaM page cache / DMA 数据路径
  GPU pages buffer
  GPU-visible request/status tensor
  Triton refill 数据转换
```

当前状态不是最终 GPU-initiated，但已经具备下一步接 GPU worker 的接口基础：

```text
BaMKVStore
  -> native_executor
     -> BaMRowCtxKVExecutor       # 默认，稳定可跑
     -> BaMGPUWorkerKVExecutor    # 当前主线，底层 worker_backend=kv_cq_service_v1

环境变量:
  VLLM_BAM_KV_EXECUTOR=rowctx
  VLLM_BAM_KV_EXECUTOR=gpu_worker
```

一句话主线：

```text
不要继续在 Python early-prefetch 上堆复杂度；
保留现有 rowctx 稳定路径；
让 gpu_worker 从 fallback 变成真实 KV worker 入口；
再把 submit / poll / completion / refill 逐步下沉到 GPU；
最终减少 LMCache tensor 中转，直接回填 vLLM paged KV cache。
```

### 1.2 当前最新口径（2026-06-30）

如果只看当前最重要的工程结论，可以先记住下面这组口径：

```text
1. 当前版本已经真实跑到我们实现的 direct placement / merged refill 逻辑。

2. 之前 direct placement 失败的一个直接原因是：
   BaM cache 默认只有 64MB，
   在 4 chunk / 448 pages 的真实 batch read 下太小，
   会触发 submit_error_code=1:
     BAM_SUBMIT_ERR_FIND_SLOT_TIMEOUT

3. 把 BaM cache 默认调到 512MB 后，
   submit 路径已经稳定穿过，
   能完整走到：
     PREFIX_HIT
     READ_BEGIN
     MERGED_REFILL_STEP
     DIRECT_PLACEMENT

4. merged refill 的真正热点不是“4 个 step 都慢”，
   而是首个 step 的一次性 Triton/JIT 初始化成本。

5. 给 merged refill 补 warmup 后，
   当前 steady-state placement 已经降到：
     read_ms    ≈ 12.603
     refill_ms  ≈ 1.021
     transfer_ms≈ 0.769
     place_ms   ≈ 1.790
     request_2_elapsed_s ≈ 2.0111

6. 因此当前 direct placement v0 的主要结论是：
   placement/refill 这段已经基本打通，
   后续瓶颈不再主要在 merged refill，
   而应更多转向 read 侧、rebuild/prefix 侧，以及
   “BaM pages -> final vLLM KV cache” 的进一步收缩。
```

### 1.2.1 当前 wave 收口方式（2026-07-04）

direct placement 这条线最近又往 GPU-initiated 主线收了一步，重点不是改数据格式，而是先把“等待边界”收紧：

```text
旧逻辑：
  start_batch()
    -> execution.wait()
    -> torch.cuda.synchronize(device)
    -> 整个 wave 结束

新逻辑：
  start_batch()
    -> execution.advance_ready()
    -> execution.wait_until_launched_range_cache_ready()
    -> 只轮询当前 wave 自己的 completion events
    -> 当前 wave 结束
```

这一步的意义：

```text
1. 不再用整卡 synchronize 把本 wave 之外的 CUDA 工作一起卡住
2. execution 层已经显式具备两类等待语义：
   - wait_until_launched_range_cache_ready()
   - wait_until_contiguous_cache_ready(target_chunks)
3. store 层仍然保持“本请求内同步收口”，因此当前主线不会引入
   “返回后后台继续改写同一份 kv_cache” 的竞态
4. 但后续如果要继续推进真正的 chunk-ready -> chunk-consumable，
   可以直接复用这层 execution 接口，而不必重新拆 direct placement 主流程
```

一句话理解：

```text
当前还不是“完全异步返回”，
但已经从“整卡同步等待”收成了“只等本 wave 的 completion event”，
这是继续往 GPU-initiated 推进时一个更安全、也更贴近主线的中间态。
```

2026-07-06 进一步收敛后的当前主线语义：

```text
当前请求命中了多少连续 prefix
  -> 先把这段 prefix 长度显式记成 return target
  -> direct placement 仍然可以有自己的 launch 范围
  -> 但 store 主路径优先等待：
       contiguous cache-ready frontier >= return target
  -> 再生成 ret_mask 返回给 LMCache / vLLM
```

这意味着当前主线已经不再优先按“本 wave launch 了多少 chunk”来决定返回时机，
而是优先按“当前请求准备返回给正常推理引擎多少 prefix”来收口。

对接现成推理框架时，应该把它理解成：

```text
ret_mask 语义
  == 当前这轮真实恢复完成、并且可以立刻被 attention 消费的连续 prefix
```

而不是：

```text
ret_mask 语义
  == 这次内部 launch 了哪些 chunk
```

2026-06-28 最新判断：

```text
BaM KV I/O 已经不是当前主要瓶颈。

最新 replay 中：
  gpu_worker + kv_cq_service_v1:
    batch_size=8
    read_ms=1.015
    total_ms=3.327
    amortized_ms=0.438/chunk
    mean_bw_gib_s=31.243

  rowctx:
    batch_size=8
    read_ms=1.321
    total_ms=4.925
    amortized_ms=0.653/chunk
    mean_bw_gib_s=20.938

真实 vLLM 中：
  batch_size=7
  read_ms=1.324
  refill_ms=427.900
  request_2_elapsed_s=2.0261

结论：
  kv_cq_service_v1/gpu_worker 已经优于 rowctx。
  后续不应继续死扣 SQ/CQ 的小优化。
  主线应转向减少 refill/rebuild：让 BaM 读出的 KV 直接落到 vLLM paged KV cache。
```

### 1.1 当前 baseline 固化

2026-06-28 当前应固化的 replay baseline：

```text
backend=bam_kv_fast_path_batch
VLLM_BAM_KV_EXECUTOR=gpu_worker
worker_backend=rowctx_compat  # 该 baseline 跑于 kv_cq_service_v1 接入前
NUM_CHUNKS=8
shape=[2, 28, 256, 512]
page_bytes=128KB
pages_per_chunk=112
```

最新稳定结果：

```text
batch_size=8
submit_ms=0.217
poll_ms=0.238
get_ms=0.571
read_ms=1.117
refill_ms=1.011
total_ms=2.901
bw_gib_s=37.702

amortized read:
  0.382 ms/chunk
  35.772 GiB/s
```

这组结果对应日志：

```text
evaluation/logs/bam_vs_gds_trace_replay/20260628_042744/run.log
```

2026-06-28 `kv_cq_service_v1` replay 验证结果：

```text
backend=bam_kv_fast_path_batch
VLLM_BAM_KV_EXECUTOR=gpu_worker
worker_backend=kv_cq_service_v1
NUM_CHUNKS=8
shape=[2, 28, 256, 512]

batch_size=8
submit_ms=0.211
poll_ms=0.081
get_ms=0.555
read_ms=0.856
refill_ms=0.953
total_ms=2.459
bw_gib_s=44.483

amortized read:
  0.324 ms/chunk
  42.259 GiB/s
```

对应日志：

```text
evaluation/logs/bam_vs_gds_trace_replay/20260628_044913/run.log
```

结论：

```text
kv_cq_service_v1 已接通 replay。
worker_backend 已从 rowctx_compat 变成 kv_cq_service_v1。
默认性能路径不读回 GPU debug table，因此 chunk_gpu_status/completion_status 为 none。
正确性无报错，性能未回退。
```

2026-06-28 真实 vLLM + LMCache prefer-load 验证结果：

```text
日志:
  evaluation/logs/single_gpu_lmcache_no_prefix_reuse_qwen25/20260628_045413/run.log

配置:
  VLLM_BAM_LMCACHE_SHADOW_ENABLE=1
  VLLM_BAM_LMCACHE_PREFER_LOAD_ENABLE=1
  VLLM_BAM_KV_FAST_PATH=1
  VLLM_BAM_KV_EXECUTOR=gpu_worker

关键路径:
  [LMCACHE_BAM_KV_FAST_PATH_PREFETCH_ENQUEUE]
  [LMCACHE_BAM_KV_FAST_PATH_BATCH_READ]
  executor=gpu_worker worker_backend=kv_cq_service_v1
  [LMCACHE_BAM_KV_FAST_PATH_BATCH_CONSUME] hit=True
  [LMCACHE_BAM_VERIFY] exact_equal=True
  [LMCACHE_BAM] prefer-load hit
  [LMCACHE_REBUILD]
  [XFORMERS_PREFIX]

batch read:
  batch_size=4
  submit_ms=0.702
  poll_ms=0.126
  get_ms=0.412
  read_ms=1.250
  refill_ms=462.512
  total_ms=464.142

端到端:
  request_1_elapsed_s=1.8031
  request_2_elapsed_s=2.0717
```

结论：

```text
真实 vLLM 正式路径已经走到 kv_cq_service_v1。
BaM KV fast path 的数据正确性通过 exact_equal=True。
request_2 已经通过 LMCache rebuild / XFormers prefix 路径使用 retrieve 到的 KV。
当前正式路径里的 462ms refill 仍主要是首次 batch refill/JIT/初始化口径，不代表稳态 BaM IO。
```

对比曾经尝试的“CPU poll 默认读取 GPU status / completion table”版本：

```text
total_ms=4.075
bw_gib_s=26.839
```

结论：

```text
CPU/Python poll 如果需要立即返回 bool/status，默认不能从 CUDA memory 读
gpu_status 或 completion_table。

即使只 D2H 读取 4 字节 status，也会同步 GPU 队列，破坏 1ms 级 replay 的
性能口径。

当前 baseline 固化为：
  CPU poll hot path 读取 C++ host-side record.status
  GPU status / chunk status / completion table 继续写入
  但只作为 debug、日志验证和未来 persistent GPU worker ABI
```

当前调试开关：

```text
VLLM_BAM_KV_DEBUG_STATUS=1
```

开启后 Python 会读回：

```text
gpu_status:        [1] int32 CUDA
gpu_chunk_status:  [batch] int32 CUDA
completion_table:  [batch, 4] int64 CUDA
```

该开关只用于定位问题，不应作为默认性能测试口径。

## 2. 三篇论文给出的约束

这次路线调整参考了 AGIO、Tutti、TARDIS 三类思路。它们共同强调：GPU-initiated 不只是“GPU 能发 I/O”，更重要的是 I/O 的异步化和控制面/数据面分离。

### 2.1 AGIO

AGIO 的核心启发：

```text
GPU I/O 要分离 initiation 和 completion。
GPU 发起 I/O 后不应该原地同步等待。
如果有计算可以 overlap，就让计算继续跑。
如果没有足够计算，也可以继续发更多 I/O 来提高并行度。
```

映射到 BaM KVCache：

```text
错误方向:
  GPU submit 一个 chunk
  GPU 原地等这个 chunk 完成
  完成后再 submit 下一个 chunk

正确方向:
  GPU 批量 submit 多个 chunk
  GPU 或 GPU worker 写 completion/status
  refill 消费 ready chunk
  submit / complete / refill 尽量流水化
```

### 2.2 Tutti

Tutti 的核心启发：

```text
CPU-prepared, GPU-executed。
CPU 负责 metadata、mapping、request preparation。
GPU 负责 I/O execution、completion queue、数据搬运。
SQ/CQ/request table 应该尽量 GPU-visible。
```

映射到 BaM KVCache：

```text
CPU:
  做 prefix/chunk lookup
  决定本轮要读哪些 chunk
  生成 request table
  做粗粒度调度和错误处理

GPU/BaM:
  consume request table
  submit BaM/NVMe I/O
  poll completion
  写 status/completion
  refill pages
```

### 2.3 TARDIS

TARDIS 的核心启发：

```text
KVCache 不应该被当成普通文件或任意 feature row。
KVCache 更像 GPU-centric mapped object。
读写接口应该围绕 KV chunk / KV block / KV cache layout 设计。
```

映射到 BaM KVCache：

```text
不要把 KVCache 长期硬塞进 read_feature_* 语义。
保留 GNN/CNN 通用 feature path。
新增 KVCache 专用 fast path。
后续直接面向 vLLM paged KV cache 做回填。
```

### 2.4 对当前路线的综合约束

三篇论文合并后，对我们当前工程的要求是：

```text
1. request table 可以由 CPU 准备，但必须 GPU-visible。
2. submit 和 completion 必须解耦，不能同步发起后原地等待。
3. CPU 不应逐 chunk/逐 completion 参与热路径。
4. KVCache 要走专用 object/chunk path，不继续套通用 feature path。
5. 最终目标是 GPU worker + completion/status table + fused refill。
```

### 2.5 对 refill/rebuild 的结论

Tutti / TARDIS / AGIO 给出的共同启发不是“把现有 refill kernel 优化一点”，
而是尽量避免读回后再走通用 rebuild/refill。

```text
当前 vLLM-BaM 正式路径:
  SSD
    -> BaM 128KB pages
    -> 中间 tensor [2, layers, tokens, hidden]
    -> Triton refill
    -> vLLM paged KV cache
    -> LMCache rebuild / XFormers prefix

目标路径:
  SSD
    -> BaM KV pages
    -> vLLM paged KV cache 目标 block
    -> attention 直接使用
```

论文思路映射：

```text
TARDIS:
  KVCache 更像 GPU-centric mapped object。
  关键是让存储对象和 GPU 侧 KV 对象直接对应，减少 CPU/框架级 rebuild。

Tutti:
  GPU-native KV object store + layer-wise I/O pipeline。
  关键不是 SSD 带宽本身，而是减少 CPU-centric I/O 发起、细粒度同步、
  以及恢复后重新接回框架的开销。

AGIO:
  initiation 和 completion 解耦。
  I/O 完成后应被后续 GPU 数据路径消费，而不是让 CPU 在每个阶段同步推进。
```

因此，下一阶段主线是：

```text
不要继续把 BaM 读出的 pages 还原成 LMCache 通用 tensor 后再 rebuild。
先做 direct placement：
  BaM pages -> GPU scatter/direct placement -> vLLM paged KV cache blocks。

再做 layer-wise pipeline：
  layer i 计算时预取 / 放置 layer i+1 或后续 layer 的 KV。

最后再做 persistent GPU worker：
  GPU 侧 submit / poll / completion / placement 形成闭环。
```

## 3. KVCache 数据组织

当前 vLLM-BaM 路径把 LMCache 的一个 KV chunk 组织成固定 128KB BaM page。

典型 Qwen2.5-7B fp16 chunk 形状：

```text
[2, 28, actual_tokens, 512]
```

含义：

```text
2              -> K/V
28             -> layer 数
actual_tokens  -> 当前 chunk 实际 token 数
512            -> hidden dim
dtype          -> float16
```

写入 BaM 前按固定 slot token 容量组织：

```text
[2, 28, 256, 512]
```

固定 slot 的原因：

```text
BaM 侧需要稳定的 chunk -> page 映射。
真实 token 不足 256 时只在逻辑上 pad。
metadata 仍记录 actual_tokens。
读回 refill 时只还原有效 token。
```

128KB page 映射：

```text
每个 token 向量大小 = 512 * 2B = 1024B
每个 128KB page 可容纳 = 128KB / 1024B = 128 tokens
每层 K 需要 2 个 page
每层 V 需要 2 个 page
一个完整 chunk = 2(K/V) * 28(layer) * 2(page/layer) = 112 pages
```

一个满 chunk 的物理组织：

```text
[112, 128KB]
```

page id 映射公式：

```text
bam_page_id =
    chunk_base_page
  + kv_id * num_layers * pages_per_kv_layer
  + layer_id * pages_per_kv_layer
  + token_page_id
```

例子：

```text
chunk_base_page = 784
kv_id = 0                 # K
layer_id = 3
token_offset = 150
page_token_capacity = 128
token_page_id = 150 // 128 = 1

bam_page_id = 784 + 0 * 28 * 2 + 3 * 2 + 1 = 791
```

当前 K 和 V 没有混在同一个 page 里，而是按如下顺序组织：

```text
K all layers/pages
V all layers/pages
```

这套组织简单、稳定，也贴合当前 LMCache chunk layout。后续如果直接回填 vLLM paged KV cache，再评估 K/V interleave 或 layer-wise layout 是否更适合 attention locality。

## 4. 当前调用流程

正式在线路径入口：

```text
vllm/distributed/kv_transfer/kv_connector/lmcache_connector.py
```

整体流程：

```text
vLLM scheduler
  -> LMCacheConnector 查询本次请求可 retrieve 的 prefix/chunk
  -> LMCache engine retrieve / prefetch
  -> LMCache storage backend get/put
  -> vllm/bam/lmcache_bam_storage.py wrapper
  -> BaM sync / prefetch / KV fast path
```

KV fast path 路径：

```text
LMCacheBaMStorageManager.get()
  -> LMCacheBaMStore.consume_kv_fast_path_tensor()
  -> LMCacheBaMStore.load_chunk_tensors_kv_fast_path_batch()
  -> vllm/bam/lmcache_bam_kv_fast_path.py
  -> BaM_IOStack/gids_module/bam_kv_store.py
  -> BaM_IOStack/gids_module/bam_row_store.py
  -> BaM_IOStack/gids_module/gids_nvme.cu
```

真实 vLLM 中 batch 收集流程：

```text
LMCache engine.prefetch(tokens, mask)
  -> storage_manager.prefetch(key)
  -> enqueue_kv_fast_path_prefetch_key()
  -> 收集本轮可能 retrieve 的 chunk keys

第一次 get(key)
  -> consume_kv_fast_path_tensor()
  -> 一次性 batch read pending keys
  -> 当前 key 命中后回填 LMCache memory_obj
```

当前仍然由 CPU 做调度决策。GPU-initiated 优化的是 I/O 读取链路本身，不是替代 vLLM scheduler 或 LMCache metadata lookup。

## 5. 已实现路径

### 5.1 BaM sync baseline

用途：

```text
验证 BaM 作为 SSD KV 后端的最小正确性。
作为后续 prefetch / KV fast path 的对照。
```

特点：

```text
CPU 调用 load_rows()
一次读取完整 chunk pages
读后 decode/refill 回 LMCache tensor
```

### 5.2 Page-level prefetch/refill

核心文件：

```text
vllm/bam/lmcache_bam_prefetch.py
vllm/bam/lmcache_bam_refill.py
vllm/bam/lmcache_bam_storage.py
```

数据流：

```text
LMCache chunk_hash
  -> BaMChunkMetadata
  -> BaMPageReadPlan(page_ids on GPU)
  -> BaMPageReadHandle(rowctx request + output pages)
  -> submit / poll / complete
  -> [page_count, 128KB] pages
  -> GPU refill
  -> [2, num_layers, actual_tokens, hidden]
```

定位：

```text
适合作为 correctness scaffold 和对照实验。
不再作为主线继续增加 Python early-prefetch 复杂度。
```

### 5.3 KV fast path batch

核心文件：

```text
BaM_IOStack/gids_module/bam_kv_store.py
BaM_IOStack/gids_module/bam_row_store.py
BaM_IOStack/gids_module/gids_nvme.cu
BaM_IOStack/gids_module/include/bam_nvme.h
vllm-bam/vllm/bam/lmcache_bam_kv_fast_path.py
vllm-bam/vllm/bam/lmcache_bam_storage.py
```

数据流：

```text
[(chunk_hash, metadata), ...]
  -> [BaMKVRequest(page_offset, page_count, actual_tokens), ...]
  -> BaMKVRequestTable [batch, 4] CUDA
  -> BaMRowCtxKVExecutor
  -> BaM rowctx batch read
  -> pages: [batch * 112, 128KB]
  -> GPU refill
  -> {chunk_hash: [2, 28, actual_tokens, 512]}
```

当前 request table：

```text
request_table: [batch, 4] int64 CUDA

每行:
  [chunk_id, page_offset, page_count, actual_tokens]
```

当前 status table：

```text
gpu_status:       [1] int32 CUDA
gpu_chunk_status: [batch] int32 CUDA

状态:
  INIT
  SUBMITTED
  IO_DONE
  CONSUMED
  ERROR
```

最新验证中已经出现：

```text
status=SUBMITTED->IO_DONE->CONSUMED
gpu_status=SUBMITTED->IO_DONE->CONSUMED
chunk_gpu_status=8xSUBMITTED->8xIO_DONE->8xCONSUMED
request_table=gpu
```

### 5.4 真实 vLLM + KV fast path

启用开关：

```text
VLLM_BAM_LMCACHE_SHADOW_ENABLE=1
VLLM_BAM_LMCACHE_PREFER_LOAD_ENABLE=1
VLLM_BAM_KV_FAST_PATH=1
```

当前行为：

```text
prefetch 阶段收集 keys
get 阶段触发 batch read
命中后 populate LMCache memory_obj
开启 verify 时可对比 LMCache 原始数据
```

已观察到的关键信号：

```text
[LMCACHE_BAM_KV_FAST_PATH_PREFETCH_ENQUEUE]
[LMCACHE_BAM_KV_FAST_PATH_BATCH_READ]
[LMCACHE_BAM_KV_FAST_PATH_BATCH_CONSUME] hit=True
[LMCACHE_BAM] prefer-load hit
[LMCACHE_BAM_VERIFY] exact_equal=True
[LMCACHE_REBUILD]
[XFORMERS_PREFIX]
```

结论：

```text
真实 vLLM 正式路径已经接通。
当前端到端收益还不稳定，主要因为 CPU 串行控制和首次 Triton JIT 仍在。
下一步应该推进 GPU worker，而不是继续堆 Python early-prefetch。
```

## 6. 当前 executor 分层

当前 `bam_kv_store.py` 中已经有两个 executor：

```text
BaMRowCtxKVExecutor
  当前默认执行层
  复用 BaM rowctx submit / poll / consume
  已验证稳定可跑

BaMGPUWorkerKVExecutor
  未来 GPU worker 的接口骨架
  当前安全 fallback 到 BaMRowCtxKVExecutor
  通过 VLLM_BAM_KV_EXECUTOR=gpu_worker 显式启用
```

这一步暂时不会带来性能变化。它解决的是代码组织问题：上层 `BaMKVStore` 只依赖统一 executor 接口，后续把 `gpu_worker` 从 fallback 改成真实 GPU worker 时，不需要再改 LMCache/vLLM 调用链。

当前验证口径：

```text
py_compile 通过
BaMKVStore / BaMRowCtxKVExecutor / BaMGPUWorkerKVExecutor import 通过
submit_native_batch / poll_native_batch / wait_native_batch / consume_native_batch 存在
VLLM_BAM_KV_EXECUTOR=rowctx 可以选择 rowctx executor
VLLM_BAM_KV_EXECUTOR=gpu_worker 可以选择 gpu_worker executor
```

## 7. BaM 底层可复用能力

BaM 底层已经有 GPU-side page cache / I/O primitive，不是纯 CPU read。

代表性 device-side 接口在：

```text
BaM_IOStack/bam/include/page_cache.h
```

已有能力：

```text
read()
read_submit_async()
read_wait_async()
read_single_thread_poll()
read_post_poll_light()
```

当前 rowctx kernel 位于：

```text
BaM_IOStack/gids_module/gids_kernel.cu
```

已有三段式思想：

```text
submit:
  read_feature_kernel_submit_async_rowctx()
  -> ptr.read_submit_async(...)

poll:
  read_feature_kernel_single_page_single_thread_poll_rowctx()

get:
  read_feature_kernel_get_feature_light_rowctx()
  -> ptr.update_page_post_poll_light(...)
```

这些能力可以继续复用，但 KVCache fast path 不应继续完全套用 `read_feature_*` 的通用 feature 语义。

## 8. GPU worker 设计与实现状态

下一步目标不是“再包装一层 Python”，而是让 `VLLM_BAM_KV_EXECUTOR=gpu_worker` 进入真正的 KV worker 路线。

### 8.1 GPU worker v0

目标：

```text
让 gpu_worker 不再只是 Python fallback。
新增独立 KV worker 底层入口。
第一版内部仍可复用 rowctx primitive。
外部语义必须是 KV worker。
```

建议接口：

```text
kv_worker_submit(request_table, gpu_status, gpu_chunk_status, completion_table)
kv_worker_poll(handle_or_status)
kv_worker_consume(handle, pages)
```

验收标准：

```text
VLLM_BAM_KV_EXECUTOR=rowctx
VLLM_BAM_KV_EXECUTOR=gpu_worker

两条路径都能跑通同一个 replay。
结果一致。
日志明确显示 executor=rowctx 或 executor=gpu_worker。
request_table=gpu。
默认性能路径不读回 GPU status table。
开启 VLLM_BAM_KV_DEBUG_STATUS=1 时，chunk_gpu_status / completion_table 正常推进。
```

当前实现状态：

```text
已完成 C++ KV worker façade：
  kv_worker_submit_from_table()
  kv_worker_poll_batch()
  kv_worker_poll_request(request_id)
  kv_worker_consume_batch()
  kv_worker_backend_name()

2026-06-28 已推进到 worker_backend=kv_cq_service_v1。

含义：
  submit 阶段仍复用稳定的 rowctx request submit。
  poll 阶段进入 KV worker 专用 CQ service façade。
  host-side record.status / GPU-visible status / completion_table 由统一 helper 更新。
  Python/vLLM 接口不变。

这还不是 persistent GPU worker。
它是把后续要替换的 CQ service 点从通用 rowctx 语义中拆出来。
```

当前 GPU-visible 表：

```text
request_table:     [batch, 4] int64 CUDA
gpu_status:        [1] int32 CUDA
gpu_chunk_status:  [batch] int32 CUDA
completion_table:  [batch, 4] int64 CUDA

completion_table 每行:
  [chunk_id, status, bytes_done, error_code]
```

2026-06-28 清理：

```text
BaMKVStore 中 executor 化之前的旧 native helper 已清理。
当前主线统一为：

BaMKVStore
  -> BaMRowCtxKVExecutor 或 BaMGPUWorkerKVExecutor
  -> BaMRowStore.kv_worker_* / kv_submit_*
  -> C++ kv_cq_service_v1 或未来 persistent GPU worker

新增 request-level poll façade：
  Python 调 kv_worker_poll_request(request_id)
  C++ 内部通过 KV worker CQ service 推进一次 request
  返回 host-side record.status，避免 D2H 同步

2026-06-28 继续推进后的结论：
  kv_worker_poll_request() 已成为 request-level poll façade。
  Python 不再依赖 rowctx 返回 ready_id 的细节。
  默认性能路径返回 C++ host-side record.status。
  GPU-visible status/completion table 继续写入，但不参与默认 CPU poll。

为什么不在默认 poll 中读取 GPU status/completion table：
  当前 CPU/Python poll 必须立即返回 bool/status。
  读取 CUDA memory 中的 gpu_status 或 completion_table 会触发 D2H 同步。
  即使只有 4 字节，也会让 1ms 级 replay 明显变慢。
  因此 GPU table 只作为 debug、日志校验和未来 persistent worker ABI。

相关调试开关：
  Python 层:
    VLLM_BAM_KV_DEBUG_STATUS=1
  C++ 层:
    GIDS_KV_POLL_FROM_GPU_STATUS=1
    GIDS_KV_REDUCE_COMPLETION_STATUS=1

上述开关只用于验证 GPU table，不应作为默认性能口径。
```

### 8.2 GPU-visible SQ/CQ

参考 Tutti，下一步要从单纯 status tensor 演进到更明确的 queue/table：

```text
KV request table / submission queue:
  chunk_id
  page_offset
  page_count
  actual_tokens
  output_page_offset
  flags

KV completion queue:
  request_id
  chunk_id
  status
  bytes_done
  error_code
```

CPU 只做：

```text
prepare request table
launch worker / ring doorbell
粗粒度 query batch status
error handling
```

GPU worker 做：

```text
read request table
submit BaM/NVMe IO
poll completion
write CQ/status
```

### 8.3 异步化

参考 AGIO，关键不是“GPU poll”本身，而是 submit 和 completion 解耦。

目标流水：

```text
batch N:
  GPU worker submit IO

batch N-1:
  GPU worker / refill kernel consume ready pages

vLLM 当前计算:
  尽量与上一批 IO overlap
```

避免的错误形态：

```text
submit chunk
wait chunk
refill chunk
submit next chunk
```

### 8.4 直接回填 vLLM paged KV cache

长期目标：

```text
BaM pages
  -> vLLM paged KV cache blocks
```

当前仍是：

```text
BaM pages
  -> LMCache tensor
  -> vLLM 使用
```

这一步收益大，但会碰 vLLM KV layout 和 attention 路径。建议在 GPU worker + status/completion 稳定后再做。

## 9. 分阶段实施计划（已完成与进行中）

阶段 1：KV 专用 batch read microbench。

```text
状态：已完成

输入:
  N 个 chunk 的 page_offset / page_count / actual_tokens

输出:
  [N, 112, 128KB] pages
```

阶段 2：接 Triton refill。

```text
状态：已完成

[N, 112, 128KB] pages
  -> refill_pages_to_lmcache_tensor()
  -> [N, 2, layers, tokens, hidden]
```

阶段 3：接 LMCache/vLLM。

```text
状态：已完成第一版

LMCache prefer-load hit
  -> KV fast path batch read
  -> consume ready KV
  -> populate memory_obj
```

阶段 4：GPU-visible request/status table。

```text
状态：第一版已完成

request_table: [batch, 4] CUDA
gpu_status: [1] CUDA
gpu_chunk_status: [batch] CUDA
```

阶段 5：executor 分层。

```text
状态：已完成

BaMRowCtxKVExecutor
BaMGPUWorkerKVExecutor
VLLM_BAM_KV_EXECUTOR=rowctx/gpu_worker
```

阶段 6：真实 GPU worker v0。

```text
状态：已完成第二版，当前 backend=kv_cq_service_v1

gpu_worker 已走独立 KV worker façade：
  kv_worker_submit()
  kv_worker_poll()
  kv_worker_consume()

kv_cq_service_v1 仍复用稳定的 rowctx 底层 primitive。
但 poll/service 语义已经从“通用 rowctx ready_id”拆成“KV request lifecycle”。

下一次 replay 验收日志应出现：
  executor=gpu_worker worker_backend=kv_cq_service_v1
```

阶段 7：GPU-side completion/poll。

```text
状态：已完成第一步，待继续下沉

已经替换 C++ kv_worker_poll_batch() 的内部入口：
  旧: kv_worker_poll_batch() -> kv_try_poll_batch() -> rowctx_compat poll
  新: kv_worker_poll_batch() -> kv_worker_service_cq_once()

kv_worker_service_cq_once() 当前仍复用：
  service_registered_completions_burst()
  registered_request_ready_at()
  iostack.mark_front_ready()

但状态推进已经统一到：
  kv_mark_record_status_and_tables()

当前已经新增 kv_worker_poll_request(request_id)，把 Python 对 ready_id/host map
细节的依赖下沉到 C++。默认 request-level poll 返回 C++ host-side
record.status，避免 D2H 同步；GPU-visible status/completion table 保留为
可选验证路径和未来 worker ABI。

下一步继续推进：
  当前: CPU 调 kv_worker_poll_request()，C++ 内部 KV CQ service 复用 rowctx primitive
  目标: GPU worker 自己 service CQ 并写 completion_table，CPU 只低频 query
  再下一步: persistent worker 常驻 GPU，进一步减少 CPU poll 调用频率

短期仍允许 CPU 粗粒度调用 poll。
目标是让高频 CQ service 和 per-chunk completion 写入在 GPU/底层 worker 内完成。
```

阶段 8：persistent worker + fused refill。

```text
状态：后续目标

GPU worker:
  submit -> poll/CQ -> IO_DONE -> refill -> REFILL_DONE
```

阶段 9：直接回填 vLLM paged KV cache。

```text
状态：当前下一阶段主线

减少 LMCache tensor 中转。
更贴近 TARDIS/Tutti 的 GPU-centric KV cache。
```

## 9.1 长期路线细化

长期目标不是继续给 Python wrapper 增加逻辑，而是逐步让 GPU worker 接管 I/O
热路径。当前和目标的区别如下：

```text
当前：
  CPU 决定要读哪些 chunk
  CPU 调 submit / poll / consume
  C++ rowctx_compat 推进 SQ/CQ
  GPU 写 pages buffer
  CPU launch refill

目标：
  CPU 只做 vLLM/LMCache 控制面和粗粒度错误处理
  CPU 填 GPU-visible request table
  GPU worker 读取 request table
  GPU worker 发起 BaM/NVMe IO
  GPU worker poll CQ 并写 completion table
  GPU worker 或 refill kernel 把 ready pages 写入目标 KV buffer
```

推荐路线已经从“继续优化 CQ service”调整为“direct placement 优先”：

```text
第一步：Direct Placement v0
  CPU 仍做 prefix/chunk lookup 和 vLLM block 分配。
  新增 KVPlacementPlan，描述 chunk pages 应该落到哪些 vLLM KV blocks。
  BaM 仍批量读 128KB pages。
  GPU kernel 直接把 pages scatter 到 vLLM paged KV cache 目标 block。
  保留当前 LMCache tensor refill 作为 fallback。

第二步：BaM KV 专用底层 read interface
  不再只暴露“读 row 到中间 buffer”。
  新增 KV request table:
    request_id
    ssd_page_offset / page_count
    dst_ptr 或 placement plan id
    layer_id / kv_id / token range
    status
  第一版仍允许 CPU submit/poll，但 table 必须 GPU-visible。

第三步：Layer-wise pipeline
  不再以完整 chunk 为唯一恢复单位。
  layer i attention 计算时，预取或放置 layer i+1 / 后续 layer 的 KV。
  目标是用 attention compute slack 隐藏 SSD I/O 和 placement 开销。

第四步：Persistent GPU worker
  GPU 读取 request queue。
  GPU 发起 BaM/NVMe I/O。
  GPU poll CQ 并写 completion table。
  GPU 或后续 kernel 消费 ready pages。
  CPU 只做 request 级调度、metadata 生成和低频错误处理。
```

这条路线和 AGIO/Tutti/TARDIS 的对应关系：

```text
AGIO:
  submit 和 completion 解耦，避免 GPU/CPU 原地同步等待。

Tutti:
  CPU-prepared, GPU-executed；CPU 准备 request table，GPU 执行 I/O。

TARDIS:
  KVCache 作为 GPU-centric object，不继续长期套通用 feature path。
```

## 10. 当前可跑分支

LMCache 原生 SSD baseline：

```text
用于确认当前 vLLM + LMCache V0 + SSD baseline。
```

LMCache-style GDS replay：

```text
用于与 BaM replay 进行 GDS 风格对比。
```

BaM sync：

```text
稳定 baseline，CPU 同步读完整 chunk。
```

BaM page-level prefetch：

```text
验证 submit/poll/get/refill 拆分，作为 KV fast path 前置 scaffold。
```

BaM KV fast path：

```text
当前主线，已接 replay 和真实 vLLM。
```

BaM KV fast path batch：

```text
当前最重要的 microbench 和真实 vLLM batch read 路径。
```

BaM gpu_worker v0：

```text
已从 fallback 骨架推进到独立 façade。
当前底层 worker_backend=kv_cq_service_v1。
replay 和真实 vLLM 已验证通过。
当前应作为 direct placement 的底层 I/O 基座。
```

BaM direct placement：

```text
下一阶段主线。
目标是减少 LMCache tensor 中转和 vLLM rebuild/refill 开销。
第一版先做 GPU scatter 到 vLLM paged KV cache blocks。
```

## 11. 保留与归档边界

继续保留并推进：

```text
BaMRowCtxKVExecutor
BaMGPUWorkerKVExecutor
BaMKVRequest / BaMKVRequestTable / BaMKVNativeBatchHandle
vllm/bam/lmcache_bam_kv_fast_path.py
vllm/bam/lmcache_bam_refill.py
LMCache-style GDS replay baseline
```

不再作为主线继续扩展：

```text
raw slab GDS backend
bam_cold_write 相关 two-process 临时逻辑
继续在 Python 层堆复杂 early-prefetch
vllm-bam 中和 Mooncake 强耦合的引用
把 GNN/CNN 通用 feature 状态机直接改成 KV 专用状态机
```

注意：

```text
BaM_IOStack 中服务 vllm-mooncake 的 Mooncake 适配代码不属于本主线清理范围。
这里说的是 vllm-bam 主线不要主动依赖 Mooncake。
```

## 12. 2026-06-30 排障收敛与当前主线

这一轮联调最重要的收敛，不是“某个 timeout 被绕过去了”，而是：

```text
KV direct placement / KV fast path 的轮询模型已经收敛，
可跑主线已经明确，
当前主要瓶颈也已经从 I/O 侧转移到了 placement/refill 侧。
```

这一节先回答三个问题：

```text
1. 之前为什么会卡住
2. 现在真正保留下来的轮询逻辑是什么
3. 当前跑通后，性能瓶颈到底在哪里
```

### 12.1 为什么 per-row/page 自己 poll completion 这条路不成立

这轮排障已经确认：

```text
当前 BaM registered rowctx / KV 路径里，
CQ completion 的消费职责不能下沉成：
  每个 row/page 一个线程，
  自己拿着自己的 (queue, cid) 去 poll / dequeue completion。
```

更合理、也和现有 BaM SQ/CQ 设计一致的模型是：

```text
一个 logical queue 对应一个 CQ consumer
一个线程只服务自己那条 CQ
completion 先按 queue 维度出队
再通过 (logical_queue_idx, cid) 找回对应 ctx
最后把 page/chunk/request 状态向上推进
```

也就是说，当前系统的正确分层仍然是：

```text
page 级发起 submit
queue 级消费 completion
ctx/page 级回填状态
request 级聚合 ready
```

而不是：

```text
page 级 submit
page 级 poll
page 级 dequeue
```

### 12.2 根本原因是什么

根本原因不是 Python 层 while 死循环，也不是 LMCache fallback 逻辑本身，
而是 CQ 消费职责放错了层级。

当前稳定逻辑里，completion 的服务方式本质上是：

```text
service_registered_completions_burst()
  -> service_registered_cq_window_kernel()
  -> 1 thread : 1 logical queue
```

在这条路径中，每条 logical queue 上的线程会串行推进：

```text
1. 从该 CQ 头部读取 completion
2. 取出 cid
3. 用 (logical_queue_idx, cid) 查 ctx_lookup
4. 找到对应 s_ctx
5. finalize 这个 page 对应的完成状态
6. 再由 ready-check 聚合整个 request 是否完成
```

它的隐含前提是：

```text
同一条 CQ 的 head/head_mark/doorbell，
当前实现默认由单个 queue-level consumer 串行推进。
```

所以之前那条实验路径会卡住，不是因为 GPU 不能并发轮询，而是因为：

```text
可以 many threads 并发轮询 many queues
但不应该 many threads 并发消费 one queue
```

更准确地说：

```text
允许：
  1 thread : 1 logical queue

不应做：
  many threads : 1 logical queue
```

### 12.3 当前真正保留下来的轮询逻辑

经过这轮清理后，KV 主线里真正保留下来的轮询逻辑已经收敛成：

```text
submit
  -> queue-level CQ service
  -> request-ready 聚合
  -> consume
```

执行过程可以按下面四步理解。

第一步，submit：

```text
request_table[num_chunks, 4]
  -> expand 成 d_row_ids[total_pages]
  -> rowctx submit
  -> 每个 page 生成一个 s_ctx
     其中至少包含：
       ctx.queue
       ctx.cid
       ctx.page_trans
       ctx.isHit
```

同时建立 lookup：

```text
(logical_queue_idx, cid) -> s_ctx*
```

第二步，poll：

```text
service_registered_completions_burst()
  -> service_registered_cq_window_kernel()
```

这个 kernel 的执行模型固定为：

```text
1 thread : 1 logical queue
```

每个线程只负责自己那条 CQ，不会有多个线程同时消费同一条 CQ。

在每条 queue 上，核心动作是：

```text
cq_try_peek_head()
  -> 读取当前 CQ head completion

cq_dequeue_head_serial()
  -> 串行推进 CQ/SQ head/tail/doorbell

put_cid()
  -> 归还 cid 到 SQ 可复用池

ctx = ctx_lookup[(logical_queue_idx, cid)]
  -> 找回原来的 page ctx

finalize_registered_ctx_completion()
  -> finalize 这个 page 对应的完成状态
```

第三步，ready-check：

```text
registered_request_ready_at()
  -> check_registered_request_ready_kernel()
```

这里已经不再直接碰 CQ，而只是做纯状态聚合：

```text
如果 ctx 还没 ready
  -> refresh_registered_ctx_from_valid_page()
  -> 看底层 page 状态是否已完成

如果还有任意 ctx 没完成
  -> pending_count += 1
```

host 侧只看：

```text
pending_count == 0 -> request ready
pending_count > 0  -> request not ready
```

第四步，consume：

```text
request_status == IO_DONE
  -> kv_worker_consume_batch()
  -> 从 BaM page cache 取回 batch pages
  -> direct placement / refill 回上层 KV cache
```

所以当前主线已经不是：

```text
page 自己拿着 ctx.cid 去 poll 自己的 completion
```

而是：

```text
queue 统一消费 completion
ctx/page 只是被 completion 回填
```

### 12.4 blocking poll 和 try-poll 的关系

接口层虽然还保留了两个名字：

```text
service_registered_poll_compatible()
service_registered_try_poll()
```

但底层已经共享同一套轮询模型：

```text
service_registered_completions_burst()
+ registered_request_ready_at()
```

区别只在外层控制语义：

```text
poll_compatible:
  while not ready:
    service CQ
    check ready

try_poll:
  只做一轮
  如果没 ready 就先返回
```

所以当前“轮询机制”本身已经统一，差别只是：

```text
阻塞 façade
vs
单步 try façade
```

### 12.5 这条思路会不会引入 CPU 参与

会，但要区分 “CPU 参与控制面” 和 “CPU 参与 CQ completion 消费”。

当前可跑版本里：

```text
CPU 仍然参与：
  1. LMCache/vLLM 命中判断和 request_table 生成
  2. 调 pybind 触发 submit
  3. 调 pybind 触发 poll
  4. 低频读取 request_status / host status
  5. 调 pybind 触发 consume
```

但 CPU 不再参与：

```text
1. 不逐 page 等 completion
2. 不自己消费 CQ
3. 不做 completion -> ctx 的分发
4. 不逐 chunk 维护完成状态
```

所以更准确的说法是：

```text
这条 per-logical-queue CQ worker 路线不会引入 CPU 参与 CQ 消费本身；
但在当前 v1 形态里，CPU 仍然是粗粒度控制面。
```

如果以后推进到 persistent GPU worker，则还可以进一步变成：

```text
CPU:
  只负责更粗粒度调度和 request 提交

GPU:
  常驻 worker 自己 service CQ
  自己写 completion/status table
  CPU 只在必要时 query 结果
```

### 12.6 当前可跑主线已经是什么

最新单卡联调日志已经表明，direct placement / KV fast path 的“新主线”已经能完整跑通。

这里的“跑通”具体指：

```text
1. request_2 的 prefix 命中被正确识别
2. BaM direct placement 真的进入了 batch read
3. submit / poll / ready / consume 全链路完成
4. 没有再出现：
   - REGISTERED_SUBMIT_ROWCTX_SUBMIT_KERNEL timed out
   - submit_error_code=1
   - DIRECT_RETRIEVE failed; fall back
```

当前可跑主线可以概括为：

```text
LMCache/vLLM
  -> direct placement prefix hit
  -> BaM KV request-table batch submit
  -> rowctx_compat_blocking
       （内部已收敛为 queue-level CQ service + ready 聚合）
  -> 从 BaM page cache 取回 batch pages
  -> placement / refill 到 vLLM paged KV cache
```

也就是说：

```text
当前可跑通的不是“旧的 per-row poll 实验路径”，
而是“保留 rowctx submit primitive，但 poll/ready 已收敛到 queue-level CQ service”
这条主线。
```

本轮推荐保留的成功日志是：

```text
evaluation/logs/single_gpu_lmcache_no_prefix_reuse_qwen25/20260630_041745/run.log
```

### 12.7 当前性能结论：I/O 已通，瓶颈转移到 placement/refill

当前这段排障可以压缩成一条清晰时间线：

```text
阶段 A：direct placement 跑通，但 placement 很重
  20260630_041745
  read_ms   = 11.956
  refill_ms = 502.319
  place_ms  = 503.945
  request_2 = 2.0826

阶段 B：引入 merged refill 后，性能明显回退
  20260630_050817
  read_ms   = 17.218
  refill_ms = 820.919
  place_ms  = 821.662
  request_2 = 2.4064

阶段 C：逐 step 计时后确认，
  慢点集中在首个 merged refill step，
  后续 3 个 step 已是亚毫秒级
  20260630_165157
  step0 = 437.792 ms
  step1 = 0.225 ms
  step2 = 0.152 ms
  step3 = 0.125 ms
  refill_ms = 438.707
  request_2 = 2.0264

阶段 D：补 merged refill warmup 后，
  首个 step 的一次性开销被移出热路径
  20260630_warmup_check
  step0 = 0.214 ms
  step1 = 0.152 ms
  step2 = 0.127 ms
  step3 = 0.126 ms
  read_ms   = 12.603
  refill_ms = 1.021
  place_ms  = 1.790
  request_2 = 2.0111
```

这条时间线对应的结论是：

```text
1. 当前 direct placement v0 已经真实跑通，不是 fallback 幻觉。
2. BaM I/O 早就不是主要瓶颈，read_ms 已经稳定在 ~12ms。
3. merged refill 的 steady-state 本身并不慢。
4. 之前的性能回退，主因是首个 merged refill step 的一次性 Triton/JIT 初始化成本。
5. 给 merged refill 补 warmup 后，placement/refill 已经下降到 1~2ms 量级。
```

对应的工程动作也可以压缩成两步：

```text
第一步：
  把 _bam_pages_to_lmcache_kernel_with_token_offset(...)
  里随 chunk 变化的量：
    total_elements
    num_layers
    actual_tokens
    total_output_tokens
    token_offset
  从 tl.constexpr 改成运行时参数，
  避免同一批 token_offset=0/256/512/768 触发多份 specialization。

第二步：
  在 BaMDirectKVPlacer 里新增：
    _maybe_warmup_merged_refill(plan)
  用真实 pages / 真实 layout / 真实 token 参数做一次安全预热，
  把首个 step 的一次性 Triton 初始化成本前移。
```

因此截至当前版本，性能判断应更新为：

```text
1. placement/refill 这段已经基本打通。
2. 当前 direct placement v0 的 steady-state placement 成本已很低。
3. 后续优化重点不应继续放在 merged refill，
   而应更多转向：
     - read_ms ~12ms 是否还能继续压缩
     - LMCache rebuild / XFORMERS_PREFIX_FALLBACK
     - 从 merged LMCache tensor 进一步收缩到 final vLLM KV cache
```

这轮结果说明：

```text
1. warmup 已经真正生效。
2. 首个 merged refill step 的一次性开销已被成功移出热路径。
3. 当前 direct placement v0 的 steady-state placement/refill
   已经下降到 1~2ms 量级。
4. 因此当前性能瓶颈已经不再主要在 merged refill，
   后续应更多关注：
     - read_ms ~12ms 还能否继续压缩
     - LMCache rebuild / XFORMERS_PREFIX_FALLBACK
     - 从 merged LMCache tensor 进一步收缩到 final vLLM KV cache
```

### 12.7.1 当前 direct placement 的实际数据通路

当前版本虽然已经叫 direct placement，但它仍然是一个过渡形态：

```text
BaM pages
  -> merged LMCache KV tensor
  -> one LMCache connector transfer
  -> vLLM paged KV cache
```

它还不是最终目标里的：

```text
BaM pages
  -> 直接写 vLLM paged KV cache
```

更具体地说，当前一轮真实 request 的数据流如下。

#### 例子：1261 token，请求前缀命中 1024 token

当前脚本里：

```text
chunk_size = 256
prefix hit = 1024 tokens
因此命中 4 个 chunk
```

也就是：

```text
chunk0 -> tokens [0,256)
chunk1 -> tokens [256,512)
chunk2 -> tokens [512,768)
chunk3 -> tokens [768,1024)
剩余 [1024,1261) 走正常 prefill
```

#### 第 1 步：LMCache connector 识别 prefix hit

控制面流程：

```text
vLLM scheduler
  -> LMCacheConnector
  -> LMCache token_database / chunk lookup
  -> 找到 4 个可恢复 chunk hash
```

日志上对应：

```text
LMCACHE_BAM_DIRECT_RETRIEVE_BEGIN
LMCACHE_BAM_DIRECT_PLACEMENT_PREFIX_HIT
```

这一步 CPU 已经知道：

```text
1. 要恢复哪 4 个 chunk
2. 每个 chunk 在本轮 request 的 token 区间
3. 当前 request 对应的 slot_mapping
```

#### 第 2 步：把 chunk 转成 batch read 请求

当前每个 chunk 在 BaM 中固定存成：

```text
[112, 128KB]
```

因为当前 layout 是：

```text
[2, 28, 256, 512]
page_token_capacity = 128
pages_per_kv_layer  = 2
pages_per_chunk     = 2 * 28 * 2 = 112
```

所以本轮 4 个 chunk 一次性提交给 BaM 的真实读请求是：

```text
batch_size     = 4
pages_per_chunk= 112
total_pages    = 448
```

#### 第 3 步：BaM 返回 pages，而不是返回 KV tensor

当前 `read_pages_batch()` 的结果不是 LMCache tensor，而是：

```text
每个 chunk:
  [112, 128KB] uint8 CUDA tensor
```

这说明当前 direct placement 的真正输入对象是：

```text
BaM page batch
```

而不是已经 decode 好的 KV。

#### 第 4 步：build placement plan

`BaMDirectKVPlacer._build_plan()` 会把：

```text
result + chunk_start + slot_mapping
```

整理成 plan entry。

对上面的 4 个 chunk，plan 大致是：

```text
entry0:
  chunk_start = 0
  actual_tokens = 256
  slot_mapping = slot_mapping[0:256]

entry1:
  chunk_start = 256
  actual_tokens = 256
  slot_mapping = slot_mapping[256:512]

entry2:
  chunk_start = 512
  actual_tokens = 256
  slot_mapping = slot_mapping[512:768]

entry3:
  chunk_start = 768
  actual_tokens = 256
  slot_mapping = slot_mapping[768:1024]
```

总 token 数：

```text
total_tokens = 1024
```

#### 第 5 步：merged refill

当前不会为每个 chunk 单独创建一个 KV tensor 再 cat，而是直接创建：

```text
merged tensor:
  [2, 28, 1024, 512]
```

然后 4 个 chunk 依次写入：

```text
chunk0 pages -> merged[:, :,   0:256, :]
chunk1 pages -> merged[:, :, 256:512, :]
chunk2 pages -> merged[:, :, 512:768, :]
chunk3 pages -> merged[:, :, 768:1024, :]
```

日志中的：

```text
token_offset=0
token_offset=256
token_offset=512
token_offset=768
```

说的就是这个过程。

也就是说，当前 `refill_pages_to_lmcache_tensor_into()` 做的是：

```text
[112, 128KB] pages
  -> 按 kv_id/layer/token_page/token_in_page 解码
  -> 写到 merged KV tensor 的目标 token 区间
```

#### 第 6 步：one transfer 到 vLLM paged KV cache

merged refill 完成后，再做一次：

```text
multi_layer_kv_transfer(
  merged_kv_tensor,
  merged_slot_mapping,
  kv_caches
)
```

这里：

```text
merged_kv_tensor   = [2, 28, 1024, 512]
merged_slot_mapping= [1024]
```

`slot_mapping` 的作用是告诉 LMCache connector kernel：

```text
merged 第 i 个 token
最终应该写到 vLLM paged KV cache 的哪个 physical slot
```

#### 第 7 步：vLLM 后续消费

回填结束后：

```text
前 1024 token 直接复用恢复出的 KV
剩余 237 token 继续正常 prefill / attention
```

因此当前 direct placement v0 的完整数据流可以概括成：

```text
LMCache prefix hit
  -> 4 个 chunk hash
  -> BaM batch read 448 pages
  -> build placement plan
  -> merged refill [2, 28, 1024, 512]
  -> one connector transfer
  -> vLLM paged KV cache
  -> 剩余未命中 token 正常 prefill
```

当前版本的定位可以一句话总结为：

```text
控制面已经 plan 化，
I/O 已经 batch 化，
placement 仍处于“BaM pages -> merged LMCache tensor -> one transfer”
这个过渡阶段。
```

补充一条 2026-06-30 当天代码侧的最新推进：

```text
direct placement 的 fused 实验路径已经从：

  单个 chunk
    -> 每个 layer
       -> 每个 K/V
          -> 多次 kernel launch

收缩成：

  单个 chunk
    -> 一次 fused kernel launch
       -> 同时覆盖全部 layer + K/V
```

这一步的意义不是“已经替代当前 lmcache 主线”，而是先把 direct placement v1
需要的最内层数据搬运收紧：

```text
旧 fused:
  Python for chunk
    Python for layer
      Python for K/V
        Triton launch

新 fused:
  Python for chunk
    Triton launch
```

当前仍保留逐 chunk 的 Python 层，原因是：

```text
1. 这样可以继续复用现有 placement plan / chunk 边界；
2. 便于和当前稳定的 lmcache 实现做 A/B；
3. 后续如果继续往 batch-level fused placement 收缩，
   只需要替换“逐 chunk 执行器”这一层，而不用改上层 direct placement 入口。
```

因此可以把这一步理解成：

```text
先把 chunk 内部的数据面收紧，
再考虑把多个 chunk 合并成更大的 batch placement kernel。
```

### 12.8 和 baseline 的对比应该怎么理解

当前更适合拿来做工程对照的 baseline 有两组：

```text
1. 原生 vLLM V0 no-prefix-reuse baseline
2. LMCache SSD-only no-prefix-reuse baseline
```

归档文档：

```text
evaluation/SINGLE_GPU_LMCACHE_SSD_BASELINE_QWEN25.md
```

较早归档的数值是：

```text
原生 vLLM V0:
  request_1_elapsed_s = 2.1716
  request_2_elapsed_s = 2.2151

LMCache SSD-only:
  request_1_elapsed_s = 2.0224
  request_2_elapsed_s = 1.6895
```

和当前脚本口径更接近、也更适合直接对照的一轮是：

```text
2026-06-30 04:12
  request_1_elapsed_s = 1.8076
  request_2_elapsed_s = 1.5966
```

但这轮实际上是：

```text
prefix 命中识别成功
direct placement submit 失败
最终 fallback 到 LMCache/原 prefer-load 路径
```

而最新真正“新主线跑通”的结果是：

```text
2026-06-30 04:18
  request_1_elapsed_s = 1.8180
  request_2_elapsed_s = 2.0826
```

所以当前对比结论应该明确写成：

```text
1. 和原生 vLLM baseline 相比：
   当前 direct placement 新主线在功能上已经更完整，
   但端到端还没有形成稳定性能优势。

2. 和 LMCache SSD-only baseline 相比：
   当前 BaM direct placement 也还没有体现出端到端收益，
   request_2 甚至可能更慢。

3. 根因并不在 BaM I/O：
   11.956ms 的 read_ms 说明底层 batch read 已经很快。

4. 当前真正拖慢 request_2 的主因是 placement/refill：
   大约 500ms 量级，
   远大于底层 submit/poll/get 的开销。
```

换句话说，当前系统已经从：

```text
能不能跑通 BaM direct placement
```

进入到了：

```text
能不能把 direct placement 后半段做轻
```

这个阶段。

## 13. 下一阶段改进路线图（2026-06-30）

基于前面的收敛结论，当前推荐下一步已经不是继续验证 `kv_cq_service_v1`
或者继续深挖 `submit/poll`，而是把系统从：

```text
CPU 组织 request
  -> GPU/BaM 读回 pages
  -> 再做一大段 placement/refill/rebuild
```

推进到：

```text
GPU-visible request queue
  -> GPU 异步发起 I/O
  -> GPU 统一消费 completion
  -> GPU 直接把 KV 放到最终可用布局
  -> 尽量与 layer 计算 overlap
```

这是 AGIO、Tutti、TARDIS 三篇工作在当前工程里的共同落点。

### 13.1 为什么现在不该再优先优化 submit/poll

最新日志已经说明：

```text
底层 BaM batch read:
  read_ms = 11.956

direct placement 后半段:
  refill_ms = 502.319
  place_ms  = 503.945
```

所以当前瓶颈已经不是：

```text
submit timeout
per-row poll
CQ dequeue 组织
```

而是：

```text
pages 读出来之后，
怎么更便宜地进入 vLLM 真正要消费的 paged KV cache 布局
```

这正好和三篇论文的关注点一致：

```text
AGIO:
  initiation / completion 解耦，GPU 不应原地同步等待 I/O

Tutti:
  CPU-prepared, GPU-executed，减少 CPU-centric I/O orchestration

TARDIS:
  KV 是 GPU-centric object，不该先恢复成通用 tensor 再 rebuild
```

### 13.2 推荐路线图

#### 13.2.1 Direct Placement v1：先打掉中间态

当前最值得优先做的是把：

```text
BaM pages
  -> 中间 tensor
  -> refill
  -> rebuild
  -> vLLM paged KV cache
```

改得更接近：

```text
BaM pages
  -> placement kernel
  -> vLLM paged KV cache target blocks
```

核心目标：

```text
减少 LMCache 通用 tensor 中转
减少 Python 侧组织
减少 rebuild/refill 的额外恢复成本
```

这一步最贴近 TARDIS 的启发：

```text
KVCache 应按最终消费布局组织，
而不是按临时恢复格式组织。
```

#### 13.2.2 GPU-visible KVPlacementPlan

下一步建议把当前 placement plan 收敛成 GPU 可直接消费的 descriptor：

```text
src_page
page_count
layer_id
kv_id
token_start / token_end
dst_block_id
dst_block_offset
dst_ptr / stride
```

这样做的好处是：

```text
CPU:
  仍负责 metadata lookup / block 分配 / request preparation

GPU:
  直接按 plan 做 scatter / placement
```

这一步最贴近 Tutti 的 `CPU-prepared, GPU-executed`。

#### 13.2.3 Layer-wise / Segment-wise placement

当前还是：

```text
prefix hit 若干 chunk
  -> 全 batch read
  -> 全 batch place
```

下一步更合理的是：

```text
layer 0 ready -> 先 place layer 0
layer 1 ready -> 再 place layer 1
...
```

更进一步：

```text
attention 计算 layer i
同时预取 / 放置 layer i+1 的 KV
```

这一点同时对应：

```text
AGIO:
  让 I/O 和 useful computation overlap

TARDIS:
  layer-wise async swapping

Tutti:
  用 scheduling/slack-aware pipeline 隐藏 I/O latency
```

#### 13.2.4 Persistent GPU worker

当前虽然已经有 `gpu_worker` façade，但本质上还是：

```text
CPU 提交
CPU 调 poll
CPU 再 consume
```

下一步应逐渐推进成：

```text
CPU:
  提交 request table
  低频看 request status

GPU worker:
  读 request queue
  submit I/O
  service CQ
  写 completion/status
  触发 placement
```

这一步最贴近 AGIO：

```text
不是 GPU“也能发 I/O”而已，
而是 GPU 自己维护异步 I/O 生命周期。
```

#### 13.2.5 双队列：I/O queue + placement queue

建议后续不要只保留一个 request table，而是拆成两个阶段：

```text
I/O request queue
  -> 描述从 SSD/BaM 读取哪些 page

Placement queue
  -> 描述这些 page 应该写到哪些 vLLM KV blocks
```

这样做的价值：

```text
1. I/O completion 和 placement 解耦
2. ready pages 不必等整批 chunk 全完成后再统一恢复
3. placement 可以独立调度，甚至按 layer 优先级推进
```

这是 AGIO 中 initiation/completion decouple 在我们系统里的最自然映射。

#### 13.2.6 最后再做 slack-aware scheduling

这一步现在还不该最先做，但后面一定值得做。

等 direct placement v1 跑稳后，再考虑：

```text
哪些 layer/chunk 优先读
哪些 request 可以后放
哪些 placement 应该让路给 attention compute
哪些 I/O 可以藏到 compute slack 后面
```

这一步主要对应 Tutti。

### 13.3 分优先级的推进顺序

如果只做最有价值的三步，推荐顺序是：

```text
1. Direct Placement v1
   目标：减少 placement/refill/rebuild 中间态

2. Layer-wise placement
   目标：不要等整批 chunk 全完成后才一起恢复

3. Persistent GPU worker
   目标：把 submit/poll/place 真正变成 GPU-side async pipeline
```

一句话版路线图：

```text
先把“读回来后怎么放进去”做轻
再把“什么时候放进去”做异步
最后把“谁来推进整个生命周期”彻底下沉到 GPU
```

### 13.4 和当前代码最直接对应的改动点

按当前代码结构，下一步最值得优先动的点是：

```text
1. place_bam_results_to_vllm_kvcache()
   当前 placement/refill 的最大热点

2. direct placement plan 的构造与传递
   逐步变成 GPU-visible KVPlacementPlan

3. kv_worker_submit / poll / consume 的边界
   为后续 persistent worker 留出 stage 分界

4. request_table 扩展为：
   request_table + placement_table

5. consume 粒度
   从 whole chunk 走向 layer / segment
```

### 13.5 为什么当前 Direct Placement v0 仍然要保留

虽然主线已经应该往 v1 走，但当前 v0 仍然有保留价值：

```text
1. 它已经证明：
   BaM batch read + queue-level CQ service + direct retrieve 可以跑通

2. 它提供了：
   精确的功能正确性 scaffold

3. 它仍然是：
   后续 v1 / layer-wise / GPU worker 改造的 fallback 基线
```

因此当前不应删除 v0，而是应把它当成：

```text
可运行基线 + 正确性对照组
```

### 13.6 Direct Placement v0（保留说明）

第一版目标：

```text
把 BaM 读出的 128KB pages 直接写入 vLLM paged KV cache 的目标 block。
尽量绕开当前:
  BaM pages -> LMCache tensor -> Triton refill -> rebuild/prefix
这条重恢复路径。
```

新增中间层：

```text
KVPlacementPlan:
  chunk_hash
  actual_tokens
  source page_offset / page_count
  layer_id
  kv_id                       # K 或 V
  source token range
  target vLLM block_id
  target block offset
  target pointer / stride info
```

第一版 CPU/GPU 分工：

```text
CPU:
  LMCache chunk 命中判断
  vLLM block table / slot mapping 读取
  生成 KVPlacementPlan
  调用 BaM batch read

GPU/BaM:
  BaM 读取 128KB pages
  GPU placement kernel 按 plan 写入 vLLM paged KV cache

保留 fallback:
  当前 load_chunk_tensors_kv_fast_path_batch()
  当前 lmcache_bam_refill.py
```

这一步的验证标准：

```text
1. replay direct placement 能 exact_equal。
2. 真实 vLLM 日志里仍能 prefer-load hit。
3. read_ms 不明显回退。
4. refill_ms 或 rebuild 相关耗时明显下降。
5. 如果 direct placement 失败，可以通过开关回退到当前 KV fast path batch。
```
