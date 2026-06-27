# GPU-initiated BaM 实现思路

日期：2026-06-25

最近整理：2026-06-27

本文记录当前 `vllm-bam` 中 BaM 接入 LMCache/vLLM KVCache 的现状、已经实现
的路径，以及下一阶段向 GPU-initiated / KVCache fast path 演进的主线。

这份文档只保留当前仍有价值的方案。早期探索中过渡性太强、已经证明不是主线的
方案不再展开，避免后续实现时被旧路线干扰。

## 1. 当前结论

当前 BaM 接入已经跑通了以下几类路径：

- LMCache 原生 SSD baseline。
- LMCache-style GDS replay baseline。
- BaM sync 读写路径。
- BaM page-level prefetch/refill 路径。
- BaM batch prefetch replay。
- 真实 vLLM + LMCache + BaM prefer-load 路径。

当前路径本质上仍然是：

```text
CPU 做控制面调度
GPU/BaM 做数据面读取与数据搬运
```

它还不是最终形态的：

```text
GPU 自己维护 request queue / poll completion / refill KV cache
```

下一步主线不是继续在 Python 层给现有 row store 包更多逻辑，而是新增一条
KVCache 专用 fast path：

```text
LMCache chunk metadata
  -> KV 专用 request descriptor
  -> BaM KV batch read / worker
  -> [chunk, 112, 128KB] pages
  -> GPU refill
  -> LMCache tensor 或 vLLM paged KV cache
```

这个方向对 KVCache 路径来说是比较彻底的改动，但不应该破坏 BaM 原有
GNN/CNN/feature 路径。推荐策略是：

```text
原 GNN/CNN/feature path:
  保留 read_feature_* / rowctx / bam_row_store.py 语义

新增 KVCache fast path:
  新增独立 request descriptor / queue / worker / refill
  通过开关接入 vLLM/LMCache
```

## 2. 当前 KVCache 数据组织

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

为了让每个 BaM IO 请求对应一个 128KB row/page，写入 BaM 前会按 slot token
容量组织：

```text
[2, 28, 256, 512]
```

其中：

```text
每个 token 向量大小 = 512 * 2B = 1024B
每个 128KB page 可容纳 = 128KB / 1024B = 128 tokens
每层 K 需要 2 个 page
每层 V 需要 2 个 page
一个完整 chunk = 2(K/V) * 28(layer) * 2(page/layer) = 112 pages
```

因此当前 BaM 中一个满 chunk 的物理组织是：

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
kv_id = 0               # K
layer_id = 3
token_offset = 150
page_token_capacity = 128
token_page_id = 150 // 128 = 1

bam_page_id = 784 + 0 * 28 * 2 + 3 * 2 + 1 = 791
```

目前 K 和 V 没有混在同一个 page 里，而是按：

```text
K all layers/pages
V all layers/pages
```

顺序组织。这个组织方式简单、稳定，也贴合当前 LMCache chunk layout。

## 3. 当前 vLLM 内调用流程

当前正式在线路径的入口在 LMCache connector：

- `vllm/distributed/kv_transfer/kv_connector/lmcache_connector.py`

大致流程：

```text
vLLM scheduler
  -> LMCacheConnector 查询本次请求可 retrieve 的 prefix/chunk
  -> LMCache engine retrieve
  -> LMCache storage backend get/put
  -> vllm/bam/lmcache_bam_storage.py wrapper
  -> BaM sync 或 prefetch 路径
```

BaM storage 侧主要入口：

- `vllm/bam/lmcache_bam_storage.py`

其中：

```text
LMCacheBaMStore.store_chunk()
  -> 把 LMCache KV tensor 写成 [pages, 128KB]
  -> BaMRowStore.store_rows()

LMCacheBaMStore.load_chunk_tensor()
  -> sync baseline
  -> BaMRowStore.load_rows()

LMCacheBaMStore.load_chunk_tensor_prefetch()
  -> page-level prefetch/refill
  -> lmcache_bam_prefetch.py
```

当前 prefetch/refill 中间层：

- `vllm/bam/lmcache_bam_prefetch.py`
- `vllm/bam/lmcache_bam_refill.py`

当前流程：

```text
prepare_request()
  -> CPU 根据 chunk metadata 生成 GPU page_ids 请求表

submit_request()
  -> CPU 调 BaM rowctx submit 接口
  -> GPU kernel 发起 BaM page read

poll_request()
  -> CPU 调 poll 接口推进/检查 completion

complete_request()
  -> CPU 调 get 接口
  -> GPU 把 [page_count, 128KB] pages 写入输出 buffer

refill_request()
  -> CPU launch Triton refill kernel
  -> GPU 把 pages 还原成 [2, layers, tokens, hidden]
```

因此当前 CPU/GPU 分工是：

```text
CPU:
  vLLM scheduler
  LMCache prefix/chunk 命中判断
  chunk metadata 查找
  request table 构造
  submit/poll/complete/refill kernel launch
  fallback/error/log

GPU/BaM:
  BaM page read
  BaM page cache / DMA 数据路径
  pages buffer 写入
  Triton refill 数据转换
```

## 4. 当前已实现的 BaM prefetch/refill 层

当前已经实现了一层保守 GPU-initiated scaffold。它没有直接修改 attention kernel，
而是把 LMCache 命中后的完整 chunk 读取改造成 page-level pipeline。

核心数据流：

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

核心文件：

```text
vllm/bam/lmcache_bam_prefetch.py
  LMCacheBaMPagePlanner
  LMCacheBaMPagePrefetcher
  LMCacheBaMPagePipeline
  LMCacheBaMBatchReadRequest

vllm/bam/lmcache_bam_refill.py
  refill_pages_to_lmcache_tensor()

vllm/bam/lmcache_bam_storage.py
  LMCacheBaMStore.load_chunk_tensor_prefetch()
  LMCacheBaMStorageManager.prefetch()
  LMCacheBaMStorageManager.load_prefetched_chunk_tensor()
```

已经验证过的 replay 结果中，batch prefetch 在去掉首次 Triton JIT 后表现正常：

```text
bam_prefetch_batch, batch size = 4
total_ms ≈ 3.283
amortized per chunk ≈ 0.851ms
bw_gib_s ≈ 16.658
```

真实 vLLM early-prefetch 路径也已经接通，但当前收益有限：

```text
request_2 从约 2.03s 到约 2.08s
约 2.5% 变慢
```

主要原因不是 BaM page read 本身慢，而是：

```text
当前 early-prefetch 只把 submit 提前了一点
poll / complete / refill 仍然在 get() 中串行消费
CPU 仍然逐 chunk 推进状态
首次 Triton refill 还有 JIT 开销
```

所以当前这层更适合作为 correctness scaffold 和 replay microbench，不应该继续
在 Python 层无限加复杂度。

## 5. BaM 底层已有能力

BaM 底层已经有 GPU-side page cache / IO primitive，不是纯 CPU read。

关键 primitive 位于：

- `BaM_IOStack/bam/include/page_cache.h`

代表性 device-side 接口：

```cpp
T read(size_t i)
T read_submit_async(size_t i, s_ctx &ctx)
T read_wait_async(size_t i, s_ctx &ctx)
T read_single_thread_poll(size_t i, s_ctx &ctx)
T read_post_poll_light(size_t i, s_ctx &ctx)
```

当前 rowctx kernel 位于：

- `BaM_IOStack/gids_module/gids_kernel.cu`

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

Python row store 暴露：

- `BaM_IOStack/gids_module/bam_row_store.py`

```python
prefetch_rows()
poll_prefetch()
get_prefetched_rows()
```

这些能力可以继续复用，但 KVCache fast path 不应该继续完全套用
`read_feature_*` 的通用 feature 语义。

## 6. 为什么需要 KVCache fast path

当前通用 feature path 适合 GNN/CNN 这类场景：

```text
任意 row_ids
任意 dim/cache_dim
通用 scatter/gather
通用 feature 输出
```

KVCache 场景则更加规则：

```text
固定 128KB page
固定 chunk-level request
固定 page_count，Qwen2.5-7B fp16 当前为 112
固定 [2, layers, tokens, hidden] layout
读后通常要 refill 到 LMCache tensor 或 vLLM paged KV cache
```

如果继续把 KVCache 强行塞进 feature path，会保留很多 KV 场景不需要的开销：

```text
构造 row_ids tensor
传 dim/cache_dim/key_off
通用 get_feature_light
Python 逐 chunk submit/poll/get
读后再单独 refill
```

因此下一步应该新增 KV 专用路径，让 KVCache 成为 BaM 底层的一等公民。

## 7. 改动力度分级

针对 KVCache 的改动可以分三档。

轻量改动：

```text
继续使用现有 read_feature / rowctx / poll / get
只在 vLLM 侧更早 submit、更好 batch
```

优点是风险小；缺点是 CPU 仍然在逐 chunk 推进 IO，GPU-initiated 收益有限。

中等改动：

```text
新增 KVCache 专用接口
复用 BaM 现有 NVMe queue / PRP / page cache / CQ service primitive
新增 KV request descriptor / status table / batch read
```

这是当前推荐的起步力度。

彻底改动：

```text
新增 KV 专用 GPU-visible request queue
新增 persistent GPU worker
GPU 负责 submit / CQ service / poll / completion / refill
后续直接回填 vLLM paged KV cache
```

这是最终方向，但不建议第一步就把全部状态机和 attention backend 一起改掉。

推荐路线：

```text
先做中等改动
  -> KV 专用 batch read microbench 正确
  -> 接现有 Triton refill
  -> 接真实 LMCache/vLLM

再演进到彻底改动
  -> GPU-visible queue/status
  -> persistent worker
  -> GPU-side poll/completion
  -> fused refill / paged KV cache refill
```

## 8. 下一步主线：KVCache 专用 fast path

BaM_IOStack 侧建议新增：

```text
BaM_IOStack/gids_module/
  gids_kv_cache.h
  gids_kv_cache.cu
  bam_kv_store.py
```

职责：

```text
gids_kv_cache.h:
  定义 BaMKVRequest / BaMKVWorkerState / status enum

gids_kv_cache.cu:
  实现 KV batch submit / CQ service / get pages
  后续加入 persistent worker / fused refill

bam_kv_store.py:
  Python 薄封装
  暴露 start / submit_chunks / query / consume / stop
```

`gids_nvme.cu` 只做 pybind 桥接，不改变原 feature path 语义：

```text
新增:
  kv_worker_start()
  kv_submit_chunk_batch()
  kv_query_status()
  kv_consume()
  kv_worker_stop()

不修改语义:
  read_feature()
  read_feature_submit_async_registered_rowctx()
  service_registered_try_poll()
  read_feature_get_feature_light_registered_rowctx()
```

vllm-bam 侧建议新增：

```text
vllm/bam/lmcache_bam_kv_fast_path.py
```

职责：

```text
LMCache key / BaM metadata
  -> KVRequest(page_offset, page_count, actual_tokens)
  -> BaMKVStore.submit_chunks()
  -> consume ready pages or tensor
  -> populate LMCache memory_obj
```

现有 `lmcache_bam_storage.py` 只做分发：

```text
if VLLM_BAM_KV_FAST_PATH=1:
    use lmcache_bam_kv_fast_path
else:
    use current sync/prefetch path
```

## 9. KV request descriptor

KV 场景不再传通用 `row_ids/index_ptr/dim/cache_dim`，而是传 chunk-level
descriptor：

```cpp
struct BaMKVRequest {
  uint64_t request_id;
  uint64_t chunk_id;
  uint64_t page_offset;
  uint32_t page_count;      // Qwen2.5-7B fp16 当前为 112
  uint32_t page_bytes;      // 128KB
  uint32_t actual_tokens;
  uint32_t status;

  uint8_t* pages_out;       // [page_count, 128KB]
  void* kv_out;             // 可选：[2, layers, tokens, hidden] 或 paged KV
};
```

第一版可以先实现：

```text
CPU 一次提交 N 个 BaMKVRequest
GPU 展开 N * 112 pages
复用 BaM primitive 读回 pages
输出 [N, 112, 128KB]
```

这一版仍可由 CPU 触发 submit/query/consume，但接口已经不再是通用 feature
接口。等 correctness 和性能稳定后，再把 request queue / status table 变成
GPU-visible，由 GPU worker 自己推进。

## 10. KV fast path 与原 GNN/CNN 路径的隔离

必须隔离：

```text
原 feature outstanding queue:
  继续服务 read_feature_* / GNN / CNN

KV request queue:
  只服务 KV fast path
```

可以共享：

```text
NVMe controller / queues
PRP / DMA / memory registration
page cache primitive
CQ service 基础能力
device-side read_submit_async / post_poll primitive
```

不应共享：

```text
request descriptor
request table
状态机语义
get/refill 输出语义
Python API
```

如果需要扩展底层 ctx，也优先新增 wrapper：

```cpp
struct BaMKVCtx {
  s_ctx base;
  uint32_t chunk_slot;
  uint32_t page_in_chunk;
};
```

不要改变已有 `s_ctx` 字段含义，避免破坏 GNN/CNN。

## 11. GPU-initiated 中 CPU 仍然负责什么

即使做 GPU-initiated，CPU 也不会完全消失。KVCache 场景里 CPU 仍然负责控制面：

```text
vLLM scheduler
请求排队与 batch 组织
LMCache prefix/chunk lookup
chunk metadata 生命周期
fallback / error handling / logging
CUDA kernel 或 worker 启停
资源分配和释放
```

GPU-initiated 优化的是细粒度 IO 热路径：

```text
CPU 不再逐 page/chunk submit
CPU 不再逐 completion poll
CPU 不再逐 chunk 推进 get/refill
```

目标分工：

```text
CPU:
  粗粒度提交“这批 chunk 需要读”

GPU/BaM:
  展开 page 请求
  submit BaM/NVMe IO
  poll completion
  mark ready
  refill pages
```

这与 TARDIS/Tutti 的启发一致：CPU 仍保留调度和系统管理，但数据面请求推进
应尽量放到 GPU 侧。

## 12. 推荐实现阶段

第一阶段：KV 专用 batch read microbench。

```text
输入:
  N 个 chunk 的 page_offset / page_count / actual_tokens

输出:
  [N, 112, 128KB] pages

目标:
  验证 correctness
  对比当前 sync / prefetch batch
  确认不会影响原 read_feature_* 路径
```

第二阶段：接现有 Triton refill。

```text
[N, 112, 128KB] pages
  -> refill_pages_to_lmcache_tensor()
  -> [N, 2, layers, tokens, hidden]
```

第三阶段：接 LMCache/vLLM。

```text
LMCache prefer-load hit
  -> KV fast path submit
  -> consume ready KV
  -> populate memory_obj
```

第四阶段：GPU-visible queue / status table。

```text
CPU enqueue descriptors
GPU 推进 submit / poll / completion
CPU query 只读 status
```

第五阶段：persistent worker + fused refill。

```text
GPU worker:
  submit -> CQ service -> IO_DONE -> refill -> REFILL_DONE
```

第六阶段：直接回填 vLLM paged KV cache。

```text
BaM pages
  -> vLLM paged KV cache blocks
  -> 减少 LMCache tensor 中转
```

## 13. 暂不作为主线的方案

以下方案不删除其代码中的有用部分，但不再作为后续主线展开：

```text
继续只优化 LMCacheBaMStorageManager.get() -> load_rows()
```

原因：这条路径始终是 CPU 同步 chunk 级接口，适合作为 baseline，不适合作为
GPU-initiated 主线。

```text
继续在 Python 层堆 early-prefetch 复杂逻辑
```

原因：当前已经证明 submit 提前不等于真正 overlap；poll/complete/refill 仍在
get 中串行消费时，端到端收益很有限。

```text
立即修改 xFormers attention kernel 做 demand load
```

原因：改动面太大，且当前 V100/xFormers/vLLM paged KV layout 下风险高。
应先把 KV fast path 和 refill 跑稳。

```text
把原 GNN/CNN 状态机直接改成 KV 专用状态机
```

原因：会破坏通用 feature path。更稳妥的是新增 KV 状态机或 KV worker。

```text
bam_cold_write 这类为 two-process cold read 临时创建的路径
```

原因：它不是稳定主线，并且曾触发底层 GPU illegal memory access。当前不要把它
作为主线继续扩展。

## 14. 当前推荐的一句话主线

```text
保留原 BaM feature/GNN/CNN 路径；
新增 KVCache 专用 fast path；
第一步先做 KV batch read microbench；
跑稳后接现有 Triton refill；
再接 LMCache/vLLM；
最后把 request queue、poll、completion、refill 逐步下沉到 GPU worker。
```

这样既能保持 BaM 仓库的通用性，也能让 vLLM KVCache 数据面真正朝
GPU-initiated 方向推进。
