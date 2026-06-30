# BaM SQ/CQ 与 rowctx/KV fast path 审计

审计日期：2026-06-28

审计范围：

- BaM 底层 NVMe controller、SQ、CQ 初始化和 GPU 侧 doorbell 逻辑。
- BaM registered rowctx submit/poll/consume 生命周期。
- 当前 vLLM/LMCache KV fast path 对 BaM rowctx 的复用方式。
- 不审计 Mooncake/vLLM-Mooncake 路径；当前结论只面向 `vllm-bam` 主线。

## 总结结论

当前 BaM 已经具备 GPU 侧发起 NVMe SQ 请求和 GPU 侧服务 CQ completion 的底层 primitive：`sq_enqueue()` 会在 GPU 上写 SQ entry 并 ring SQ doorbell，`cq_try_peek_head()`/`cq_dequeue()` 会在 GPU 上检查 CQ phase bit、移动 CQ head 并 ring CQ doorbell。

但当前 vLLM KV fast path 还不是完整的 persistent GPU-initiated I/O worker。现在的 KV 路径是：

```text
CPU 粗粒度调度
  -> 生成 GPU request/status table
  -> CPU 调 pybind submit
  -> BaM GPU kernel 发起 SQ read
  -> CPU 调 pybind poll
  -> GPU kernel 服务 CQ
  -> CPU 同步检查 pending_count 并标记 READY
  -> CPU 调 pybind consume
  -> GPU kernel 把 BaM page cache 拷到 pages buffer
  -> GPU refill 回 LMCache KV tensor
```

所以更准确的定位是：**数据面已有 GPU 侧 NVMe submit/CQ service primitive；控制面仍由 CPU 逐 batch 推进 submit/poll/consume。**

下一阶段不要一上来改 NVMe SQ/CQ 的基础实现。更稳的路线是先在 KV 层新增独立的 GPU-visible completion/status table，把 rowctx_compat 当前写 status 的位置抽象成 KV 专用完成表。等这个表稳定后，再把“CPU 调 poll kernel + CPU 同步 pending_count”替换为 GPU-side worker 或 persistent polling。

## 关键源码位置

- `BaMRowStore` 初始化 BaM controller：`/home/xhk/llm-inference/BaM_IOStack/gids_module/bam_row_store.py:91`
- `GIDS_Controllers::init_GIDS_controllers()`：`/home/xhk/llm-inference/BaM_IOStack/gids_module/gids_nvme.cu:330`
- `Controller` 打开 `/dev/libnvm*`、创建 queue pair：`/home/xhk/llm-inference/BaM_IOStack/bam/include/ctrl.h:171`
- `QueuePair` 创建 SQ/CQ 和 GPU helper buffer：`/home/xhk/llm-inference/BaM_IOStack/bam/include/queue.h:134`
- GPU SQ enqueue：`/home/xhk/llm-inference/BaM_IOStack/bam/include/nvm_parallel_queue.h:168`
- GPU CQ peek/dequeue：`/home/xhk/llm-inference/BaM_IOStack/bam/include/nvm_parallel_queue.h:873`
- registered rowctx submit：`/home/xhk/llm-inference/BaM_IOStack/gids_module/gids_nvme.cu:667`
- registered CQ service：`/home/xhk/llm-inference/BaM_IOStack/gids_module/gids_nvme.cu:829`
- registered readiness check：`/home/xhk/llm-inference/BaM_IOStack/gids_module/gids_nvme.cu:938`
- KV batch submit from GPU request table：`/home/xhk/llm-inference/BaM_IOStack/gids_module/gids_nvme.cu:1324`
- KV batch poll/consume：`/home/xhk/llm-inference/BaM_IOStack/gids_module/gids_nvme.cu:1411`
- Python KV executor：`/home/xhk/llm-inference/BaM_IOStack/gids_module/bam_kv_store.py:247`
- vLLM/LMCache KV fast path：`/home/xhk/llm-inference/vllm-bam/vllm/bam/lmcache_bam_kv_fast_path.py:61`

## 初始化链路

vLLM 侧 `BaMRowStore` 会创建 `GIDS_Controllers`，调用：

```text
BaMRowStore.__init__()
  -> GIDS_Controllers.init_GIDS_controllers(num_ssd, 4096, 128, ssd_list)
  -> BAM_Feature_Store_long.init_controllers(...)
```

对应位置是 `/home/xhk/llm-inference/BaM_IOStack/gids_module/bam_row_store.py:91`。

BaM C++ 侧 `init_GIDS_controllers()` 会为每个 SSD 创建一个 `Controller`：

```text
GIDS_Controllers::init_GIDS_controllers()
  -> new Controller(ctrls_paths[ssd_id], namespace, cudaDevice, queueDepth, numQueues)
```

对应位置是 `/home/xhk/llm-inference/BaM_IOStack/gids_module/gids_nvme.cu:330`。

`Controller` 构造函数做的事情：

- 打开 `/dev/libnvm*`。
- 初始化 NVMe controller 和 admin queue。
- `cudaHostRegister(ctrl->mm_ptr, ..., cudaHostRegisterIoMemory)` 把 controller MMIO 映射给 CUDA。
- `reserveQueues(MAX_QUEUES, MAX_QUEUES)` 申请 SQ/CQ。
- 根据硬件和 `numQueues` 得到 `n_qps`。
- 为每个 queue pair 创建 `QueuePair`，再把 `QueuePair` 拷到 `d_qps`，让 GPU kernel 能直接访问。

对应位置是 `/home/xhk/llm-inference/BaM_IOStack/bam/include/ctrl.h:171`。

`QueuePair` 构造函数做的事情：

- 创建 CQ 和 SQ。
- 设置 SQ/CQ 的 DMA memory。
- 初始化 GPU helper buffer：`sq.tickets`、`sq.tail_mark`、`sq.cid`、`cq.head_mark`、`cq.pos_locks`。
- 当前 SQ size 固定为 `4096`，CQ size 取 `min(queueDepth, cap)`。

对应位置是 `/home/xhk/llm-inference/BaM_IOStack/bam/include/queue.h:134`。

## SQ 提交流程

KV path 最终复用 rowctx submit。当前 batch 入口是：

```text
BaMRowCtxKVExecutor.submit()
  -> _prepare_request_table()
  -> row_store.kv_submit_chunk_batch_from_table()
  -> C++ kv_submit_chunk_batch_from_table_with_status_tables()
```

`request_table` 是 CUDA tensor，形状为：

```text
[batch, 4] int64
col0: chunk_id
col1: page_offset
col2: page_count
col3: actual_tokens
```

这个表目前已经是 GPU-visible，位置是 `/home/xhk/llm-inference/BaM_IOStack/gids_module/bam_kv_store.py:390`。

C++ 侧会把 request table 展开成 rowctx 需要的 row id：

```text
request_table [batch, 4]
  -> kv_expand_request_table_kernel
  -> d_row_ids [batch * pages_per_chunk]
  -> read_feature_submit_async_registered_rowctx(d_row_ids, total_pages, ...)
```

对应位置是 `/home/xhk/llm-inference/BaM_IOStack/gids_module/gids_nvme.cu:1324`。

registered rowctx submit 会：

- `cudaMalloc(d_row_ctxs)`，一页一个 `s_ctx`。
- `iostack.register_outstanding()` 在 host deque 里登记 request metadata。
- 启动 `read_feature_kernel_submit_async_rowctx()`。
- 建立 `(logical_queue_idx, cid) -> s_ctx*` 的 lookup table，供 CQ completion 回填 ctx。

对应位置是 `/home/xhk/llm-inference/BaM_IOStack/gids_module/gids_nvme.cu:667`。

真正发起 NVMe read 的 GPU 调用链是：

```text
read_feature_kernel_submit_async_rowctx()
  -> bam_ptr.read_submit_async()
  -> update_page_submit_async()
  -> acquire_page_submit_async()
  -> read_data_submit_async()
  -> sq_enqueue()
```

`read_feature_kernel_submit_async_rowctx()` 在 `/home/xhk/llm-inference/BaM_IOStack/gids_module/gids_kernel.cu:238`。

`acquire_page_submit_async()` 在 cache miss 时会：

- 给 page 设置 `BUSY`。
- 分配 BaM page cache slot。
- 记录 `ctx.page_trans`。
- 根据 backing page 计算 LBA。
- 选择 controller 和 queue。
- 调用 `read_data_submit_async()`。

对应位置是 `/home/xhk/llm-inference/BaM_IOStack/bam/include/page_cache.h:2210`。

`read_data_submit_async()` 在 GPU 上构造 NVMe read command：

```text
get_cid()
nvm_cmd_header()
nvm_cmd_data_ptr(prp1, prp2)
nvm_cmd_rw_blks(starting_lba, n_blocks)
sq_enqueue()
ctx.cid = cid
ctx.has_pending_io = 1
```

对应位置是 `/home/xhk/llm-inference/BaM_IOStack/bam/include/page_cache.h:5038`。

`sq_enqueue()` 的关键行为：

- GPU 线程通过 `sq->in_ticket.fetch_add()` 获取 SQ slot。
- 把 64B NVMe command 写入 SQ memory。
- 使用 `tail_mark` 和 `tail_lock` 合并/推进 tail。
- 通过 `st.mmio.relaxed.sys.global.u32` 写 SQ doorbell。

对应位置是 `/home/xhk/llm-inference/BaM_IOStack/bam/include/nvm_parallel_queue.h:168` 和 `/home/xhk/llm-inference/BaM_IOStack/bam/include/nvm_parallel_queue.h:302`。

这个部分是真正 GPU 侧发起 NVMe SQ 请求。

## CQ 服务与 ready 判断

当前 `gpu_worker` 和 `rowctx` executor 的 poll 最终都走：

```text
row_store.kv_worker_poll()
  -> row_store.kv_poll_batch()
  -> C++ kv_try_poll_batch()
  -> service_registered_try_poll()
```

对应位置是：

- `/home/xhk/llm-inference/BaM_IOStack/gids_module/bam_row_store.py:486`
- `/home/xhk/llm-inference/BaM_IOStack/gids_module/gids_nvme.cu:1411`
- `/home/xhk/llm-inference/BaM_IOStack/gids_module/gids_nvme.cu:1110`

`service_registered_try_poll()` 做两件事：

- 调用 `service_registered_completions_burst()` 服务 CQ。
- 调用 `registered_request_ready_at()` 判断当前 front request 是否所有 ctx 都完成。

`service_registered_completions_burst()` 会启动 `service_registered_cq_window_kernel()`。这个 kernel 一条线程处理一个 logical queue，循环最多处理 `max_events_per_queue` 个 completion。

GPU CQ service kernel 的核心流程：

```text
service_registered_cq_window_kernel()
  -> cq_try_peek_head()
  -> cq_dequeue()
  -> put_cid()
  -> ctx_lookup[(logical_queue_idx, cid)]
  -> finalize_registered_ctx_completion()
```

对应位置是 `/home/xhk/llm-inference/BaM_IOStack/gids_module/gids_kernel.cu:421`。

`cq_try_peek_head()` 会读取 CQ head entry 的 phase bit，判断是否有 completion：

```text
head -> loc
cpl_entry = cq->vaddr[loc].dword[3]
phase bit 匹配则返回 cid/pos/head
```

对应位置是 `/home/xhk/llm-inference/BaM_IOStack/bam/include/nvm_parallel_queue.h:873`。

`cq_dequeue()` 会：

- 加锁 CQ position。
- 推进 CQ head。
- 通过 `st.mmio.relaxed.sys.global.u32` 写 CQ doorbell。
- 释放 position lock。

对应位置是 `/home/xhk/llm-inference/BaM_IOStack/bam/include/nvm_parallel_queue.h:900`。

这个部分是真正 GPU 侧服务 CQ completion。

但 ready 判断目前还不是纯 GPU control。`registered_request_ready_at()` 会：

```text
cudaMemset(d_registered_try_pending, 0)
check_registered_request_ready_kernel<<<...>>>()
cudaDeviceSynchronize()
cudaMemcpy(pending_count, D2H)
return pending_count == 0
```

对应位置是 `/home/xhk/llm-inference/BaM_IOStack/gids_module/gids_nvme.cu:938`。

这意味着：CQ entry 的读取和处理在 GPU 上，但每次 poll 是否完成仍由 CPU 调 kernel、CPU 同步、CPU 读取 pending_count 后决定。

## consume/get 流程

KV batch ready 后，当前 consume 仍要求 FIFO front request：

```text
row_store.kv_consume_chunk_batch()
  -> while not kv_poll_batch(): pass
  -> store.kv_consume_chunk_batch(out_rows_i64.data_ptr())
  -> C++ read_feature_get_feature_light_registered_rowctx()
```

对应位置是 `/home/xhk/llm-inference/BaM_IOStack/gids_module/bam_row_store.py:569` 和 `/home/xhk/llm-inference/BaM_IOStack/gids_module/gids_nvme.cu:1433`。

底层 get kernel 是：

```text
read_feature_kernel_get_feature_light_rowctx()
  -> update_page_post_poll_light()
  -> 把 page cache 中的数据拷贝到 out_tensor
```

对应位置是 `/home/xhk/llm-inference/BaM_IOStack/gids_module/gids_kernel.cu:261`。

对于 KV fast path，`out_tensor` 是：

```text
[batch * pages_per_chunk, 128KB] uint8 CUDA
```

vLLM 侧再调用 refill：

```text
[page_count, 128KB] pages
  -> refill_pages_to_lmcache_tensor()
  -> [2, layers, tokens, hidden] LMCache KV tensor
```

对应位置是 `/home/xhk/llm-inference/vllm-bam/vllm/bam/lmcache_bam_kv_fast_path.py:108`。

## 当前 KV fast path 的分层

当前 Python 层有两层 executor：

```text
BaMRowCtxKVExecutor
  -> 直接复用 rowctx batch primitive

BaMGPUWorkerKVExecutor
  -> 调 row_store.kv_worker_* façade
  -> 当前 worker_backend=rowctx_compat
```

`BaMGPUWorkerKVExecutor` 的意义是接口分离，不是说底层已经完全换成 persistent GPU worker。它当前已经不直接 fallback 到 `BaMRowCtxKVExecutor.submit/poll/consume`，而是走 `kv_worker_submit/poll/consume` 三个独立入口。后续只要替换 `BaMRowStore.kv_worker_*` 内部实现，上层 vLLM/LMCache 不需要改。

对应位置是 `/home/xhk/llm-inference/BaM_IOStack/gids_module/bam_kv_store.py:598` 和 `/home/xhk/llm-inference/BaM_IOStack/gids_module/bam_row_store.py:453`。

## CPU/GPU 职责划分现状

| 模块 | 当前 CPU 参与 | 当前 GPU 参与 | 备注 |
|---|---|---|---|
| vLLM/LMCache 命中判断 | CPU 决定要 retrieve 哪些 chunk | 无 | 这是调度控制面，短期仍应由 CPU 做 |
| KV request table | CPU 构造 chunk descriptor | request table 是 CUDA tensor | 当前是 GPU-visible，但不是 GPU 自主生成 |
| KV row id 展开 | CPU 调 pybind 入口 | CUDA kernel 展开 request table | 已减少 Python 逐 page 构造 |
| SQ submit | CPU launch submit kernel | GPU 构造 NVMe command、写 SQ、ring SQ doorbell | 底层已经是 GPU 发起 SQ |
| CQ service | CPU 调 poll 入口并 launch service kernel | GPU peek/dequeue CQ、finalize ctx | CQ 数据面在 GPU，推进节拍在 CPU |
| ready 判断 | CPU 同步并 D2H 读取 pending_count | GPU kernel 统计 pending ctx | 这是当前最大 CPU control 缝隙 |
| consume/get | CPU 调 consume 入口 | GPU 从 page cache 拷贝 pages buffer | FIFO front request 约束仍在 |
| refill | CPU launch kernel | GPU 把 pages 还原 KV tensor | 当前逐 chunk refill |
| 日志/验证 | CPU 读 GPU status tensor | GPU status tensor 保存状态 | 只用于验证和日志，不是数据面瓶颈 |

## 可以复用的部分

以下部分建议保留并继续复用：

- `Controller` 和 `QueuePair` 初始化逻辑：已有 MMIO 映射、SQ/CQ 创建、doorbell 映射，不需要为 KV 另起一套。
- `sq_enqueue()`：已经实现 GPU 侧 SQ entry 写入和 doorbell。
- `cq_try_peek_head()` / `cq_dequeue()`：已经实现 GPU 侧 CQ phase 检查、head 推进和 doorbell。
- 128KB page 作为 KV row：与当前 BaM 硬件单次 IO 上限匹配。
- KV request table `[batch, 4]`：足够表达 chunk 级请求，后续 persistent worker 可以直接消费。
- KV executor façade：`kv_worker_*` 已经把上层接口与 rowctx_compat 分开，是后续替换底层 worker 的好边界。

## 风险点与不建议直接动的部分

以下部分不建议作为下一步第一刀：

- 不建议直接重写 `sq_enqueue()` 或 `cq_dequeue()`。它们牵涉 NVMe queue 正确性、doorbell、ticket、phase bit 和 SQ/CQ head/tail 一致性，改错会影响所有 BaM 场景。
- 不建议直接改通用 page cache 状态机。GNN/CNN/read_feature 路径仍依赖它，KV 专用优化最好新增 fast path，不要破坏通用路径。
- 不建议把 host-side outstanding deque 立即删除。当前 consume 依赖 FIFO front request，先新增 KV completion/status table，再逐步弱化 FIFO 假设更稳。
- 不建议把验证日志里的 GPU status D2H 当成主性能瓶颈。它只在日志/校验阶段发生，真正要优化的是 poll readiness 的同步路径和逐 batch 串行推进。

## 推荐下一步

第一步：新增 KV 专用 completion/status table 抽象。

目标不是马上替换 NVMe CQ，而是把当前 `host map + gpu_status + gpu_chunk_status` 组织成明确的 KV completion table：

```text
request_table:    [batch, 4] int64 CUDA
completion_table: [batch, 4] int64/int32 CUDA

completion_table col0: chunk_id
completion_table col1: status
completion_table col2: bytes_done
completion_table col3: error_code
```

当前 rowctx_compat 仍然可以由 CPU/C++ 在 `SUBMITTED -> IO_DONE -> CONSUMED` 阶段写这张表。这样做的意义是把后续 GPU worker 需要写的接口先固定下来。

第二步：把 ready 判断从 `pending_count D2H` 迁移到 GPU-visible completion table。

当前 `registered_request_ready_at()` 每次都会 `cudaDeviceSynchronize()` 并把 `pending_count` 拷回 CPU。下一阶段可以先让 poll 只读 batch/chunk status，减少对通用 rowctx readiness 细节的依赖。

第三步：新增 KV 专用 CQ service kernel。

这一步仍可复用现有 `cq_try_peek_head()` 和 `cq_dequeue()`，但 completion 写入不再只回填 `s_ctx`，而是直接写 KV completion table：

```text
CQ completion cid
  -> kv_ctx_lookup[cid]
  -> mark completion_table[chunk/page] done
  -> when all pages done, mark chunk IO_DONE
```

第四步：做 persistent GPU worker。

worker 常驻 GPU，循环读取 request table，发起 SQ，poll CQ，写 completion table。CPU 只负责粗粒度提交 request table 和在 vLLM scheduler 中决定何时需要哪些 chunk。

第五步：减少 consume/refill 边界。

当前 consume 先把 BaM page cache 拷到 `[batch * pages_per_chunk, 128KB]` pages buffer，再 refill 到 LMCache tensor。最终更激进的版本可以让 KV worker 直接写入 vLLM paged KV cache 或 LMCache tensor 目标位置，减少一次中间 buffer 和一次 kernel。

## 下一步实现边界建议

建议新增代码时保持三层边界：

```text
vLLM/LMCache 层
  只负责 chunk metadata、命中判断、调用 fast path。

BaM KV executor 层
  维护 request_table / completion_table / status snapshot。
  暴露 submit / poll / consume。

BaM 底层 worker 层
  当前 rowctx_compat 写 completion table。
  后续 GPU worker 复用同一张表和同一组接口。
```

这样可以保证：

- 原生 LMCache SSD 路径不受影响。
- 已经能跑的 BaM sync / prefetch / KV fast path 不被打坏。
- GNN/CNN 的通用 read_feature 路径不需要跟着 KVCache 改。
- 后续从第 2 档 KV fast path 迁移到第 3 档 persistent GPU worker 时，上层接口不用推倒重来。

## 当前判断

我现在不建议声称“我们已经熟知并重写了 BaM SQ/CQ 全细节”。更准确的说法是：

- 已经审计清楚现有 SQ/CQ 的关键初始化、submit、doorbell、CQ peek/dequeue 和 rowctx lifecycle。
- 已经确认当前 KV fast path 的真实底层仍是 `rowctx_compat`。
- 已经确认可复用的底层 GPU SQ/CQ primitive。
- 已经确认下一步应先做 KV completion/status table，而不是直接改通用 SQ/CQ 状态机。

这条路线最稳，也最符合前面 TARDIS/Tutti/AGIO 给出的启发：CPU 仍做粗粒度调度和策略决策，GPU 负责高频 I/O 数据面和 completion 推进；先把 GPU-visible request/completion ABI 固定，再逐步把 poll/control 迁到 GPU。
