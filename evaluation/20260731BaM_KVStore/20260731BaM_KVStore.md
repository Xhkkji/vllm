# BaM KVStore 后续实现与分支管理建议

日期：2026-07-31

## 1. 目标定位

后续目标是把 BaM 建设成一个与 LMCache 并列的独立 KV 存储后端，而不是继续把
BaM 包装在 LMCache 的 chunk 读写路径下面。

新链路采用 vLLM 原生 block 作为逻辑数据单元：

- vLLM scheduler 负责请求调度、prefix block 命中判断和 GPU physical block 分配。
- BaM KVStore 负责逻辑 block 到 SSD LBA 的索引、SSD 空间分配和请求提交。
- BaM_IOStack 的 GPU worker 负责 NVMe SQ submit、CQ polling 和 ready flag 发布。
- NVMe DMA 直接读写 vLLM 已经分配好的 paged KV cache，不经过 BaM data cache。
- CPU 不轮询 CQ，也不搬运 KV 数据；CPU 只负责控制面调度和推进计算。

目标读取链路如下：

```text
vLLM Scheduler
  -> 查询连续命中的 KV block
  -> 分配目标 vLLM GPU physical blocks
  -> 生成 {SSD LBA, length, destination block/offset}
  -> GPU persistent worker 提交 NVMe read
  -> NVMe DMA 直接写入 vLLM paged KV cache
  -> GPU CQ worker 发布 block/group ready flag
  -> CPU 只读 request ready 状态
  -> CPU 发起 attention
  -> Attention 直接消费已经恢复的 KV block
```

目标写入链路如下：

```text
vLLM 完成 KV block 计算
  -> 生成 {source block/offset, SSD LBA, length}
  -> GPU persistent worker 提交 NVMe write
  -> NVMe DMA 直接从 vLLM paged KV cache 写入 SSD
  -> GPU CQ worker发布 write-complete
  -> BaM KVStore 提交 block index，使该 block 对后续请求可见
```

这里“不再使用 BaM cache”指不再保留 GB 级、承载 KV payload 的 BaM page cache。
SQ/CQ、request table、PRP list、ready flag 等少量 GPU 控制内存仍然需要保留。

## 2. 数据组织方式

### 2.1 逻辑粒度

SSD 上以一个 vLLM token block 作为逻辑对象。一个逻辑 block 包含当前 TP rank
负责的全部 attention layer 的 K/V 数据：

```text
Logical KV Block N
  Layer 0 K
  Layer 0 V
  Layer 1 K
  Layer 1 V
  ...
  Layer L-1 K
  Layer L-1 V
```

第一版不改变 vLLM attention backend 的显存布局。每层 K/V 在 vLLM GPU cache
中可能不是一个跨层连续区域，因此一个“逻辑 block”可以展开成多个直接 DMA
fragment。逻辑上按 block 管理，不要求第一版用一条 NVMe 命令搬完整个 block。

以 Qwen2.5-7B、FP16、block size 16、28 个 attention layer、4 个 KV head、
head size 128 为例：

- 每层单独的 K 或 V fragment 约为 16KB。
- 每层 K+V 约为 32KB。
- 一个完整 vLLM block 约为 896KB。
- 第一版可以按 layer/KV fragment 批量提交，优先保证直接落址和正确性。
- 后续再根据 IOPS、SQ 深度和命令开销决定是否做 PRP 聚合或连续 block 合并。

### 2.2 SSD 索引

vLLM `physical_block_id` 只表示当前进程中的显存槽位，会被释放和复用，不能直接
作为跨请求 prefix cache 的持久 key。因此需要一个轻量索引：

```text
(model/layout/tp-rank/token-block-hash) -> SSD extent/LBA
```

索引只保存 key、LBA、长度、有效 token 数和状态，不保存 KV payload，因此不属于
BaM data cache。

第一版建议只缓存完整 vLLM block：

- 完整 block 可以稳定计算内容 hash，并被不同请求复用。
- prompt 尾部未满 block 暂不写入 SSD，由 vLLM 重算。
- 写请求的全部 fragment 完成后才能提交索引，避免半写入 block 被读取。

## 3. 新链路的模块边界

建议新建独立模块，而不是在现有 `bam_kv_store.py` 中继续增加第四套兼容模式：

```text
vllm-bam
  BaMConnector
    - scheduler-side lookup
    - block allocation metadata
    - worker-side load/save orchestration

  BaMDirectBlockStore
    - block key / SSD extent index
    - block request generation
    - request lifecycle

BaM_IOStack
  BaMDirectKVIO
    - register external CUDA allocations
    - build PRP from vLLM KV cache mappings
    - GPU-visible read/write request table
    - persistent SQ submit / CQ poll
    - ready/error status publication
```

建议第一版 request descriptor 收敛为：

```text
request_id
operation             # read / write
ssd_lba
length
dma_mapping_id
gpu_buffer_offset
logical_block_id
fragment_id
status/error
```

descriptor 不再包含 LMCache chunk、128KB page、refill tensor 或 xFormers metadata。

## 4. 当前代码中可以复用的部分

### 4.1 vllm-bam

可以复用：

- vLLM scheduler、block manager 和 physical block 分配语义。
- `CacheEngine` 已分配的每层 GPU KV cache 及其生命周期。
- `BaMKVLayout` 中根据模型配置描述 block 大小和层数的思路。
- Connector factory 和 V1 KV connector 抽象。
- `get_num_new_matched_tokens()`、`update_state_after_alloc()`、
  `build_connector_meta()` 等 scheduler-side 接口。
- `start_load_kv()`、`wait_for_layer_load()` 和 `save_kv_layer()` 等 worker-side
  接口。
- 已有 baseline 文档、数据集 runner 和仍然有效的对照实验脚本。

需要注意：当前正式验证主要基于 V0/V100。长期接口更适合 V1 Connector；如果
V100 环境暂时无法稳定运行 V1，应先让底层 `BaMDirectBlockStore` 与 V0/V1 解耦，
再做一个最小 V0 调度适配层，不要继续依赖 LMCache worker-side metadata rebuild。

### 4.2 BaM_IOStack

可以复用：

- NVMe controller、SQ、CQ 和 doorbell 基础实现。
- 每个线程负责一个 CQ 的 queue ownership 约束。
- GPU persistent service 的启动、停止和 request slot 生命周期。
- GPU-visible request/status/completion table。
- 已验证的 CQ completion、错误状态和 ready/frontier 发布思路。
- `nvm_dma_map_device()` 对已有 CUDA device pointer 的注册能力。
- 当前 1 CTA service + 多 CTA 协作版本中的状态收口经验。

vLLM KV cache 应在 `CacheEngine` 初始化后注册一次，DMA mapping 生命周期与整块
KV cache allocation 保持一致；不能按 request 反复注册和注销。

## 5. 需要重构的部分

### 5.1 直接 DMA 映射

当前 BaM 的 PRP 地址主要指向 BaM 自己分配的 page cache。新实现需要支持：

1. 接收 vLLM 每层 KV tensor 的 base pointer 和 allocation size。
2. 使用 `nvm_dma_map_device()` 注册完整 allocation。
3. 保存每个 controller 对应的 IO address table。
4. 根据 layer、K/V、physical block id 计算目标 offset。
5. 为每个 fragment 构造直接指向 vLLM KV cache 的 PRP。

必须在初始化时验证指针对齐、映射范围、tensor stride 和 fragment 边界，禁止在
设备侧根据未经校验的地址盲写。

### 5.2 独立 BaM Connector

新 Connector 不应导入 LMCache engine，也不应依赖 LMCache chunk key。它负责：

- 根据 token block 和模型布局生成 block key。
- 查询可以连续命中的 prefix blocks。
- 在 vLLM 分配目标 physical blocks 后生成 IO metadata。
- 启动 GPU-side read/write。
- 将 ready handle 绑定到当前 batch/layer group。
- 在 block 被重用或 request 结束时正确处理引用和 SSD extent 生命周期。

### 5.3 GPU 同步

NVMe CQ completion 只表示设备 IO 已完成；在发布 ready flag 前还要保证 DMA 写入
对后续 GPU kernel 可见。新链路需要明确：

- CQ worker 在观察到 completion 后执行必要的 system/device fence。
- ready flag 使用 release 语义发布。
- CPU 只读取 GPU worker 已发布的 request 状态，不读取或推进 NVMe CQ。
- CPU 观察到 request ready 后发起 attention，attention 直接消费最终 KV cache。
- persistent service 必须保留足够 GPU 资源，并在启动前完成模型 workspace 和
  临时显存分配，避免运行期间的 `cudaMalloc/cudaFree` 与常驻 kernel 互相等待。

第一版可以先做 request 级“全部 block ready 后 forward”；正确后再推进
layer-group ready 和 IO-compute overlap。

## 6. 新分支中可以清理的旧逻辑

### 6.1 vllm-bam

新链路验证通过后，可以从新分支删除：

- LMCache BaM shadow write / prefer-load wrapper。
- LMCache chunk、128KB page、prefetch、refill 和 storage adapter。
- `lmcache_bam_direct_placement.py` 中的旧 placement plan。
- `lmcache_bam_kv_fast_path.py` 中的 chunk request 生命周期。
- `BaM cache -> output_pages -> vLLM KV` 两次搬运路径。
- 旧 fused placement 和 runtime one-copy 的上层适配。
- xFormers prefix fallback、query scatter 和 metadata rebuild 实验逻辑。
- deferred retrieve、minimum poll count 和 idle-stop benchmark 逻辑。
- V0 CPU swap shadow、prefer BaM swap-in 和 reference verify 实验分支。
- 对应的 `VLLM_BAM_LMCACHE_*`、`VLLM_BAM_DIRECT_PLACEMENT_*`、
  xFormers debug 和 metadata attachment 环境变量。

`gds_baseline` 可以继续作为 evaluation 对照，但不应被新的生产路径依赖。

### 6.2 BaM_IOStack

新 direct KV IO 路径不再需要：

- KV chunk/page-offset descriptor。
- `output_pages` staging buffer。
- BaM cache page 到 vLLM paged KV cache 的 mover scatter。
- materialized 与 one-copy 两套 consume/finalize。
- runtime placement attachment 和 slot mapping attachment。
- attention metadata rebuild 字段。
- BaM page-cache refcount、cache-page release 和 deferred allocation cleanup。
- LMCache refill 专用状态和 direct-placement debug 字段。

不能直接删除的通用部分：

- `page_cache.h` 和 BaM 通用 page cache。
- GNN/CNN/feature store 使用的 rowctx 路径。
- 通用 controller、queue、buffer 和 DMA API。

这些功能仍属于 BaM_IOStack 的其他使用场景。新 KV 路径应通过独立类避免实例化
page cache，而不是破坏通用 BaM 功能。

## 7. 1+4 CTA 模式在新链路中的变化

当前 1+4 CTA 模式的语义是：

```text
1 CTA/CQ service
  +
4 mover CTA 将 BaM cache page scatter 到 vLLM KV cache
```

新链路由 NVMe DMA 直接写入 vLLM KV cache，因此 mover scatter 应被删除。可以
复用的是：

- persistent service 控制块。
- CQ ownership 和每线程一个 CQ 的约束。
- request slot、completion 和 ready 状态机。
- 最后完成者负责状态收口的经验。

如果后续需要多个 CTA，应让 CTA 服务不同 CQ、SSD 或 request shard，而不是继续
搬运 KV payload。第一版先用最少 CTA 验证正确性，再根据 SQ/CQ backlog、SSD
带宽和 GPU 占用决定 CTA 数量。

## 8. 推荐实施顺序

### 阶段 0：冻结基线

- 保留两个旧开发分支作为当前可运行版本。
- 给两个仓库的兼容提交分别打 tag，并在文档中记录提交配对。
- 新开发只在 `feature/bam-kvstore-vllm-connector` 上进行。

### 阶段 1：外部显存直接 DMA primitive

- 注册一块已有 CUDA tensor，而不是由 BaM 分配 cache。
- 完成指定 LBA 到指定 tensor offset 的 read/write。
- 验证随机 offset、跨 GPU page 边界和多请求 exact equality。

### 阶段 2：vLLM block roundtrip

- 注册 vLLM 每层 KV cache allocation。
- 根据随机 physical block id 生成全部 layer/K/V fragment。
- 完成 SSD write -> 清零目标 block -> SSD read -> exact equality。
- 覆盖非连续 block id、block id 复用和多 block batch。

### 阶段 3：BaMDirectBlockStore

- 实现 SSD extent allocator。
- 实现 block key 到 LBA 的内存索引。
- 只支持完整 block 和固定模型布局。
- 写完成后原子提交索引。

### 阶段 4：vLLM Connector 接入

- 注册独立 `BaMConnector`，与 LMCacheConnector 并列。
- scheduler 查询命中并分配目标 GPU blocks。
- worker 生成 direct IO request table。
- GPU ready 后推进 attention。

### 阶段 5：删除旧 KV 实验链路

- 删除 LMCache chunk/refill/direct-placement 依赖。
- 删除旧环境开关和失效脚本。
- 收缩 C++ runtime slot 和 Python handle。
- 保留必要 baseline 文档，删除只服务旧实现的生产入口。

### 阶段 6：性能优化

- 批量提交多个 block fragment。
- 多 CQ/多 SSD 并行。
- 合并连续 block 或构造更高效的 PRP list。
- layer-group prefetch。
- 多请求 prefill 与 KV restore overlap。

## 9. 正确性与性能验收标准

每个阶段至少验证：

- 所有 layer、K/V、block 的 byte-level exact equality。
- 随机且非连续 physical block id。
- block id 被释放并重新分配后不会读到旧 KV。
- partial block 不会被错误标记为完整命中。
- write 未完成前索引不可见。
- CQ error 能传播到 request，而不是永久停在 wait。
- CPU 不轮询 NVMe CQ、不搬运 KV；只允许批量读取 request ready 状态。
- 数据路径中不存在 BaM payload cache 和额外 GPU scatter。
- persistent worker 停止、进程退出和异常 cleanup 不会卡住。

性能测试至少分开记录：

- 纯 SSD -> vLLM KV direct read bandwidth。
- request submit 和 CQ completion latency。
- 单 block 与多 block batch 的 IOPS/带宽。
- KV restore time。
- TTFT、decode throughput 和端到端 request latency。
- persistent CTA 对 attention compute 的资源影响。

## 10. 分支管理建议

### 10.1 当前分支

截至 2026-07-31：

| 仓库 | 新开发分支 | 起点提交 | 对应旧稳定分支 |
|---|---|---|---|
| `vllm-bam` | `feature/bam-kvstore-vllm-connector` | `ffcee8f90` | `xhk/bam-sync-swap-v100` |
| `BaM_IOStack` | `feature/bam-kvstore-vllm-connector` | `f099e4c` | `xhk/bam-vllm-swapout` |

两个新分支当前分别与旧稳定分支指向同一提交。后续在新分支提交的删除和重构不会
改变旧分支内容，因此旧分支可以继续作为 1+4 CTA、LMCache prefer-load 和历史
正确性实验的可运行归档。

### 10.2 建议 tag

在开始重构前，可以分别创建不可移动的 tag：

```bash
cd /home/xhk/llm-inference/vllm-bam
git tag baseline/bam-lmcache-one-copy-20260731 ffcee8f90

cd /home/xhk/llm-inference/BaM_IOStack
git tag baseline/bam-iostack-one-copy-20260731 f099e4c
```

tag 名称不同，但日期一致，并在本文件中记录配对关系。

### 10.3 两个仓库必须成对切换

新旧 Python/CUDA API 会逐步分离，不能只切换一个仓库：

```text
新 vllm-bam feature branch
  <-> 新 BaM_IOStack feature branch

旧 vllm-bam stable branch
  <-> 旧 BaM_IOStack stable branch
```

建议在每次跨仓库 ABI 变更的提交信息中记录另一个仓库对应的 commit hash，并在
测试日志中同时打印两个仓库的 commit。

### 10.4 提交组织

不要把“新接口、数据面重构、删除旧代码”放进同一个大提交。建议按以下顺序提交：

1. `Add external CUDA memory DMA registration API`
2. `Add direct KV block read/write primitives`
3. `Add vLLM block layout and SSD index`
4. `Add standalone BaM KV connector`
5. `Remove LMCache chunk and refill compatibility path`
6. `Remove legacy direct-placement runtime switches`
7. `Add direct-block regression and performance tests`

这样出现回归时可以单独定位接口、数据面或清理提交。

### 10.5 旧分支和新分支的使用方式

- 旧分支只用于复现实验、性能对照和查阅旧实现，不再继续开发。
- 新分支只保留新的 direct-block 主线，不新增旧模式 fallback。
- 不需要在新分支建立庞大的 `legacy/` 目录；Git 历史和旧分支已经保存旧实现。
- evaluation 文档可以保留历史结果，但失效脚本应明确标记或从新分支删除。
- BaM_IOStack 的通用 GNN/CNN/page-cache 功能继续保留，不能因 KV 清理而破坏。

如需同时查看和测试新旧代码，建议使用 `git worktree` 分别检出，而不是在同一
工作目录频繁切换。这样也可以避免编译产物和 Python extension 在两个 ABI 之间
互相污染。

## 11. 当前阶段结论

这次重构不需要重新实现整个 BaM_IOStack。最有价值、可以直接继承的是 GPU
NVMe SQ/CQ、persistent service、request/status table 和 CUDA memory DMA
mapping。真正需要重写的是当前绑定 LMCache chunk、BaM page cache、refill 和
direct scatter 的 KV 上层数据面。

最终主线应保持为：

```text
vLLM block scheduling
  -> thin SSD index
  -> GPU-visible direct IO descriptor
  -> NVMe DMA directly to/from vLLM KV cache
  -> GPU ready flag
  -> attention compute
```

在这个边界下，后续 layer-group prefetch、多请求 IO-compute overlap 和更完整的
GPU-initiated 调度都可以继续建立在同一套 block/request ABI 上，而不需要再次
经过 LMCache chunk 或 BaM cache 中转。

### 11.1 2026-07-31 direct-block 验证结果

当前第一版已经验证：

- Qwen2.5-7B、FP16、28 层、block size 16 的 vLLM-layout KV allocation。
- 每个 logical block 展开为 56 个 16KB layer/K/V fragment，共 917,504 bytes。
- NVMe DMA 直接读写最终 vLLM-layout physical block，不经过 BaM payload cache。
- GPU persistent service 独立完成 CQ polling；CPU 只检查 request ready 并发起
  后续 GPU marker，service 在 marker 执行时仍保持运行。
- 1-block roundtrip：56 fragments，exact_equal=1。
- 4-block batch roundtrip：224 fragments、3,670,016 bytes，exact_equal=1。
- 4-block 实测 write/read 均约 1.26 GiB/s；该数据只用于当前小批量正确性测试，
  不作为正式吞吐 baseline。
## 11. 新调用链代码标记

> 本目录中的新路径统一以 `【BaM KVStore 直通调用链】` 标记。该标记表示函数
> 属于 `vLLM block metadata -> GPU submit -> SSD DMA -> vLLM KV cache -> GPU
> ready -> CPU launch attention` 主线，不属于旧 LMCache chunk/page cache 路径。

## 12. 真实 CacheEngine 接入进度（2026-07-31）

当前已经把独立 direct block 数据面接入 vLLM V0 `CacheEngine`：

```text
Scheduler blocks_to_swap_out: GPU block -> storage block
  -> CacheEngine.swap_out
  -> BaMVLLMDirectKVStore
  -> BaMDirectBlockStore 展开 layer/K/V fragments
  -> GPU submit + persistent CQ poll
  -> SSD write 完成

Scheduler blocks_to_swap_in: storage block -> GPU block
  -> CacheEngine.swap_in
  -> SSD DMA 直接写入真实 vLLM paged KV cache
  -> GPU 发布 request ready
  -> CPU 只检查 ready
  -> Worker 继续发起 attention
```

实现约束如下：

- 新路径仅由 `VLLM_BAM_DIRECT_KVSTORE_ENABLE=1` 显式开启，默认关闭。
- 开启后，CPU block id 只作为 SSD storage block id，不再分配 CPU KV payload。
- vLLM GPU KV cache 使用 64KB 对齐 owner，但 attention 看到的 shape/stride
  与原生 XFormers paged KV cache 一致。
- BaM controller 初始化和 DMA registration 延迟到模型 warmup、CUDA Graph
  capture、workspace allocation 全部完成之后。
- persistent CQ worker 仍然延迟到第一批 submit 才启动。
- SSD extent 位于 namespace 尾部实验区域之前，并额外保留 64MB tail guard。
- 旧 LMCache SSD、GDS、BaM page cache 和普通 V0 CPU swap 路径没有改默认行为。

真实 `CacheEngine` smoke 使用 Qwen2.5-7B、FP16、block size 16、XFormers
layout，结果如下：

```text
cache_engine_layout=(2, 8, 8192)
stride=(65536, 8192, 1)
compute_launch_after_ready=1
cpu_kv_payload_cache_layers=0
bam_page_cache=0 staging=0 refill=0
exact_equal=1
[DIRECT_KV_CACHE_ENGINE] PASS
```
