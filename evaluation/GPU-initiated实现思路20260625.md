# GPU-initiated BaM 实现思路

日期：2026-06-25
最近整理：2026-07-29

本文只保留当前 `vllm-bam` 中与 LMCache / vLLM KVCache 主线直接相关、并且仍有工程价值的实现思路。
目标不是记录所有历史尝试，而是回答下面四个问题：

```text
1. 现在到底已经实现到哪里了？
2. 当前真实跑通的数据通路是什么？
3. 当前性能瓶颈到底在哪里？
4. 下一步应该沿哪条主线继续推进？
```

建议阅读顺序：

```text
1. 先看“1. 当前主线结论”和“1.-1 2026-07-27 one-copy 最新修正结论”
2. 再看“3. 当前真实数据通路”
3. 再看“4. 当前异步/轮询逻辑到底是什么”
4. 再看“14. 相关 SSD/KVCache 工作的评测数据集与后续 baseline 选择”
5. 最后看“15. 2026-07-23 当前收束结论与后续创新点”
```

---

## 1. 当前主线结论

### 1.-0 2026-07-29 decode-only NVTX / Nsight 最新结论

截至 2026-07-29，`gpu_worker_persistent_one_copy` 的一个状态机误报已经修正：

```text
问题：
  gpu_worker_submit 后原本强制校验 request frontier status == SUBMITTED；
  但 persistent GPU worker 是异步推进的，submit 返回前 frontier 可能已经被推进到
  IO_DONE / CONSUMED。

修正：
  只在 gpu_worker_submit 阶段允许 frontier 从 SUBMITTED 合法前进到
  IO_DONE / CONSUMED；
  仍然拒绝 ERROR 或未知状态。

影响：
  该修正只调整 Python 层状态校验语义，不改变 BaM / CUDA 数据面。
```

修正后完成的关键验证：

```text
BaM one-copy 1+4 CTA
  NUM_SAMPLES=1, REPEAT_READ=1, MAX_TOKENS=16:
    正常跑通

BaM one-copy 1+4 CTA + Nsight Systems:
  NUM_SAMPLES=1, REPEAT_READ=1, MAX_TOKENS=16:
    正常跑通，能够生成包含 NVTX / CUDA trace 的 nsys-rep

BaM one-copy 1+4 CTA 长 decode 压力：
  NUM_SAMPLES=5, REPEAT_READ=1, MAX_TOKENS=128:
    requests=10
    write_avg_s=3.5267
    read_avg_s=3.0458
    avg_request_s=3.2863
```

同口径 GDS 对照：

```text
LMCache GDS
  NUM_SAMPLES=5, REPEAT_READ=1, MAX_TOKENS=128:
    requests=10
    write_avg_s=3.4125
    read_avg_s=3.0890
    avg_request_s=3.2508

BaM vs GDS read speedup @128 tokens:
  1.014x
```

这说明：

```text
1. 修正后，之前 max_tokens=128 下的 frontier status mismatch 不再复现；
2. 长 decode 下，BaM SSD->GPU read path 的端到端优势被 decode 计算明显稀释；
3. 继续只优化 KV read path，难以在长 decode 场景下获得显著端到端收益。
```

#### decode-only NVTX / Nsight 结论

为判断“多轮 decode 中 CPU 调度/切换是否是瓶颈”，新增了默认关闭的 NVTX 观测开关：

```text
LONGBENCH_NVTX_TRACE=1
  标记 longbench_request:write/read

VLLM_BAM_NVTX_TRACE=1
  标记 vllm_engine_schedule
  标记 vllm_engine_execute_model
  标记 vllm_recv_kv:prefill/decode
  标记 vllm_forward:prefill/decode
  标记 vllm_logits:prefill/decode
  标记 vllm_sample:prefill/decode
```

NVTX trace 口径：

```text
dataset:
  LongBench TriviaQA lt4k bucket
model:
  Qwen2.5-7B-Instruct
NUM_SAMPLES:
  1
REPEAT_READ:
  1
MAX_TOKENS:
  16
paths:
  BaM one-copy 1+4 CTA
  LMCache GDS
```

read request 的 decode-only 对比结果：

| 路径 | read request | decode token span | GPU busy / token | token 间 gap | schedule 总耗时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BaM one-copy 1+4 CTA | 1082.8 ms | 约 22 ms/token | 约 21 ms/token，95%+ | 约 0.9 ms | 1.4 ms / 16 steps |
| LMCache GDS | 1090.0 ms | 约 22 ms/token | 约 21 ms/token，95%+ | 约 0.9 ms | 1.4 ms / 16 steps |

BaM read request 细分：

```text
read request duration:
  1082.779 ms

read request GPU kernel busy:
  434.772 ms
  busy_ratio=0.402

vLLM decode steps:
  decode_forward count=15
  decode_forward avg=14.94 ms/token
  decode_logits avg=0.10 ms/token
  decode_sample avg=7.04 ms/token

per-token step span:
  约 21.8 - 22.7 ms

per-token GPU busy:
  约 20.8 - 21.7 ms
  busy_ratio≈95%

token 间 inter_gap:
  约 0.9 ms

vllm_engine_schedule:
  16 次合计约 1.4 ms
  单步约 0.08 - 0.09 ms
```

GDS read request 呈现基本一致的形态：

```text
decode_forward avg:
  约 14.5 - 15.3 ms/token

decode_sample:
  约 6.7 - 7.9 ms/token

per-token GPU busy_ratio:
  约 95%+

token 间 inter_gap:
  约 0.9 ms

vllm_engine_schedule:
  16 次合计约 1.4 ms
```

因此当前结论是：

```text
当前单请求 decode 下，CPU scheduler / 多轮 decode 切换不是主要瓶颈。

虽然整个 read request 的 GPU busy ratio 只有约 39% - 40%，但低 busy 主要来自
read/prefill/restore/sampling 等阶段混合后的整体口径；在真正的 per-token
decode forward/logits/sample 区间内，GPU kernel busy ratio 已经接近 95%。

token 间确实存在约 0.9 ms gap，但相比单 token 约 22 ms 的执行时间不是主导项。
```

当前更值得优先推进的方向应调整为：

```text
1. layer 级 KV prefetch / IO-compute overlap；
2. SSD read 与 transformer layer 计算的流水线重叠；
3. sampling / launch overhead 优化；
4. CUDA Graph bucket / 取消 enforce-eager 的对照实验；
5. 多请求 batch 下调度开销是否被摊薄的进一步测试。
```

暂时不建议立刻大改成完整 GPU-side decode loop：

```text
理由：
  现有 NVTX trace 不支持“CPU 调度是 decode 主瓶颈”的判断；
  当前 decode token 内部 GPU 已经高度忙碌；
  大改 GPU-side decode loop 的工程风险高，收益证据不足。
```

### 1.-1 2026-07-27 one-copy 最新修正结论

截至 2026-07-27，`gpu_worker_persistent_one_copy` 不能再简单视为
no-debug 条件下的稳定正确基线。

最新排查结论是：

```text
one-copy 数据面确实触发了 BaM -> vLLM paged KV cache 的 direct placement；
prefix hit / runtime attach / runtime ready / deferred retrieve done 都能正常出现；
但在 GIDS_KV_DEBUG=0 的 normal hot path 下，write/read 文本级一致性不稳定。

GIDS_KV_DEBUG=1 会把结果从 1/4 exact 改成 4/4 exact；
VLLM_BAM_DIRECT_PLACEMENT_VERIFY_RUNTIME_WRITE=1 单独打开不能修复。

因此当前问题不是“没有命中 prefix cache”，也不是“没有触发 KV read/write”，
而是 native KV worker runtime / one-copy direct placement 的时序或同步边界问题。
```

本轮使用的核心测试条件：

```text
dataset:
  LongBench TriviaQA
bucket:
  4k_8k
model:
  Qwen2.5-7B-Instruct
pipeline:
  gpu_worker_persistent_one_copy
cache_size_mb:
  1024
chunk_capacity:
  128
NUM_SAMPLES:
  4
REPEAT_READ:
  1
MAX_TOKENS:
  32
```

#### 现象 1：no-debug 复现 mismatch

日志：

```text
evaluation/lmcache_ssd_read_paths_baseline/logs_longbench_triviaqa/bam_one_copy_nodebug_4samples_cap128/20260727_170804/run.log
evaluation/lmcache_ssd_read_paths_baseline/logs_longbench_triviaqa/bam_one_copy_nodebug_4samples_cap128/20260727_170804/metrics.jsonl
```

结果：

```text
exact_pairs=1/4
answer_hits=8/8
```

判断：

```text
1. 4 个样本的 answer substring 都命中；
2. 但 4 组里只有第 4 组 write/read generated_text 逐字一致；
3. 前 3 组 read 的答案开头正确，后续 Passage 文本漂移；
4. 这说明当前问题不是完全读不到 KV，
   更像是 prefix KV / metadata / stream visibility 的部分时序不一致。
```

关键链路信号：

```text
pipeline=gpu_worker_persistent_one_copy
finalize_mode=runtime_direct
chunk_capacity=128
active_chunk_slots=73
chunk_slot_evictions=0
prefix_hit_chunks=16/17
ret_mask_tokens=4096/4352
RUNTIME_ATTACH present
RUNTIME_READY present
DEFERRED_RETRIEVE_DONE present
```

因此不能把文本差异解释为“没命中前缀缓存”或“没有触发 KV 读写”。

#### 现象 2：完整 debug 版本通过

日志：

```text
evaluation/lmcache_ssd_read_paths_baseline/logs_longbench_triviaqa/bam_one_copy_debug_4samples_cap128/20260727_170357/run.log
evaluation/lmcache_ssd_read_paths_baseline/logs_longbench_triviaqa/bam_one_copy_debug_4samples_cap128/20260727_170357/metrics.jsonl
```

结果：

```text
exact_pairs=4/4
answer_hits=8/8
```

这说明：

```text
1. chunk_capacity=128 本身不是充分复现条件；
2. active_chunk_slots 到 73 时也可以正确；
3. debug / verify 组合会改变时序，使 race 被掩盖。
```

#### 现象 3：隔离开关定位到 GIDS_KV_DEBUG

只开 runtime write verify：

```text
evaluation/lmcache_ssd_read_paths_baseline/logs_longbench_triviaqa/bam_one_copy_isolate_runtime_verify_4samples_cap128/20260727_172108/metrics.jsonl

GIDS_KV_DEBUG=0
GIDS_KV_REF_DEBUG=0
VLLM_BAM_DIRECT_RETRIEVE_SLOT_BLOCK_ALIGN_VERIFY=0
VLLM_BAM_DIRECT_PLACEMENT_VERIFY_RUNTIME_WRITE=1

exact_pairs=1/4
answer_hits=8/8
```

只开 GIDS_KV_DEBUG：

```text
evaluation/lmcache_ssd_read_paths_baseline/logs_longbench_triviaqa/bam_one_copy_isolate_gids_debug_4samples_cap128/20260727_172512/metrics.jsonl

GIDS_KV_DEBUG=1
GIDS_KV_REF_DEBUG=0
VLLM_BAM_DIRECT_RETRIEVE_SLOT_BLOCK_ALIGN_VERIFY=0
VLLM_BAM_DIRECT_PLACEMENT_VERIFY_RUNTIME_WRITE=0

exact_pairs=4/4
answer_hits=8/8
```

结论：

```text
VLLM_BAM_DIRECT_PLACEMENT_VERIFY_RUNTIME_WRITE=1 不是关键同步来源；
GIDS_KV_DEBUG=1 是关键差异。

GIDS_KV_DEBUG 不只是打印日志，它会触发 native runtime 路径里的
KV_RUNTIME_DIRECT_PLACE_PROBE、状态观测以及额外 stream/device sync 副作用。
这些副作用很可能把 normal hot path 的 race 等住了。
```

#### 当前根因判断

当前根因应优先收敛到：

```text
BaM_IOStack/gids_module/gids_nvme.cu
  GPU persistent service 标记 cache_ready / consumable 的时机
  direct placement 写入 vLLM paged KV cache 的可见性
  host poll 看到 RUNTIME_READY 后进入 vLLM forward 的同步边界
  cleanup / retire 与后续 attention 消费之间的 stream ordering
```

暂时不应优先怀疑：

```text
1. LMCache 没命中 prefix；
2. ret_mask 完全没返回；
3. BaM cache 容量不够；
4. chunk_slot_evictions；
5. direct_placement_finalize 分支；
6. runtime write verify repair 分支。
```

#### 下一步建议

下一步不要继续扩大 debug 开关，而是增加一个最小、可控、可关闭的同步实验开关。

建议优先验证两个位置：

```text
位置 A：
  native persistent worker 在完成 direct placement 写入后、
  写 cache_ready / consumable / host_status=READY 前，
  增加最小 stream/event ordering。

位置 B：
  host poll 看到 RUNTIME_READY 后、
  finalize 返回给 vLLM attention 消费前，
  等待对应 runtime stream/event，而不是依赖 GIDS_KV_DEBUG 的 probe/sync 副作用。
```

验证标准：

```text
1. GIDS_KV_DEBUG=0；
2. 不打开 raw verify / alignment verify；
3. 仍使用 4 samples + cap128；
4. exact_pairs 从 1/4 恢复为 4/4；
5. read_ms / poll_ms 不因全设备同步明显退化。
```

在修复前，当前正确性口径应调整为：

```text
rowctx_baseline:
  正确性 oracle

gpu_worker_persistent_materialized:
  当前默认可靠 fast path / 回归口径

gpu_worker_persistent_one_copy:
  当前目标实验线；
  no-debug hot path 存在同步 race；
  只能作为待修复路径，不能作为最终正确性结论。
```

截至 2026-07-23，当前主线可以概括成一句话：

```text
BaM 读回、direct placement 和现有 attention 消费链路已经基本打通；
当前 cta=4 one-copy 已经固定为可正确跑通的稳定基线；
request-scoped ref_count release 已经验证正常；
512MB BaM cache 压力下，正常后台常驻模式可以稳定跑完；
下一步策略应收敛在 vLLM / LMCache 上层，
存储后端只作为 LMCache I/O 数据面和可选预取缓存区；
不应继续堆临时 rescue / fallback 分支。
```

当前最新可跑通的 LongBench-TriviaQA one-copy 日志：

```text
evaluation/lmcache_ssd_read_paths_baseline/logs_longbench_triviaqa/bam_one_copy/20260722_231109/run.log

pipeline=gpu_worker_persistent_one_copy
worker_backend=kv_persistent_service_v0
finalize_mode=runtime_direct
cache_size_mb=512
chunk_capacity=64
chunk_slot_evictions=169
requests=24
samples=12
read_avg_s=0.9472
```

这次结果说明：

```text
1. 当前不是路径回退；
2. normal persistent service 模式下没有打开 ref debug / stop-service 路径；
3. BaM cache page -> vLLM paged KV cache 的 one-copy scatter 正确性已经恢复；
4. 512MB BaM cache 压力下没有 submit failure / runtime hang；
5. LongBench-TriviaQA 4k_8k bucket 的 read 平均约 0.95s/request。
```

ref_count 生命周期已经通过 debug 路径单独验证：

```text
evaluation/lmcache_ssd_read_paths_baseline/logs_longbench_triviaqa/bam_one_copy/20260722_225047/run.log

total_cache_pages=4096
borrowed_pages_submitted=23408
borrowed_pages_released=23408
borrowed_pages_outstanding=0
borrowed_pages_underflow=0
submit_requests=12
release_requests=12
active_runtime_slots=0
```

当前已经明确跑通的真实链路：

```text
request_1:
  vLLM 正常 prefill
    -> LMCache chunk 生成
    -> LMCache shadow write 到 BaM

request_2:
  LMCache prefer-load 命中共享 prefix
    -> BaM direct placement 读回 prefix 对应 chunk
    -> KV 恢复到 vLLM paged KV cache
    -> xformers prefix fallback 消费 prefix + query
    -> request_2 正常继续执行
```

### 1.0 2026-07-17 三条 KV 链路定版

当前不再把所有实验都继续塞进同一条“fast path”里，而是明确保留三条链路：

```text
1. rowctx_baseline
   作用：
     稳定基线、正确性对照、回归兜底
   数据流：
     rowctx batch read
       -> materialized pages
       -> 已验证正确的 materialized placement
       -> vLLM paged KV cache

2. gpu_worker_persistent_materialized
   作用：
     当前输出正确的 fast 路径，也是默认性能/回归口径
   数据流：
     gpu_worker submit
       -> GPU persistent service 轮询/推进 read/stage
       -> cleanup 后停止 idle service
       -> host materialized finalize
       -> vLLM paged KV cache

3. gpu_worker_persistent_one_copy
   作用：
     最激进 one-copy 实验线，保留用于继续推进最终 GPU-resident 数据面。
     当前它不是默认正确性口径，仍可能带 correctness repair / verify。
   目标数据流：
     gpu_worker submit
       -> GPU persistent service 轮询/推进 read/stage
       -> GPU persistent service 直接写最终 vLLM paged KV cache
       -> host 只做 cleanup-only finalize
```

最新端到端正确输出对应日志：

```text
evaluation/logs/single_gpu_lmcache_no_prefix_reuse_qwen25/20260717_001108/run.log
```

这次日志里需要固定看的判断点：

```text
VLLM_BAM_KV_EXECUTOR=gpu_worker
GIDS_KV_GPU_WORKER_RUNTIME_ENABLE=1
GIDS_KV_GPU_WORKER_PERSISTENT_ENABLE=1
VLLM_BAM_DIRECT_PLACEMENT_RUNTIME_ONE_COPY=0
worker_backend=kv_persistent_service_v0
BAM_KV_GPU_WORKER_CLEANUP_ONLY_DONE
LMCACHE_BAM_RUNTIME_IDLE_STOP source=materialized_finalize active_count=0 stopped=true
impl=lmcache fused_ms=0.000
```

因此，当前正确 fast 路径不是退回 LMCache 原始路径，也不是 one-copy 已经完成，
而是：

```text
GPU 后台负责 poll/read/stage
host 只在 service 空闲后执行已验证正确的 materialized placement
```

代码里对应的命名已经收束为：

```text
pipeline=rowctx_baseline
pipeline=gpu_worker_persistent_materialized
pipeline=gpu_worker_persistent_one_copy
```

后续判断“跑的是哪条链路”，优先看日志：

```text
[LMCACHE_BAM_DIRECT_PLACEMENT_PIPELINE] pipeline=...
[LMCACHE_BAM_DIRECT_PLACEMENT_READ] ... pipeline=... finalize_mode=...
[LMCACHE_BAM_DIRECT_PLACEMENT] ... pipeline=...
```

当前代码也按这个口径做了函数级收束：

```text
start / poll / finalize 对外入口仍保持稳定：
  start_direct_placement_request()
  poll_direct_placement_request()
  finalize_direct_placement_request()

read 收口阶段拆成两条：
  _consume_materialized_read_request()
    服务：
      rowctx_baseline
      gpu_worker_persistent_materialized

  _finalize_one_copy_read_request()
    服务：
      gpu_worker_persistent_one_copy

写端 finalize 阶段拆成两条：
  _finalize_materialized_pipeline()
    服务：
      rowctx_baseline
      gpu_worker_persistent_materialized

  _finalize_persistent_one_copy_pipeline()
    服务：
      gpu_worker_persistent_one_copy
```

这次收束后的约束是：

```text
1. one-copy 的 cleanup / verify / dense workspace 准备
   不再和 materialized pages consume 混在一个大函数里。

2. materialized 路径只负责：
   consume pages
   materialized placement
   不再理解 one-copy 的 runtime attachment 细节。

3. one-copy 路径只负责：
   cleanup-only finalize
   optional correctness repair / verify
   不再回退到 host materialized consume。

4. 外部判断链路时看 pipeline 名称，
   内部 finalize mode 只作为局部实现细节保留。
```

这里要特别强调：

```text
results_materialized 只是内部 finalize mode，
不是“旧回退路径”的同义词。

它同时承载：
  rowctx_baseline
  gpu_worker_persistent_materialized

真正区分三条链路时看 pipeline 名称。
```

### 1.2 2026-07-15 分支收束结果

这一轮的重点不是继续堆新功能，而是先把 kvcache 主线收束干净。

当前代码组织明确收成三层：

```text
1. storage / kv fast path
   负责：
   - chunk -> page 请求翻译
   - start / poll / finalize
   - runtime direct placement 与 results materialized 两条 finalize 主线

2. adapter / rebuild
   负责：
   - 何时走 BaM direct retrieve
   - 根据 ret_mask 重建当前 request 的 model input
   - 发布 request authoritative ready 语义

3. attention consume
   负责：
   - prefix 恢复后的最终消费
   - 当前 V100 主线统一走 xformers prefix fallback
```

这轮明确收掉了几类已经没有工程价值的残留：

```text
1. xformers 里的 zero-alibi prefix kernel 实验线
   这条路径在当前 V100 主线已确认无效，只会增加入口分叉。

2. adapter 里“看起来像会直接消费 runtime attachment 大 tensor，
   实际固定回落到本地 rebuild”的布尔分支
   现在显式写成：authoritative ready 语义保留，
   slot_mapping / block_table 仍统一走本地薄重建。

3. manager 层对外暴露、但当前主线完全没有调用的 frontier 查询接口
   当前权威 ready 观察口已经收敛到 poll()；
   frontier/get_frontier 不再作为上层主线接口继续保留。

4. 启动脚本里会反向覆盖代码主线推导的冗余开关
   尤其是 runtime metadata attachment，不再由脚本默认硬压成 0，
   而是交给代码侧按 one-copy 主线自动推导。
```

因此，当前真正保留下来的“有语义差异的主分支”只剩三组：

```text
1. retrieve 入口
   - request-handle direct retrieve
   - legacy direct retrieve（仅作为兼容回退）

2. finalize 路径
   - runtime_direct
   - results_materialized

3. attention consume 路径
   - 当前主线：xformers prefix fallback
   - 后续待新增：显式的 dense consume backend
```

这一步的意义不是“已经完成终局实现”，而是把后续要推进的主线先整理成：

```text
retrieve / poll / finalize
  与
consume backend

明确解耦
```

后面如果继续推进新的 dense consume backend，或者继续把控制面下沉到 GPU
persistent service，就不需要再绕过一堆历史实验小分支。

### 1.1 2026-07-14 最新状态更新

这一轮需要单独写清楚，因为它解决的是“卡死/跑不完”的问题，
但还没有解决“数据完全正确”的问题。

当前最新结论可以概括为：

```text
1. direct placement + persistent service 这条主线已经能完整跑完 request_2。
2. 最近一次卡死的根因已经进一步收敛，不是在 poll 本身。
3. 当前能跑通，靠的是一个兼容性修复；这不是最终架构终点。
4. 数据正确性仍然没有完全通过，request_2 输出仍然出现明显损坏。
```

最新成功跑完的日志：

```text
evaluation/logs/single_gpu_lmcache_no_prefix_reuse_qwen25/20260714_002105/run.log
```

这轮日志里最关键的判断点有三个：

```text
1. direct placement 已经完成 submit/read/retrieve 主线，没有再次卡死在 persistent poll。
2. 日志出现：
     stage=single_seq_runtime_metadata_fast_path_idle_stop stopped=true
   说明在进入旧 rebuild 兼容路径前，已经先把空闲 runtime service 停掉。
3. 日志出现：
     LMCACHE_REBUILD_SEMANTICS rebuilt_context_tokens=1024 expected_context_tokens=1024
   说明“重建出来的上下文 token 数”在控制面语义上已经对齐。
```

这轮卡死根因的最新判断是：

```text
不是 BaM poll 卡住；
不是 direct placement read 没完成；
也不是新的 single-seq metadata fast path 自己死锁。

真正的问题是：
  当 attachment 还不具备 authoritative 语义时，
  链路会回退到旧的 rebuild 路径；
  而旧 rebuild 路径与仍在运行的 persistent service 存在执行冲突，
  从而表现成“看起来像是卡在 direct retrieve 后半段”。
```

因此，这一轮实际落地的是一个“窄兼容修复”：

```text
当 single-seq runtime metadata fast path 发现 attachment 还不能 authoritative 返回时，
先停止 idle 的 runtime service，再进入旧 rebuild。
```

这一步的意义是：

```text
1. 先把“跑不完/卡死”的问题从主线上挪开。
2. 证明当前 persistent KV 主线与旧 rebuild 的冲突点确实在这里。
3. 给后续继续把 rebuild / metadata 发布彻底收进 GPU runtime 留出空间。
```

但是，这一轮不能被误判为“正确性已经完成”。

最新日志中的输出对比是：

```text
20260714_002105:
  request_1:
    Qwen2.5-7B-Instruct 是一款基于大规模预训练模型的指令调优语言模型...
  request_2:
    Q22-2BInstruct 是一款基于 7B 亿参数训练的模型...

20260712_204826:
  request_1 == request_2
  二者都保持正常的 Qwen2.5-7B-Instruct 描述文本
```

这说明当前状态是：

```text
1. 链路已经功能性跑通。
2. request_2 的恢复结果仍然不正确。
3. 问题更像是“恢复出的 KV 内容或其消费结果不对”，
   而不是“高层 block table token 数量统计错了”。
```

从工程判断上，当前应把这轮结果记成：

```text
功能状态：
  跑通

正确性状态：
  未通过

修复性质：
  兼容性过渡修复，不是最终 GPU-resident 终局实现
```

2026-07-11 最新一轮已经再次验证：

```text
1. 走的是 gpu_worker + kv_persistent_service_v0
2. poll 已经由 GPU persistent service 推到 IO_DONE
3. consume 已经出现 READ_CONSUME_DONE，不再卡死
4. direct placement / rebuild / request_2 整条链路已经跑通
```

这轮日志里最关键的几个判断点是：

```text
backend=kv_persistent_service_v0
service_running=True
READ_CONSUME_BEGIN
READ_CONSUME_DONE
worker_backend=kv_persistent_service_v0
request_2_elapsed_s 正常输出
```

因此它不是“路径回退后碰巧能跑”，而是同一条 persistent KV 主线已经被修通到：

```text
submit
  -> GPU persistent poll
  -> consume direct cache load
  -> fused placement
  -> xformers prefix fallback 消费
  -> request_2 正常完成
```

当前需要把“性能判断”和“功能主线”分开理解：

```text
1. BaM 读回不是当前主瓶颈。
2. CPU 轮询当前也不是最大的端到端性能热点。
3. attention consume 侧仍然存在可继续收缩的结构性开销。
4. 但“CPU poll 不是性能热点”不等于“控制面不需要下沉”。
5. 当前功能目标已经明确为：
   GPU 持续维护 CQ completion、request frontier 和生命周期状态；
   host 只做轻量 submit、非阻塞 observe，以及消费已经完成的数据。
```

这里再把最终希望收敛到的接口契约明确写死：

```text
submit(request):
  CPU/host 准备 request descriptor
  调用 GPU 侧 submit 入口
  立即返回 request_handle
  不阻塞等待 I/O

poll/get_status(request_handle):
  可以由 CPU 或 GPU 发起
  只能读取 request/runtime/frontier 状态
  只能回答“是否 ready / ready 到哪里”
  不能在检查过程中再触发新的数据搬运

consume/get_ready_view(request_handle):
  只能取得已经完成写入的数据句柄、view 或可消费范围
  不能再承担 page read / refill / placement / cache write

GPU persistent service:
  负责 CQ 轮询
  负责 completion -> ctx/page/chunk/request 状态推进
  负责把数据直接搬到 LMCache cache / 最终可消费布局
  负责发布 consumable frontier
```

也就是说，最终主线要避免的是：

```text
CPU poll 一边检查、一边推进状态
CPU consume 时再补做数据搬运
finalize 时再重新组织 pages / refill / placement
```

当前已经具备的基础：

```text
1. GPU-visible request_table / completion_table / frontier_table
2. start / poll / finalize 三段式 request-handle 边界
3. GPU worker runtime slot
4. persistent service CTA 骨架
5. queue-level CQ service 和 (queue, cid) -> ctx lookup
```

本轮还额外做了一次代码收敛，明确把下列偏线实验从 KV 主线里清理掉：

```text
1. 延后 frontier launch 的实验控制面
2. frontier chunk limit / followup wave 这类双波 direct placement 运行时分支
3. poll 侧顺手推进 placement 的混合语义
4. 仅服务过渡验证、对后续 GPU-resident 主线无帮助的脚本与测试入口
```

也就是说，当前保留下来的 direct placement 运行时语义已经进一步收口为：

```text
一次 request
  -> 一次连续 prefix 命中收集
  -> 一次 BaM read submit
  -> poll 只观察 read frontier
  -> finalize 做一次 placement consume
  -> ret_mask 返回完整连续可消费前缀
```

但整条链路还没有达到“GPU 全权状态管理”。当前更准确的职责边界是：

```text
CPU / host:
  仍负责 prefix/chunk lookup
  仍负责 request table / metadata 准备
  可以轻量发起 submit
  仍负责高层调度、错误处理和最终 consume 入口
  persistent 路径下应只读观察 request 是否 ready

GPU / BaM:
  负责 page read
  persistent service 已开始负责 queue-level CQ 推进
  负责 completion -> ctx/page 状态回填
  负责刷新 GPU-visible runtime slot / frontier

仍需继续下沉：
  placement / finalize 状态收口
  cache_ready / consumable frontier
  ret_mask 对应的连续 prefix 可见性
```

把“当前已经做到什么”和“还没做到什么”再用一句话写死：

```text
当前已经做到：
  GPU 负责主要的 CQ poll、page completion 状态推进，以及实际的数据搬运 kernel

当前还没完全做到：
  CPU 仍然负责高层 request 调度、submit 触发、阶段 observe，以及在合适时机 launch
  consume / placement / 后续计算相关 kernel
```

也就是说，当前形态更准确的表述是：

```text
GPU 管 poll
GPU 管底层数据搬运
CPU 只做轻量控制和阶段切换
真正把这些数据用于 attention 计算的是后续另一个计算 kernel
```

从这条契约回看当前代码，真正还没收干净的核心差距只有两点：

```text
1. submit 仍然没有完全做到“异步立返”
2. consume/finalize 还没有直接写到最终 LMCache / paged KV 最终布局
```

补充一条 2026-07-11 已经完成的关键收口：

```text
KV consume 主线已经不再回退到旧的 registered rowctx get。

现在的语义是：
  poll:
    只观察 request/runtime/frontier 是否到达 IO_DONE

  consume:
    直接根据 submit 阶段展开好的 row_ids
    从 BaM cache 读到当前 pages staging buffer
    然后完成 request 生命周期收尾

  persistent service:
    不再为了 consume 被强制 stop
    ctx 释放改成 heartbeat 驱动的延迟回收
```

这意味着当前 KV 主线已经切掉了一层最重的历史兼容堆叠：

```text
旧路径：
  poll -> stop persistent service -> old rowctx get -> free ctx

现路径：
  poll(只读)
    -> direct cache load
    -> unregister runtime slot
    -> 延迟回收 ctx
```

---

## 2. 当前真正保留的主线路径

### 2.1 写路径

当前写路径已经稳定：

```text
vLLM / LMCache 产生一个 KV chunk
  -> 按当前 KV layout 切成固定 128KB page
  -> shadow write 到 BaM
  -> 记录 chunk_hash -> BaM page metadata
```

关键结论：

```text
写路径现在不是主要问题。
当前系统的难点已经从“怎么写进去”转移到“怎么更薄地取出来并直接消费”。
```

### 2.2 读路径

当前主线读路径已经不再依赖 LMCache 原始 disk fallback：

```text
LMCache prefer-load 命中
  -> 收集 prefix 对应 chunk_hash
  -> 走 BaM KV fast path batch read
  -> direct placement / fused placement
  -> ret_mask 返回“当前已真正可消费的连续 prefix”
  -> 上层 attention 消费
```

### 2.3 当前返回语义

当前 `ret_mask` 的语义已经收口到：

```text
当前这轮真正恢复完成、并且可以立刻被 attention 消费的连续 prefix。
```

它不再表示：

```text
内部 launch 了哪些 chunk
```

而是表示：

```text
当前请求真实可用的连续 prefix frontier 到了哪里
```

这点非常重要，因为它更贴近真实推理引擎的需要。

---

## 3. 当前真实数据通路

### 3.1 KV chunk 与 BaM page 的组织方式

当前主线已经从早期零散布局收敛成固定 page 布局：

```text
LMCache KV chunk
  -> 固定切成 128KB BaM page
  -> 一个满 chunk 对应固定数量的 page
  -> page metadata 由 chunk_hash 定位
```

当前典型 Qwen2.5-7B fp16 口径：

```text
chunk_size_tokens = 256
page_bytes = 128KB
pages_per_chunk = 112
hidden_dim = 512
num_layers = 28
```

### 3.2 direct placement 当前在做什么

当前 direct placement 不是“先把 BaM 数据还原成通用 LMCache tensor，再交回上层”，
而是更接近下面这条线：

```text
BaM page read
  -> GPU page buffer
  -> 根据 plan 组织 prefix 对应 chunk/page
  -> 直接放置到 vLLM paged KV cache / fallback 消费路径所需目标布局
```

### 3.3 一个完整例子

以当前单请求共享前缀场景为例：

```text
request_1:
  prompt 长度 1261 tokens
  chunk_size = 256
  因此会写出：
    [0,256)
    [256,512)
    [512,768)
    [768,1024)
    [1024,1261)   # 最后一段不满 chunk

request_2:
  与 request_1 共享前 1024 tokens
```

当前主线执行过程：

```text
1. LMCache connector 识别出：
     前 4 个 chunk 可复用
     第 5 段是 miss

2. storage 收集 4 个 prefix chunk 的 metadata

3. BaM KV fast path 发起 batch read：
     4 chunks
     448 pages

4. direct placement 把这 4 个 chunk 恢复到 vLLM 所需 KV 目标位置

5. ret_mask 返回：
     1024 个 prefix token 已可消费

6. 后续 493 个 query token 继续按正常 prefill 流程执行
```

最终 attention 看到的是：

```text
context = 1024 recovered prefix tokens
query   = 493 current request tokens
total kv len = 1517
```

---

## 4. 当前异步/轮询逻辑到底是什么

### 4.1 当前同时保留稳定 v1 和 persistent 推进路径

当前主线的本质是：

```text
BaM 底层 I/O 接口本身是异步能力；
稳定 v1 路径仍由 host 调用 poll 推进 queue-level CQ service；
persistent 路径则由常驻 GPU service CTA 持续推进 CQ，
host poll 开始收缩为只读 runtime slot / frontier 状态。
```

两条路径当前共享相同的请求返回语义：

```text
submit 是异步风格
poll / ready / consume 也是分阶段的
但当前请求在返回给上层前，
仍然会等待自己命中的连续 prefix 真正 consumable
```

因此当前既不能简单描述成“同步 I/O”，也还不能描述成
“整条链路已经完全 GPU-initiated”：

```text
native read / CQ completion:
  persistent 路径已经开始由 GPU 主导

placement / finalize / ret_mask:
  仍然保留明显的 host 控制面
```

### 4.2 为什么当前要这样做

原因不是“不会做异步”，而是当前 runtime 契约还不允许太激进：

```text
如果请求已经把某一批 live handle / live placement 交给后台继续推进，
而上层又开始进入下一轮调度，
就会遇到：
  同一份 kv_cache 目标区域还在被后台写
  上层已经准备读/写/复用它
```

这会引入非常危险的竞态。

所以当前主线选择的是：

```text
先把“当前要返回给推理引擎的连续 prefix frontier”收口好，
再返回。
```

### 4.3 当前最值得保留的轮询语义

当前真正保留下来的不是早期那些 per-row / per-page 轮询实验，
而是下面这层更接近主线的语义：

```text
execution.advance_ready()
execution.wait_until_launched_range_cache_ready()
execution.wait_until_contiguous_cache_ready(target_chunks)
```

一句话理解：

```text
当前已经从“整卡同步”等待
收成了“只等本请求需要的 cache-ready frontier”。
```

这不是最终 GPU-resident runtime，
但已经是后续继续往 persistent service kernel 推进时一个更健康的中间态。

### 4.4 最新的 poll / consume 边界

persistent 路径最近已经开始把轮询接口收窄成更清晰的职责：

```text
poll:
  只观察 runtime slot / frontier / request status
  不再负责推进 CQ
  不再顺手修改 placement front-ready 状态

consume:
  只消费已经完成的 BaM pages
  在旧 rowctx 语义仍需要时，
  保留 consume 前的最小兼容桥接
```

对应到当前实现方向：

```text
kv_worker_poll_request():
  persistent 打开时逐步收窄成只读 runtime slot 状态

kv_get_batch_status():
  优先读取 GPU worker runtime slot 的 request 状态

mark_front_ready():
  从 poll 热路径移到 consume 前的最小兼容位置
```

这一收敛的意义是把原来混在一起的三件事拆开：

```text
旧语义:
  poll = observe + 推进底层 + 修改可消费状态

目标语义:
  GPU service = 推进底层和维护状态
  host poll    = observe
  consume      = 使用已经完成的数据
```

### 4.5 当前 consume 的真实语义

当前 `consume` 已经不是“再去等待 IO 完成”，而是：

```text
request 已经到 IO_DONE
  -> 直接根据 submit 阶段展开好的 row_ids + rowctx
  -> 从 BaM cache 做一次 direct cache load
  -> 把数据放到当前 pages staging / placement 输入 buffer
  -> 然后做 request 生命周期收尾
```

对应到当前代码，关键位置是：

```text
kv_consume_chunk_batch():
  BaM_IOStack/gids_module/gids_nvme.cu

kv_direct_copy_pages_from_cache():
  BaM_IOStack/gids_module/gids_nvme.cu

read_feature_kernel_get_feature_light_rowctx():
  BaM_IOStack/gids_module/gids_kernel.cu
```

它和更早版本的最大区别是：

```text
旧版本：
  consume 里还会绕回旧 registered rowctx get 兼容尾巴

当前版本：
  consume 直接走 post-poll light rowctx load
  不再重新发 IO
  不再重新做 page wait
```

### 4.6 为什么前一版会报错，而最新一版能跑通

前一版报错时，日志表现是：

```text
poll 已经到 IO_DONE
但 READ_CONSUME_BEGIN 之后卡住
最终报 KV_DIRECT_CACHE_LOAD timeout
```

这说明问题不在：

```text
submit
poll
persistent service 是否启动
```

而在：

```text
consume 阶段的 direct cache load 语义
```

后面之所以能跑通，核心不是“回退到旧路径”，而是 consume 这条线被重新收口成了更干净的语义：

```text
1. 改成 post-poll light rowctx load
   不再用会重入 acquire_page()/wait 的通用 read()

2. consume 前先把 request 从 runtime service 观察面摘掉
   避免后台 persistent service 和前台 consume 同时碰同一批 ctx/page

3. direct cache load、placement、rebuild、request_2 最终都成功收尾
```

因此当前更准确的判断是：

```text
不是路径回退
而是同一条 persistent KV 主线下，
consume 阶段的语义冲突被收敛后，链路已经跑通
```

这里还要明确一个容易误解的点：

```text
xformers prefix fallback 仍然会出现
```

这不代表：

```text
退回到了 LMCache disk fallback
退回到了旧 rowctx blocking 主线
```

它只是说明：

```text
BaM direct placement 已经把 prefix KV 恢复好了
后续 attention 消费当前仍由 xformers fallback 路径执行
```

---

## 5. 最近一轮 xformers fallback 收缩

这一节是 2026-07-08 最新进展的核心。

### 5.1 为什么现在重点转到 xformers fallback

因为最近几轮 profile 已经反复说明：

```text
BaM direct placement steady-state:
  read_ms       ≈ 4~6 ms
  prepare_ms    ≈ 0.3 ms
  direct_total  ≈ 6~7 ms
```

而 request_2 端到端仍然在 1.5s 左右，
所以当前大头显然不在 BaM 读回。

### 5.2 第一阶段：packed prefix + query direct scatter

这一步已经实现并验证过：

```text
prefix:
  packed paged cache -> full workspace

query:
  contiguous query KV -> direct scatter -> full workspace
```

对应热路径日志：

```text
prefix_mode=packed_direct_to_workspace
query_mode=direct_scatter
```

这一步的意义：

```text
1. prefix 不再走老的 cache_view + gather + copy 组合
2. query 不再走 Python segment copy
3. 证明“更薄的数据面组织”方向是对的
```

### 5.3 第二阶段：single-request packed compose

为了继续验证“不要先拆成 prefix gather + query scatter，而是直接按最终消费布局 compose”这条思路，
又补了一条更专门化的快路径：

```text
single_request_packed_compose
  + in_kernel_compose
```

对应热路径日志：

```text
prefix_mode=single_request_packed_compose
query_mode=in_kernel_compose
```

这条路径当前只在下面口径打开：

```text
1. 单个 prefill request
2. prefix 存在
3. query 存在
4. packed prefix gather 条件满足
5. query direct scatter 条件满足
```

它做的事情是：

```text
不再分成：
  prefix kernel
  query kernel

而是把：
  prefix KV from packed cache
  query KV from contiguous tensor

一次性 compose 到最终 full KV buffer
```

### 5.4 这一步的意义与边界

它的意义是：

```text
1. 证明“直接按最终消费布局 compose”是可行的
2. 再把 fallback 数据面收薄一层
3. 给多请求通用版提供最小可工作的语义骨架
```

它的边界也很明确：

```text
1. 目前只覆盖单请求主场景
2. 不能作为最终面向真实推理引擎 batch prefill 的正式答案
3. 它的价值主要是验证“按最终消费布局 compose”这条方向成立，
   并提前把消费侧数据面收紧，为后续 GPU-resident frontier /
   persistent service kernel 做准备
```

---

## 6. 最近性能结论

### 6.1 当前主线 steady-state 结论

2026-07-08 最新几轮结论可以概括为：

```text
1. BaM direct placement 已经很轻。
2. xformers fallback 已经不是老的 Python copy-heavy 版本。
3. 单层 fallback 已经被收缩到亚毫秒级。
4. 端到端 request_2 仍然没有出现数量级改善，
   说明后续需要继续削减 attention consume 侧的结构性开销。
```

### 6.2 开启 fallback profile 时的解释方式

最近很多对比都开了：

```text
VLLM_BAM_XFORMERS_PREFIX_FALLBACK_PROFILE=1
```

因此要注意：

```text
profile 会引入 CUDA event + synchronize 开销；
这类日志更适合看“结构”和“占比”，
不适合直接作为最终吞吐口径。
```

### 6.3 2026-07-08 最新结构性判断

最新两阶段 profile 的含义如下：

第一阶段：

```text
prefix_mode=packed_direct_to_workspace
query_mode=direct_scatter
```

结论：

```text
query scatter 已经跑到真实热路径；
但它不是最终最大头。
```

第二阶段：

```text
prefix_mode=single_request_packed_compose
query_mode=in_kernel_compose
```

结论：

```text
单层 fallback 又薄了一点；
但收益是“继续收缩”，不是“根本改观”。
```

因此当前最重要的性能结论是：

```text
当前不该回头再抠 BaM poll，
也不该继续长时间停留在单请求 / 多请求 compose 这种
“CPU 仍主导整体运行时契约”的局部收缩上；
主线应转向 GPU-resident frontier / persistent service kernel。
```

### 6.4 性能热点与控制面下沉并不矛盾

当前 profile 说明 BaM I/O 和 host poll 不是最大的耗时项，
这个结论仍然保留；但它只回答“时间主要花在哪里”，
不回答“异步生命周期应该由谁管理”。

```text
性能优化视角:
  attention consume / fallback 仍有继续收薄空间

系统实现视角:
  completion / frontier / request state 仍需要迁到 GPU 常驻服务
```

因此后续不应继续堆叠 host 侧 tracker、wave、mirror 等过渡桥接，
而应在保留现有可跑数据面的基础上，直接收敛控制面所有权。

---

## 7. 已明确归档、不再作为主线推进的路线

下面这些路线已经有明确结论，不应再回头作为主线：

### 7.1 per-row / per-page 自己 poll completion

结论：

```text
不是当前主线。
复杂、脆弱，而且与真实 CQ 使用语义不匹配。
```

### 7.2 CPU 热路径直接频繁读 GPU status/completion table

结论：

```text
会引入额外 D2H / 同步代价，
破坏毫秒级 replay 与 steady-state 口径。
```

### 7.3 回头死抠 V100 上的 PagedAttention.forward_prefix()

结论：

```text
这条路前面已经尝试过，
在当前 V100 环境下不稳定，
不应再作为主线优先项。
```

### 7.4 继续把 KVCache 当成通用 feature row/object 来复用

结论：

```text
会把 KVCache 路径越做越重；
当前必须坚持 KV fast path / direct placement 这条专用线。
```

---

## 8. 下一步主线

### 8.1 当前为什么要优先转向 GPU-resident frontier

当前更合理的下一步不是先补多请求通用 compose，
而是优先推进：

```text
GPU-resident frontier
  + persistent service kernel
```

原因：

```text
1. 当前真正限制系统继续往前的，不是 compose helper 还不够通用，
   而是 runtime 仍然是“CPU 负责收口、CPU 决定何时返回”的形态。
2. 只要这个契约不改，即使先做出 multi-request compose，
   整体仍然停留在 CPU 主导 frontier 的中间态。
3. 当前已经有 enough evidence 说明：
   BaM I/O 已经通
   direct placement 已经轻
   xformers fallback 已经被收薄
   所以主矛盾已经转向“谁来维护 frontier / consumable state”。
```

### 8.2 当前要下沉到 GPU 的到底是什么

这里所说的 GPU-resident frontier，不是简单指“某个 kernel 在 GPU 上跑”，
而是指下面这组状态的拥有权开始从 CPU 迁到 GPU 侧：

```text
1. 哪些 chunk/page 已经 launch
2. 哪些 chunk/page 已经 read-ready
3. 哪些 chunk 已经 cache-ready
4. 当前连续 consumable frontier 到了哪里
5. 是否可以继续触发下一波 launch / placement / consume
```

也就是说，后续真正要做的不是“把更多 copy kernel 写漂亮”，
而是把下面这条控制链收进 GPU：

```text
submit
  -> completion observe
  -> ready frontier advance
  -> cache-ready / consumable frontier advance
  -> 触发后续 placement / consume
```

### 8.3 persistent service kernel 的目标语义

当前更贴近主线的理解方式是：

```text
CPU:
  仍然准备 request metadata / request table
  可以轻量发起 submit
  仍然负责高层调度和错误处理
  只做非阻塞 observe
  只消费已经完成的数据

GPU persistent service side:
  持有 outstanding request frontier state
  持续服务 logical queue 对应的 CQ
  通过 (queue, cid) 找回 completion 对应的 ctx
  更新 page / chunk / request 状态
  推进 chunk_ready -> chunk_consumable
  维护连续 prefix frontier
```

一句话说：

```text
submit 可以由 GPU 发起，也可以先由 CPU 轻量发起；
但 completion 轮询、frontier 推进和 request 状态管理，
应尽量由 GPU persistent service 全权负责。
```

这里的“全权负责”不要求 CPU 从系统中完全消失。
CPU 仍可准备 metadata、创建请求和处理异常；
关键是它不再靠反复调用 poll 来驱动 I/O 生命周期前进。

### 8.4 为什么当前这些 compose 收缩仍然有价值

虽然下一步主线改成了 GPU-resident frontier / persistent service kernel，
但前面已经做完的这些 `packed gather / direct scatter / single-request compose`
并没有白做，它们的价值在于：

```text
1. 它们把消费侧数据面提前收薄了
2. 让后续 persistent service kernel 不必一边接 frontier state machine，
   一边还背着一堆 Python copy / 中间 buffer 组织
3. 它们证明：
   “按最终消费布局直接生成/放置 KV”这条方向是成立的
```

所以这些工作现在应该被理解成：

```text
不是下一步主线本身；
而是 GPU-resident frontier 主线的前置铺路工作。
```

### 8.5 这条路线与 AGIO / Tutti / TARDIS 的对应关系

当前主线与三篇论文给出的约束是一致的：

```text
AGIO:
  initiation / completion 解耦
  不再把 I/O 当同步函数用到底

Tutti:
  CPU 准备 metadata
  GPU 消费 GPU-visible request/state

TARDIS:
  KVCache 走专用对象路径
  不再长期复用通用 feature 语义
```

一句话总结接下来的工程方向：

```text
先利用现有已收薄的数据面，
把 frontier / completion / consumable state machine 下沉到 GPU；
再在 persistent service kernel 稳定后，
根据需要补多请求通用 compose / 更通用的消费组织方式。
```

### 8.6 2026-07-08 当前已落地的最小 frontier ABI

在真正写 persistent service kernel 之前，已经先把一层更稳的 request 级 ABI
接到了现有 KV 路径上：

```text
request_table
  + gpu_status
  + gpu_chunk_status
  + gpu_completion_table
  + gpu_frontier_table   <- 新增
```

其中 `gpu_frontier_table` 当前先固定为 7 列：

```text
[status,
 launch_frontier_chunks,
 read_ready_frontier_chunks,
 cache_ready_frontier_chunks,
 consumable_frontier_chunks,
 total_chunks,
 error_code]
```

当前 rowctx_compat 版本还不能稳定提供“逐 chunk completion 到达”的 frontier，
所以这一版先用保守语义：

```text
SUBMITTED:
  launched = total_chunks
  read_ready/cache_ready/consumable = 0

IO_DONE:
  launched = total_chunks
  read_ready = total_chunks
  cache_ready/consumable = 0

CONSUMED:
  launched = total_chunks
  read_ready/cache_ready/consumable = total_chunks
```

这样做的价值不是“现在就拿它直接替代上层 direct-placement frontier”，
而是先把下面这件事固定下来：

```text
KV request 有一张稳定的、GPU-visible 的 frontier 状态表
后续 persistent service kernel 只需要把这张表的更新粒度细化
而不需要再重新改 Python / pybind / request handle 结构
```

同时，2026-07-08 这一轮又往前推了一步：

```text
frontier_table 不再只是“host 跟着 batch status 手工写”
而是开始优先由 completion_table 归约得到
```

当前由于 rowctx_compat 仍然是“整批同态 completion”，
所以表面上看 frontier 结果和之前一致；
但语义已经更接近后续 persistent service kernel：

```text
future:
  service kernel 持续更新 completion row
  frontier reducer 从 completion row 推出 launch/read_ready/cache_ready/consumable
```

也就是说，这一步的价值不在于立刻改变性能，
而在于把 frontier 的“真数据源”从 host status 继续往 GPU completion 表上收。

### 8.7 最新的主线收敛与代码简化原则

当前复杂度主要来自历史过渡阶段叠加，并不是最终目标本身复杂：

```text
1. tracker + frontier_table + runtime mirror 多套状态源并存
2. request / wave / execution 多层对象都承担部分状态推进
3. native read frontier 与 placement frontier 尚未完全统一
4. finalize 仍然承担较重的状态收口职责
```

后续整理应坚持下面的优先级：

```text
1. 冻结 completion_table / frontier_table / runtime slot 为主事实源
2. host tracker 只保留派生视图或兼容用途，不再成为主状态机
3. persistent GPU service 持续维护 request 生命周期
4. host 只保留 submit + observe + consume
5. placement / consumable frontier 再沿同一事实源继续下沉
```

这意味着后续不再优先新增新的 host bridge、mirror 或轮询分支。
如果旧逻辑只服务已经否掉的 per-row/page poll，或者与 persistent 主线重复，
应在确认不影响稳定 v1、KVCache 可跑路径以及原有 CNN/GNN 路径后清理。

### 8.8 2026-07-14 当前代码收束判断：哪些必须保留，哪些必须硬切，哪些可以清理

这一节不是泛泛地说“后面再优化”，而是把当前代码里真正影响主线推进的结构问题直接列出来。

先给一句总判断：

```text
当前最大的阻碍，不是 BaM CQ 轮询能力不够，也不是 SSD I/O 还没打通，
而是“一次搬运主线”和“旧的 materialize / host placement / fallback 主线”
仍然共存在同一套函数内，导致：
  状态语义混杂
  回退语义混杂
  校验语义混杂
  性能口径混杂
```

因此下一步不应继续在原结构上叠加小补丁，
而应先做一次明确的“主线收束”。

#### 8.8.1 必须保留的部分

下面这些部分已经是当前主线的稳定地基，不应再回退或重写成旧语义。

1. `request-handle` 三段式边界
   位置：
   `vllm/bam/lmcache_bam_storage.py`
   `start_direct_placement_request()`
   `poll_direct_placement_request()`
   `finalize_direct_placement_request()`

   保留原因：

   ```text
   这是当前把“一站式 blocking retrieve”拆成可继续下沉 runtime 的最小边界。
   后续不管是 host 同步 finalize，还是 GPU-resident runtime 周期性推进，
   都应该复用这套 start / poll / finalize 契约。
   ```

2. `BaMKVRequestTable` 这套 GPU-visible request ABI
   位置：
   `BaM_IOStack/gids_module/bam_kv_store.py`
   `class BaMKVRequestTable`

   保留原因：

   ```text
   request_table / gpu_status / gpu_chunk_status / gpu_completion_table /
   gpu_frontier_table / pages
   已经是当前 KV 主线最清晰的一套共享事实源。
   后续 persistent service 继续下沉，也应继续围绕这组表推进，
   不应再重新发明另一套 Python 侧 request 描述结构。
   ```

3. `gpu_worker + persistent service` 设备侧主循环
   位置：
   `BaM_IOStack/gids_module/gids_nvme.cu`
   `kv_worker_runtime_persistent_service_kernel`

   保留原因：

   ```text
   当前真正有工程价值的 GPU-resident 主线就在这里：
     queue-level CQ service
       -> runtime slot refresh
       -> direct placement
       -> metadata fill
       -> request consume/retire
   这条路径虽然还不够收干净，但方向本身已经是对的。
   ```

4. `cleanup-only finalize` 语义
   位置：
   `BaM_IOStack/gids_module/bam_kv_store.py`
   `finalize_runtime_attached_native_batch()`

   保留原因：

   ```text
   这是“GPU 后台已经搬完数据，host 只做 request 生命周期收尾”的关键契约。
   后续只应继续把它收薄，不应再回退到 host consume 重新承担数据搬运。
   ```

5. `poll` 只读化方向
   位置：
   `BaM_IOStack/gids_module/bam_row_store.py`
   `kv_worker_poll()`

   保留原因：

   ```text
   persistent 打开时，host poll 已经开始收窄成：
     只看 request status / frontier
   不再顺手推进旧状态机。
   这是后续“CPU 只 observe，不再驱动生命周期”的必要方向。
   ```

6. `slot_mapping + block_table` 的大控制面优先收敛思路
   位置：
   `LMCache-v0-torch26/lmcache/integration/vllm/vllm_adapter.py`
   `_build_single_seq_runtime_metadata_fast_path_*`

   保留原因：

   ```text
   当前已经验证：
   真正值得继续往 persistent service / runtime attachment 下沉的是
     slot_mapping
     block_table
   而不是 context_lens/query_start_loc 这类很小的标量控制面。
   这个优先级判断是对的，应明确保留。
   ```

#### 8.8.2 必须硬切的部分

这里的“硬切”意思不是立刻删代码，
而是要在运行模式和控制流上明确切开，避免同一次 request 混用两条主线。

1. `runtime one-copy` 主线与 `host materialize placement` 主线必须分模式
   位置：
   `vllm/bam/lmcache_bam_storage.py`
   `_start_direct_placement_request()`
   `_finalize_direct_placement_request()`

   当前问题：

   ```text
   同一个 direct retrieve request 里，
   可能先按 runtime-direct 的语义 attach/submit，
   然后又因为某个条件掉回：
     blocking batch read
     results materialized
     host-side placement
   ```

   需要硬切成：

   ```text
   模式 A: runtime_one_copy
     submit -> persistent poll -> cleanup-only finalize
     不允许再局部回退到 host placement

   模式 B: legacy_materialized
     submit/read -> materialize pages/results -> host placement
     不假装自己在跑 one-copy
   ```

   只有这样，日志、正确性和性能口径才会一致。

2. `attach runtime direct placement 失败` 的处理必须从“局部回退”改成“整条回退”
   位置：
   `vllm/bam/lmcache_bam_storage.py`
   `_try_attach_runtime_direct_placement()`
   `_finalize_direct_placement_request()`

   当前问题：

   ```text
   attach 失败后，当前 request 仍可能继续沿同一函数走 host placement 分支。
   这会让“当前到底是不是 one-copy request”变得不再明确。
   ```

   需要改成：

   ```text
   one-copy 模式下：
     attach 失败 -> 当前 request 整体退出 one-copy
     由更外层统一决定是否回退到 legacy direct retrieve
   ```

3. `submit 失败 -> blocking batch read` 只能存在于 legacy 模式
   位置：
   `vllm/bam/lmcache_bam_storage.py`
   `_start_direct_placement_request()`

   当前问题：

   ```text
   现在 async/native submit 一旦失败，会局部改走 blocking batch read。
   这在排障期有价值，但在 one-copy 主线中会污染语义：
   用户以为自己在测 persistent request runtime，
   实际上中途已经改成 blocking 读。
   ```

   因此应硬切：

   ```text
   runtime_one_copy:
     submit 失败 -> 整体失败/整体回退

   legacy_materialized:
     才允许 blocking batch read 兜底
   ```

4. 校验路径必须与主链完全解耦
   位置：
   `vllm/bam/lmcache_bam_storage.py`
   `_verify_runtime_direct_placement_write_against_materialized_chunks()`
   `BaM_IOStack/gids_module/gids_nvme.cu`
   `output_pages_mirror_enable`

   当前要求已经明确：

   ```text
   校验只能复用当前 live request 已有的 pages mirror；
   不允许再次触发任何额外 BaM 读。
   ```

   后续还应继续硬切：

   ```text
   默认性能主线：
     verify off
     mirror off

   调试校验主线：
     verify on
     mirror on
     但不改变主 request 的控制流分支
   ```

5. runtime metadata attachment 必须从“总开关摇摆”变成“只下沉大控制面”
   位置：
   `LMCache-v0-torch26/lmcache/integration/vllm/vllm_adapter.py`
   `_runtime_metadata_attachment_enabled()`
   `single_seq_runtime_metadata_fast_path`

   当前判断已经很明确：

   ```text
   全量 metadata attachment 收益不大，风险很高。
   当前真正值得 authoritative 化的是：
     slot_mapping
     block_table
   ```

   所以后续需要硬切掉“全量 metadata 一起信任”的思路，
   收成：

   ```text
   大控制面:
     继续下沉到 runtime attachment

   小标量控制面:
     host 侧按最终语义直接构造
   ```

#### 8.8.3 确认可删除或归档的部分

下面这些部分不是说此刻必须立刻从仓库抹掉，
而是已经确认不应继续参与 KV 主线判断，后续应尽量隔离或清理。

1. 只服务 `per-row / per-page 自己 poll completion` 的实验逻辑
   结论：

   ```text
   已在第 7 节归档，不应再进入当前 KV one-copy 主线。
   ```

2. `poll` 内顺手推进 placement / tracker 的混合语义
   位置：
   `vllm/bam/lmcache_bam_storage.py`
   `wave/execution.advance_ready()` 一类 host bridge

   结论：

   ```text
   当前真正的目标是：
     GPU service 维护状态
     host poll 只观察状态
   因此 host poll 不应继续承担 placement frontier 推进职责。
   ```

3. 只为旧测试桩存在的 `_build_stats()` / 旧签名兼容桥
   位置：
   `vllm/bam/lmcache_bam_storage.py`
   `_wait_direct_placement_wave()` 周围

   结论：

   ```text
   如果后续不再依赖这些旧测试桩，
   应逐步收掉 `_build_stats()` fallback 和旧 execution 签名兼容桥，
   避免 direct placement 主线继续背历史接口。
   ```

4. 只为旧同步 direct retrieve 保留的一站式 helper 主入口
   位置：
   `LMCache-v0-torch26/lmcache/integration/vllm/vllm_adapter.py`
   `_try_bam_direct_retrieve()`
   `vllm/bam/lmcache_bam_storage.py`
   `direct_place_chunks_to_vllm_kvcache()`

   结论：

   ```text
   当前 request-handle API 已经成熟到足以作为主线。
   后续应继续把同步 one-shot helper 收缩成薄封装，
   不再让它重新承担主控制面组织。
   ```

5. `output_pages_ptr` 作为主数据面 staging 的旧语义
   位置：
   `BaM_IOStack/gids_module/include/bam_nvme.h`
   `BaM_IOStack/gids_module/gids_nvme.cu`

   结论：

   ```text
   当前 one-copy 主线下，
   output_pages 只应保留两种用途：
     非 direct-placement fallback
     verify-only mirror
   不应再让它回到“主数据面必经 staging buffer”的地位。
   ```

#### 8.8.4 代码层面的实际收束顺序

后续真正动代码时，建议按下面顺序收，不要反过来。

第一步，先切模式，不要再混：

```text
runtime_one_copy
legacy_materialized
```

具体动作：

```text
1. `lmcache_bam_storage.py`
   把 `_finalize_direct_placement_request()` 里的 runtime-direct 分支和
   host materialize 分支彻底拆成两个 helper

2. `lmcache_bam_storage.py`
   把 submit-fail blocking fallback 挪出 one-copy 主线

3. adapter 层
   让 request-handle API 成为默认主入口；
   legacy direct retrieve 只作为显式回退
```

第二步，再统一 completion 语义：

```text
GPU runtime 发布 authoritative request completion row
host finalize 只读这一条
```

具体目标：

```text
host 不再自己拼：
  status
  frontier
  authoritative flag
  cleanup_only_done
  metadata_ready_flag
```

第三步，再收 metadata 主线：

```text
只继续 authoritative 化：
  slot_mapping
  block_table
```

小控制面继续保持：

```text
context_lens
seq_lens
query_start_loc
selected_token_indices
```

由 host 按最终语义轻量构造。

第四步，最后再删兼容桥：

```text
1. 删除 one-copy 主线不再触达的 blocking batch fallback
2. 删除仅服务旧测试桩的 wait/stats 兼容桥
3. 删除 poll 中仍残留的 host-side ready 推进桥接
4. 让 output_pages staging 退出 one-copy 正常热路径
```

#### 8.8.5 当前最应该避免的错误推进方式

就当前代码状态来说，真正还差的补齐点其实已经很少，核心只剩下面几项：

```text
1. runtime_one_copy 和 legacy_materialized 必须继续硬切
   - runtime mode 不再允许同一次 request 内部偷偷回退到 blocking read
   - finalize 失败要么显式报错，要么由外层统一决定是否重试/回退

2. GPU persistent service 负责 poll / completion / 数据搬运的主语义要保持单一
   - host 侧只做轻量 submit、观察 frontier、构造 ret_mask
   - 不再继续把 host-side placement / ready 推进混进 poll 里

3. return 语义必须继续绑定到连续 consumable prefix
   - 命中了多少连续 prefix chunk，就只返回这多少连续 prefix chunk
   - 不能为了图省事把“已命中但尚未 consumable”的 chunk 也暴露出去

4. 旧的 per-row / per-page poll 试验桥、host-side bridge、额外 BaM 读校验
   都只应保留为归档或 debug，不应继续参与主线判断
```

后续推进时，应明确避免下面这些“看起来在修 bug，实际上在继续堆叠”的做法：

```text
1. 在同一个 finalize 里继续多加 if/else，把 runtime-direct 和 legacy 分支缝在一起
2. 在 poll 阶段继续顺手更新更多 host tracker 状态
3. 为了 debug 再引入新的额外 BaM 读路径
4. 为了修 metadata 个别字段，再把全量 metadata attachment 打开
5. 为了暂时跑通，再让 one-copy request 局部掉回 host placement
```

一句话说：

```text
后续主线不是“继续补桥”，而是“把桥拆掉，让模式和所有权清楚起来”。
```

---

## 9. 当前一句话总结

如果只记一段话，可以记下面这段：

```text
当前 vllm-bam 的 BaM KV 路径已经不是“能不能读出来”的问题，
而是“怎么让 prefix 命中后的 KV 以更薄、更贴近最终 attention 消费的形态被使用”的问题。

最近已经完成：
  packed prefix direct gather
  query direct scatter
  single-request packed compose

这些都证明方向是对的。

但下一步主线不应先停留在“补多请求通用 compose”这种局部收缩上，
而应直接转向：
  GPU-resident frontier state machine
  persistent service kernel

也就是说：
先把运行时控制平面下沉，
再根据 persistent 路径的实际需要补更通用的 compose/consume 语义。
```

---

## 10. 当前轮询契约

这一章开始不再按时间线堆实验过程，而是只保留当前代码主线真正还在用的契约。

当前 KV 主线的轮询模型已经明确收敛成：

```text
submit
  -> queue-level CQ service
  -> request-ready 聚合
  -> consume / finalize
```

它不是早期那种：

```text
page 自己轮询自己的 completion
```

而是：

```text
queue 统一消费 completion
ctx/page/chunk/request 只是被 completion 回填
```

当前真正应该记住的执行边界只有四步：

```text
1. submit
   request_table -> row/page request -> ctx 建立

2. service CQ
   1 thread : 1 logical queue
   queue 独占消费自己的 CQ

3. ready 聚合
   只看 request / frontier 是否已经达到当前返回目标

4. finalize / consume
   把已经 ready 的结果真正收口成上层可见语义
```

当前 CPU 与 GPU 的职责边界是：

```text
CPU:
  prefix/chunk lookup
  request table / metadata 构造
  submit
  poll 返回值观察
  最终调度

GPU:
  queue-level CQ service
  completion -> ctx/page/chunk/request 状态推进
  frontier 刷新
  direct placement / prefix consume backend
```

稳定 v1 与 persistent 路径的区别，不再是“是否使用不同数据路径”，
而只是：

```text
稳定 v1:
  host poll 仍会间接推进底层 ready 状态

persistent:
  GPU service CTA 持续推进底层状态
  host poll 只读 runtime slot / frontier
```

---

## 11. 当前 backend 拆分

当前代码已经明确拆成三层 backend 选择。

### 11.1 storage / direct placement finalize backend

在 [lmcache_bam_storage.py](/home/xhk/llm-inference/vllm-bam/vllm/bam/lmcache_bam_storage.py)
里，`finalize` 不再自己夹杂一串路径判断，而是先确定 `read_finalize_mode`，
再分发到显式 backend handler：

```text
runtime_direct
  -> runtime_direct_cleanup

results_materialized
  -> materialized_host_finalize
```

两条 backend 的职责很清楚：

```text
runtime_direct_cleanup:
  GPU 后台已经完成 one-copy
  前台只做 cleanup、发布 consumable frontier 和返回语义

materialized_host_finalize:
  前台先 consume 出 materialized 结果
  再执行 host prepare + placement frontier wave
```

这样后面如果继续加新的 finalize consume backend，
只需要补：

```text
1. backend 名字
2. selector
3. handler
```

而不需要继续把逻辑塞进 `_finalize_direct_placement_request()`。

### 11.2 xformers prefix consume backend

在 [xformers.py](/home/xhk/llm-inference/vllm-bam/vllm/attention/backends/xformers.py)
里，`_run_prefix_attention_fallback()` 现在也已经拆成显式 backend 选择。

当前只保留两层非常窄的 backend：

```text
prefix backend:
  packed_direct_to_workspace
  gather_then_copy

query backend:
  direct_scatter
  segment_copy
```

对应语义是：

```text
prefix backend:
  决定 prefix 命中的 paged KV 怎么进入 full workspace

query backend:
  决定本轮 query KV 怎么进入 full workspace
```

当前主线里不再把 compose 当成 active backend。
`single_request_packed_compose` 仍保留函数壳和测试口径，
但主线 selector 不会再把它当成当前 consume 主线的一部分。

### 11.3 adapter 层保留的主分支

在 [vllm_adapter.py](/home/xhk/llm-inference/LMCache-v0-torch26/lmcache/integration/vllm/vllm_adapter.py)
里，目前只保留两类真正有语义差异的入口：

```text
1. request-handle direct retrieve
2. legacy direct retrieve（兼容回退）
```

request-handle 主线下，上层已经只把 `poll()` 当成权威 ready 观察口，
不再继续保留 `get_frontier()` 作为上层对外主接口。

---

## 12. 当前进度与仍未完成的问题

当前已经完成的部分：

```text
1. request-handle 生命周期已经拉起：
   start / poll / finalize

2. direct placement finalize 已拆成显式 backend

3. xformers prefix fallback consume 已拆成显式 backend

4. runtime metadata fast path 当前只承担：
   request authoritative ready 语义
   + 本地薄重建入口

5. manager / adapter 中已经清掉一批无效观察接口与实验分支

6. dense_prefix_workspace_consume 已经正式落地：
   storage/finalize
     -> 复用 live request pages
     -> 还原旧两次搬运语义下的 dense chunk tensor
     -> 挂到单请求 rebuilt attn metadata
     -> xformers fallback 显式切到 dense consume backend
```

当前仍未完成、但已经被收缩清楚的问题：

```text
1. request-handle / persistent service / consumable 发布主线已经能跑完
2. BaM read、最终 paged KV 写入、xformers gather、xformers attention
   都已经通过抽样校验
3. 当前输出仍然错误，剩余嫌疑已经转到 rebuild 控制面语义
4. GPU persistent service 后续仍要继续接管更多控制面，
   但当前 correctness 阻塞点不再优先指向 paged-KV 写端 ABI
```

因此当前真正的主线判断应该是：

```text
retrieve / poll / finalize 主线已经足够清楚；
真正剩下的问题不再是“poll 会不会卡死”，
也不再优先是“paged-KV 最后一跳是否写错”；
下一步重点不该再是补更多小开关，
而是核对 prefix 命中之后交回 vLLM 的
input token / position / slot_mapping / sampling metadata 是否仍保持原生语义。
```

### 12.1 最新正确性收敛结论：paged-KV 写端 ABI 已被抽样校验排除

2026-07-16 这一轮 verify 比前一版更进一步，已经把之前怀疑的
“flat token-row scatter / paged-KV ABI 写错”从当前主嫌里移开。
最新日志里同时出现了下面几类证据：

```text
1. request-handle 主线已正常走到：
   submit
   -> persistent poll
   -> consumable=4/4
   -> cleanup_only_runtime_direct

2. BaM 写入/读回/live pages 解码正确：
   LMCACHE_BAM_WRITE_READ_VERIFY_OK

3. runtime direct 最终写入 vLLM paged KV cache 后，
   按 packed key/value ABI 抽样回读正确：
   LMCACHE_BAM_DIRECT_PLACEMENT_VERIFY_RUNTIME_WRITE_OK

4. xformers fallback 从 paged KV cache gather 出来的 prefix
   与 dense chunk reference 完全一致：
   XFORMERS_PREFIX_GATHER_VERIFY_OK

5. xformers attention 输出与同一份 query/full_key/full_value
   上的 PyTorch reference 一致：
   XFORMERS_ATTENTION_REF_VERIFY_OK
```

这说明当前问题已经从：

```text
“后台轮询有没有跑起来”
“BaM 数据有没有读错”
“最终 paged KV 写端 ABI 有没有错”
“xformers gather / attention 有没有错”
```

收敛成：

```text
“prefix 命中后，adapter 重建给 vLLM 的控制面语义是否正确”
```

更具体地说，当前日志里仍然能看到：

```text
single_seq_runtime_metadata_fast_path_skip
reason=runtime_direct_temporarily_bypassed_for_correctness
```

也就是说，数据面已经走了 `gpu_runtime_direct`，但模型真正继续执行时，
metadata 仍回到旧的：

```text
build_partial_prefill_input()
```

这条路径会重新构造：

```text
input_tokens
input_positions
slot_mapping
block_tables
query_start_loc
context_lens_tensor
selected_token_indices
```

现在的下一步应该优先验证：

```text
1. 已恢复 1024 prefix token 后，
   suffix query tokens 是否正好是原请求的 [1024, 1517)

2. suffix input_positions 是否仍是 [1024, 1517)

3. slot_mapping 是否对应这些 suffix token 的真实 vLLM slot

4. selected_token_indices 是否指向 rebuild 后最后一个 query token，
   而不是仍然保留原 full-prefill 的 token index

5. sampling metadata / seq_group 的 query_len 是否与 rebuilt query_len 一致
```

### 12.2 为什么旧的 paged-KV 写端判断需要修正

前一版判断认为：

```text
用 flat token-row scatter 的方式
去写一个真实读侧会按 packed-key ABI 解码的缓存
```

这个判断在当时是合理的，因为那时只看到输出错误，还没有同时完成：

```text
最终 paged KV 回读 verify
xformers gather verify
xformers attention reference verify
```

但现在这三层都已经通过抽样校验，所以这个结论要降级为：

```text
它曾经是一个合理怀疑；
但按最新证据，它不再是当前乱码输出的第一嫌疑。
```

### 12.3 当前最可能的问题：rebuild 语义与 vLLM 原生语义错位

当前请求 2 的典型语义是：

```text
full request tokens: 1517
prefix hit tokens: 1024
需要继续 prefill 的 suffix query tokens: 493
```

因此正确的 rebuild 结果应该是：

```text
input_tokens:
  原始 request token[1024:1517]

input_positions:
  1024, 1025, ..., 1516

context_lens:
  1024

query_lens:
  493

seq_lens:
  1517

selected_token_indices:
  对单请求采样来说，应落在 rebuilt query 的最后一个 token，
  即 492，而不是原始 full prefill 的 1516
```

如果这里任意一项错位，就会出现一种很典型的现象：

```text
KV 数据本身是对的；
attention 读到的 prefix 也是对的；
attention 对当前 query 的计算也是自洽的；
但模型整体输出仍然乱码。
```

这说明错误更可能发生在：

```text
attention 输入之前的 token/position 构造
或 attention 输出之后的 sampling/index 语义
```

### 12.4 下一步诊断

下一步不再继续扩大 paged-KV verify，而是在
`build_partial_prefill_input()` 内补轻量语义日志：

```text
1. 打印 boundary tokens：
   full token[1018:1030]
   rebuilt suffix token 前几个/后几个

2. 比较：
   full_tokens[num_computed:]
   与 model_input.input_tokens[start:end]

3. 打印/校验 positions：
   expected [num_computed, num_token)
   actual rebuilt input_positions

4. 打印 slot_mapping 前后样本和 block table 形状

5. 打印原始 selected_token_indices 与 rebuilt selected_token_indices
```

---

## 13. 下一步主线

后面建议只沿下面这条主线继续推进：

### 13.1 当前已经有两条显式 consume backend

当前 consume 侧已经正式拆成：

```text
paged_kv_consume
dense_prefix_workspace_consume
```

当前含义是：

```text
1. paged_kv_consume
   继续服务当前真实 vLLM paged-KV 主线

2. dense_prefix_workspace_consume
   直接消费 live request pages materialize 出来的 dense chunk tensor
   语义对齐之前已经验证过的两次搬运路径

3. 一旦两者结果不同
   问题就会被压缩到 paged-KV 写入/解释/消费语义

4. 最新 verify 已经表明：
   现在真正错的不是 poll / BaM read / paged-KV write / xformers gather，
   而更像是 prefix hit 后重建给 vLLM 的控制面语义
```

### 13.2 下一步先把 rebuild 语义核准，再继续下沉 GPU ownership

最终目标仍然是：

```text
BaM 负责存
GPU service 负责取 + 放 + 更新状态
CPU 只负责 submit / observe / 调度
```

在当前阶段，更具体地说是：

```text
1. 先核准 build_partial_prefill_input() 的语义：
   input_tokens / positions / slot_mapping / selected_token_indices

2. 如果 rebuild 语义错：
   直接修正这条控制面路径，不再继续误查数据面

3. 如果 rebuild 语义也正确：
   再把比较点前移到每层 attention 输入/输出之后，
   查当前 query K/V 写入或后续 sampling 语义

4. correctness 稳定后，再继续把 dense consume / metadata workspace
   往 GPU persistent service 内收

5. 最终让 CPU 只保留 submit / observe / 调度
```

### 13.3 这条主线上应继续避免的方向

```text
1. 再回头长时间抠 CPU poll 小优化
2. 继续在主流程里叠更多布尔实验开关
3. 在没有明确 backend 边界的情况下继续把 paged-KV / dense / compose 混写
4. 在没有 staging / commit 语义前直接做跨轮次 live handle 可见性扩展
```

一句话总结当前第 10 章之后真正需要保留的信息：

```text
当前主线已经从“路径能不能跑通”转到“backend 怎么拆干净”；
后续实现应该继续沿显式 backend 和 GPU-resident runtime 契约推进，
而不是继续在单一路径里堆实验分支。
```

### 13.4 后续 GPU-initiated submit 主线

这里需要先明确一个边界：

```text
GPU-initiated 的价值，不只是“第一下 submit 是否由 CPU 发起”
而是整条 I/O 控制面是否继续依赖 CPU 高频介入
```

也就是说，即便当前第一波 seed submit 仍由 CPU 发起，只要后续已经逐步变成：

```text
GPU 负责 poll CQ
GPU 负责更新 frontier
GPU 负责数据放置
GPU 负责发布 consumable 状态
CPU 不再为每个 chunk/page 高频介入
```

它就已经和“只有 GDS 数据直达、但控制面仍由 CPU 高频驱动”的形态有本质区别。

当前更值得推进的，不是立刻执着于“第一下 submit 也必须 GPU 发”，
而是把“同一条请求内部的 follow-up submit”收给 GPU persistent service。

推荐主线：

```text
CPU：
  1. 新请求准入
  2. 第一波 seed submit

GPU persistent service：
  1. 轮询 CQ
  2. 更新 request / chunk frontier
  3. 负责 place / dense consume / 状态发布
  4. 根据 frontier 自己决定下一波要读哪些 chunk/page
  5. 直接发起 follow-up submit
```

#### 13.4.1 什么叫由 frontier 驱动 follow-up submit

所谓：

```text
GPU 根据当前 active sequence frontier 直接发起下一批需要的读
```

意思不是 GPU 被动等待 CPU 再下命令，而是：

```text
1. GPU 已经知道这条 sequence 当前推进到哪里
   例如：
     launch_frontier=4
     read_ready_frontier=4
     consumable_frontier=4

2. GPU 也知道这条 sequence 后面还需要哪些 chunk

3. 当 frontier 到达某个阈值后
   GPU 后台 service 自己决定：
     现在继续提交 chunk 5,6
     或继续提交 chunk 5,6,7,8
```

这样 follow-up submit 的触发条件就来自 GPU 自己维护的 frontier，
而不是 CPU 重新 poll 一次、再回到 host 侧做 descriptor 组装和 submit。

#### 13.4.2 推理里哪些流程最适合继续下沉成 GPU submit

最值得做的是下面几类：

```text
1. 同一条请求内部的后续 chunk / wave submit
   这是最直接、也最值钱的 GPU-submit 主线

2. 局部预取 / follow-up prefetch
   当前请求已经命中前缀后，GPU 可继续为后续 chunk 提前发起读

3. decode 阶段的小步迭代 I/O
   这类场景轮次多、步长小，CPU 若每轮都重新介入 submit，控制面会很重

4. dense consume / workspace 继续下沉后的 follow-up read
   一旦 dense consume 进一步并入 GPU service，自然就应由同一个 service
   继续决定后续还需要提交哪些 chunk
```

#### 13.4.3 哪些部分短期仍可能保留 CPU 参与

短期内不必强求全部搬空：

```text
1. 新外部请求的准入与 batch 调度
   这仍更接近推理引擎自己的 CPU 调度逻辑

2. 第一波 seed submit
   只要 chunk metadata / key lookup 仍主要在 host 侧，这一步保留 CPU 更自然

3. 粗粒度策略决策
   例如这一轮是否先返回部分 prefix，是否让位给其它请求
```

因此更现实、也更贴合当前代码结构的中间目标是：

```text
CPU 只介入一次 seed submit
GPU 接管同一请求内部的后续 follow-up submit / poll / place / publish
```

#### 13.4.4 当前代码离这一步还差什么

按现在这套 request/frontier/runtime 组织，后续真正要补的是：

```text
1. 让 persistent service 持有“下一波 chunk descriptor 从哪里开始”的权威状态

2. 把 follow-up submit 需要的 metadata
   例如：
     chunk_id -> page_offset / page_count / actual_tokens
   组织成 GPU 可直接消费的 descriptor 表

3. 把当前 request-level frontier
   真正变成 follow-up submit 的触发条件，而不只是 ready/return 观察口

4. 把 dense consume 进一步往 persistent service 一侧内收
   避免 cleanup 后再由前台组织过多后处理
```

一句话总结这部分后续工作：

```text
后续不是简单追求“把第一下 submit 也搬到 GPU”
而是要把“同一请求内部的后续 I/O 决策权”真正从 CPU 挪到 GPU runtime。
```

### 13.5 one-copy 下一步：先校验 repair 前的 GPU 原始写入

当前三条 KV 链路已经收束为：

```text
1. rowctx_baseline
   稳定基线，保留用于回归对照。

2. gpu_worker_persistent_materialized
   当前输出正确的 fast path。
   GPU persistent service 负责 poll/read/stage；
   最终 paged KV cache 写入仍走已验证正确的 materialized placement。

3. gpu_worker_persistent_one_copy
   最激进实验线。
   目标是 GPU persistent service 直接完成：
     BaM cache -> vLLM paged KV cache
```

one-copy 当前的问题不是 submit/poll 主线是否能推进，而是：

```text
GPU service 原始 scatter 到 vLLM paged KV cache 的结果
是否完全符合 vLLM 官方 packed KV cache ABI
```

此前 one-copy 为了保证端到端输出正确，会在 finalize 末尾调用：

```text
_rewrite_runtime_direct_prefix_into_paged_kv_cache_with_official_write()
```

这一步会用 vLLM 官方 `PagedAttention.write_to_paged_cache()` 覆盖最终
paged KV cache。它能保证 correctness，但也会把 GPU 原始 one-copy scatter
的错误盖掉，导致后续 verify 看到的是 repair 后结果，而不是 raw runtime write。

因此新的排查顺序必须改成：

```text
1. cleanup-only runtime direct finalize 完成
2. 在 official-write repair 之前，校验 GPU 原始 one-copy 写入
3. 若 raw verify 失败，直接根据 mismatch 定位 CUDA scatter
4. 若 raw verify 通过，再考虑移除 official-write repair
5. 只有 raw 写端确认正确后，one-copy 才能作为真正热路径推进
```

代码上已经补了两个边界清晰的 helper：

```text
_resolve_one_copy_runtime_verify_expected_tensors()
  优先复用 live request pages 生成的 expected dict；
  如果不存在，就把 repair 已经准备好的 dense prefix tuple
  按 chunk_hash 轻量组回 dict，避免为了校验再读一次 BaM。

_verify_one_copy_raw_runtime_write_before_repair()
  只在 VLLM_BAM_DIRECT_PLACEMENT_VERIFY_RUNTIME_WRITE=1 时启用；
  在 official repair 前调用已有 packed verifier；
  日志使用 RAW_RUNTIME_WRITE_VERIFY_BEGIN/DONE 明确标记。
```

下一步运行 one-copy 时，应先打开 raw runtime write verify：

```text
VLLM_BAM_DIRECT_PLACEMENT_RUNTIME_ONE_COPY=1
VLLM_BAM_DIRECT_PLACEMENT_REQUIRE_RUNTIME_ONE_COPY=1
VLLM_BAM_DIRECT_PLACEMENT_VERIFY_RUNTIME_WRITE=1
VLLM_BAM_DIRECT_PLACEMENT_VERIFY_RUNTIME_WRITE_ALLOW_LIVE_REFILL=1
```

若日志出现 mismatch，优先看：

```text
chunk_hash / layer / token_idx / slot_id / head / dim
host_reference
official_oracle
```

判断顺序是：

```text
host_reference=match 且 official_oracle=match
  说明 expected tensor、slot_mapping 和官方 packed 写入都正确，
  根因基本收敛到 gids_nvme.cu 的 one-copy scatter。

host_reference 或 official_oracle 自己 mismatch
  说明 verifier 输入或 slot/packed 解释仍有问题，不能直接怪 CUDA scatter。
```

这一步的目的不是新增长期分支，而是把 one-copy 的问题从“端到端乱码”
收敛成一个具体的底层写端公式问题。修对之后，应删除或默认关闭
official-write repair，让 one-copy 真正变成：

```text
BaM cache -> vLLM paged KV cache
```

的一次搬运主线。

### 13.6 2026-07-17 最新收敛：错误已从数据面转移到 metadata 交接面

最新 one-copy 强校验日志显示，下面这些环节已经可以从主要嫌疑里排除：

```text
1. BaM 写入 store / 从 store 读回 chunk
2. persistent service 的 submit / poll / cleanup
3. one-copy 后最终 paged KV cache 内容
   官方 write verify 已覆盖 28 层抽样通过
4. slot_mapping 与 block_table 对齐
5. xformers 从 paged KV cache gather prefix
6. xformers 每层 attention 输出抽样 reference
```

也就是说，当前乱码不应继续优先怀疑：

```text
BaM page read
BaM cache -> output pages buffer
最终 paged KV cache 写入内容
xformers prefix gather / attention 本身
```

真正暴露出来的问题是两条路径的上层控制面不一致：

```text
正确 materialized 路径：
  direct retrieve finalize
  -> single_seq_runtime_metadata_fast_path
  -> dense_prefix_attached=false
  -> xformers 从 paged KV cache 消费 prefix

错误 one-copy 路径：
  direct retrieve finalize
  -> 历史 correctness bypass
  -> build_partial_prefill_input()
  -> dense_prefix_attached=true
  -> 控制面与正确 fast path 分叉
```

因此当前重构方向已经调整为：

```text
1. one-copy 不再回退到旧 build_partial_prefill_input()
2. one-copy 和 materialized 统一走 single_seq_runtime_metadata_fast_path
3. dense prefix tensor 只保留为 storage 内部 repair / verify 材料
4. 最终 attention 消费统一回到 vLLM paged KV cache
5. 先把 correctness 跑通，再继续移除 official-write repair
```

这一步不是放弃 one-copy，而是把 one-copy 的错误面收窄：

```text
数据面：
  GPU persistent service 仍负责读、写、发布 consumable

控制面：
  adapter 只构造与正确 materialized 路径一致的 model_input / metadata

调试/repair：
  dense prefix 不再参与最终 xformers 输入，只服务校验和官方写端覆盖
```

后续验证时重点看日志是否出现：

```text
pipeline=gpu_worker_persistent_one_copy
stage=single_seq_runtime_metadata_fast_path
dense_prefix_attached=false
```

并确认不再出现：

```text
reason=runtime_direct_temporarily_bypassed_for_correctness
stage=build_partial_prefill_input_begin
dense_prefix_attached=true
```

### 13.7 2026-07-17 晚间更新：三条 KV 链路最新定版

截至 `20260717_231209` 这轮日志，KV 路径已经进一步收束为三条边界清楚的链路。这里不覆盖前面历史排查记录，只更新当前应以哪套语义判断代码是否跑在主线。

#### 13.7.1 当前保留的三条链路

```text
1. rowctx_baseline
   作用：
     稳定正确性基线、回归对照、排错参照。

   数据流：
     CPU / Python 层组织 chunk/page request
       -> BaM rowctx batch read
       -> 读回 materialized pages / chunk tensor
       -> 已验证正确的 materialized placement
       -> vLLM paged KV cache
       -> xFormers prefix fallback 从 paged KV cache 消费 prefix

   当前定位：
     这是“正确但不激进”的基线，不再作为 GPU-resident 主线继续加新逻辑。
     后续 one-copy 或 persistent 出问题时，仍用它判断写入布局和输出语义。

2. gpu_worker_persistent_materialized
   作用：
     当前稳定 fast 路径之一，用于验证 GPU persistent service 的 poll/read/stage 主线。

   数据流：
     CPU 轻量 submit request table / metadata
       -> GPU worker submit BaM row/page 请求
       -> GPU persistent service 维护 completion / frontier / staged 状态
       -> service 发布 cache_ready / consumable
       -> 前台 cleanup / finalize
       -> 使用已验证正确的 materialized placement 写入 vLLM paged KV cache
       -> xFormers prefix fallback 从 paged KV cache 消费 prefix

   当前定位：
     GPU 已经接管 BaM I/O 的轮询和状态推进；
     但最后写入 paged KV cache 仍走 materialized placement，
     所以它不是 one-copy 数据面，只是 persistent service 的稳定过渡路径。

3. gpu_worker_persistent_one_copy
   作用：
     当前最接近目标的数据面主线。

   数据流：
     CPU 轻量 submit request table / metadata
       -> GPU worker submit BaM row/page 请求
       -> GPU persistent service 维护 completion / frontier / staged 状态
       -> GPU persistent service 按 vLLM 官方 paged KV ABI 直接 scatter
          BaM cache page 到最终 vLLM paged KV cache
       -> GPU 发布 cache_ready / consumable
       -> CPU / adapter 只观察 poll 返回值与 consumable frontier
       -> cleanup-only runtime direct finalize
       -> xFormers prefix fallback 从 paged KV cache 消费 prefix

   当前定位：
     `BaM cache -> vLLM paged KV cache` 这段已经是真 one-copy，
     最新日志显示输出正确，并且没有回退到 output pages staging 或 live pages decode。
```

#### 13.7.2 最新 one-copy 跑通的判断依据

最新可用日志：

```text
evaluation/logs/single_gpu_lmcache_no_prefix_reuse_qwen25/20260717_231209/run.log
```

关键判断点如下：

```text
[LMCACHE_BAM_DIRECT_PLACEMENT_PIPELINE]
  pipeline=gpu_worker_persistent_one_copy
  runtime_one_copy=true
  runtime_attach=true

[KV_RUNTIME_DIRECT_PLACE_PROBE]
  official_flat=1
  use_output_pages=0

[LMCACHE_BAM_DIRECT_PLACEMENT_READ_FRONTIER]
  cache_ready_chunks=4/4
  consumable_chunks=4/4
  host_status=3

[LMCACHE_BAM_DIRECT_PLACEMENT_READ_CONSUME_DONE]
  mode=runtime_direct

[LMCACHE_BAM_DIRECT_PLACEMENT_READ]
  executor=runtime_direct_cleanup
  worker_backend=kv_persistent_service_v0
  pipeline=gpu_worker_persistent_one_copy
  finalize_mode=runtime_direct

[LMCACHE_BAM_DIRECT_PLACEMENT]
  impl=gpu_runtime_direct
  refill_ms=0.000
  transfer_ms=0.000
  fused_ms=0.000
  place_ms=0.000
```

这些日志合在一起说明：

```text
1. 没有走 LMCache 原始 fallback。
2. 没有走 output_pages staging mirror。
3. 没有走 BaM cache -> output pages buffer -> vLLM paged KV cache 的两次搬运。
4. 没有进入 LIVE_PAGES_DECODE / materialized_refill 调试校验路径。
5. 最终返回给 vLLM 的 prefix 命中长度是 4 个 chunk / 1024 tokens。
6. request 2 输出语义正常。
```

因此当前可以把 one-copy 数据面定为：

```text
BaM cache page
  -> GPU persistent service direct scatter
  -> vLLM paged KV cache
```

而不是：

```text
BaM cache page
  -> output pages buffer
  -> host / Triton / official write repair
  -> vLLM paged KV cache
```

#### 13.7.3 哪些 `fallback` 不是 KV 数据路径回退

当前日志里仍会出现几个 `fallback` 字样，容易误判。这里统一口径：

```text
1. defer to later BaM read instead of LMCache fallback
   含义：prefetch 阶段不走 LMCache fallback，而是把读取延后交给 BaM direct placement。
   结论：不是回退。

2. XFORMERS_PREFIX_FALLBACK
   含义：V100 上 `PagedAttention.forward_prefix()` 不可用，
        attention 消费侧仍使用 xFormers fallback 从 paged KV cache 读取 prefix。
   结论：这是 attention backend fallback，不是 BaM / KV 数据搬运路径回退。

3. fallback_local_slice_validation / fallback_local_rebuild_validation
   含义：adapter 在构造 slot_mapping / block_table 时保留的本地校验或兼容命名。
   结论：只影响 metadata rebuild / validation，不代表 KV 数据重新 materialize。
```

真正判断是否跑回旧 KV 数据路径，应优先看：

```text
pipeline=...
finalize_mode=...
impl=...
use_output_pages=...
refill_ms / transfer_ms / fused_ms / place_ms
LIVE_PAGES_DECODE 是否出现
results_materialized 是否出现
```

#### 13.7.4 当前 CPU / GPU 职责边界

当前已经实现的职责划分如下：

```text
CPU / adapter:
  1. 根据 LMCache chunk metadata 和 vLLM 当前 request metadata，组织第一波 request table。
  2. 把 slot_mapping、block_table、chunk_start、kv_cache_ptrs 等 runtime metadata attach 给 BaM runtime。
  3. 非阻塞地通过 poll 观察 consumable frontier。
  4. 当 consumable_chunks 达到 return_target_chunks 后，返回 ret_mask 给 vLLM。
  5. 做 cleanup-only finalize，不再承担 one-copy 数据搬运。

GPU persistent service:
  1. 维护 BaM CQ / completion 的推进。
  2. 更新 request / chunk / frontier 状态。
  3. 在 one-copy 路径中，直接把 BaM cache page scatter 到 vLLM paged KV cache。
  4. 发布 cache_ready / consumable，作为 CPU adapter 的权威 ready 语义。
```

所以当前已经不是“CPU 负责数据搬运，只是用了 BaM 异步接口”的形态。更准确地说：

```text
CPU 仍负责第一波 submit 和 metadata attach；
GPU persistent service 已经负责 BaM I/O completion、数据写入和 consumable 状态发布；
CPU 后续只观察是否可以让当前 request 继续计算。
```

#### 13.7.5 xFormers 消费侧的当前位置

one-copy 已经解决的是数据进入 paged KV cache 的路径；后续 attention 消费仍是：

```text
vLLM paged KV cache
  -> xFormers prefix fallback gather / packed_direct_to_workspace
  -> xFormers attention
```

最新日志中：

```text
selected_prefix=packed_direct_to_workspace
selected_query=direct_scatter
prefix_copy_ms=0.000
```

这说明 xFormers 消费侧已经比早期 gather-then-copy 更薄，但它仍不是最终 GPU-resident frontier / persistent service 的终局。当前阶段先把 one-copy 数据面定住，下一步再考虑继续减少 xFormers fallback 和 metadata rebuild 的结构性开销。

#### 13.7.6 当前仍未完成的问题

当前 one-copy 数据面已经跑通，但还没到最终 GPU-initiated 目标。剩余问题按优先级是：

```text
1. 进程退出 / cleanup 尾部仍可能残留 root Python 进程占用 GPU。
   这不是数据通路 correctness 问题，但会影响连续测试。

2. 第一波 seed submit 仍由 CPU 发起。
   当前 GPU 接管的是 poll / place / consumable 发布，
   还没有把“下一波 I/O 决策权”完全下沉到 GPU。

3. metadata rebuild 仍在 adapter 侧发生。
   目前它已经不参与 KV 数据搬运，但仍会构造 slot_mapping / block_table。

4. xFormers prefix fallback 仍是 attention 消费侧路径。
   V100 上 PagedAttention.forward_prefix 不可用，
   因此短期内继续保留 xFormers fallback，长期再考虑更贴合 paged KV 的消费 kernel。
```

后续推进顺序建议保持：

```text
1. 固化 one-copy 正确路径，减少默认调试开关，避免 verify 重新引入 live pages decode。
2. 处理尾部 cleanup / 进程退出问题，保证 benchmark 可重复。
3. 将 CPU poll 进一步收缩为只读 consumable 观察口。
4. 继续推进 GPU-resident frontier / follow-up submit，
   把同一请求内部下一波 I/O 的决策权从 CPU 挪到 GPU runtime。
```

#### 13.7.7 one-copy 正确修复口径与性能恢复口径

当前 one-copy 之所以能从“输出乱码”收敛到“输出正确”，核心不是靠回退或 repair 覆盖，而是修正了两层语义。

第一层是数据面写入 ABI 修正：

```text
错误旧思路：
  runtime direct scatter 自己按 key/value packed view 推导目标地址。

正确新思路：
  严格复刻 LMCache 官方 multi_layer_kv_transfer(direction=false)
  对 vLLM paged KV cache 的物理写入语义。
```

当前每层 vLLM KV cache 底层可以视作：

```text
layer_cache: [2, page_buffer_size, hidden_dim]
```

因此 one-copy scatter 的目标地址必须是：

```text
dst = kv * page_buffer_size * hidden_dim
    + slot_id * hidden_dim
    + hidden
```

这对应最新日志中的：

```text
official_flat=1
use_output_pages=0
```

`official_flat=1` 说明写入公式已经按官方 flat paged-buffer ABI；
`use_output_pages=0` 说明没有再走 output pages staging，数据是直接从 BaM cache page 写入最终 vLLM paged KV cache。

第二层是控制面统一：

```text
错误旧思路：
  one-copy 写了 paged KV cache，
  但 adapter 后续可能继续走 dense prefix / partial prefill / repair 相关分支，
  导致 xFormers 看到的 metadata 与真实 paged KV cache 消费路径不一致。

正确新思路：
  one-copy 和 materialized 正确路径统一回到
  single_seq_runtime_metadata_fast_path，
  dense_prefix_attached=false，
  xFormers 只从 vLLM paged KV cache 消费 prefix。
```

因此当前正确链路是：

```text
BaM cache page
  -> GPU persistent service 按官方 flat ABI direct scatter
  -> vLLM paged KV cache
  -> adapter 构造一致的 slot_mapping / block_table
  -> xFormers prefix fallback 从 paged KV cache 消费 prefix
```

而不是：

```text
BaM cache page
  -> output pages / dense prefix / official repair
  -> 再喂给 xFormers
```

性能上，最新正确性日志和之前约 `1.55 iter/s` 左右的口径不能直接比较，因为两者冷启动条件不同。

当前正确性日志 `20260717_231209`：

```text
request_2_elapsed_s=2.4046
read_ms=364.906
poll_ms=359.361
first_xformers_prefix_total_ms=467.662
```

之前更快的 `20260717_024543` 最终 request：

```text
request_2_elapsed_s=1.9983
read_ms=474.937
poll_ms=469.423
first_xformers_prefix_total_ms=0.983
```

这个对比说明：

```text
1. 最新 one-copy 的 BaM read/poll 反而更短：约 365ms vs 475ms。
2. 最新端到端更慢，主要不是 one-copy 数据搬运退化。
3. 差距主要来自 xFormers prefix fallback 冷启动：
   最新 request 自己承担约 468ms；
   之前更快版本已经被前面的 hidden prewarm / steady run 预热，
   最终 request 只剩约 1ms。
```

所以恢复到约 `1.55 iter/s` 左右，第一步不是重写 one-copy scatter，而是恢复相同的 steady-state 测试口径：

```text
1. 打开或保留 hidden prewarm：
   DIRECT_RETRIEVE_PREWARM_AFTER_REQUEST1=1

2. 不用 correctness verify wrapper 做性能口径：
   VLLM_BAM_DIRECT_PLACEMENT_VERIFY_RUNTIME_WRITE=0
   VLLM_BAM_DIRECT_PLACEMENT_VERIFY_RUNTIME_WRITE_ALLOW_LIVE_REFILL=0
   VLLM_BAM_WRITE_READ_VERIFY=0
   VLLM_BAM_DIRECT_RETRIEVE_CROSS_SOURCE_FULL_COMPARE=0
   VLLM_BAM_XFORMERS_VERIFY_PREFIX_GATHER=0
   VLLM_BAM_XFORMERS_VERIFY_ATTENTION_OUTPUT=0

3. 关闭非必要调试日志：
   GIDS_KV_DEBUG=0
   VLLM_BAM_XFORMERS_PREFIX_FALLBACK_PROFILE=0

4. 用 steady-state 或至少第二次 direct retrieve 的结果做性能判断，
   不把首次 xFormers gather / kernel 初始化算进最终主口径。
```

如果恢复 warmed steady-state 后仍低于历史值，再继续查真正的结构性开销：

```text
1. persistent service poll/read 是否稳定在 350ms 以内，
   如果回到 450ms 以上，优先查 BaM cache 命中、CQ service loop、debug sync。

2. metadata rebuild 是否仍需要 idle_stop，
   当前为了避免 V100 上后台 service 与前台 rebuild 争用，
   `single_seq_runtime_metadata_fast_path_idle_stop` 可能会停止 idle service。
   这保证正确性，但不是最终 GPU-resident 形态。

3. xFormers fallback 是否还有首次 400ms 级 warmup，
   若 benchmark 必须比较 steady-state，应在计时前显式预热 prefix fallback。

4. 进一步性能目标不是再加 repair 或 host fallback，
   而是继续把 metadata rebuild / follow-up submit / frontier 状态推进下沉到
   GPU runtime，减少 CPU 侧同步与前台 rebuild。
```

一句话结论：

```text
当前 one-copy 正确性已经修在主线数据面上；
性能回退主要是 cold-start / profile 口径差异，不是 one-copy 又多了一次搬运。
先恢复 hidden prewarm + 关闭调试校验，再用 warmed steady-state 重新对比，
才是和之前 1.55 iter/s 左右结果一致的比较方式。
```

### 13.7.8 2026-07-17 清理后定版：三条 KV 链路与冗余支线收束

本轮代码清理的原则是：保留已经有工程意义的三条 KV 链路，删除 one-copy 修错过程中临时堆出来的 verify / repair / live-pages decode 支线。清理后的主线不再依赖“先写错再 repair”的语义，也不再在常规启动脚本里透传大量调试开关。

当前保留的三条链路如下：

```text
1. rowctx_baseline
   CPU/rowctx batch read
     -> materialized pages
     -> BaMDirectKVPlacer / LMCache transfer
     -> vLLM paged KV cache

2. gpu_worker_persistent_materialized
   GPU persistent service 负责 poll/read/stage
     -> 前台 consume 出 materialized pages
     -> BaMDirectKVPlacer / LMCache transfer
     -> vLLM paged KV cache

3. gpu_worker_persistent_one_copy
   GPU persistent service 负责 poll/read/direct scatter/state publish
     -> BaM cache page 按官方 flat paged-buffer ABI 直接写入 vLLM paged KV cache
     -> 前台只做 cleanup、ret_mask 与 consumable frontier 发布
```

已经从主线代码中清理掉的内容：

```text
1. runtime-write verify：
   不再从 live request pages 重新 materialize expected tensor。

2. official-write repair：
   不再调用 PagedAttention.write_to_paged_cache 覆盖 one-copy 写入结果。

3. cross-source compare：
   direct retrieve finalize 不再额外回头读 LMCache 原始 storage 做逐块对照。

4. live-pages tensor decode：
   KV fast path 不再保留 load_chunk_tensors_from_live_request_pages()
   这类只服务 one-copy 调试校验的入口。
```

清理后 one-copy 的正确性边界更清楚：

```text
BaM cache page
  -> GPU persistent service direct scatter
  -> vLLM paged KV cache
  -> xFormers 从 paged KV cache 消费 prefix
```

如果输出错误，不能再靠前台 repair 覆盖，而应直接回到底层 scatter ABI、slot_mapping、block table 和 xFormers paged-cache consume 语义排查。

启动脚本也同步收束：

```text
run_single_gpu_lmcache_no_prefix_reuse_qwen25.sh
  只保留基础 BaM/LMCache/direct-placement/runtime 参数；
  不再透传 runtime-write verify、official-write verify、cross-source compare。

run_single_gpu_lmcache_no_prefix_reuse_qwen25_gpu_worker_persistent.sh
  固定 gpu_worker + runtime + persistent，默认 materialized fast path。

run_single_gpu_lmcache_no_prefix_reuse_qwen25_gpu_worker_persistent_verify.sh
  名字仍沿用，但当前语义是 one-copy 快速启动 wrapper：
  默认打开 runtime one-copy + require one-copy，不再打开重型 verify。
```

后续如果要继续做性能恢复，应该基于这三条干净链路比较，而不是重新打开旧的 verify/repair 支线。

### 13.7.9 2026-07-18 BaM 底层 KV façade 收束

在上层三条 KV 链路已经定版之后，BaM 底层仍残留了一些早期实验接口：

```text
1. page-offset submit fallback
2. status-only / no-status submit wrapper
3. GIDS_KV_WORKER_POLL_IMPL 动态 poll 分支
4. worker submit 失败后静默回退 rowctx batch
```

这些接口早期用于逐步接入 KV fast path，但当前会带来一个问题：

```text
日志上看起来是 gpu_worker / one-copy，
底层却可能因为某个旧扩展符号或开关走到另一条语义。
```

因此本轮只对 KV 专用 façade 做收束，不碰 CNN/GNN/DNN 的通用 rowctx/read_feature 路径。

收束后的底层入口如下：

```text
rowctx_baseline:
  BaMRowCtxKVExecutor
    -> BaMRowStore.kv_submit_chunk_batch_from_table()
    -> C++ kv_submit_chunk_batch_from_table_with_completion()
    -> kv_try_poll_batch()
    -> kv_consume_chunk_batch()

gpu_worker_persistent_materialized:
  BaMGPUWorkerKVExecutor
    -> BaMRowStore.kv_worker_submit()
    -> C++ kv_worker_submit_from_table()
    -> kv_worker_poll_request()
    -> persistent service / rowctx_compat 状态推进
    -> kv_worker_consume_batch()

gpu_worker_persistent_one_copy:
  BaMGPUWorkerKVExecutor
    -> BaMRowStore.kv_worker_submit()
    -> C++ kv_worker_submit_from_table()
    -> attach runtime placement / attention metadata
    -> persistent service direct scatter
    -> kv_worker_cleanup_batch()
```

本轮清理后的关键约束：

```text
1. rowctx baseline 也固定从 request-table ABI 出发，
   不再退回 page_offsets 路径。

2. kv_worker_submit 必须显式传入 status/chunk_status/completion/frontier table，
   缺少任意一张表直接报错，不再静默 fallback。

3. C++ kv_worker_poll_batch 保留 ABI，但内部固定为 rowctx_compat_blocking。
   persistent service 路径不依赖它，而是通过 kv_worker_poll_request()
   只读 runtime slot 状态。

4. 上层日志不再打印已废弃的 poll impl 开关，
   改为打印 kv executor/runtime/persistent 三个真正影响分支的参数。
```

这样三条链路的区别被压回到明确的执行层：

```text
rowctx_baseline:
  是否使用 worker = 否
  是否 persistent = 否
  是否 one-copy = 否

gpu_worker_persistent_materialized:
  是否使用 worker = 是
  是否 persistent = 是
  是否 one-copy = 否

gpu_worker_persistent_one_copy:
  是否使用 worker = 是
  是否 persistent = 是
  是否 one-copy = 是
```

后续如果 one-copy 或 materialized 再出问题，排查时不应再怀疑旧 submit/poll
fallback 是否偷偷生效，而应直接看当前分支自己的状态机、placement attachment、
frontier/consumable 发布和 xFormers 消费侧。

### 13.7.10 2026-07-18 启动开关收束：三条分支只暴露一个用户层选择器

当前启动层也进一步收束，不再要求命令行同时传：

```text
VLLM_BAM_KV_EXECUTOR
GIDS_KV_GPU_WORKER_RUNTIME_ENABLE
GIDS_KV_GPU_WORKER_PERSISTENT_ENABLE
VLLM_BAM_DIRECT_PLACEMENT_RUNTIME_ONE_COPY
VLLM_BAM_DIRECT_PLACEMENT_REQUIRE_RUNTIME_ONE_COPY
```

这些变量仍然会导出给底层代码使用，但不再作为用户手工组合的主入口。
用户层只保留一个分支选择器：

```text
VLLM_BAM_KV_BRANCH=rowctx_baseline
VLLM_BAM_KV_BRANCH=gpu_worker_persistent_materialized
VLLM_BAM_KV_BRANCH=gpu_worker_persistent_one_copy
```

主脚本 `run_single_gpu_lmcache_no_prefix_reuse_qwen25.sh` 根据分支名统一派生低层开关：

```text
rowctx_baseline:
  VLLM_BAM_KV_EXECUTOR=rowctx
  GIDS_KV_GPU_WORKER_RUNTIME_ENABLE=0
  GIDS_KV_GPU_WORKER_PERSISTENT_ENABLE=0
  VLLM_BAM_DIRECT_PLACEMENT_RUNTIME_ONE_COPY=0
  VLLM_BAM_DIRECT_PLACEMENT_REQUIRE_RUNTIME_ONE_COPY=0

gpu_worker_persistent_materialized:
  VLLM_BAM_KV_EXECUTOR=gpu_worker
  GIDS_KV_GPU_WORKER_RUNTIME_ENABLE=1
  GIDS_KV_GPU_WORKER_PERSISTENT_ENABLE=1
  VLLM_BAM_DIRECT_PLACEMENT_RUNTIME_ONE_COPY=0
  VLLM_BAM_DIRECT_PLACEMENT_REQUIRE_RUNTIME_ONE_COPY=0

gpu_worker_persistent_one_copy:
  VLLM_BAM_KV_EXECUTOR=gpu_worker
  GIDS_KV_GPU_WORKER_RUNTIME_ENABLE=1
  GIDS_KV_GPU_WORKER_PERSISTENT_ENABLE=1
  VLLM_BAM_DIRECT_PLACEMENT_RUNTIME_ONE_COPY=1
  VLLM_BAM_DIRECT_PLACEMENT_REQUIRE_RUNTIME_ONE_COPY=1
```

这样做的目的不是删除底层能力，而是减少错误组合：

```text
1. 不再出现 executor=gpu_worker 但 persistent 没开的半配置。
2. 不再出现 one-copy 打开但 require-one-copy 没开的模糊语义。
3. 日志先打印 vllm_bam_kv_branch，再打印派生后的低层值，方便确认真实路径。
4. 旧命令仍兼容：未显式设置 VLLM_BAM_KV_BRANCH 时，主脚本会根据旧低层变量推断分支。
```

对应 wrapper 也收束为只声明分支：

```text
run_single_gpu_lmcache_no_prefix_reuse_qwen25.sh
  默认等价 rowctx_baseline；
  也可显式传 VLLM_BAM_KV_BRANCH=rowctx_baseline。

run_single_gpu_lmcache_no_prefix_reuse_qwen25_gpu_worker_persistent.sh
  默认只传：
    VLLM_BAM_KV_BRANCH=gpu_worker_persistent_materialized

run_single_gpu_lmcache_no_prefix_reuse_qwen25_gpu_worker_persistent_verify.sh
  文件名沿用历史 verify 命名；
  当前语义是 one-copy 快速启动 wrapper，默认只传：
    VLLM_BAM_KV_BRANCH=gpu_worker_persistent_one_copy
    DIRECT_RETRIEVE_PREWARM_AFTER_REQUEST1=0
```

仍然保留的非分支调试开关：

```text
GIDS_KV_DEBUG=1
VLLM_BAM_XFORMERS_PREFIX_FALLBACK_PROFILE=1
VLLM_BAM_XFORMERS_PREFIX_BACKEND=...
VLLM_BAM_XFORMERS_QUERY_BACKEND=...
```

这些只用于日志或 xFormers fallback 诊断，不再决定三条 KV 主链路本身。

### 13.7.11 2026-07-18 性能瓶颈重判：persistent service 不应同时承担控制面和大数据搬运

最新 one-copy 正确版本已经确认：

```text
pipeline=gpu_worker_persistent_one_copy
worker_backend=kv_persistent_service_v0
impl=gpu_runtime_direct
finalize_mode=runtime_direct
```

这说明当前 request_2 没有路径回退，已经走到：

```text
BaM cache page
  -> GPU persistent service direct scatter
  -> vLLM paged KV cache
  -> xFormers 从 paged KV cache 消费 prefix
```

但最新性能仍然停在约：

```text
request_2_elapsed_s ~= 1.84s
read_ms ~= 339ms
poll_ms ~= 332ms
poll_iters ~= 208
```

同时，清理掉 Python rebuild stage 日志和 device-side direct-place probe 后，
性能只小幅改善，说明当前大头不是日志，也不是 xFormers fallback 的单点问题。

当前底层实现的关键结构是：

```text
kv_worker_runtime_persistent_service_kernel<<<1, 128>>>:
  1. service CQ / completion
  2. refresh runtime slots
  3. 扫描 runtime slot
  4. 如果 request IO_DONE，直接在同一个 CTA 内执行：
       BaM cache -> vLLM paged KV cache one-copy scatter
  5. 发布 CONSUMED / consumable frontier
```

这条实现把原先 CPU 侧多线程同步忙等轮询，收缩成了 GPU 侧一个轻量
persistent service CTA 轮询 CQ。这个方向是对的：

```text
轮询 / completion / request 状态机:
  适合由少量 GPU 常驻线程负责
```

但同一个 CTA 又承担了大规模数据搬运：

```text
4 chunks * 112 pages/chunk * 128KB/page ~= 56MB
```

这就把 56MB 级别的数据面压进了 128 个线程里。128 线程适合控制面，
不适合做几十 MB 的 scatter copy，因此 persistent one-copy 正确版本虽然
语义更接近最终目标，性能却会被单 CTA 搬运吞吐限制。

另一个次要低效点是 runtime slot 扫描：

```text
runtime_capacity 默认 1024
即使当前只有 1 个活跃 request，
service 也会在 consume 阶段串行扫描 1024 个 slot，
并在 slot 循环中反复 __syncthreads()
```

这个会放大 service loop 成本，但不是 300ms 级瓶颈的主因。主因仍然是：

```text
控制面 service CTA 正在做数据面大搬运
```

#### 四篇工作的启发

BaM 的启发：

```text
BaM 的核心不是“CPU 帮 GPU 做 I/O”，而是让 GPU 能以高并发方式发起
细粒度 storage access，并通过 GPU-side software cache / high-throughput
queues 合并请求、降低 I/O amplification。
```

映射到当前 KVCache 主线：

```text
1. 继续保留 BaM 的 GPU-side CQ / cache / request table 能力。
2. 不应把 CPU poll 重新拉回热路径。
3. 但也不应把大数据搬运限制在单个 service CTA 内。
```

Tutti 的启发：

```text
Tutti 把 SSD-backed KV cache 做成 GPU-centric KV object store，
目标是把 CPU 从 HBM <-> SSD 的关键数据路径和 I/O 控制路径中移走。
同时它强调 GPU-native object abstraction 和 slack-aware I/O scheduling。
```

映射到当前主线：

```text
1. 当前 chunk/page/slot_mapping/kv_cache_ptrs 应继续收成 GPU-visible object descriptor。
2. CPU 可以负责高层 request admission / vLLM block 分配，但不应参与 page/chunk 数据搬运。
3. I/O worker 不能无约束抢占 SM；需要能限制数据搬运 CTA 数量，避免和 attention 争资源。
```

TARDIS 的启发：

```text
TARDIS 的方向是 GPU-centric KV cache service。
对 LLM 推理来说，KV cache 不应继续被当作普通 feature row 或通用 tensor
读写，而应有面向层、chunk、request frontier 的专用服务语义。
```

映射到当前主线：

```text
1. 保留 KV 专用 request/frontier/completion table，不再回退通用 row 语义。
2. 后续应支持 layer-wise / chunk-wise consumable，而不是只把整批 request
   当作一个 blocking read。
3. attention 需要哪一层、哪一段 prefix，GPU runtime 应能按 frontier 逐步发布。
```

AGIO 的启发：

```text
AGIO 强调把 I/O initiation 和 completion 解耦，让 GPU 线程发起异步 I/O 后，
可以继续推进其它工作，而不是同步等待 completion。
```

映射到当前主线：

```text
1. CPU submit 之后立刻返回，只观察 consumable 状态，这是正确方向。
2. GPU service 不应在一次 request 内同步卡死等待所有后续工作都完成。
3. 应把 read completion、copy job、consumable publish 拆成异步流水。
```

#### 最终推荐架构

最终不应该是：

```text
一个 persistent service CTA:
  poll CQ
  refresh state
  搬 56MB KV
  fill metadata
  publish consumable
```

而应该拆成两个 GPU 角色：

```text
GPU persistent control service:
  少量常驻线程 / 1 个 service CTA
  负责：
    - CQ poll
    - page/chunk/request 状态机
    - frontier / completion / consumable 发布
    - 生成 copy job

GPU data mover workers:
  多 CTA / 可配置线程资源
  负责：
    - 读取 copy job queue
    - 并行执行 BaM cache -> vLLM paged KV cache
    - 搬完后更新 cache_ready / consumable
```

对应数据流：

```text
CPU / vLLM scheduler:
  1. prefix 命中分析
  2. 分配 vLLM paged KV blocks
  3. 生成 GPU-visible request descriptor
  4. submit seed request 后立即返回
  5. 后续只 poll consumable frontier

GPU control service:
  1. 轮询 BaM CQ
  2. 发现 page ready
  3. 更新 page/chunk/read_ready 状态
  4. 当 chunk/page 达到可搬运条件，写入 copy_job_queue
  5. 观察 data mover 完成状态
  6. 发布 chunk_consumable / request_consumable

GPU data mover:
  1. 从 copy_job_queue 取 job
  2. 根据 request_table / chunk_start / slot_mapping / kv_cache_ptrs
     计算源 BaM cache page 和目标 vLLM paged KV cache 地址
  3. 多 CTA 并行 scatter
  4. 写回 job_done / chunk cache_ready
```

这样可以同时满足两个目标：

```text
1. 轮询和状态管理仍然由 GPU 全权负责；
2. 大规模数据搬运不再被 128-thread service CTA 限制。
```

#### 分阶段落地建议

第一阶段：恢复正确 one-copy 的性能基线。

```text
persistent service CTA:
  只做 CQ poll / 状态机 / 发布 IO_DONE

host observe:
  看到 IO_DONE 后只 launch 一个宽 scatter kernel
  不做数据搬运、不做 CPU rebuild

wide scatter kernel:
  多 CTA 并行执行 BaM cache -> vLLM paged KV cache
  完成后发布 CONSUMED / consumable
```

这一步仍有一次 CPU kernel launch，但 CPU 不参与数据搬运。它的价值是快速验证：

```text
当前 330ms 是否主要来自单 CTA 搬运。
```

如果宽 scatter kernel 能把 read/poll+place 拉回几十毫秒量级，就说明方向正确。

第二阶段：把 wide scatter 的启动权下沉。

```text
GPU control service:
  发现 chunk ready 后，不再等 CPU launch
  直接向 GPU-resident copy_job_queue 发布 job

GPU data mover persistent workers:
  常驻或按资源预算运行
  自己取 job 并执行 copy
```

这一步才是真正贴近最终 GPU-initiated runtime 的版本：

```text
CPU:
  submit seed request
  observe consumable

GPU:
  poll
  issue follow-up work
  move data
  publish consumable
```

第三阶段：做推理引擎友好的 frontier。

```text
request-level consumable:
  当前请求命中多少连续 prefix，就等待多少连续 prefix consumable 后返回。

chunk/layer-level consumable:
  后续可扩展到 layer-wise 或 chunk-wise publish，
  让 attention 与 KV restore 有机会流水化。
```

这一步对应 Tutti / TARDIS 更强调的 serving 场景：不是单请求一次性搬完所有数据，
而是在不破坏 vLLM 调度语义的前提下，把 I/O 和计算做成可重叠流水。

#### 当前代码下一步应该怎么改

短期不建议继续优化单 CTA 内部的 copy 循环，因为即使把 byte/uint4 细节调好，
本质仍然是：

```text
1 个 CTA 搬几十 MB
```

更合理的下一步是：

```text
1. 把 persistent service 中 direct placement 的职责降级：
   从“直接搬数据”改成“发布 ready/copy job”。

2. 新增或复用一个宽 one-copy scatter kernel：
   输入仍然是当前已经验证正确的 runtime descriptor：
     request_table
     ctx_ptr / ctx_count / ctx_stride
     chunk_start
     slot_mapping
     kv_cache_ptrs
     page layout 参数

3. 先由 CPU 在 observe 到 IO_DONE 后 launch 这个宽 kernel，
   用最少改动验证性能。

4. 验证通过后，再把 launch 替换成 GPU-resident copy_job_queue +
   persistent data mover workers。
```

需要保持的约束：

```text
1. 不回退到 output_pages staging。
2. 不恢复 official-write repair。
3. 不让 CPU rebuild / CPU copy 回到 one-copy 热路径。
4. rowctx_baseline 和 materialized 分支继续保留，用作正确性对照。
5. CNN/GNN/DNN 的通用 BaM 路径不参与这轮重构。
```

因此，新的最终主线应该从：

```text
GPU persistent service = poll + copy + publish
```

收束成：

```text
GPU runtime = control service + data mover workers
```

这才同时符合 BaM / Tutti / TARDIS / AGIO 给出的共同方向：

```text
CPU 从关键 I/O 控制路径和数据路径中退出；
GPU 负责异步 I/O、状态机和数据移动；
控制面轻量常驻；
数据面宽并行、按资源预算执行；
推理侧只消费已经发布的 consumable KV。
```

### 13.7.12 2026-07-19 cta=4 one-copy 恢复与 CQ 根因确认

本轮回退后，`GIDS_KV_GPU_WORKER_MOVER_CTAS=4` 一度再次卡在 request 2：

```text
request_id=2
runtime_row=(1, 1, 1, 2, ...)
host_status=1
read_ready_chunks=4/4
cache_ready_chunks=0/4
consumable_chunks=0/4
```

这个状态的含义是：

```text
1. request 已经 submit 并 attach 到 runtime slot；
2. high-level frontier 已经知道本轮应该读 4 个 prefix chunk；
3. 但 GPU runtime slot 仍停在 SUBMITTED；
4. persistent service 没有把底层 CQ completion 推进到 IO_DONE / CONSUMED。
```

最终确认这不是没有重新编译，也不是路径回退。

证据：

```text
BAM_ROW_STORE_IMPORT:
  bam_feature_store=/home/xhk/llm-inference/BaM_IOStack/gids_module/build/BAM_Feature_Store/__init__.py

BAM_Feature_Store.so:
  /home/xhk/llm-inference/BaM_IOStack/gids_module/build/BAM_Feature_Store/BAM_Feature_Store.so

request_1:
  pipeline=gpu_worker_persistent_one_copy
  worker_backend=kv_persistent_service_v0
  finalize_mode=runtime_direct
```

真正根因是 BaM 底层 CQ service 的 missing `ctx_lookup` 处理方式被改偏了。

错误改法：

```text
cq_try_peek_head()
  -> 如果 cid 对应的 ctx_lookup[slot] == nullptr
       保留 CQ head
       break
```

这个写法看起来像是在避免 completion 丢失，但当前 BaM CQ/CID 语义里，
CQ head 可能对应旧 request、stale entry 或已经不再有 lookup 的 completion。
如果保留这个 head，就会永久堵住对应 queue，后续真正属于 request 2 的
completion 无法被 service 到，所以 request 2 一直停在 `SUBMITTED`。

恢复后的正确语义与 `22b7cc7` 已验证版本一致：

```text
cq_try_peek_head()
  -> dequeue CQ head
  -> put_cid()
  -> 如果 cid >= capacity，跳过
  -> 查 ctx_lookup[logical_queue, cid]
  -> 如果 ctx_lookup 缺失，跳过
  -> 如果 ctx_lookup 存在，finalize_registered_ctx_completion()
```

这条规则的工程含义：

```text
1. CQ head 不能被一个 missing lookup 永久堵住；
2. 当前 CQ entry 没有 request/generation 级别匹配字段，
   因此不能把 missing lookup 当成“未来一定会变有效”的 completion；
3. 要彻底解决潜在竞态，应在底层引入 request/generation 匹配，
   而不是在没有 generation 的情况下保留 CQ head；
4. 当前阶段保持 22b7cc7 的已验证语义，优先保证 request 连续推进。
```

恢复后最新日志：

```text
evaluation/logs/single_gpu_lmcache_no_prefix_reuse_qwen25/20260719_023812/run.log
```

关键结果：

```text
request_id=2
[BAM_KV_GPU_WORKER_POLL_READY] poll_iters=4 poll_ms=14.360
runtime_row=(1, 4, 3, 2, ...)
read_ms=22.154
request_2_elapsed_s=1.5297
```

因此当前 cta=4 one-copy 链路已经恢复为：

```text
CPU:
  1. submit request table
  2. attach runtime placement / metadata descriptor
  3. 非阻塞 observe runtime status
  4. 看到 CONSUMED 后做 cleanup-only finalize

GPU persistent service CTA:
  1. 轮询 BaM CQ
  2. dequeue completion
  3. refresh runtime slot
  4. request 到 IO_DONE 后发布 direct_move 任务
  5. 等 mover CTA 完成
  6. fill attention metadata
  7. 发布 CONSUMED / consumable frontier

GPU mover CTA1..N:
  1. 观察 direct_move_state
  2. atomic claim linear worker id
  3. 按官方 vLLM paged KV ABI 执行 direct scatter
  4. atomicAdd direct_move_done_ctas
```

当前应固定下来的实现约束：

```text
1. 不恢复 Python poll rescue。
2. 不在 one-copy finalize 中 stop persistent service。
3. 不恢复 official-write repair 覆盖 one-copy 结果。
4. 不把 output_pages staging 混回 one-copy 正常热路径。
5. CQ service 缺失 ctx_lookup 时不能堵 CQ head。
6. 后续若要彻底修 CQ/lookup 竞态，应新增 generation/request 匹配语义。
```

当前三条链路仍应这样保留：

```text
rowctx_baseline:
  正确性和回归兜底。

gpu_worker_persistent_materialized:
  GPU persistent poll/read/stage + host materialized placement。
  用于判断 one-copy scatter 以外的 persistent 主线是否正常。

gpu_worker_persistent_one_copy:
  GPU persistent poll + mover CTA direct scatter + cleanup-only finalize。
  当前已经恢复到 request_2_elapsed_s≈1.53s 的正确性能基线。
```

2026-07-19 已完成的基线收束：

```text
1. 主启动脚本已经按分支派生 mover CTA：
   gpu_worker_persistent_one_copy -> GIDS_KV_GPU_WORKER_MOVER_CTAS=4
   其它分支 -> GIDS_KV_GPU_WORKER_MOVER_CTAS=0

2. one-copy wrapper 不再重复透传 debug/profile/mover/prewarm 默认值，
   只声明 VLLM_BAM_KV_BRANCH=gpu_worker_persistent_one_copy。

3. vLLM / BaM 注释已从“激进实验 / verify repair”收束为
   “cta=4 one-copy 稳定基线”。
```

后续不建议继续围绕 CQ missing lookup 加临时保护分支。
更合理的方向是：

```text
1. 如果继续优化性能，应从 control service / data mover worker 解耦入手；
2. 如果继续优化稳定性，应在 BaM 底层补 request/generation 级 CQ 匹配，
   而不是在 service loop 里用保留 CQ head 的方式猜测 completion 所属关系。
```

---

## 14. 相关 SSD/KVCache 工作的评测数据集与后续 baseline 选择

这一节用于记录后续做 BaM one-copy / LMCache SSD / GDS / GPU-initiated
对比时应该参考哪些公开 workload。这里不改变当前实现主线，只约束后续
benchmark 不要只停留在单条固定 prompt。

### 14.1 相关工作的 workload 口径

当前能直接对应到 KV cache / SSD-backed KV cache 的工作，大致分成两类：

```text
1. 长上下文 QA / 摘要 / 多文档检索类数据集
   目的：
     拉长 prompt，制造大量可复用 prefix KV，
     观察 SSD read、KV restore、prefix consume 对 TTFT 的影响。

2. 多轮对话 / 请求 trace 类数据集
   目的：
     制造跨请求 prefix reuse、cache hit/miss、warmup/steady-state，
     更接近 serving 系统里的真实缓存复用行为。
```

各工作的评测 workload 目前整理如下：

| 工作 | 使用的数据集 / workload | 对我们的意义 |
| --- | --- | --- |
| SolidAttention | 性能侧使用 `WikiText-2` 构造不同长度 prompt；准确率侧使用 `OpenCompass`，包括 `Winogrande`、`ARC-Challenge`、`MMLU`、`GSM8K`、`LongBench`；长上下文任务里包含 `2WikiMQA`、`TriviaQA`、`HotpotQA`、`MultiFieldQA`、`MuSiQue`、`NarrativeQA`、`Qasper`、`GovReport` | 适合参考它的“合成长 prompt + LongBench 正确性”组合；但它更偏 sparse attention / 本地 SSD offload，不完全等价于我们的 BaM KV restore |
| Tutti | `LEval`、`LooGLE` | 最贴近 SSD-backed KV cache 主线；适合用于长上下文、长 prefix 读回、GPU bubble 和 storage bandwidth 对比 |
| TARDIS | 公开可访问材料里暂未稳定确认完整 dataset 列表 | 目前只把它作为 GPU-centric KV cache service 的系统设计参考，不把 dataset 作为确定依据写入实验计划 |
| LMCache | 合成 multi-round QA、`LongBench` long-context QA、vLLM random benchmark；centralized storage server 实验里可重点看 `LongBench-TriviaQA` | 这是和我们当前 LMCache/BaM connector 最直接的 baseline 对照，应优先对齐它的 LongBench 与 random workload |
| HCache | `ShareGPT4`、`L-Eval` | 适合补多轮对话与 long-context/RAG 两类复用场景 |
| CachedAttention / AttentionStore | `ShareGPT` sessions | 适合评估多轮对话里的 prefix reuse、cache hit、warmup/steady-state |
| KVPR | 主要使用 synthetic sequence settings，例如固定 batch size、prompt length、generation length | 更适合做机制拆解，不适合作为真实 SSD KV cache workload 主线 |

### 14.2 当前项目建议采用的 benchmark 层次

结合上面的工作，后续 benchmark 不建议一次性铺太大，而应分三层固定：

```text
第一层：固定 prompt 双请求样例
  用途：
    保持当前 one-copy / LMCache SSD / GDS 通路可快速回归。
  当前已经使用：
    request_1 写入或建立缓存；
    request_2 命中共享 prefix 后读回 4 个 chunk。
  主要指标：
    request_2_elapsed_s
    read_ms / poll_ms / get_ms
    输出正确性

第二层：LongBench / LEval / LooGLE 长上下文样例
  用途：
    对齐 SolidAttention、Tutti、LMCache 这类论文口径，
    观察长 prefix 下 read bandwidth、placement、attention consume 的占比。
  建议优先顺序：
    LongBench-TriviaQA
    LongBench-GovReport
    LEval
    LooGLE

第三层：ShareGPT / ShareGPT4 多轮 trace
  用途：
    评估 serving 场景里的 cache reuse、warmup、steady-state、
    多请求调度和未来 GPU follow-up submit。
  主要指标：
    TTFT
    request throughput
    cache hit rate
    storage read bandwidth
    GPU bubble / SM occupancy
    p50 / p90 / p99 latency
```

这三层的关系是：

```text
固定 prompt 双请求：
  验证链路是否正确，定位单次 read/placement 开销。

LongBench / LEval / LooGLE：
  验证长上下文下 KV restore 是否有扩展性。

ShareGPT / ShareGPT4：
  验证多请求、多轮、prefix reuse 场景下是否真的能服务化。
```

### 14.3 与当前 BaM one-copy / LMCache SSD baseline 的对应关系

当前已经新增的 baseline 文件夹：

```text
evaluation/lmcache_ssd_read_paths_baseline/
```

这组 baseline 最初用于第一层固定 prompt 双请求样例，现在已经扩展到
LongBench-TriviaQA bucket 化 manifest：

```text
ssd_cpu_gpu:
  原生 LMCache V0 local_disk
  数据路径：
    SSD -> CPU MemoryObj -> multi_layer_kv_transfer -> vLLM paged KV cache

gds_gpu:
  LMCache-style GDS wrapper
  数据路径：
    SSD/cufile -> CUDA chunk tensor -> LMCache MemoryObj
    -> multi_layer_kv_transfer -> vLLM paged KV cache

bam one-copy:
  当前 cta=4 稳定基线
  数据路径：
    BaM cache -> vLLM paged KV cache
```

当前 LongBench 侧已经新增并使用的主要脚本：

```text
evaluation/lmcache_ssd_read_paths_baseline/
  run_longbench_triviaqa_ssd_read_paths_qwen25.sh
    测原生 LMCache local_disk / GDS wrapper 路径

  run_longbench_triviaqa_bam_one_copy_qwen25.sh
    测 BaM one-copy 主路径

  run_longbench_triviaqa_bam_one_copy_eviction_qwen25.sh
    normal persistent service + 512MB BaM cache 压力测试

  run_longbench_triviaqa_bam_one_copy_refcount_debug_qwen25.sh
    ref_count debug 专用，允许 stop service / 打 debug stats

  run_longbench_triviaqa_bam_one_copy_refcount_eviction_debug_qwen25.sh
    压 cache + ref_count debug 专用，不作为正式性能口径
```

截至 2026-07-19 的固定 prompt request 2 对比：

```text
原生 LMCache SSD:
  request_2_elapsed_s=2.0372

LMCache-style GDS:
  request_2_elapsed_s=2.0507
  4 个 chunk GDS read 合计约 39.97ms

BaM one-copy cta=4:
  request_2_elapsed_s=1.5252
  read_ms=19.926
  poll_iters=4
```

这个结果说明：

```text
1. 当前 BaM one-copy request 2 端到端约比两条 LMCache SSD 路径快 0.51s；
2. GDS 纯 chunk read 与 BaM read/direct placement 的差距只有约 20ms；
3. 因此端到端差距不只来自 SSD 读带宽，
   还需要继续拆 LMCache retrieve、MemoryObj 包装、
   multi_layer_kv_transfer 和 xformers prefix consume 的上层固定开销。
```

后续扩展 baseline 时，建议先保持这个目录结构：

```text
evaluation/lmcache_ssd_read_paths_baseline/
  logs_longbench_triviaqa/
    ssd_cpu_gpu/
    gds_gpu/
    bam_one_copy/
  RESULT_20260719.md
```

然后再新增子目录，而不是把长上下文、多轮 trace 和固定 prompt 混到一起：

```text
evaluation/lmcache_ssd_read_paths_baseline/
  fixed_prompt/
  long_context/
  sharegpt_trace/
```

### 14.4 后续测试优先级

短期优先做：

```text
1. 给原生 LMCache SSD 路径补 chunk 级 timing：
   LocalDiskBackend.get_blocking()
   MemoryObj 构造 / tensor 转换
   multi_layer_kv_transfer(direction=false)

2. 给 GDS wrapper 路径拆 timing：
   cufile / POSIX read
   CUDA chunk tensor populate
   MemoryObj 包装
   multi_layer_kv_transfer

3. 固定 one-copy cta=4 作为 BaM 当前稳定基线，
   暂时不要把 cta=2 这类已知会卡住的配置放进正式对比。
```

中期再做：

```text
1. 接 LongBench-TriviaQA / GovReport。
2. 接 LEval / LooGLE。
3. 接 ShareGPT / ShareGPT4 trace。
```

如果当前只选一个数据集来同时覆盖 SSD-backend 性能和后续类似 SolidAttention 的预取机制，
优先级应定为：

```text
LongBench-TriviaQA  >  LongBench-GovReport  >  WikiText-2 sliced prompt
```

原因是：

```text
1. TriviaQA 属于长上下文 QA，输入足够长，输出通常较短，
   适合先看 SSD read / KV restore / prefix consume 的纯性能。

2. GovReport 更偏长文档摘要，适合在 TriviaQA 之后拉长 prefix，
   观察 read bandwidth 和 consume 开销的扩展性。

3. WikiText-2 sliced prompt 更适合专门验证 SolidAttention 式预取，
   但不适合作为第一条 SSD-backend baseline。
```

这些 workload 的使用目标不是替代当前固定 prompt 样例，而是补齐论文对齐：

```text
固定 prompt:
  工程回归。

LongBench / LEval / LooGLE:
  长上下文论文口径。

ShareGPT / ShareGPT4:
  多轮 serving 口径。
```

### 14.5 参考链接

```text
SolidAttention FAST'26:
  https://www.usenix.org/system/files/fast26-zheng.pdf

Tutti:
  https://arxiv.org/abs/2605.03375

TARDIS:
  https://dl.acm.org/doi/10.1145/3725783.3764393

LMCache:
  https://arxiv.org/abs/2410.05004

CachedAttention:
  https://prongs1996.github.io/assets/pdf/CachedAttention.pdf

KVPR:
  https://arxiv.org/abs/2411.17089
```

### 14.6 LongBench-TriviaQA 的收集与处理建议

如果当前目标是先测 SSD-backend 链路，再为后续 SolidAttention 式预取做铺垫，
LongBench-TriviaQA 建议按下面方式落地：

```text
1. 数据源选择
   优先使用 LongBench 的 TriviaQA 任务。
   原始 LongBench 的官方说明里，TriviaQA 是按阅读理解式长文档问答构造的，
   数据格式已经标准化为 input / context / answers / length / dataset / language / _id。

2. 下载方式
   直接通过 Hugging Face datasets 加载任务切片。
   当前可见的公开镜像包括 LongBench 的 HF 数据集卡，
   官方仓库 README 也给出了 load_dataset('THUDM/LongBench', 'triviaqa', split='test')
   这样的用法。
   如果环境里镜像名不同，优先以当前 HF dataset card 为准。

3. 预处理目标
   把每条样本整理成我们自己的 benchmark manifest，
   至少保留：
     _id
     dataset
     length
     prompt_text
     answers
     context_len / prompt_len
     prefix_hash

4. prompt 组织
   使用 LongBench 原始 task template，
   将 context 和 input 拼成最终 prompt，
   不要在这个阶段额外做会改变 token 边界的二次改写。
   后面如果要测 prefetch，最好在 tokenization 之后再做 block 化，
   这样更容易和 SSD / KV cache 的 page 边界对齐。

5. 样本切分
   先固定一个小而稳定的子集做系统回归，
   再按 length 分桶扩展。
   建议顺序：
     - 先跑 50~100 条固定样本
     - 再按 0-4k / 4k-8k / 8k+ 分桶
     - 最后再扩大到全量 test split

6. 评测方式
   如果测性能：
     重点看 request_2_elapsed_s、read_ms、poll_ms、get_ms、TTFT。
   如果测正确性：
     对 answers 做归一化 EM / token overlap。
   如果测预取：
     额外记录每个样本的 token 长度、chunk 数、是否能复用前缀。

7. 和 SolidAttention 预取的衔接
   LongBench-TriviaQA 适合先验证“长上下文 KV restore 是否正确且稳定”。
   如果要进一步验证 block predictor / prefetch 策略，
   后面再补 WikiText-2 sliced prompt 更合适，
   因为它更容易人为控制长度和 block 命中模式。
```

---

## 15. 2026-07-26 当前收束结论与后续创新点

这一节只记录当前已经完成并应固定下来的事实，以及下一阶段围绕
GPU-initiated / SSD-backed KV cache prefetch 的判断。它不覆盖前面历史排查，
也不要求立即改代码。

### 15.1 当前已经完成的工程状态

当前已经可以固定下来的部分：

```text
1. cta=4 one-copy 作为稳定基线
   链路：
     LMCache prefer-load
       -> BaM KV fast path
       -> GPU persistent service
       -> mover CTA one-copy scatter
       -> vLLM paged KV cache

2. request-scoped ref_count release
   submit 阶段借用 BaM page；
   GPU service 在 one-copy scatter 完成后 release；
   host cleanup 只负责 runtime slot / ctx buffer 生命周期。

3. LongBench-TriviaQA baseline
   数据集已经按 Qwen2.5 tokenizer token 数组织成 bucket manifest；
   当前主测 bucket 是 4k_8k；
   固定 prompt baseline 和 LongBench baseline 不再混在同一个目录层次里。

4. 正常后台常驻 eviction pressure 测试
   使用 512MB BaM cache、64 shadow chunks、12 samples；
   不打开 ref debug / stop-service；
   能稳定跑完 24 requests。

5. GPU-initiated descriptor prefetch 主线
   CPU early-submit 原型已经收束掉：
     不再在 connector/storage 预取阶段调用 BaM native submit；
     不再保存 `_kv_batch_submitted_requests` 这种已提交 handle 表；
     connector 只提前生成 retrieve plan；
     BaM store 只把 plan 准备成 compact KV descriptor；
     direct placement start 在统一 read 边界消费 descriptor，再 submit/poll/finalize。
   这保留了“上层提前规划，下层 request 生命周期接管”的接口，
   但避免把 CPU early-submit 误称为 GPU-initiated。
   后续真正 GPU-initiated 化时，应把这份 descriptor 写入 GPU-side ring，
   由 GPU persistent service 消费 descriptor 并发起 SSD read。
```

关键日志口径：

```text
ref_count debug:
  logs_longbench_triviaqa/bam_one_copy/20260722_225047/run.log
  borrowed_pages_submitted=23408
  borrowed_pages_released=23408
  borrowed_pages_outstanding=0

normal persistent eviction pressure:
  logs_longbench_triviaqa/bam_one_copy/20260722_231109/run.log
  cache_size_mb=512
  chunk_capacity=64
  chunk_slot_evictions=169
  requests=24
  samples=12
  read_avg_s=0.9472
```

因此当前不应再引入：

```text
1. one-copy 正常路径里的 fallback ref decrement；
2. submit failure 后的 host 兜底 release；
3. 正常性能脚本里的 stop-service lifecycle stats；
4. 为观察 ref_count 而长期保留的同步 debug kernel。
5. connector-level `engine.prefetch()` 作为 one-copy GPU-initiated 主线；
   它不能覆盖当前 deferred one-copy read 主线，且容易在 write / miss 阶段
   生成不可消费的 prefetch 日志。
```

### 15.2 LMCache 淘汰与 BaM 淘汰是否对齐

当前两层淘汰机制是正确性兼容，但不是同一个策略状态机。

```text
LMCache 层：
  管逻辑 KV chunk。
  维护 chunk_hash -> page_offset / metadata。
  VLLM_BAM_LMCACHE_SHADOW_CHUNKS=64 时会触发 chunk slot eviction。
  日志里的 chunk_slot_evictions 属于这一层。

BaM 层：
  管 SSD page -> GPU page cache 的物理缓存。
  victim 条件是：
    UNLOCKED && ref_count == 0 && !BUSY
  BaM 不理解 LMCache chunk / prompt / request 语义。
```

因此：

```text
LMCache chunk 被淘汰：
  不代表 BaM page 立即失效；
  只代表逻辑 metadata 不再命中。

BaM page 被替换：
  不代表 LMCache chunk metadata 失效；
  数据仍然在 SSD / BaM store 中，后续 miss 后可以重新读取。
```

当前合理定位是：

```text
LMCache:
  逻辑 KV cache 控制面
  prefix / chunk / request reuse / metadata eviction

BaM:
  SSD-backed 数据面
  GPU page cache / I/O / one-copy scatter / ref_count lifecycle
```

后续如果要做策略对齐，不建议把两层 cache 合并，而应补 hint 接口：

```text
LMCache evict chunk metadata
  -> BaM mark_cold(page_offset, page_count)

LMCache predict future chunk
  -> BaM prefetch_pages(page_offset, page_count, priority, deadline)

BaM victim selection
  -> 在 ref_count==0 && !BUSY 的候选里优先替换 cold page
```

### 15.3 BaM 直接接 vLLM 还是接 LMCache

如果只做 vLLM swap / block-level paging，BaM 直接接 vLLM 更短：

```text
vLLM scheduler / block manager
  -> BaM SSD backend
  -> vLLM paged KV cache
```

适合：

```text
1. vLLM 内部 swap out / swap in；
2. preemption block restore；
3. 单 engine 内部 block 生命周期；
4. 极限性能数据面优化。
```

但当前项目目标更接近 SSD-backed KV cache reuse 和后续 prefetch 策略，
因此继续接在 LMCache 上更合理：

```text
vLLM
  -> LMCache connector / token database / chunk metadata
  -> BaM backend
  -> SSD / GPU page cache
```

原因：

```text
1. LMCache 已经有 chunk_hash / prefix reuse / store-retrieve 语义；
2. 预取策略需要知道未来 request / token range / chunk；
3. 多级流水线更像 LMCache 的控制面职责；
4. BaM 不应理解 prompt / request / prefix 语义。
```

当前建议：

```text
短期主线：
  vLLM -> LMCache -> BaM
  用于 SSD-backed KV reuse、LongBench baseline、prefetch 策略实验。

长期性能线：
  vLLM block manager -> BaM
  用于 vLLM swap / preemption / block-level SSD paging。
```

### 15.4 SolidAttention 式预取是否是常规方向

结论：

```text
SSD-backed KV cache 上做流水线预取是常规且必要的方向；
创新点不在“有没有 prefetch”，而在：
  预测什么；
  以什么粒度预取；
  如何和 attention / layer compute overlap；
  如何减少 CPU 在细粒度 I/O submit/poll 上的参与。
```

SolidAttention 的可借鉴点：

```text
1. KV Consolidator
   把小粒度 KV entry 合并，增大 SSD transfer unit。

2. Speculative Prefetcher
   利用相邻 iteration 的重要 KV 选择局部性提前预取。

3. SSD-aware Scheduler
   把 computation / I/O 组织成能 overlap 的细粒度任务。
```

但对当前 LMCache-backed SSD KV 主线来说，不能只照搬 sparse-attention 语义。
更合适的落点是：

```text
vLLM:
  负责 request 调度、attention 执行语义、layer / token frontier。

LMCache:
  负责 chunk/block 生命周期、命中判断、淘汰策略、预取计划。

存储后端:
  按 LMCache 请求执行 page/chunk I/O；
  后端 cache 可以作为预取缓存区；
  不决定哪些 KV 应该被预取、淘汰或参与 attention。

接口之间：
  只传 compact prefetch plan / priority / deadline / hint；
  不共享完整 cache 状态机；
  不把上层策略下沉到 I/O 后端。
```

### 15.5 其他相关最新成果的定位

当前应这样理解几类工作：

```text
Tutti:
  方向最接近 GPU-initiated SSD-backed KV cache。
  核心是 GPU-native object store、GPU I/O control path、
  slack-aware I/O scheduling。
  公开 vLLM main 中能看到 multi-tier KV offload / objectstore tier PR，
  但当前没有明确找到 Tutti 本体的 GPU-initiated SSD data path 实现。

TARDIS:
  更偏 GPU-centric KV cache service。
  可以作为“GPU service 管理 KV cache / I/O”的系统设计参考，
  不把它的 dataset 口径作为当前实验主依据。

LMCache:
  适合做语义控制层。
  当前 BaM 接在 LMCache 上，是为了复用 chunk metadata、
  prefix reuse、backend abstraction 和后续 prefetch policy。

DualPath:
  更偏分布式 / 多轮 agentic workload 的 KV loading。
  对后续跨节点或 decode-side bandwidth 复用有参考价值，
  但不是当前单机 BaM page-cache 主线。

Asynchronous KV Cache Prefetching:
  目标层级是 GPU 内部 L2 / cache hierarchy。
  它说明 KV prefetch 是跨层通用优化，但和 SSD-backed page prefetch
  是不同层级的问题。
```

### 15.6 值得尝试的 GPU-initiated 小创新点

当前不要追求“整个推理过程完全 GPU-only”。vLLM scheduler 仍然更适合在 CPU
侧做全局 request admission、continuous batching、block allocation 和 prefix
lookup。更合理的目标是：

```text
CPU off critical path for KV restore / prefetch.
```

优先级建议：

```text
1. GPU-initiated layer-wise KV prefetch
   CPU/LMCache 生成 coarse plan；
   后端执行层按 layer frontier 预取后续 layer / chunk pages；
   目标是把 SSD I/O 藏进 attention compute slack。

2. GPU-side request expansion
   CPU 只写 compact descriptor：
     chunk_id, page_offset, page_count, layer range, priority
   后端执行层展开成 per-page / per-layer I/O requests。

3. GPU-side I/O coalescing
   后端执行层对多个 chunk/page request 按 LBA 或 page_offset 合并；
   减少 tiny random I/O 和 CPU submit 开销。

4. LMCache -> storage backend cache hint
   LMCache 负责 hot/cold/prefetch-candidate 语义；
   后端只把 hint 用到 victim selection / prefetch priority。

5. Frontier-driven partial readiness
   后端执行层持续更新 chunk/layer readiness；
   上层只消费已经 ready 的 frontier，
   后续再考虑和 attention pipeline 做更细粒度 overlap。
```

推荐下一阶段主线：

```text
先不动 vLLM scheduler 主体；
先在 LMCache 与存储后端之间增加 prefetch plan 和 page hint；
先让后端执行层承担更多细粒度 I/O 控制；
再评估是否需要把 layer-wise readiness 暴露给 attention consume。
```

### 15.7 类 SolidAttention 预取机制的落地边界

这里的预取机制应先按“透明优化”理解，而不是一开始就改变模型
attention 语义。

当前 dense attention 语义：

```text
每个 decode token 会 attend 到它可见范围内的完整历史 KV；
LMCache / 存储后端只改变 KV 如何提前读回，
不改变 attention kernel 会消费哪些 KV。
```

对应实现边界：

```text
vLLM:
  提供 request / layer / token frontier；
  保持原有 dense attention 正确性；
  必要时暴露 query、block table、layer id 作为预测输入。

LMCache:
  生成 chunk/block 级 prefetch plan；
  管理 prefetch 命中、失效、淘汰和正式 retrieve 兜底；
  统计 prefetch hit、late prefetch、wasted prefetch、retrieve wait time。

存储后端:
  执行预取读写；
  提供缓存区；
  返回 readiness / stats；
  不负责策略判断。
```

是否需要修改推理模型，取决于目标：

```text
1. 只做访问预测和预取
   不需要修改模型。
   Qwen2.5 可以继续作为主测模型；
   attention 仍然按 dense 语义计算完整可见 KV。

2. 用 query / attention score 辅助预测
   通常也不需要修改模型结构。
   需要改的是 vLLM 执行链路：
     在 attention 或 model runner 附近拿到 query / layer id / block table；
     把预测结果传给 LMCache prefetch planner。

3. 真正启用 sparse attention
   这会改变计算语义。
   需要适配 attention metadata、selected block table、kernel、mask、
   prefix cache 和正确性评测；
   对普通 dense 模型可能带来准确率下降，
   必要时才考虑蒸馏、微调或使用原生支持 sparse pattern 的模型。
```

因此推荐阶段划分：

```text
第一阶段：
  dense attention 不变；
  只做 LMCache 层 prefetch plan 和正式 retrieve 兜底；
  目标是证明 SSD wait time / TTFT / TPOT 有收益。

第二阶段：
  引入 query-aware 或 layer-aware predictor；
  predictor 只作为 prefetch hint；
  错误预测不影响输出正确性。

第三阶段：
  如果透明预取收益不足，
  再评估是否改 vLLM attention backend 做真正 sparse attention。
```

### 15.8 已收束：从 CPU early-submit 到 descriptor-plan

上一版最小原型不是最终 GPU-initiated submit，而是 CPU early-submit +
direct placement handle reuse：

```text
connector receive 入口:
  1. 根据 model_input / retrieve_status 构造 retrieve tokens/mask；
  2. 通过 storage_manager.submit_bam_gpu_initiated_prefetch_plan()
     把本轮可能 retrieve 的 prefix chunks 提前规划出来；
  3. BaM store 复用 direct placement 同源 key 规则收集 entries；
  4. CPU 调 submit_chunk_pages_batch_request()；
  5. 把已经提交的 native handle 存入 `_kv_batch_submitted_requests`。

direct placement start:
  1. 再次按正式 direct placement 语义收集同一批 keys；
  2. submit_chunk_pages_kv_fast_path_batch_request(keys)
     先查 `_kv_batch_submitted_requests`；
  3. batch_key 匹配时 pop 出已提交 handle；
  4. 后续 attach / poll / cleanup 继续走 one-copy 主线。
```

代码落点：

```text
vllm/distributed/kv_transfer/kv_connector/lmcache_connector.py
  _maybe_submit_lmcache_bam_gpu_initiated_prefetch_plan()
  LMCacheConnector.recv_kv_caches_and_hidden_states()
  LMCacheConnector._maybe_submit_gpu_initiated_prefetch_plan()

vllm/bam/lmcache_bam_storage.py
  LMCacheBaMStore.submit_kv_fast_path_prefetch_plan()
  LMCacheBaMStore.submit_kv_fast_path_prefetch_keys()
  LMCacheBaMStore.submit_chunk_pages_kv_fast_path_batch_request()
  LMCacheBaMStorageManager.submit_bam_gpu_initiated_prefetch_plan()
```

这个原型的意义：

```text
1. 证明 direct placement 不必现场 submit；
2. 证明上层提前准备的 handle 可以被下层 request 生命周期接管；
3. 给后续“CPU 写 descriptor，GPU service 发起 read”留下接口替换点。
```

这个原型的边界：

```text
1. submit 仍然由 CPU 发起；
2. `_kv_batch_submitted_requests` 存的是已提交 native handle，
   不是 GPU 待消费 descriptor；
3. connector 和 direct placement 仍会各做一次 key / entries 收集；
4. 当前单请求串行 LongBench 测试里，可重叠窗口较小，
   很难产生明显性能收益。
```

因此它应作为生命周期原型保留，但不应被包装成最终创新点。
后续论文/实验叙事应明确写成：

```text
CPU early-submit prototype:
  validates cross-layer request-handle ownership.

GPU-initiated async prefetch target:
  moves fine-grained submit / poll / readiness update to GPU resident service.
```

当前代码已经把这条原型收束为 descriptor-plan 主线：

```text
connector receive 入口:
  1. 根据 model_input / retrieve_status 构造 retrieve tokens/mask；
  2. 调 storage_manager.stage_bam_gpu_initiated_prefetch_plan()；
  3. BaM store 复用 direct placement 同源 key 规则收集 prefix hit entries；
  4. 只调用 prepare_chunk_pages_batch_request() 生成 KV descriptor；
  5. 把 descriptor plan 存入 `_kv_batch_prefetch_plans`。

direct placement start:
  1. 按正式 direct placement 语义收集同一批 keys；
  2. 用 batch_key 取走 `_kv_batch_prefetch_plans` 中的 prepared descriptor；
  3. 当前兼容执行层在统一 read-submit 边界调用
     submit_prepared_chunk_pages_batch_request()；
  4. 后续 attach / poll / cleanup 继续走 one-copy 主线。
```

代码落点更新：

```text
vllm/distributed/kv_transfer/kv_connector/lmcache_connector.py
  _maybe_stage_lmcache_bam_gpu_initiated_prefetch_plan()
  LMCacheConnector.recv_kv_caches_and_hidden_states()
  LMCacheConnector._maybe_stage_gpu_initiated_prefetch_plan()

vllm/bam/lmcache_bam_kv_fast_path.py
  LMCacheBaMKVPreparedBatchRead
  LMCacheBaMKVFastPath.prepare_chunk_pages_batch_request()
  LMCacheBaMKVFastPath.submit_prepared_chunk_pages_batch_request()

vllm/bam/lmcache_bam_storage.py
  LMCacheBaMStore.stage_kv_fast_path_prefetch_plan()
  LMCacheBaMStore.stage_kv_fast_path_prefetch_keys()
  LMCacheBaMStore._take_prepared_kv_prefetch_plan()
  LMCacheBaMStorageManager.stage_bam_gpu_initiated_prefetch_plan()
```

当前边界：

```text
1. connector/storage 阶段不再 CPU submit；
2. prepared descriptor 仍由 CPU/Python 构造；
3. 当前底层 Python native API 尚未暴露 GPU-side descriptor ring，
   所以正式 submit 仍发生在 direct placement start 的兼容执行点；
4. 这一步主要是清理语义和接口，为下一步 GPU persistent service 消费
   descriptor 后发起 SSD read 铺路，不应期待单独产生明显性能收益。
```

### 15.9 下一步：异步 demand-load / GPU-side descriptor 主线

下一步真正要做的是把当前“已提交 handle 表”升级成
“GPU-visible descriptor / readiness 表”。目标链路不是：

```text
CPU submit native read
  -> 保存 handle
  -> direct placement 复用 handle
```

而是：

```text
CPU / LMCache:
  生成 coarse prefetch plan；
  写入 GPU-visible descriptor ring；
  不再 per request 调 native submit。

GPU persistent service:
  从 descriptor ring 取任务；
  查 BaM cache / page metadata；
  miss 时发起 SSD read；
  完成后把 KV 写入 HBM / vLLM paged KV cache；
  更新 readiness / frontier table。

vLLM / LMCache consume:
  查询 ready frontier；
  ready 则直接走 direct placement / attention；
  not ready 则 defer / poll / 正式 retrieve 兜底。
```

这条链路对应用户层直觉：

```text
Attention / pre-attention planner 需要 block
  -> 查 resident / readiness table
  -> hit:
       直接消费 HBM / vLLM paged KV cache 中的 block
  -> miss:
       写入 async load descriptor
       GPU service 发起 SSD read
       KV block 进入 HBM / paged KV cache
       更新 ready frontier
  -> 后续 attention 消费 ready block
```

dense attention 下需要注意：

```text
1. 不能因为 block miss 就跳过该 block；
2. miss block 必须 ready 后才能执行完整 dense attention；
3. 因此第一版应把 readiness gate 放在 attention 前，
   不建议直接改 flash-attn kernel 做同步 SSD wait；
4. 后续若引入 sparse/block attention，才适合让 attention 只消费 selected blocks。
```

建议新增或收束的代码组织：

```text
vllm/bam/lmcache_bam_prefetch.py
  新增：
    BaMPrefetchPlan
    BaMPrefetchDescriptor
    BaMPrefetchState
    BaMPrefetchStats
  职责：
    把 chunk-level plan 翻译成 page-level descriptor；
    记录 prefetch hit / late / wasted / wait time；
    不做策略判断，不直接理解 prompt 语义。

vllm/bam/lmcache_bam_storage.py
  保持总入口和状态 owner：
    接收 connector / LMCache 的 plan；
    维护 chunk metadata / resident state；
    管理 async prefetch request 与 direct placement request 的绑定；
    发布 request frontier。

vllm/bam/lmcache_bam_kv_fast_path.py
  新增异步执行接口：
    enqueue_prefetch_descriptors()
    poll_prefetch_request()
    attach_prefetch_to_direct_placement()
    cancel_or_release_prefetch()

native BaM runtime / persistent service
  新增：
    GPU-visible descriptor ring
    resident / readiness table
    GPU-side descriptor consumer
    completion / frontier update path

vllm/vllm_flash_attn/flash_attn_interface.py
  第一阶段不改；
  attention 仍假设 block_table 指向 ready 的 paged KV cache。
```

阶段划分：

```text
第一阶段：
  dense attention 不变；
  CPU 写 descriptor ring / readiness metadata；
  GPU persistent service 消费 descriptor 并发起 read；
  direct placement / retrieve 通过 ready frontier 兜底。

第二阶段：
  layer-wise prefetch；
  Layer i compute 时，GPU service 预取 Layer i+1 / 后续 chunk pages；
  目标是把 SSD wait 藏进 attention / MLP compute slack。

第三阶段：
  query-aware / block-aware predictor；
  predictor 只作为 prefetch hint；
  错误预测不影响 dense attention 正确性。

第四阶段：
  如果透明预取收益不足，再评估 sparse/block attention；
  这时才考虑修改 attention backend / selected block table / mask 语义。
```

### 15.10 Attention 计算流水线与 GPU-initiated 落点

当前已经跑通的新逻辑应定位为 descriptor-plan 版本，而不是完整的
GPU 发起 submit：

```text
当前已实现：
  LMCache / connector 根据 retrieve 状态提前生成 compact KV descriptor；
  descriptor 绑定 context-chunk readiness frontier；
  descriptor 被 direct placement start 复用；
  正式 native submit 仍发生在统一 read-submit 边界；
  dense attention 仍然只消费已经恢复到 vLLM paged KV cache 的 KV。

当前未实现：
  layer-wise descriptor / layer-ready frontier；
  GPU 常驻服务从 GPU-visible ring 消费 descriptor；
  GPU 在 miss 时真正发起 SSD read；
  attention kernel 内部按 block miss/resident 状态做 demand-load。
```

因此当前 one-copy 主线仍是“attention 前恢复 KV”，而不是
“attention 访问 KV 时发现 miss 再由 GPU 发起 I/O”：

```text
LMCache retrieve / BaM one-copy restore
  -> KV 写入 vLLM paged KV cache
  -> LMCache 返回 ret_mask
  -> vLLM model forward
  -> 每层 QKV projection
  -> attention kernel 按 dense 语义读取 paged KV cache
  -> MLP / 下一层
```

如果要把预取点继续前移，优先不要直接改
`vllm/vllm_flash_attn/flash_attn_interface.py`。第一版更稳的切入点是
vLLM / LMCache 上层的 layer frontier 和 context chunk frontier，让 attention
看到的仍然是 ready KV：

```text
Layer i-1 attention / MLP compute
  与下面动作重叠：
GPU service 根据 descriptor 预取 Layer i 所需 KV pages
  -> 写入 paged KV cache
  -> 更新 layer-ready frontier
Layer i attention
  -> 等待或消费 layer-ready frontier
  -> dense attention 正常执行
```

这条链路的好处是：

```text
1. 不改变模型和 dense attention 正确性；
2. 不要求 flash-attn kernel 在内部等待 SSD I/O；
3. 能把 SSD wait 尽量藏到上一层 compute / MLP / 调度间隙里；
4. readiness frontier 可以继续复用到后续 GPU-side descriptor ring；
5. 如果预取失败或来不及，仍可退回正式 retrieve 等待，不影响输出。
```

可流水线化的几个层级按推荐顺序如下：

```text
1. Layer-wise KV restore prefetch
   最适合当前主线。
   粒度是 layer 或 layer group；
   上一层计算时预取下一层 KV；
   attention 入口只检查 ready frontier。

2. Chunked-prefill / context-chunk prefetch
   适合长上下文 prefill。
   计算 context chunk i 时预取 chunk i+1；
   prefill compute window 较大，更容易覆盖 SSD latency。

3. Decode next-batch / next-token prefetch
   适合 continuous batching。
   对单请求、小 batch 的收益有限；
   对多请求 steady-state 更有价值。

4. Attention tile-level demand-load
   GPU-initiated 语义最强，但改动最大。
   attention tile 访问 block 前检查 resident / ready；
   miss 时写 descriptor 并等待完成；
   需要重做 kernel 调度、partial softmax、mask 和 block table 正确性。

5. Query-aware / sparse-like prefetch
   第一阶段只作为 prefetch hint。
   dense attention 仍消费完整可见 KV；
   只有在决定改变计算语义时，才引入 selected block table / sparse mask。
```

最终主线应收束成四步：

```text
第一步：
  保留当前 descriptor-plan + context-chunk frontier；
  去掉 CPU early-submit handle 叙事；
  明确它只是 GPU descriptor ring 的前置接口。

第二步：
  在当前 context frontier 基础上继续增加 layer frontier；
  vLLM / LMCache 根据下一层或下一 chunk 生成 prefetch plan；
  attention 前只等待 ready frontier，不改 attention kernel。

第三步：
  native runtime 增加 GPU-visible descriptor ring；
  GPU persistent service 负责消费 descriptor、submit read、更新 completion；
  CPU 只负责 coarse plan 和全局调度。

第四步：
  在透明预取收益不足时，再评估 query-aware hint 或真正 sparse attention。
  这一步才涉及 attention metadata、selected block table、mask 和 kernel 语义。
```

当前代码组织上，下一步建议保持分层：

```text
vLLM connector / model runner:
  只提供 request、layer、context chunk 的执行 frontier；
  不直接理解底层 page cache 细节。

LMCache policy / metadata:
  决定哪些 chunk/layer 值得预取；
  负责 prefetch hit、late、waste、wait time 统计。

storage backend:
  把 prefetch plan 翻译成 compact descriptor；
  维护 resident / readiness / ref_count；
  不决定上层调度策略。

attention backend:
  第一阶段不改；
  只要求进入 attention 前相关 KV block 已经 ready。
```

### 15.11 先做 transformer layer 级流水线

如果当前优先目标是验证“前向过程中能否把 SSD/BaM I/O 隐藏到计算里”，
下一步应先做 transformer layer 级流水线，而不是一上来补 block 级 IO。

原因是 transformer forward 的天然执行顺序就是按层推进：

```text
Layer 0 attention
Layer 0 MLP
Layer 1 attention
Layer 1 MLP
...
```

因此最直接的流水线窗口是：

```text
计算 Layer i
  同时预取 Layer i+1 的 KV pages

Layer i+1 attention 前
  只等待 Layer i+1 / layer_group ready
```

这一步不减少 dense attention 的总 IO 量，
但可以验证 I/O wait 是否能被上一层 attention / MLP compute 覆盖。
它回答的是：

```text
同样读完整 prefix KV，
能不能把等待时间藏起来？
```

而 block 级 IO / sparse attention 回答的是：

```text
能不能少读一部分 prefix KV？
```

这两个问题有递进关系，但不应该混在第一版里一起做。

#### 为什么当前布局适合先做 layer 级

当前 BaM page layout 本身已经接近 layer-major / token-page：

```text
page_id =
  page_offset
  + kv_id * num_layers * pages_per_kv_layer
  + layer_id * pages_per_kv_layer
  + token_page_id
```

所以某个 layer 的 K/V page range 可以直接算出来：

```text
K layer i:
  page_offset + i * pages_per_kv_layer
  page_count = pages_per_kv_layer

V layer i:
  page_offset + num_layers * pages_per_kv_layer
              + i * pages_per_kv_layer
  page_count = pages_per_kv_layer
```

如果设置：

```text
layer_group_size = 1
```

就是逐层流水线。
如果设置：

```text
layer_group_size = 2 / 4
```

就是 layer-group 流水线，可以减少 descriptor 数量和 submit/poll 开销。

#### 这一阶段不需要 block 级 IO

先做 layer-wise pipeline 时，需要的是：

```text
chunk_hash + layer_id / layer_group_id -> page range
```

暂时不需要：

```text
chunk_hash + block_id -> page range
```

也就是说，第一版 layer 流水线只补：

```text
1. layer_group descriptor
   chunk_hash
   layer_start
   layer_count
   K page range
   V page range

2. layer frontier
   read_ready_layer_group
   cache_ready_layer_group
   consumable_layer_group

3. layer-wise direct placement
   只把当前 layer_group 的 pages scatter 到 vLLM paged KV cache

4. attention 前 ready gate
   Layer i attention 前确认 Layer i / layer_group 的 KV ready
```

不需要先改：

```text
LMCache chunk key；
token database；
prefix matching；
sparse selected block table；
flash-attn kernel 内部的 block miss handling。
```

#### 推荐的近期实现边界

近期实现应保持：

```text
LMCache:
  仍然按 chunk 做 key / prefix hit / fallback。

vLLM / LMCache adapter:
  在 forward/layer 边界附近生成 layer prefetch hint；
  不直接理解 BaM page cache 细节。

BaM storage / fast path:
  根据 chunk metadata 推导 layer_group page ranges；
  生成 layer_group descriptor；
  维护 layer_group frontier。

attention backend:
  第一阶段不改 kernel；
  只在进入 Layer i attention 前检查 ready frontier。
```

这一阶段的实验重点：

```text
1. layer_group_size = 1 / 2 / 4 的 submit/poll 开销对比；
2. I/O wait 是否能和上一层 compute 重叠；
3. 单请求 LongBench 是否有收益；
4. 多请求 / continuous batching 下是否更容易压出收益；
5. frontier miss 时是否只等待当前 layer_group，而不是等待整个 chunk。
```

#### 与后续 block-packed 的关系

layer-wise pipeline 和 block-packed layout 是递进关系：

```text
第一阶段：
  layer_group 粒度
  目标是 overlap，不减少总 IO

第二阶段：
  block 或 block_group 粒度
  目标是服务 sparse attention / selected block prefetch
  可以减少总 IO
```

所以当前路线应写成：

```text
先做 layer 级流水线，验证前向计算能否覆盖 I/O；
再做 block 级数据组织，服务 sparse-attention 和 selected-block read。
```

### 15.12 第三条路：block-packed physical layout

当前 LMCache 的逻辑 chunk 不应该直接改成 block 语义。
更合理的长期组织方式是：

```text
LMCache 逻辑层：
  chunk_key -> 一个完整 KV chunk

BaM 物理层：
  chunk_key + block_id + layer_group_id
    -> chunk 内部一小段 block-packed pages
```

也就是说：

```text
chunk 外壳不变；
block 内核变细；
dense attention 可以汇总全部 blocks；
sparse attention 可以只取 selected blocks。
```

#### 当前布局的问题

当前 BaM page layout 更接近 layer-major / token-page：

```text
page_id =
  page_offset
  + kv_id * num_layers * pages_per_kv_layer
  + layer_id * pages_per_kv_layer
  + token_page_id
```

语义上类似：

```text
K, layer0, token 0..127
K, layer0, token 128..255
K, layer1, token 0..127
...
V, layerN, token 128..255
```

在 Qwen2.5-7B fp16、hidden_dim=512 的典型设置下：

```text
一个 token vector = 512 * 2B = 1KB
一个 128KB page = 128 tokens
vLLM block size = 16 tokens

所以：
  1 个 BaM page = 8 个 vLLM blocks
```

因此即使 metadata 能算出某个 block 位于哪个 page，
如果只需要 16-token block，也仍然要读它所在的整页 128 tokens。
这会导致 block-level 调度有语义价值，但 IO 省不下来。

直接把 `LMCache chunk_size` 改成一个 vLLM block 也不是好方案：

```text
chunk_size = 16 tokens
pages_per_kv_layer = 1
pages_per_chunk = 2 * num_layers
```

对于 28 层模型，一个 16-token chunk 就需要约 56 个 128KB pages。
原来一个 256-token chunk 约 112 pages；
改成 16 个小 chunk 后会变成：

```text
16 * 56 = 896 pages
```

大量 page 是 padding，metadata / submit / poll / eviction 也都会放大。
所以不应把 LMCache chunk 直接缩成 block。

#### block-packed 布局

第三条路是重排 BaM 内部物理 page，让一个 128KB page 更贴近 attention block：

```text
page = block_id + layer_group_id
```

例如：

```text
block_size = 16 tokens
hidden_dim = 512
dtype = fp16

一个 layer 的 K 或 V block:
  16 * 512 * 2B = 16KB

一个 layer 的 K+V block:
  32KB

一个 128KB page 可以放：
  4 个 layer 的 K+V block
```

于是可以设：

```text
layer_group_size = 4
pages_per_block = ceil(num_layers / layer_group_size)
```

物理顺序可以组织为：

```text
page 0:
  block 0, layers 0..3, K/V

page 1:
  block 0, layers 4..7, K/V

...

page 7:
  block 0, layers 28..31, K/V

page 8:
  block 1, layers 0..3, K/V
```

对于 28 层模型：

```text
pages_per_block = ceil(28 / 4) = 7
一个 256-token chunk = 16 blocks
完整 chunk pages = 16 * 7 = 112 pages
```

所以 dense 读完整 chunk 时，总 page 数不变；
但 sparse / block pipeline 只读部分 block 时，可以真的减少 IO：

```text
selected blocks = 4
read pages = 4 * 7 = 28 pages

full chunk read = 112 pages
```

#### metadata 设计

LMCache key 仍然只对应 chunk：

```text
chunk_hash
```

BaM metadata 增加 block-packed layout 字段：

```text
chunk_hash
  page_offset
  actual_tokens
  slot_num_tokens
  block_size
  num_blocks
  num_layers
  layer_group_size
  pages_per_block
  pages_per_chunk
  layout_kind = block_packed
```

定位公式：

```text
block_id = token_offset // block_size
layer_group_id = layer_id // layer_group_size

page_id =
  page_offset
  + block_id * pages_per_block
  + layer_group_id
```

这允许：

```text
dense:
  selected_blocks = all blocks
  selected_layer_groups = all layer groups

sparse:
  selected_blocks = predictor / sparse attention 需要的 blocks
  selected_layer_groups = 当前 layer 或 layer group
```

#### descriptor 设计

当前 chunk-level descriptor 类似：

```text
read chunk:
  page_offset
  page_count = pages_per_chunk
```

block-packed 后应新增 block-level descriptor：

```text
read block group:
  chunk_hash
  block_start / block_count
  或 selected_block_ids
  layer_group_start / layer_group_count
  page_ids / page_ranges
```

为了避免 tiny random IO，需要在 submit 前做 coalesce：

```text
selected blocks: 0,1,2,3
layer_groups: all

可以合并成连续 page range：
  page_offset + 0 * pages_per_block
  count = 4 * pages_per_block
```

如果 sparse block 分布不连续，则生成多个 range，
但仍应由 runtime / storage backend 负责 range coalescing，
不要让上层 attention 直接操作 page id 细节。

#### encode / decode / scatter 改动

当前 encode 逻辑是：

```text
[2, layers, tokens, hidden]
  -> [2, layers, token_page, page_token_capacity, hidden]
  -> pages
```

block-packed encode 应改成：

```text
[2, layers, tokens, hidden]
  -> pad tokens 到 block_size 对齐
  -> [2, layers, blocks, block_size, hidden]
  -> [blocks, layer_groups, 2, layer_group_size, block_size, hidden]
  -> pages
```

当前 direct placement / scatter kernel 假设读回的是完整 chunk pages，
并按：

```text
kv_id, layer, token
```

从 page buffer 里还原。

block-packed scatter 需要改成：

```text
block_id
layer_group_id
layer_in_group
kv_id
token_in_block
hidden
```

然后写入 vLLM paged KV cache：

```text
slot = slot_mapping[chunk_start + block_id * block_size + token_in_block]
layer = layer_group_id * layer_group_size + layer_in_group
```

这意味着应该新增一个 block-packed direct-placement kernel，
不要把它硬塞进当前完整 chunk scatter kernel。

#### 与 dense / sparse attention 的关系

dense attention:

```text
需要当前 chunk 的全部 blocks；
runtime 可以按 block frontier 逐步读；
attention 前必须保证需要的 block 全 ready；
最终语义等价于完整 chunk restore。
```

sparse attention:

```text
predictor / sparse metadata 选出 selected blocks；
只提交 selected block descriptors；
只等待 selected block frontier；
attention backend 读取 selected block table / sparse mask。
```

因此 block-packed layout 不是为了改变 LMCache 语义，
而是为了让 BaM 的物理 page 内容和后续 sparse/block attention 的消费粒度对齐。

#### 推荐实施顺序

第一步：只加 metadata / 公式，不改 IO

```text
新增 block/page 映射 helper；
给每个 chunk 计算 block_id -> page range；
先用于日志、frontier、调度验证。
```

第二步：page-aligned block-group read

```text
在现有 layout 下按 page_token_capacity 聚合 blocks；
selected block 先映射到 selected pages；
验证 block-aware scheduler / frontier；
接受第一版收益被 128KB page 粒度稀释。
```

第三步：block-packed physical layout 分支

```text
新增 layout_kind = block_packed；
新增 encode_pages_block_packed / decode_pages_block_packed；
新增 block-packed KV read descriptor；
新增 block-packed direct-placement scatter kernel。
```

第四步：接 GPU-side descriptor ring

```text
CPU / policy 写 selected block descriptors；
GPU persistent service 消费 descriptors；
miss 时发起 SSD read；
完成后更新 block frontier / layer frontier。
```

第五步：接 sparse attention

```text
vLLM / attention metadata 携带 selected block table；
attention backend 只消费 selected ready blocks；
错误预测不应影响 correctness，第一版可保留 dense fallback。
```

#### 当前判断

```text
能快速做的：
  block -> page range metadata；
  page-aligned block-group frontier；
  调度和日志验证。

真正有 IO 收益的：
  block-packed physical layout；
  block-level descriptor；
  block-packed scatter；
  GPU-side descriptor consumer。
```

结论：

```text
定位 block 对应 page range 好做；
只读 block 并产生 IO 收益，需要 block-packed physical layout；
不要把 LMCache chunk_size 直接缩成 block；
应保持 chunk 外壳，新增 block-aware BaM runtime 数据面。
```
