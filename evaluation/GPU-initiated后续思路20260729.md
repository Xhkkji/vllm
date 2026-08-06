# GPU-initiated 后续思路

日期：2026-07-29

本文整理当前 `vllm-bam` / `BaM_IOStack` 上 GPU-initiated SSD KV Cache
链路的最新判断、性能瓶颈和后续推进顺序。重点是明确短期不应该继续扩大
GPU-side decode loop 改造，而应先围绕 CUDA Graph、layer 级预取和
IO-compute overlap 做更低风险的验证。

## 1. 当前结论

### 1.1 暂不优先大改 GPU-side decode loop

当前 NVTX / Nsight 结果不支持“CPU scheduler 是单请求 decode 主瓶颈”的判断。

在 LongBench TriviaQA `lt4k` / Qwen2.5-7B-Instruct / `MAX_TOKENS=16`
的 decode-only trace 中：

```text
BaM one-copy 1+4 CTA:
  read request duration: 约 1082.8 ms
  per-token decode span: 约 22 ms/token
  per-token GPU busy: 约 21 ms/token，busy ratio 约 95%+
  token 间 inter-gap: 约 0.9 ms
  vllm_engine_schedule: 16 次合计约 1.4 ms

LMCache GDS:
  read request duration: 约 1090.0 ms
  per-token decode span: 约 22 ms/token
  per-token GPU busy: 约 95%+
  token 间 inter-gap: 约 0.9 ms
  vllm_engine_schedule: 16 次合计约 1.4 ms
```

这说明：

```text
1. token 间确实存在 CPU/框架调度 gap；
2. 但 gap 约 0.9 ms，相比单 token 约 22 ms 的执行时间不是主导项；
3. 真正的 per-token forward/logits/sample 区间内 GPU 已经高度忙碌；
4. 当前单请求 decode 下，CPU scheduler / 多轮 decode 切换不是最优先瓶颈。
```

因此，短期不建议直接把整段 decode loop 下沉到 GPU。这个方向工程风险高，
需要重写 request lifecycle、sampling、stop criteria、streaming、scheduler
状态推进和错误处理，但当前 trace 还没有给出足够收益证据。

### 1.2 CUDA Graph / 取消 enforce-eager 初测

当前日志已经提示 CUDA graph 没启用。decode 阶段每 token 内部有大量 kernel
launch，trace 中单个 decode step 约有数百次 `cudaLaunchKernel` API 调用。

优先做 CUDA Graph / decode bucket 对照的原因：

```text
1. 改动面小于 GPU-side decode loop；
2. 直接针对 decode 阶段的 launch overhead；
3. 和 vLLM 现有优化方向一致；
4. 如果收益明显，可以作为后续 GPU 下沉的强 baseline；
5. 如果收益不明显，可以更明确地把瓶颈定位到模型计算或 sampling。
```

建议先做两组最小对照：

```text
case A:
  当前 enforce-eager / CUDA graph disabled

case B:
  取消 enforce-eager，允许 decode CUDA graph bucket

观察指标：
  per-token decode span
  cudaLaunchKernel API 总耗时
  token 间 inter-gap
  read_avg_s
  avg_request_s
  文本一致性
```

### 1.2.1 多请求 batch 下 CPU 调度测试结论

为验证“多请求 / 多轮对话 serving 场景下，CPU scheduler、Python runtime、
engine step 和 launch gap 是否会在 decode 阶段成为瓶颈”，新增了
`--batch-size` 测试口径：

```text
batch_size=1:
  保持原始逐样本 write/read 语义；

batch_size>1:
  同一批 prompts 先一起 write；
  再一起 read；
  llm.generate(prompts) 一次提交多个请求，真正触发 vLLM batch 调度。
```

当前完成的 GDS 多请求 batch 测试：

```text
dataset:
  LongBench TriviaQA lt4k

model:
  Qwen2.5-7B-Instruct

path:
  LMCache GDS

MAX_TOKENS:
  16

batch_size:
  2 / 4

NVTX:
  LONGBENCH_NVTX_TRACE=1
  VLLM_BAM_NVTX_TRACE=1
```

batch_size=2 的 read request：

```text
read batch elapsed:
  1.2640 s

vllm_engine_schedule:
  16 次合计 1.93 ms
  平均 0.12 ms/step

decode execute_model:
  15 steps
  平均 26.18 ms/step

decode forward:
  平均 15.16 ms/step

decode sample:
  平均 10.08 ms/step

decode step inter-gap:
  平均 0.23 ms

decode GPU busy ratio:
  约 93.1%

cudaLaunchKernel:
  4335 次
  合计 44.1 ms
```

batch_size=4 的 read request：

```text
read batch elapsed:
  1.7099 s

vllm_engine_schedule:
  17 次合计 3.02 ms
  平均 0.18 ms/step

decode execute_model:
  15 steps
  平均 27.60 ms/step

decode forward:
  平均 16.05 ms/step

decode sample:
  平均 10.51 ms/step

decode step inter-gap:
  平均 0.29 ms

decode GPU busy ratio:
  约 93.1%

cudaLaunchKernel:
  4700 次
  合计 50.7 ms
```

对比单请求 decode：

```text
single request:
  token inter-gap 约 0.9 ms
  vllm_engine_schedule 16 次合计约 1.4 ms
  per-token GPU busy ratio 约 95%+

batch_size=2/4:
  decode step inter-gap 降到 0.23 - 0.29 ms
  schedule 合计仍只有 1.93 - 3.02 ms
  decode GPU busy ratio 约 93%
```

因此当前结论是：

```text
多请求 batch 下，CPU scheduler / engine schedule 没有变成 decode 主瓶颈；
相反，token/step 间 gap 被 batch 调度摊薄了。

当前更重的部分仍然是：
  transformer forward
  sampling
  大量 kernel launch
  runtime sync / memcpy
  read/prefill/restore 阶段同步边界
```

这进一步支持短期不从 CPU 调度下沉入手，而是优先看 CUDA Graph / launch
overhead 是否能解释端到端收益不足；如果收益有限，则继续推进 layer 级
KV prefetch / IO-compute overlap。

需要注意的是，BaM one-copy 当前不能直接用于多请求 batch 结论：

```text
BaM one-copy batch_size=2, MAX_TOKENS=8:
  write batch 能正常完成；
  read batch 启动后，第一个 sequence 的 direct placement submit 成功；
  第二个 sequence 的 direct placement submit 后卡住；
  GPU SM 约 99%，无后续 ready 输出。
```

这说明当前 BaM runtime direct-placement 主线还缺多 sequence 并发语义。
要测 “BaM IO 调度 + SSD KV 切换 + 多请求 decode”，需要先处理：

```text
1. 多 sequence descriptor 是否应聚合成一个 native batch request；
2. 是否允许多个 runtime direct-placement request 同时 in-flight；
3. frontier / completion / runtime slot 是否支持 batch 内多个 sequence 并发推进；
4. one-copy direct placement cleanup-only finalize 是否能按 sequence 独立收口。
```

这部分不应和 CPU scheduler 优化混在一起；它是 BaM 多请求 direct-placement
并发能力问题。

### 1.2.2 CUDA Graph / 取消 enforce-eager 测试结论

为验证 decode 阶段是否被 eager kernel launch / runtime sync / framework
overhead 限制，使用 LMCache GDS 路径做了 `ENFORCE_EAGER=false` 对照。
BaM 多请求 batch 当前存在 direct-placement 并发卡住问题，因此本轮先用
GDS 代替 BaM，避免把 BaM runtime 并发问题和 CUDA Graph 问题混在一起。

测试口径：

```text
dataset:
  LongBench TriviaQA lt4k

model:
  Qwen2.5-7B-Instruct

path:
  LMCache GDS

MAX_TOKENS:
  16

batch_size:
  2 / 4

对照:
  ENFORCE_EAGER=true
  ENFORCE_EAGER=false
```

非 Nsight 端到端结果：

| 路径 | batch_size | enforce_eager | read_avg_s | total_avg_s | exact |
| --- | ---: | --- | ---: | ---: | ---: |
| LMCache GDS | 2 | true | 1.0094 | 1.3178 | 4/4 |
| LMCache GDS | 2 | false | 0.9845 | 1.3077 | 4/4 |
| LMCache GDS | 4 | true | 1.5685 | 2.2366 | 4/4 |
| LMCache GDS | 4 | false | 1.5616 | 2.2278 | 4/4 |

端到端收益：

```text
batch_size=2:
  read_avg_s 约提升 2.5%

batch_size=4:
  read_avg_s 约提升 0.4%
```

Nsight 下确认 CUDA Graph 确实生效：

```text
eager batch_size=2:
  cudaLaunchKernel: 4335 次，合计 44.1 ms
  cudaGraphLaunch: 0 次
  decode step inter-gap: 0.23 ms
  read request: 1264 ms

CUDA Graph batch_size=2:
  cudaLaunchKernel: 566 次，合计 23.8 ms
  cudaGraphLaunch: 16 次，合计 1.9 ms
  decode step inter-gap: 0.14 ms
  read request: 1295 ms
```

因此当前判断是：

```text
CUDA Graph 能明显减少 launch API 数量，也能降低 decode step 间 gap；
但端到端 read latency 基本没有明显改善。

说明当前 GDS + LongBench lt4k + Qwen2.5-7B + max_tokens=16 口径下，
eager kernel launch / CPU launch overhead 不是主要端到端瓶颈。
```

CUDA Graph 可以作为小优化保留，但不应作为当前主线。更值得优先推进的是：

```text
1. layer 级 KV prefetch / IO-compute overlap；
2. read / prefill / restore 阶段同步边界拆细；
3. sampling 开销定位；
4. BaM 多 sequence direct-placement 并发语义。
```

### 1.3 GPU-initiated 的重点应转向 layer 级 KV prefetch

当前 BaM one-copy 1+4 CTA 在短 decode / read-heavy 场景能体现 read path 优势：

```text
LongBench TriviaQA lt4k, full dataset, MAX_TOKENS=16, repeat_read=1:
  BaM read_avg_s: 0.5147
  LMCache GDS read_avg_s: 0.5903
  LMCache SSD cold+cgroup CPU path read_avg_s: 0.6952

  BaM vs LMCache GDS read speedup: 1.147x
  BaM vs CPU SSD path read speedup: 1.351x

repeat_read=3:
  BaM read_avg_s: 0.5007
  LMCache GDS read_avg_s: 0.5787
  BaM vs LMCache GDS read speedup: 1.156x
```

但在更长 decode 下，read path 优势会被 transformer 计算稀释：

```text
LongBench TriviaQA, MAX_TOKENS=128, repeat_read=1:
  BaM one-copy 1+4 CTA read_avg_s: 3.0458
  LMCache GDS read_avg_s: 3.0890
  BaM vs GDS read speedup: 1.014x
```

这说明继续只优化“整段 KV read 完成后再 forward”的路径，很难在长 decode
端到端指标上打出明显优势。GPU-initiated 更合理的切入点是：

```text
在 transformer layer 计算过程中，由 GPU 侧提前推进后续 layer / layer-group
的 KV 读取和放置，让 SSD read 被 layer compute 覆盖。
```

也就是把 readiness 粒度从 request/chunk 降到 layer 或 layer-group：

```text
当前粗粒度:
  wait whole prefix chunk ready
  -> forward

目标细粒度:
  compute layer i
  -> GPU prefetch / place layer i+1 或 layer group i+1
  -> attention 前只等待当前 layer_group ready
```

### 1.4 layer-group prefetch 机会测试

为判断 layer 级 / layer-group 级 KV prefetch 是否值得继续推进，补充了单
sequence、不同上下文长度下的 BaM one-copy 与 GDS layer trace 测试。本轮目标不是
实现真实 layer prefetch，而是先回答两个问题：

```text
1. 随着 prefix chunks 增加，KV restore 是否变成足够明显的可优化项；
2. 当前 transformer layer compute 的时间窗口，是否足以覆盖 SSD KV read。
```

测试口径：

```text
dataset:
  LongBench TriviaQA

model:
  Qwen2.5-7B-Instruct

path:
  BaM one-copy 1+4 CTA
  LMCache GDS

batch_size:
  1

repeat_read:
  1

MAX_TOKENS:
  16

debug:
  GIDS_KV_DEBUG=0
  LONGBENCH_DEBUG_LOG=0
```

BaM 单 sequence 端到端结果：

| Bucket | Prompt Tokens | Retrieved Chunks | Retrieved Tokens | Path | Frontier 粒度 | Write Request | Read Request | Avg Request | Output Tokens |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| `lt4k` | 1518 | 4 | 1024 | BaM baseline | none | 1.9829 s | 0.9622 s | 1.4726 s | 16 |
| `lt4k` | 1518 | 4 | 1024 | BaM context-frontier | context_chunk, `layer_group_size=28` | 2.0500 s | 1.0497 s | 1.5498 s | 16 |
| `4k_8k` | 4440 | 16 | 4096 | BaM baseline | none | 2.8482 s | 0.9971 s | 1.9227 s | 16 |
| `4k_8k` | 4440 | 16 | 4096 | BaM context-frontier | context_chunk, `layer_group_size=28` | 2.7521 s | 1.0013 s | 1.8767 s | 16 |
| `8k_12k` | 8235 | 31 | 7936 | BaM baseline | none | 3.7236 s | 0.9136 s | 2.3186 s | 9 |
| `8k_12k` | 8235 | 31 | 7936 | BaM context-frontier | context_chunk, `layer_group_size=28` | 3.8043 s | 0.9207 s | 2.3625 s | 9 |

BaM 内部 KV restore 分解：

| Bucket | Path | Chunks | Retrieved Tokens | Total Bytes | Submit | Poll | Poll Iters | Get | Read / Total | Placement |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lt4k` | BaM baseline | 4 | 1024 | 58,720,256 | 5.404 ms | 16.377 ms | 6 | 1.815 ms | 24.772 ms | 0.000 ms |
| `lt4k` | BaM context-frontier | 4 | 1024 | 58,720,256 | 5.374 ms | 14.998 ms | 5 | 1.834 ms | 22.949 ms | 0.000 ms |
| `4k_8k` | BaM baseline | 16 | 4096 | 234,881,024 | 5.650 ms | 48.735 ms | 17 | 1.278 ms | 56.326 ms | 0.000 ms |
| `4k_8k` | BaM context-frontier | 16 | 4096 | 234,881,024 | 5.757 ms | 50.095 ms | 16 | 2.017 ms | 58.548 ms | 0.000 ms |
| `8k_12k` | BaM baseline | 31 | 7936 | 455,081,984 | 6.000 ms | 94.685 ms | 25 | 1.817 ms | 103.114 ms | 0.000 ms |
| `8k_12k` | BaM context-frontier | 31 | 7936 | 455,081,984 | 5.853 ms | 93.849 ms | 24 | 1.323 ms | 102.518 ms | 0.000 ms |

BaM baseline 与 context-frontier 对照：

| Bucket | Baseline Read Request | Context-Frontier Read Request | Request Delta | Baseline BaM Read | Context-Frontier BaM Read | BaM Read Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `lt4k` | 0.9622 s | 1.0497 s | +9.09% | 24.772 ms | 22.949 ms | -7.36% |
| `4k_8k` | 0.9971 s | 1.0013 s | +0.42% | 56.326 ms | 58.548 ms | +3.94% |
| `8k_12k` | 0.9136 s | 0.9207 s | +0.78% | 103.114 ms | 102.518 ms | -0.58% |

关键 run 目录：

| 测试 | Run Dir |
| --- | --- |
| `lt4k` BaM baseline | `evaluation/lmcache_ssd_read_paths_baseline/logs_longbench_triviaqa/layer_probe_bam_baseline_b1_mt16_s1/20260729_121356` |
| `lt4k` BaM context-frontier | `evaluation/lmcache_ssd_read_paths_baseline/logs_longbench_triviaqa/layer_probe_existing_gpu_initiated_b1_mt16_s1/20260729_121240` |
| `4k_8k` BaM baseline | `evaluation/lmcache_ssd_read_paths_baseline/logs_longbench_triviaqa/layer_probe_bam_4k8k_baseline_b1_mt16_s1/20260729_121951` |
| `4k_8k` BaM context-frontier | `evaluation/lmcache_ssd_read_paths_baseline/logs_longbench_triviaqa/layer_probe_bam_4k8k_context_frontier_b1_mt16_s1/20260729_122054` |
| `8k_12k` BaM baseline | `evaluation/lmcache_ssd_read_paths_baseline/logs_longbench_triviaqa/layer_probe_bam_8k12k_baseline_b1_mt16_s1/20260729_122431` |
| `8k_12k` BaM context-frontier | `evaluation/lmcache_ssd_read_paths_baseline/logs_longbench_triviaqa/layer_probe_bam_8k12k_context_frontier_b1_mt16_s1/20260729_122534` |
| `4k_8k` GDS layer Nsight | `evaluation/lmcache_ssd_read_paths_baseline/logs_longbench_triviaqa/nsys_probe_gds_layer_4k8k_b1_mt16_s1/20260729_122220` |

结论：

```text
1. KV restore 随 retrieved chunks 增加明显变重：
   4 chunks 约 24.8 ms；
   16 chunks 约 56.3 ms；
   31 chunks 约 103.1 ms。

2. 当前 VLLM_BAM_GPU_INITIATED_PREFETCH=1 分支仍是 context-chunk frontier：
   granularity=context_chunk；
   layer_group_size=28；
   也就是 28 层一起 ready，不是真正 layer 级 prefetch。

3. context-frontier 分支没有带来端到端收益：
   4k_8k: 0.9971 s -> 1.0013 s；
   8k_12k: 0.9136 s -> 0.9207 s。

4. 因此继续在 request/chunk 粗粒度上包装 frontier 意义有限；
   真正需要拆的是 layer / layer-group readiness 边界。
```

GDS layer trace 用于估算模型逐层计算窗口。由于 BaM Nsight wrapper 当前不能透传
`MANIFEST_PATH` / `MAX_MODEL_LEN` 等变量，`4k_8k` 的 layer trace 暂用 GDS
路径代替；layer compute 主要由模型和 token 数决定，因此可用于判断 overlap
空间。

Nsight layer trace 结果：

| Path | Bucket | Prompt Tokens | Read Request | `recv_kv` Count | `recv_kv` Sum | Prefill Forward | Prefill Layer Sum | Decode Steps | Decode Forward Avg | Decode Layer Avg | Decode Layer Sum | Decode Sample Sum |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BaM one-copy | `lt4k` | 1518 | 1060.437 ms | 5 | 49.876 ms | 600.351 ms | 599.810 ms | 15 | 15.277 ms | 0.529 ms | 222.065 ms | 102.353 ms |
| LMCache GDS | `4k_8k` | 4440 | 1205.841 ms | 1 | 195.202 ms | 601.560 ms | 601.001 ms | 15 | 15.323 ms | 0.530 ms | 222.593 ms | 102.937 ms |

Nsight 报告文件：

| Trace | File |
| --- | --- |
| BaM one-copy `lt4k` | `evaluation/lmcache_ssd_read_paths_baseline/nsys_decode_probe/bam_layer_probe_b1_mt16_s1.nsys-rep` |
| BaM one-copy `lt4k` SQLite | `evaluation/lmcache_ssd_read_paths_baseline/nsys_decode_probe/bam_layer_probe_b1_mt16_s1.sqlite` |
| GDS `4k_8k` | `evaluation/lmcache_ssd_read_paths_baseline/nsys_decode_probe/gds_layer_probe_4k8k_b1_mt16_s1.nsys-rep` |
| GDS `4k_8k` SQLite | `evaluation/lmcache_ssd_read_paths_baseline/nsys_decode_probe/gds_layer_probe_4k8k_b1_mt16_s1.sqlite` |

结合 BaM 的同压力结果：

```text
BaM 4k_8k:
  16 chunks / 4096 retrieved tokens
  read_ms 约 56.3 ms

GDS 4k_8k:
  vllm_recv_kv 约 195.2 ms
```

这说明 BaM 相比 GDS 在纯 KV restore / SSD->GPU 数据路径上已经有明显优势：

```text
GDS recv_kv / BaM read_ms:
  195.2 / 56.3 = 约 3.47x
```

但端到端 read request 没有同比例放大收益，原因是 read request 还包含：

```text
prefill forward
decode forward
sampling
engine step
logits / output processing
runtime 同步边界
```

因此 BaM 对 GDS 的优势要分层看：

```text
read path / KV restore:
  BaM 有明确优势，尤其在 retrieved chunks 增加时更明显。

end-to-end request:
  优势容易被 transformer compute 和 sampling 稀释。

当前 context-frontier:
  只是 request/context chunk 级 ready 观测；
  不会自动带来 layer overlap，所以端到端收益不明显。
```

对 layer-group prefetch 的直接判断：

```text
1. 单层 decode compute 约 0.5 ms，太短，无法覆盖一次几十到上百 ms 的 KV read；
2. 单纯“layer i 计算时预取 layer i+1”粒度过细，收益不够；
3. 更合理的是 layer-group prefetch，例如 4 层或 7 层一组；
4. 对 8k_12k 的 31 chunks / 103 ms restore，如果能把 restore 拆成多个
   layer-group 波次，并和 prefill/decode layer compute 重叠，才可能体现
   GPU-initiated 的范式优势。
```

建议下一步实现的最小可验证形态：

```text
layer_group_size:
  4 或 7

descriptor:
  chunk_hash
  layer_start
  layer_count
  page_offset
  page_count
  target KV cache metadata

frontier:
  read_ready_layer_group
  cache_ready_layer_group
  consumable_layer_group

attention 前:
  GPU-side wait/fence 当前 layer_group ready

CPU:
  仍只推进 vLLM forward；
  不参与 SSD CQ poll 和 KV placement。
```

## 2. GPU-initiated 相比传统 GDS 的潜在瓶颈

GPU-initiated 并不天然比传统 GDS 快。它的优势来自减少 CPU 介入、减少
host bounce、以及让 IO 和计算更细粒度重叠。当前没有在所有场景明显拉开
差距，主要可能受以下瓶颈影响。

### 2.1 IO 与计算没有充分重叠

如果当前链路仍是：

```text
SSD read / KV restore 完成
  -> 再进入 transformer forward
```

那么 GPU-initiated 只能减少部分读取和放置路径开销，无法覆盖 read latency。
真正能体现范式优势的是：

```text
SSD read / KV placement
  与
transformer layer compute
  同时推进
```

### 2.2 粒度仍偏 request/chunk

当前 direct placement 的上层语义主要围绕 prefix chunk ready。对 attention
来说，真正需要的是当前 layer 或 layer-group 的 KV ready，而不是整个 chunk
全部 ready。

如果等待边界太粗，GPU-initiated 的异步能力会被同步边界抵消。

### 2.3 BaM 常驻 worker 会占用 GPU 资源

当前 1+4 CTA 模式保留一个轮询/服务侧执行单元，并通过 4 个 mover CTA 负责
数据搬运和 placement。这个模式能减少 CPU 参与，但常驻 CTA 会消耗 GPU 资源。

当 decode kernel 本身已经接近 95% busy 时，常驻 worker 可能与 attention /
sampling kernel 竞争 SM、warp、寄存器或 memory bandwidth。短 decode /
read-heavy 场景下收益更容易体现；长 decode / compute-heavy 场景下收益会被
计算主导项稀释。

### 2.4 控制面状态机不是免费开销

GPU-side request table、frontier table、completion table、metadata-ready
flag、memory fence、polling loop 都会带来控制面开销。

当每次 IO 请求较小、并发不足或复用次数不高时，这些开销可能抵消
SSD->GPU direct path 的收益。

### 2.5 数据放置仍可能是 scatter 型访存

即使 SSD 到 GPU 是 direct，最终也要写入 vLLM paged KV cache 的 block layout。
如果 placement 是大量非连续 scatter，或者需要额外 reshape / layout 转换，
那么瓶颈会从 IO path 转移到 GPU memory movement / placement kernel。

### 2.6 传统 GDS / LMCache baseline 本身已经较强

LMCache GDS 路径已经避免了传统 SSD->CPU->GPU 的一部分开销；如果测试中
还有 page cache、文件系统缓存、repeat read 或 warm cache 影响，传统路径
也会被抬高。

因此对比 CPU path 时需要使用 cold read / drop caches / cgroup 限制；对比
GDS path 时，应重点观察 read-heavy 和 overlap 场景，而不是只看长 decode
端到端平均延迟。

### 2.7 sampling / launch overhead 仍会压住端到端收益

当前 trace 中 sampling 和 kernel launch 数量都不可忽略。即使 KV read path
更快，端到端仍会被以下项限制：

```text
sampling cost
cudaLaunchKernel overhead
vLLM eager execution overhead
Python / engine step overhead
attention / MLP compute
```

所以 GPU-initiated 的后续收益评估不能只看 read_avg，也要区分：

```text
read path latency
decode token latency
launch overhead
sampling overhead
overall request latency
```

## 3. 当前代码的最新链路

### 3.1 vLLM / LMCache 接入层

当前仍基于 vLLM V0 connector 和 LMCache storage manager 扩展：

```text
vLLM LLMEngine
  -> schedule()
  -> execute_model()
  -> ModelRunner.execute_model()
  -> recv_kv_caches_and_hidden_states()
  -> LMCacheConnector
  -> LMCache engine.retrieve()
  -> LMCacheBaMStorageManager
```

其中 `LMCacheConnector` 做了两件关键事：

```text
1. 在 storage manager 外层包一层 LMCacheBaMStorageManager；
2. 在 direct placement defer runtime 开启时，把 retrieve 拆成
   start / poll / finalize 三段，允许跨 engine step 持有 in-flight request。
```

当前新增的 NVTX 标记默认关闭：

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

VLLM_BAM_LAYER_NVTX_TRACE=1
  标记 qwen2_layer:<layer_idx>:tokens=<token_count>
  用于分析逐层计算窗口和 layer-group prefetch overlap 空间
```

默认不开时，不影响普通性能测试和正确性测试。

### 3.2 write 路径

write request 仍然走 vLLM 正常 prefill，并通过 LMCache 生成 chunk：

```text
request_1 write:
  vLLM prefill
  -> LMCache chunk 生成
  -> LMCache 原始 storage 写入
  -> BaM shadow store 写入 SSD-backed BaM store
```

这条路径的作用是给后续 read request 提供可命中的 prefix chunk。

### 3.3 read / prefix reuse 路径

read request 依赖 LMCache 的 prefix hit 语义：

```text
request_2 read:
  LMCache should_retrieve 判断 prefix hit
  -> direct placement start 边界构造本轮真正要消费的 chunk descriptor
  -> BaM native batch request
  -> GPU worker submit
  -> GPU persistent service 推进 IO / frontier / placement
  -> runtime direct placement cleanup-only finalize
  -> vLLM paged KV cache 中 prefix KV 可被 attention 消费
  -> vLLM 继续执行 query 部分 forward / decode
```

当前 GPU-initiated direct-placement 主线不再复用 LMCache `engine.prefetch()`
提前 CPU submit。原因是 direct placement 的 descriptor/frontier 必须根据
真实 prefix hit 和当前 slot/block metadata 在 start 边界现场生成，否则容易
提交不会被 retrieve 消费的 pending key。

### 3.4 BaM_IOStack 执行层

当前 BaM 侧核心执行层是 `BaMGPUWorkerKVExecutor`：

```text
BaMGPUWorkerKVExecutor.submit()
  -> 构造 request_table / completion / frontier ABI
  -> row_store.kv_worker_submit()
  -> persistent service 接管 IO 请求

BaMGPUWorkerKVExecutor.poll()
  -> CPU 只检查 request/runtime 是否 ready
  -> GPU persistent service 自己推进 CQ / completion / frontier
  -> one-copy 路径等待 CONSUMED 语义

BaMGPUWorkerKVExecutor.consume()/finalize
  -> one-copy runtime direct placement 下只做 cleanup-only 生命周期收口
```

当前重要语义：

```text
1. CPU 仍负责 vLLM engine step、调度和发起计算；
2. persistent 模式下，CPU poll 的目标是观察 ready，不应负责推进 CQ 和数据搬运；
3. SSD read、completion 轮询、frontier 推进和 KV placement 由 GPU service 负责；
4. one-copy 路径的数据最终直接写入 vLLM paged KV cache；
5. host finalize 主要负责生命周期收口，而不是重新 materialize / copy KV。
```

### 3.5 1+4 CTA 当前定位

当前默认实验主线是 BaM one-copy 1+4 CTA：

```text
GIDS_KV_GPU_WORKER_MOVER_CTAS=4
```

语义上可以理解为：

```text
1 个服务/轮询侧执行单元:
  负责观察/推进 request、CQ、completion、frontier 等状态；

4 个 mover CTA:
  负责并行执行数据搬运和 direct placement；

CPU:
  不参与 GPU CQ 轮询和数据搬运；
  只在 vLLM forward 前观察 request 是否 ready，并继续发起计算。
```

这条链路已经在关闭 debug 的情况下完成过全量 `lt4k` r1/r3 正确性验证：

```text
repeat_read=1:
  write/read exact match: 25/25

repeat_read=3:
  write/read exact match: 75/75
```

### 3.6 当前 frontier 修正

最新修正是在 `gpu_worker_submit` 阶段允许 frontier 状态合法超前：

```text
原问题:
  submit 返回后校验 expected=SUBMITTED；
  但 persistent GPU worker 可能已经把 frontier 推进到 IO_DONE / CONSUMED；
  Python 层误判为 status mismatch。

当前修正:
  只在 gpu_worker_submit 阶段允许 SUBMITTED -> IO_DONE / CONSUMED；
  仍然拒绝 ERROR 或未知状态；
  不改变 CUDA / BaM 数据面。
```

该修正后，`MAX_TOKENS=128` 压力测试能够跑通：

```text
NUM_SAMPLES=5
REPEAT_READ=1
MAX_TOKENS=128
GIDS_KV_DEBUG=0
GIDS_KV_GPU_WORKER_MOVER_CTAS=4

requests=10
write_avg_s=3.5267
read_avg_s=3.0458
avg_request_s=3.2863
```

## 4. 后续推进优先级

### 4.1 第一优先级：layer-group KV prefetch

CUDA Graph 和多请求 batch 调度测试都说明，当前端到端收益不足不能主要归因于
CPU scheduler 或 eager launch overhead。因此下一步主线应转向
layer-wise / layer-group-wise 的 KV prefetch 与 IO-compute overlap。

目标不是先做完整 GPU-side decode loop，而是在现有 CPU-driven forward 框架内增加
layer 级或 layer-group 级 readiness：

```text
layer_group descriptor:
  chunk_hash
  layer_start
  layer_count
  page_offset
  page_count
  target KV cache metadata

layer frontier:
  read_ready_layer_group
  cache_ready_layer_group
  consumable_layer_group

attention consume:
  Layer i attention 前只等待 Layer i / layer_group ready
```

这样可以先验证：

```text
SSD read 是否能被 transformer layer compute 覆盖；
是否必须等待整个 chunk ready；
layer_group_size=1/2/4 时 descriptor 数量与 overlap 收益如何权衡。
```

### 4.2 第二优先级：BaM 多 sequence direct-placement 并发

当前 GDS batch=2/4 可以正常跑通，但 BaM one-copy batch=2 会在 read 阶段
第二个 sequence 的 direct placement submit 后卡住。这说明 BaM runtime
direct-placement 还缺多 sequence 并发语义。

后续如果要把 GPU-initiated 用到真实 serving batch，就需要先处理：

```text
1. 多 sequence descriptor 是否应聚合成一个 native batch request；
2. 是否允许多个 runtime direct-placement request 同时 in-flight；
3. frontier / completion / runtime slot 是否支持 batch 内多个 sequence 并发推进；
4. one-copy direct placement cleanup-only finalize 是否能按 sequence 独立收口。
```

这部分是 BaM 数据面和 runtime 状态机能力，不是 CPU scheduler 优化问题。

### 4.3 第三优先级：sampling / launch overhead

当前 trace 中 sampling 开销不可忽略。CUDA Graph 已经把 eager launch API
数量明显降下来，但端到端 read latency 改善很小，因此需要继续拆分 sampling
和 logits 的真实开销。

建议单独对比：

```text
greedy / deterministic sampling
top-k / top-p sampling
temperature=0
不同 max_tokens
不同 batch size
```

目的是判断端到端收益被 sampling 压住的程度。

### 4.4 第四优先级：CUDA Graph 作为小优化保留

CUDA Graph 已经确认可以显著减少 launch API 数量，但当前 batch=2/4 的
端到端收益只有约 0.4% - 2.5%。因此它可以作为工程小优化和 baseline
对照保留，但不应作为当前 GPU-initiated 主线。

### 4.5 可低风险借鉴 Tutti 的优化点

Tutti 的完整方案包括 GPU-native KV object store、GPU io_uring、SGL、
SM partition 和 slack-aware scheduler。当前 BaM/vLLM 代码不需要直接重构成
Tutti 形态，短期只借鉴其中能局部落地的调度思想。

#### 4.5.1 Read restore 优先于 write shadow

Tutti 的一个重要启发是：read 位于请求关键路径，write 多数情况下可以延后。
当前 BaM 链路中，LMCache shadow write 与 prefix read restore 都会使用 SSD /
BaM runtime 资源。如果 read/write 同时竞争队列、SSD 带宽或 GPU worker，
TTFT 更容易被 read 阶段拉高。

可先做最小实现：

```text
if read_restore_queue 非空:
  暂停或限速 write_shadow_queue
else:
  继续后台 shadow write
```

这个优化不改变 KV layout，也不需要 layer 级接口，主要改 storage manager /
worker queue 的 admission policy。它适合先用于确认：

```text
read-heavy / prefix-reuse 压力下，write shadow 是否干扰 BaM read restore。
```

#### 4.5.2 Prefill-window prefetch

Tutti 的 slack-aware scheduler 比较复杂，但当前可以先实现规则化版本：
当 vLLM 当前 engine step 有较长 prefill 时，把后续即将 decode 的请求
KV restore 提前提交给 BaM。

简化策略：

```text
if current_step_has_prefill and prefill_tokens >= threshold:
  submit_prefetch(next_decode_candidates)
```

候选阈值：

```text
prefill_tokens >= 1024 或 2048
```

原因是前面 Nsight 数据显示 prefill forward 通常是数百毫秒量级，而 BaM
KV restore 是几十到一百毫秒量级。多请求场景下，prefill 是更大的
IO-compute overlap 窗口，比单请求单层 decode 更容易隐藏 SSD->GPU restore。

#### 4.5.3 Layer-group readiness，而不是完整 object store

Tutti 的 layer-wise retrieve/store 接口工程量较大。当前更现实的做法是：
保留现有 LMCache chunk / BaM direct placement 主线，只把 readiness 从
request/chunk 粗粒度拆到 layer-group 粒度。

最小目标：

```text
layer_group_size = 4 或 7
request_id, layer_group_id -> ready_chunks / ready_tokens
attention 前只等待当前 layer_group ready
```

这一步比完整 GPU-native object store 小得多，但可以验证 Tutti 最核心的
IO-compute overlap 假设：

```text
整段 KV restore 是否能被拆成多个 layer-group restore；
后续 group 的 restore 是否能被当前 group / 当前请求 prefill 计算隐藏；
group_size=4/7 是否比单层 prefetch 有更好的 IO 粒度与同步开销平衡。
```

#### 4.5.4 暂不建议直接搬的 Tutti 组件

| Tutti 组件 | 暂不直接实现的原因 |
| --- | --- |
| GPU io_uring | 会重构 BaM I/O runtime 和 SQ/CQ API，风险过大 |
| SGL command path | 需要改底层 NVMe command / DMA 描述路径 |
| GPU-native object store | 会重构 LMCache chunk 到 KV object 的映射 |
| SM partition / green context | 工程复杂，先用固定 CTA topology 验证即可 |
| 完整 slack-aware scheduler | 需要先有 layer-group readiness，否则没有调度对象 |

当前建议的最小推进顺序：

```text
1. read restore 优先级，避免 write shadow 干扰关键路径；
2. prefill-window prefetch，先利用多请求 prefill 的大计算窗口；
3. layer-group readiness，将整段 restore 拆成 4/7 层一组；
4. attention 前只等待当前 layer group ready，再评估是否需要更完整的
   Tutti-like scheduler。
```

## 5. 当前分支已经完成的工作

这一分支现在不只是“想法”，而是已经把几条关键链路补到能做对照实验的状态。

### 5.1 vLLM / LMCache / BaM 路径区分已经补齐

当前已经明确区分了三类路径：

```text
1. vLLM 原生 swap：
   GPU KV <-> CPU swap space

2. LMCache SSD 路径：
   chunk 级 prefix / decode cache 读写

3. BaM / MDS 路径：
   以 SSD 为后端的同步或半同步 KV restore / store
```

这件事的意义是：后续不再把“decode 慢”笼统归因，而是能分清到底是
vLLM 原生 preemption、LMCache chunk I/O，还是 BaM/MDS runtime 在起作用。

### 5.2 已加的实验开关

已经加了一个默认关闭的实验开关，用来只在需要时允许 LMCache 在
decode-only batch 上发送 KV：

```text
VLLM_LMCACHE_SEND_DECODE_KV
```

它默认关闭，和原生 vLLM 行为保持一致；实验时再显式打开，用于验证
decode cache 写盘 / 读盘路径是否真的会被触发。

### 5.3 已补的统计口径

当前已经把 LMCache 的 retrieve / disk_read / disk_write 结果按 phase 拆开，
至少区分成：

```text
prefill
mixed
decode
```

这样后续如果真的打到 decode cache I/O，就不会再把它误归到 prefill 里。

### 5.4 已跑出的代表性结果

这条分支已经积累了几组能直接支撑判断的结果：

```text
1. 连续到达 / 在线服务曲线
   能看到 TTFT p95 和 TPOT p95 的变化，以及 read-heavy 压力下的尾延迟放大。

2. 代表性的 I/O 压力表
   能拆出 prefill compute / prefill read / decode compute / decode read / decode write。

3. 原生 vLLM swap 相关 trace
   说明当前默认 swap 还是 CPU tier，不是 SSD tier。

4. LMCache decode-send 探针
   已验证 decode-only batch 侧的发送开关确实能把路径打通，但 decode I/O 是否出现，
   仍然受 chunk 边界、cache 命中和调度形态影响。
```

### 5.5 当前已经能确定的一点

现在可以明确一点：

```text
vLLM scheduler 负责“什么时候让某个 seq_group 跑”
LMCache / BaM 负责“这一轮 KV 到底怎么搬、搬到哪一层、何时落盘”
```

所以后续如果要做更强的 SSD-aware 调度，不能只停在 scheduler 层；但
scheduler 层已经足够承载第一版 policy 实验。

## 6. 后续延伸路线

### 6.1 第一层：vLLM scheduler-aware policy

这是最容易先落地的一层。目标不是复刻 SolidAttention，而是先在 serving
层把“swap 太贵”和“waiting 太久”这对矛盾显式化。

可做的策略包括：

```text
1. preemption 优先级改成 SLO aware；
2. 根据 seq 长度 / 剩余 token / KV block 数估计 swap 成本；
3. 在 waiting / swapped / running 之间做更主动的 admission control；
4. 用更合理的 chunked prefill 配比减少后续请求饥饿。
```

这一层适合验证一个核心命题：

```text
如果 swap 成本下降，scheduler 是否会更愿意把请求切换成 swap，
从而减少长期 waiting 和尾延迟？
```

### 6.2 第二层：layer-group prefetch

如果只是 request-level 调度，仍然会卡在“整段 KV restore 完成后再 forward”。
更自然的下一步是把 readiness 从 request/chunk 往下拆到 layer-group。

目标形态是：

```text
当前 layer 计算时
  -> 提前发起后续 layer-group 的 KV restore
  -> 让 SSD I/O 和 GPU compute overlap
```

这层和 SolidAttention 的思路最接近，但不需要一次性把 attention kernel 全拆掉。

### 6.3 第三层：attention-inner microtask 调度

如果要继续往 SolidAttention 靠，就不能只在 ModelRunner 外围做文章，而要往
attention 执行内部走。

可以逐步演进成：

```text
SSD KV block ready
  -> attention 先拿到已就绪的 block
  -> missing block 再补读
  -> 新 KV 再写回 SSD
```

这一步的本质是把“store / retrieve / compute”从串行变成可重叠的 microtask
流，而不是传统的整请求 blocking restore。

### 6.4 第四层：BaM / MDS 作为底层 I/O substrate

如果目标是把上述调度真正做起来，BaM / MDS 比 LMCache chunk 路径更像一个
可持续演进的底座，因为它更接近 block 级、layer 级和异步 I/O 的控制面。

更具体地说：

```text
LMCache 负责证明 SSD backend 的存在感；
BaM / MDS 负责把 SSD I/O 变成可调度、可 overlap 的底层能力；
vLLM scheduler 负责把 serving 侧的压力、SLO 和公平性纳入决策。
```

### 6.5 一个比较稳的推进顺序

```text
1. 先把 vLLM scheduler 层的策略做出来，验证是否能缓解 waiting / TTFT 尾延迟；
2. 再把 layer-group prefetch 接进去，验证能否把 SSD I/O 藏到 compute 后面；
3. 再评估是否要继续往 attention-inner microtask 调度推进；
4. 最后再考虑把更复杂的 SSD-aware scheduler 收束成完整系统。
```

## 7. 阶段性判断

当前最稳妥的路线是：

```text
保留 BaM one-copy 1+4 CTA 作为 GPU-initiated read path 主线；
保留 LMCache GDS 和 CPU cold+cgroup 作为 baseline；
短期优先推进 layer-wise KV prefetch / IO-compute overlap；
同时补齐 BaM 多 sequence direct-placement 并发语义；
CUDA Graph 作为小优化和对照保留；
暂不重写完整 GPU-side decode loop。
```

原因：

```text
1. 当前 BaM read path 在 read-heavy 场景已经能超过 LMCache GDS 和 CPU SSD path；
2. 在 4k_8k 单 sequence trace 中，GDS recv_kv 约 195 ms，而 BaM read_ms
   约 56 ms，说明 BaM 在纯 KV restore / SSD->GPU 数据路径上有明确优势；
3. 但长 decode 下端到端优势会被 transformer compute / sampling 稀释；
4. 当前 context-frontier 仍是 request/chunk 粗粒度，不会自动产生 layer overlap；
5. 单请求和多请求 batch trace 都不支持 CPU scheduler 是主瓶颈；
6. GPU-initiated 的真正差异化价值在细粒度 IO-compute overlap；
7. CUDA Graph 能减少 launch API，但端到端收益有限；
8. 完整 GPU-side decode loop 工程风险最大，应等 layer-wise overlap 和
   BaM 多 sequence direct-placement 并发能力验证后再决定是否推进。
```
