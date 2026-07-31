# 20260728 Baseline Results

<style>
body,
.markdown-body {
  background: #ffffff !important;
  color: #111827 !important;
}

table,
thead,
tbody,
tr,
th,
td {
  background: #ffffff !important;
  color: #111827 !important;
}

table {
  border-collapse: collapse !important;
  border: 1px solid #1f2937 !important;
}

th,
td {
  border: 1px solid #1f2937 !important;
  padding: 6px 10px !important;
}

thead th {
  border-bottom: 2px solid #111827 !important;
  font-weight: 700 !important;
}

code,
pre {
  background: #f6f8fa !important;
  color: #111827 !important;
}
</style>

本文档整理 2026-07-28 在 `vllm-bam` 上完成的 LongBench-TriviaQA / Qwen2.5-7B-Instruct SSD-backed KV read baseline。测试目标是用更高压力的全量 `lt4k` 数据集，对比 BaM GPU-initiated one-copy 路径、LMCache SSD cold+cgroup 路径和 LMCache GDS 路径。

## 测试环境与公共配置

| 项目 | 配置 |
| --- | --- |
| Repo | `/home/xhk/llm-inference/vllm-bam` |
| 数据集 manifest | `/home/xhk/llm-inference/datasets/longbench/organized/triviaqa/qwen25/full/buckets/lt4k.jsonl` |
| 数据集 | LongBench TriviaQA |
| 模型 | Qwen2.5-7B-Instruct |
| bucket | `lt4k` |
| prompt mode | `full` |
| max_model_len | 4096 |
| max_tokens | 16 |
| debug | `GIDS_KV_DEBUG=0`, `LONGBENCH_DEBUG_LOG=0` |
| 主要指标 | `write_avg_s`, `read_avg_s`, `avg_request_s`, `read_p50_s`, `read_min_s`, `read_max_s`, write/read exact match, answer hit |

## PPT 展示摘要

### 数据集与测试口径

- 数据集：LongBench TriviaQA，`lt4k` bucket，全量 25 条样本。
- 模型：Qwen2.5-7B-Instruct，`max_model_len=4096`，`max_tokens=16`。
- Prompt 长度：`qwen_prompt_tokens` 平均 2904.72，p50 3028，范围 1518-3771。
- `repeat_read` 含义：每条样本先执行 1 次 write 写入 KV，再执行 N 次 read 从外部存储读回 KV；`r1` 即 `repeat_read=1`，`r3` 即 `repeat_read=3`。
- `r1` 口径：每条样本 1 write + 1 read，共 25 write + 25 read = 50 requests；三条路径 BaM / LMCache GDS / LMCache SSD cold+cgroup CPU path 都有同口径数据。
- `r3` 口径：每条样本 1 write + 3 read，共 25 write + 75 read = 100 requests；更强调 Prefix Reuse / KV read path 压力，但当前只跑了 BaM vs LMCache GDS。
- PPT 建议：主表选 `r1` 三路径对比，因为它同时包含 CPU SSD 路径、LMCache GDS 和 BaM，口径最完整；如果重点讲 read/reuse 压力，可以在结论里补充 `r3` 下 BaM 相对 LMCache GDS 的 1.156x read speedup。

### 代表性性能结果

推荐用于 PPT 主展示：`repeat_read=1` 全量三路径对比。

| 路径 | 数据通路 | requests | read_avg_s | avg_request_s | read_p50_s | exact match | 相对结论 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| BaM GPU-initiated one-copy 1+4 CTA | SSD -> GPU direct placement | 50 | 0.5147 | 0.7893 | 0.4974 | 25/25 | vs CPU path: 1.351x read speedup；vs LMCache GDS: 1.147x read speedup |
| LMCache GDS | SSD -> GPU, CPU submit/sync | 50 | 0.5903 | 0.8168 | 0.5784 | 25/25 | GDS baseline |
| LMCache SSD cold+cgroup CPU path | SSD -> CPU -> GPU | 50 | 0.6952 | 0.8280 | 0.6872 | 25/25 | CPU SSD baseline，read 前 drop caches，16GB cgroup |

结论：在三路径同口径 `repeat_read=1` 全量测试下，BaM one-copy 1+4 CTA 相比 CPU SSD cold+cgroup 路径 read 延迟降低 26.0%，相比 LMCache GDS read 延迟降低 12.8%，并保持 25/25 write/read 逐字一致。补充结果：在更强调 read/reuse 压力的 `repeat_read=3` 下，BaM 相比 LMCache GDS 达到 1.156x read speedup，端到端平均延迟降低 7.1%。

## 两页 PPT 建议内容

### Page 1：BaM GPU-initiated 路径相比传统 GDS 的优势

这一页建议强调“BaM 的优势首先体现在 KV restore / SSD->GPU 数据路径，而不是所有端到端 request 指标都会同比例提升”。主表使用全量 `lt4k` 的 `repeat_read=1` 三路径对比，口径最完整；补充表使用 `repeat_read=3` 展示 read/reuse 压力变大后 BaM 相比 GDS 的优势仍然稳定。

#### 主表：全量 `lt4k` 三路径端到端对比

| 路径 | 数据通路 | CPU 参与 | requests | read_avg_s | avg_request_s | read_p50_s | exact match | Read Speedup vs BaM |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BaM one-copy 1+4 CTA | SSD -> GPU direct placement -> vLLM paged KV | CPU 提交/推进计算；GPU 轮询与搬运 | 50 | 0.5147 | 0.7893 | 0.4974 | 25/25 | 1.000x |
| LMCache GDS | SSD -> GPU buffer -> LMCache/vLLM restore | CPU 仍参与提交、runtime 调度与同步边界 | 50 | 0.5903 | 0.8168 | 0.5784 | 25/25 | 1.147x |
| LMCache SSD cold+cgroup CPU path | SSD -> CPU memory -> GPU | CPU read path + Host 侧中转 | 50 | 0.6952 | 0.8280 | 0.6872 | 25/25 | 1.351x |

#### 补充表：更强调 Prefix Reuse 的 `repeat_read=3`

| 路径 | 数据通路 | requests | reads | read_avg_s | avg_request_s | read_p50_s | exact match | Read Speedup vs BaM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BaM one-copy 1+4 CTA | SSD -> GPU direct placement | 100 | 75 | 0.5007 | 0.6449 | 0.4942 | 75/75 | 1.000x |
| LMCache GDS | SSD -> GPU, CPU submit/sync | 100 | 75 | 0.5787 | 0.6945 | 0.5795 | 75/75 | 1.156x |

#### 数据路径层面的补充证据

`4k_8k` 单 sequence / 4440 prompt tokens / 16 retrieved chunks 的 Nsight 与 BaM 日志显示，BaM 在纯 KV restore 数据路径上优势更明显：

| 对比项 | BaM one-copy | LMCache GDS | 相对结论 |
| --- | ---: | ---: | --- |
| KV restore / read path | 56.326 ms | 195.202 ms | GDS / BaM = 3.47x |
| read request | 0.9971 s | 1.2059 s | BaM 端到端也更快，但收益被计算稀释 |
| retrieved chunks | 16 | 16 | 同一长上下文压力 |
| retrieved tokens | 4096 | 4096 | 同一 KV 读回规模 |

#### Page 1 参数解释

| 参数 | 含义 | 讲解口径 |
| --- | --- | --- |
| `read_avg_s` | 所有 read request 的平均耗时 | 最适合展示 KV Cache read/reuse 场景性能；本项目主要看这个指标 |
| `avg_request_s` | write 和 read 所有 request 的平均耗时 | 会被 write/prefill 和 decode 计算稀释，因此提升通常小于 `read_avg_s` |
| `read_p50_s` | read request 的中位数耗时 | 用于排除少数长尾，展示典型 read latency |
| `Read Speedup vs BaM` | 对照路径 `read_avg_s / BaM read_avg_s` | 数值越大，说明 BaM 相比该路径越快 |
| `exact match` | 同一样本 read 输出是否与 write 输出逐字一致 | 用于证明 KV read path 没有破坏正确性 |
| `requests` | 总请求数 | `repeat_read=1` 下 25 write + 25 read = 50；`repeat_read=3` 下 25 write + 75 read = 100 |
| `reads` | read request 数量 | Prefix Reuse / SSD-backed KV restore 的压力来源 |
| `retrieved chunks` | 从外部存储恢复的 LMCache chunks 数 | chunks 越多，SSD KV restore 压力越大 |
| `retrieved tokens` | 被恢复的 prefix token 数 | 近似表示本次 KV restore 覆盖的上下文规模 |
| `KV restore / read path` | 存储路径内部读回与放置耗时 | 更接近 SSD->GPU 数据路径真实性能，比端到端 request 更能体现 BaM/GDS 差异 |

Page 1 可讲结论：

```text
BaM 相比传统 GDS 的优势不是“模型整体立刻快数倍”，而是在 SSD-backed KV
restore 数据路径上减少 CPU 参与和中间同步边界。全量 lt4k 下 BaM 相比 GDS
read_avg 提升 1.15x；在 4k_8k 单 sequence 的 KV restore 层面，BaM 相比
GDS 约 3.47x。端到端收益较小，是因为 read request 还包含 prefill、decode、
sampling 和 engine step。
```

### Page 2：Layer 级 / Layer-group 级 KV Prefetch 是下一步优化点

这一页建议强调“当前 BaM 已经把 SSD->GPU KV restore 做快了，但现在仍然是整段 KV ready 后再 forward；下一步要把等待边界拆到 layer-group，让 IO 和 transformer compute 重叠”。

#### 表 1：KV restore 随上下文长度增加而变重

| Bucket | Prompt Tokens | Retrieved Chunks | Retrieved Tokens | BaM Read Path | Read Request | 说明 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `lt4k` | 1518 | 4 | 1024 | 24.772 ms | 0.9622 s | 短上下文，KV restore 压力较低 |
| `4k_8k` | 4440 | 16 | 4096 | 56.326 ms | 0.9971 s | 中等上下文，KV restore 已明显增加 |
| `8k_12k` | 8235 | 31 | 7936 | 103.114 ms | 0.9136 s | 长上下文，KV restore 成为可优化项 |

#### 表 2：当前 context-frontier 不是 layer 级 prefetch

| Bucket | Baseline BaM Read | Context-Frontier BaM Read | Baseline Read Request | Context-Frontier Read Request | Frontier 粒度 | 结论 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `lt4k` | 24.772 ms | 22.949 ms | 0.9622 s | 1.0497 s | `context_chunk`, `layer_group_size=28` | 内部读回略快但端到端变慢 |
| `4k_8k` | 56.326 ms | 58.548 ms | 0.9971 s | 1.0013 s | `context_chunk`, `layer_group_size=28` | 基本无收益 |
| `8k_12k` | 103.114 ms | 102.518 ms | 0.9136 s | 0.9207 s | `context_chunk`, `layer_group_size=28` | 基本无收益 |

#### 表 3：Layer-group prefetch 机会随上下文长度变化

这张表的横轴统一为上下文长度 bucket。BaM non-Nsight 日志提供当前 BaM 数据面的 KV restore 耗时，GDS Nsight trace 提供同 bucket 下的 transformer layer compute 窗口。这里不是做 BaM/GDS 横向性能对比，而是判断“随着上下文变长，整段 KV restore 相对 layer compute 是否越来越重”。

| Bucket | Prompt Tokens | Retrieved Chunks | BaM KV Restore | GDS `recv_kv` | GDS Prefill Forward | Decode Forward Avg | Decode Layer Avg | BaM Restore / Decode Step | BaM Restore / Decode Layer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lt4k` | 1518 | 4 | 24.772 ms | 80.713 ms | 638.931 ms | 15.472 ms | 0.535 ms | 1.60x | 46.3 layers |
| `4k_8k` | 4440 | 16 | 56.326 ms | 195.202 ms | 601.560 ms | 15.323 ms | 0.530 ms | 3.68x | 106.3 layers |
| `8k_12k` | 8235 | 31 | 103.114 ms | 411.933 ms | 589.641 ms | 14.983 ms | 0.518 ms | 6.88x | 199.1 layers |

#### Prefill + Layer-group Decode Prefetch 流水线思路

表 3 说明的是：当前 BaM KV restore 是一次性完成的串行等待项，且随上下文长度增长从 24.8 ms 增至 103.1 ms；同时 prefill forward 约 0.59~0.64 s，明显大于单次 KV restore。因此下一步更合理的目标不是只优化单请求的 raw restore time，而是把 KV restore 放到多请求 prefill 窗口和单请求 decode layer-group 窗口中重叠。

```text
时间轴  -------------------------------------------------------------------->

GPU Compute Stream:
Req A:  Prefill A ================================  Decode A-G0  Decode A-G1  Decode A-G2
Req B:                                          Prefill B ================================ Decode B
Req C:                                                                      Prefill C ========

GPU-Initiated BaM IO Stream:
Req B:  Restore Req B KV =========
Req C:              Restore Req C KV =========
Req A:                                Prefetch A-G1 KV   Prefetch A-G2 KV   Prefetch A-G3 KV
```

PPT 截图版流程图：

```text
┌──────────────────────────── GPU Compute ────────────────────────────┐
│ Req A Prefill ────────────────┐  Req A Decode G0 │ G1 │ G2 │ G3     │
│                               │  Req B Prefill ────────────────┐    │
└───────────────────────────────┴────────────────────────────────┴────┘

┌────────────────────── GPU-Initiated BaM IO ─────────────────────────┐
│ Restore Req B KV ───────┐ Restore Req C KV ───────┐                 │
│                         │                          │                 │
│                         └─ Prefetch Req A G1 KV ──└─ Prefetch G2 KV │
└─────────────────────────────────────────────────────────────────────┘
```

其中 `G0/G1/G2/G3` 表示连续的 transformer layer group，不是 GPU。例如 Qwen2.5-7B 有 28 层，若 `group_size=4`：

```text
G0 = Layer 0  ~ Layer 3
G1 = Layer 4  ~ Layer 7
G2 = Layer 8  ~ Layer 11
...
G6 = Layer 24 ~ Layer 27
```

调度逻辑可以概括为：

```text
1. Prefill 阶段隐藏跨请求 KV restore：
   当 Req A 正在执行长 prefill compute 时，BaM GPU worker 提前恢复 Req B/C 的 KV。

2. Decode 阶段隐藏请求内部的后续 layer group restore：
   当 Req A 正在计算 G0 时，GPU 侧预取 G1 需要的 KV；
   当 Req A 正在计算 G1 时，GPU 侧预取 G2 需要的 KV。

3. CPU 只负责推进 ready 请求进入下一轮计算：
   IO submit、completion poll、KV placement 和 ready flag 更新尽量由 GPU 侧完成。
```

这个思路下，BaM 相比传统 CPU-driven GDS 的优势不应只看单次 `read_avg`，而应看它是否更容易构建细粒度流水线：

| 路径 | IO 推进方式 | 适合的优化粒度 | 潜在问题 |
| --- | --- | --- | --- |
| 传统 GDS / CPU-driven 路径 | CPU submit / poll / callback 后推进调度 | 整段 KV restore 或较粗粒度 batch | IO 完成状态更多依赖 Host 侧感知，难和 GPU layer 执行节奏细粒度绑定 |
| BaM / GPU-Initiated 路径 | GPU worker 常驻轮询、搬运并写 ready flag | 跨请求 prefill overlap + layer-group decode prefetch | 需要控制 GPU worker 占用、ready/fence 粒度和 IO request 粒度 |

PPT 口径：

```text
Prefill 提供跨请求的大粒度隐藏窗口，layer-group decode prefetch 提供单请求内部的细粒度隐藏窗口。
BaM 的 GPU-initiated 路径把 IO 轮询与数据放置下沉到 GPU，更适合把 SSD KV restore
与 transformer layer 执行节奏绑定。它的目标不是让总 IO 字节数减少，而是降低 restore
暴露在关键路径上的比例。
```

#### 表 4：Layer-group 粒度的计算窗口估算

| Bucket | Decode Layer Avg | 4-layer Group Compute | 7-layer Group Compute | Full 28-layer Decode Forward | 判断 |
| --- | ---: | ---: | ---: | ---: | --- |
| `lt4k` | 0.535 ms | 2.14 ms | 3.75 ms | 15.47 ms | 单层太短，group 粒度更合理 |
| `4k_8k` | 0.530 ms | 2.12 ms | 3.71 ms | 15.32 ms | restore 已约等于 3.7 个 decode step |
| `8k_12k` | 0.518 ms | 2.07 ms | 3.63 ms | 14.98 ms | restore 已约等于 6.9 个 decode step |

#### 补充实验结果目录

| 测试 | Run Dir / Trace |
| --- | --- |
| GDS `lt4k` layer Nsight run | `evaluation/lmcache_ssd_read_paths_baseline/logs_longbench_triviaqa/nsys_probe_gds_layer_lt4k_b1_mt16_s1/20260730_060636` |
| GDS `lt4k` Nsight report | `evaluation/lmcache_ssd_read_paths_baseline/nsys_decode_probe/gds_layer_probe_lt4k_b1_mt16_s1.nsys-rep` |
| GDS `lt4k` SQLite | `evaluation/lmcache_ssd_read_paths_baseline/nsys_decode_probe/gds_layer_probe_lt4k_b1_mt16_s1.sqlite` |
| GDS `4k_8k` layer Nsight run | `evaluation/lmcache_ssd_read_paths_baseline/logs_longbench_triviaqa/nsys_probe_gds_layer_4k8k_b1_mt16_s1/20260729_122220` |
| GDS `4k_8k` SQLite | `evaluation/lmcache_ssd_read_paths_baseline/nsys_decode_probe/gds_layer_probe_4k8k_b1_mt16_s1.sqlite` |
| GDS `8k_12k` layer Nsight run | `evaluation/lmcache_ssd_read_paths_baseline/logs_longbench_triviaqa/nsys_probe_gds_layer_8k12k_b1_mt16_s1/20260730_060800` |
| GDS `8k_12k` Nsight report | `evaluation/lmcache_ssd_read_paths_baseline/nsys_decode_probe/gds_layer_probe_8k12k_b1_mt16_s1.nsys-rep` |
| GDS `8k_12k` SQLite | `evaluation/lmcache_ssd_read_paths_baseline/nsys_decode_probe/gds_layer_probe_8k12k_b1_mt16_s1.sqlite` |

#### Page 2 参数解释

| 参数 | 含义 | 讲解口径 |
| --- | --- | --- |
| `BaM Read Path` | BaM 内部 SSD->GPU read + one-copy direct placement 的耗时 | 衡量 BaM 数据面读回成本；随 chunks 增加从 24.8 ms 增至 103.1 ms |
| `Read Request` | 完整 read request 的端到端耗时 | 包含 KV restore、prefill、decode、sampling、engine step |
| `context_chunk` | 当前 frontier 只按上下文 chunk 发布 ready | 说明目前还没有按 transformer layer 拆分 |
| `layer_group_size=28` | 28 层作为一个整体 ready | Qwen2.5-7B 有 28 层，这等价于“整模型粒度”，不是 layer 级预取 |
| `recv_kv` | vLLM/LMCache 接收或恢复 KV 的 NVTX 区间 | 当前同步等待边界；如果太大，就应该被拆细并与计算重叠 |
| `prefill forward` | read 后执行的 prefill 计算 | 如果下一批 / 下一层 KV 能提前读，这部分计算可成为 overlap 窗口 |
| `decode forward avg` | 每个 decode step 的 transformer forward 平均耗时 | 可估算每 token 可提供多少计算时间用于隐藏 IO |
| `decode layer avg` | 单层 decode 计算耗时 | 约 0.53 ms，说明逐层粒度太细，应考虑 layer-group |
| `decode sample sum` | decode 阶段 sampling 总耗时 | 解释为什么 read path 加速不会完全转化为端到端加速 |
| `BaM Restore / Decode Step` | BaM KV restore 相当于多少个 decode step 的计算时间 | 数值从 1.60x 增至 6.88x，说明长上下文下整段 restore 越来越难被单个 decode step 自然掩盖 |
| `BaM Restore / Decode Layer` | BaM KV restore 相当于多少个单层 decode compute | 数值从 46.3 层增至 199.1 层，说明单层 prefetch 粒度过细，应尝试 layer-group |
| `4-layer / 7-layer Group Compute` | 按 4 层或 7 层聚合后的估算计算窗口 | 用于判断 layer-group 粒度是否比单层更适合做 IO-compute overlap |

Page 2 可讲结论：

```text
当前 BaM 的问题不是 SSD->GPU read path 完全没有优势，而是等待边界仍然偏粗：
必须等整段 prefix KV restore 完成后才进入 forward。随着上下文变长，BaM
read path 从 24.8 ms 增长到 103.1 ms，已经成为可优化项。但单层 decode
compute 只有约 0.53 ms；到 8k_12k 时，一次 BaM restore 已约等于 6.88 个
decode step 或 199 个单层 decode compute。这个比例说明单层预取粒度太细，
整段等待又太粗。因此下一步不应做 layer i -> layer i+1 的极细粒度预取，
而应做 4 层或 7 层一组的 layer-group prefetch，并在 attention 前用
GPU-side ready/fence 等待当前 layer group ready，让 SSD read 与 transformer
layer compute 重叠。
```

## 数据集信息

| 字段 | 值 |
| --- | --- |
| 样本数 | 25 |
| source_dataset | `triviaqa` |
| source_file | `triviaqa.jsonl` |
| language | `en` |
| length_bucket | `lt4k` |
| token_length_bucket | `lt4k` |
| prompt_mode | `full` |
| reserved_output_tokens | 32 |

### 长度分布

| 字段 | min | max | avg | p50 |
| --- | ---: | ---: | ---: | ---: |
| `length` | 1145 | 2643 | 1982.84 | 1987 |
| `raw_length` | 1145 | 2643 | 1982.84 | 1987 |
| `qwen_prompt_tokens` | 1518 | 3771 | 2904.72 | 3028 |
| `qwen_total_budget_tokens` | 1550 | 3803 | 2936.72 | 3060 |

## 测试命令

### BaM one-copy 1+4 CTA, repeat_read=1

```bash
sudo -n NUM_SAMPLES=0 REPEAT_READ=1 MAX_TOKENS=16 \
  GIDS_KV_DEBUG=0 LONGBENCH_DEBUG_LOG=0 \
  GIDS_KV_GPU_WORKER_MOVER_CTAS=4 \
  LOG_ROOT=/home/xhk/llm-inference/vllm-bam/evaluation/lmcache_ssd_read_paths_baseline/logs_longbench_triviaqa/full_lt4k_bam_one_copy_dedicated \
  /usr/local/sbin/run-bam-one-copy-qwen25
```

### LMCache SSD cold+cgroup 16G, repeat_read=1

```bash
sudo -n /usr/local/sbin/run-lmcache-ssd-cold-cgroup-qwen25 \
  --num-samples 0 \
  --repeat-read 1 \
  --max-tokens 16 \
  --log-root /home/xhk/llm-inference/vllm-bam/evaluation/lmcache_ssd_read_paths_baseline/logs_longbench_triviaqa/full_lt4k_lmcache_ssd_cold_cgroup_16g
```

该路径每次 read 前执行 `sync + drop_caches`，并在 16GB cgroup memory limit 下运行，用于压低传统 CPU page cache 对 SSD read 的加速影响。

### LMCache GDS, repeat_read=1

```bash
NUM_SAMPLES=0 REPEAT_READ=1 MAX_TOKENS=16 \
  LOG_ROOT=/home/xhk/llm-inference/vllm-bam/evaluation/lmcache_ssd_read_paths_baseline/logs_longbench_triviaqa/full_lt4k_lmcache_gds \
  bash evaluation/lmcache_ssd_read_paths_baseline/run_longbench_triviaqa_ssd_read_paths_qwen25.sh gds_gpu
```

### BaM one-copy 1+4 CTA, repeat_read=3

```bash
sudo -n NUM_SAMPLES=0 REPEAT_READ=3 MAX_TOKENS=16 \
  GIDS_KV_DEBUG=0 LONGBENCH_DEBUG_LOG=0 \
  GIDS_KV_GPU_WORKER_MOVER_CTAS=4 \
  LOG_ROOT=/home/xhk/llm-inference/vllm-bam/evaluation/lmcache_ssd_read_paths_baseline/logs_longbench_triviaqa/full_lt4k_r3_bam_one_copy_dedicated \
  /usr/local/sbin/run-bam-one-copy-qwen25
```

### LMCache GDS, repeat_read=3

```bash
NUM_SAMPLES=0 REPEAT_READ=3 MAX_TOKENS=16 \
  LOG_ROOT=/home/xhk/llm-inference/vllm-bam/evaluation/lmcache_ssd_read_paths_baseline/logs_longbench_triviaqa/full_lt4k_r3_lmcache_gds \
  bash evaluation/lmcache_ssd_read_paths_baseline/run_longbench_triviaqa_ssd_read_paths_qwen25.sh gds_gpu
```

## 原始结果目录

| 测试 | run dir |
| --- | --- |
| BaM one-copy 1+4 CTA, r1 | `evaluation/lmcache_ssd_read_paths_baseline/logs_longbench_triviaqa/full_lt4k_bam_one_copy_dedicated/20260728_232319` |
| LMCache SSD cold+cgroup 16G, r1 | `evaluation/lmcache_ssd_read_paths_baseline/logs_longbench_triviaqa/full_lt4k_lmcache_ssd_cold_cgroup_16g/20260728_232638` |
| LMCache GDS, r1 | `evaluation/lmcache_ssd_read_paths_baseline/logs_longbench_triviaqa/full_lt4k_lmcache_gds/20260728_232835` |
| BaM one-copy 1+4 CTA, r3 | `evaluation/lmcache_ssd_read_paths_baseline/logs_longbench_triviaqa/full_lt4k_r3_bam_one_copy_dedicated/20260728_233126` |
| LMCache GDS, r3 | `evaluation/lmcache_ssd_read_paths_baseline/logs_longbench_triviaqa/full_lt4k_r3_lmcache_gds/20260728_233330` |

每个目录包含：

- `run.log`: 完整运行日志。
- `metrics.jsonl`: 每个 write/read request 的结构化结果。

## 性能结果

### repeat_read=1

`repeat_read=1` 表示每个样本 1 次 write + 1 次 read，共 25 write + 25 read = 50 requests。

| 路径 | requests | writes | reads | write_avg_s | read_avg_s | avg_request_s | read_p50_s | read_min_s | read_max_s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BaM one-copy 1+4 CTA | 50 | 25 | 25 | 1.0638 | 0.5147 | 0.7893 | 0.4974 | 0.4628 | 0.9672 |
| LMCache SSD cold+cgroup 16G | 50 | 25 | 25 | 0.9609 | 0.6952 | 0.8280 | 0.6872 | 0.5791 | 1.0683 |
| LMCache GDS | 50 | 25 | 25 | 1.0432 | 0.5903 | 0.8168 | 0.5784 | 0.5137 | 1.0474 |

### repeat_read=3

`repeat_read=3` 表示每个样本 1 次 write + 3 次 read，共 25 write + 75 read = 100 requests。该配置更强调 Prefix Reuse / KV read path 压力。

| 路径 | requests | writes | reads | write_avg_s | read_avg_s | avg_request_s | read_p50_s | read_min_s | read_max_s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BaM one-copy 1+4 CTA | 100 | 25 | 75 | 1.0777 | 0.5007 | 0.6449 | 0.4942 | 0.4577 | 0.9595 |
| LMCache GDS | 100 | 25 | 75 | 1.0422 | 0.5787 | 0.6945 | 0.5795 | 0.4985 | 1.1021 |

## 正确性结果

### repeat_read=1

| 路径 | write/read exact match | read answer hits | all answer hits | drop_caches_reads |
| --- | ---: | ---: | ---: | ---: |
| BaM one-copy 1+4 CTA | 25/25 | 23/25 | 46/50 | 0 |
| LMCache SSD cold+cgroup 16G | 25/25 | 23/25 | 46/50 | 25 |
| LMCache GDS | 25/25 | 23/25 | 46/50 | 0 |

### repeat_read=3

| 路径 | write/read exact match | read answer hits | all answer hits | drop_caches_reads |
| --- | ---: | ---: | ---: | ---: |
| BaM one-copy 1+4 CTA | 75/75 | 69/75 | 92/100 | 0 |
| LMCache GDS | 75/75 | 69/75 | 92/100 | 0 |

说明：

- `write/read exact match` 表示同一个 sample 的 read request 输出是否与 write request 输出逐字一致。
- BaM 与对照路径在 r1/r3 下均完全逐字一致，说明 KV read path 没有引入文本级偏差。
- answer hit 数低于 request 总数，是模型输出本身没有包含答案子串；同一口径下 BaM 与对照一致，不是 BaM read path 导致的正确性退化。

## 相对性能对比

下表先列每条路径的原始延迟数据，再列相对 BaM 路径的性能对比；BaM 路径本身也作为 baseline 单独列出。

| 口径 | 路径 | requests | read_avg_s | avg_request_s | read speedup vs BaM | read latency reduction vs this path | total speedup vs BaM | total latency reduction vs this path |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| r1 | BaM one-copy 1+4 CTA | 50 | 0.5147 | 0.7893 | 1.000x | 0.0% | 1.000x | 0.0% |
| r1 | LMCache SSD cold+cgroup CPU path | 50 | 0.6952 | 0.8280 | 1.351x | 26.0% | 1.049x | 4.7% |
| r1 | LMCache GDS | 50 | 0.5903 | 0.8168 | 1.147x | 12.8% | 1.035x | 3.4% |
| r3 | BaM one-copy 1+4 CTA | 100 | 0.5007 | 0.6449 | 1.000x | 0.0% | 1.000x | 0.0% |
| r3 | LMCache GDS | 100 | 0.5787 | 0.6945 | 1.156x | 13.5% | 1.077x | 7.1% |

## 结论

1. 在全量 `lt4k` 数据集、关闭 debug 的压力测试下，BaM one-copy 1+4 CTA 路径正确性稳定，r1/r3 均达到 write/read 逐字一致。
2. 相比 LMCache SSD cold+cgroup 传统路径，BaM read 阶段延迟降低 26.0%，说明在尽量压低 CPU page cache 影响后，BaM GPU-side one-copy 路径能体现出 SSD read path 优势。
3. 相比 LMCache GDS 路径，BaM 在 r1 和 r3 下分别取得 1.147x / 1.156x read speedup；r3 场景下端到端平均请求延迟降低 7.1%，更能体现 KV read/reuse 压力下的优势。
4. 当前端到端收益小于 read 阶段收益，主要因为 write/prefill 和生成计算仍占较大比例；当 read/reuse 比例提高时，BaM 的端到端优势更明显。
