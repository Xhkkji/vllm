# GPU-Initiated 后续 baseline 思路 20260809

## 1. 当前判断

当前系统已经具备 GPU-Initiated / BaM MDS 异步 SSD KV I/O 的基础链路：

- BaM/MDS 侧已经支持 descriptor pool、多 logical transfer、read/write 混合压力和 direct read/write correctness。
- vLLM 侧已经有 `AsyncKVScheduler` 过渡入口，并将 async KV transfer/state/policy 初步解耦到 `vllm/core/custom_schedulers/`。
- 现在还没有完成真正的 block-aware scheduler，当前自定义调度更接近“原生 vLLM chunked-prefill 调度 + 异步 KV swap transfer queue”。

后续不建议先完整复现 Tutti。更合适的路线是：

> 以 Tutti-style storage baseline 为主线，以 Bidaw-style multi-turn workload 暴露在线多轮对话下的 read/write 动态交织、waiting 饥饿和尾延迟问题。

## 2. 和相关工作的关系

### 2.1 Tutti

Tutti 主要解决长上下文 serving 中 SSD-backed KV cache 的 I/O 路径和 compute-I/O overlap。它的 baseline 主要按 storage tier 来定：

| Baseline | 含义 |
|---|---|
| HBM | 标准 vLLM，只用 GPU HBM |
| LMCache-DRAM-LW | LMCache DRAM 扩容 + layer-wise overlap |
| LMCache-SSD | SSD offload，CPU-mediated 路径 |
| LMCache-GDS | SSD offload + GDS |

当前系统最接近 Tutti 的方向，因此主 baseline 应该优先对齐 LMCache-SSD / LMCache-GDS，而不是直接复现 Tutti 的完整 GPU io_uring、green context 和 slack-aware scheduler。

### 2.2 Bidaw

Bidaw 面向在线多轮对话，重点是历史 KV 复用和两级存储调度。它的 baseline 包括 vLLM、CachedAttention、FlashGen 和 ideal caching。

Bidaw 对当前工作的启发主要是 workload 和调度问题定义：

- 多 session 持续到达；
- 多轮历史 KV 复用；
- SSD read 与新 KV write 动态交织；
- TTFT / TPOT / SLO / waiting time 作为核心指标。

### 2.3 SolidAttention

SolidAttention 面向本地低显存长上下文推理，使用 sparse attention 和 attention-inner SSD microtask 调度。它改变 attention 计算语义，因此当前阶段不适合作为主 baseline。

当前最多把 SolidAttention 放入 related work，用来说明 attention-inner 细粒度 I/O 调度方向；端到端 baseline 暂不直接对齐。

## 3. 推荐 baseline matrix

第一阶段建议固定如下配置矩阵：

| 配置名 | Scheduler | Chunked Prefill | KV Backend | 目的 |
|---|---|---:|---|---|
| `vllm_continuous` | 原生 `Scheduler` | 关 | HBM / recompute | 原生 continuous batching baseline |
| `vllm_chunked` | 原生 `Scheduler` | 开 | HBM / recompute | 原生 chunked prefill baseline |
| `lmcache_ssd` | 原生 `Scheduler` | 开 | LMCache local SSD | 当前常见 SSD KV baseline |
| `lmcache_gds` | 原生 `Scheduler` | 开 | LMCache GDS | 当前可用的 GDS-style baseline |
| `bam_sync` | 原生 `Scheduler` 或 sync path | 开 | BaM sync direct | 同步 SSD->GPU 路径对照 |
| `bam_async` | `AsyncKVScheduler` | 开 | BaM MDS async | 当前异步 I/O 后端 |
| `bam_async_sched` | 后续 block-aware scheduler | 开 | BaM MDS async | 最终主方案 |

第一版最重要的是：

```text
vLLM chunked prefill
vs LMCache-SSD/GDS
vs BaM MDS async
```

这样能清楚回答：

1. 原生 vLLM 调度是否已经足够；
2. LMCache SSD/GDS 在多轮对话下的问题在哪里；
3. BaM 细粒度 GPU-Initiated I/O 是否能降低 restore 尾延迟；
4. 后续 block-aware scheduler 应该优化 read、write、waiting 还是 cache residency。

## 4. Baseline runner 设计

建议在 `BaM_IOStack/vllm_evaluation/` 下新建统一 baseline runner：

```text
vllm_evaluation/multiturn_baseline/
  run_baseline_matrix.sh
  multiturn_serving_eval.py
  aggregate_baseline_matrix.py
  README.md
```

其中：

- `run_baseline_matrix.sh`：负责依次运行不同配置；
- `multiturn_serving_eval.py`：负责生成多轮请求、按 arrival rate 注入、调用 vLLM；
- `aggregate_baseline_matrix.py`：负责聚合 TTFT、TPOT、SLO、I/O latency、waiting time；
- `README.md`：记录实验配置和运行方式。

每个 run 的结果目录建议统一：

```text
result/multiturn_baseline/20260809_xxxxxx/
  vllm_continuous/run_1/
    console.log
    trace.log
    summary.json
  vllm_chunked/run_1/
  lmcache_ssd/run_1/
  bam_async/run_1/
  RESULTS.md
  summary.json
```

统一控制以下参数：

```text
model
tokenizer
max_model_len
max_num_seqs
max_num_batched_tokens
prompt length
output length
arrival rate
num_sessions
num_turns
seed
```

## 5. 最小多轮对话 workload

第一版先不做复杂真实 ShareGPT pipeline，可以先构造合成但可控的多轮对话：

```text
num_sessions = 16 或 32
num_turns = 3
turn_0 prompt_len = 2048 tokens
turn_1 = turn_0 history + 128-token new user input
turn_2 = turn_0 + turn_1 history + 128-token new user input
output_len = 64 或 128 tokens
arrival = Poisson
```

这个 workload 可以稳定制造：

- 历史 KV restore；
- 新增 decode KV write；
- 多 session 交错；
- read/write 混合；
- waiting queue 压力；
- chunked prefill 与 decode 混合调度。

后续再替换为 ShareGPT / LongBench-TriviaQA 派生的真实多轮 workload。

## 6. 第一版核心指标

第一版只保留最能说明问题的指标：

| 指标 | 作用 |
|---|---|
| TTFT p50 / p95 / p99 | 观察后续轮次是否被卡 |
| TPOT p50 / p95 / p99 | 观察 decode 是否被 I/O 干扰 |
| joint SLO 达成率 | 衡量用户体验 |
| waiting time p95 | 判断是否存在请求饥饿 |
| SSD read latency p95 | 判断 restore 是否暴露在关键路径 |
| SSD write latency / backlog | 判断后台写盘是否干扰前台读 |
| read/write bytes | 统计 I/O 压力和放大 |
| KV hit rate | 判断缓存复用是否有效 |
| GPU bubble / GPU util | 判断 I/O 是否造成计算空洞 |

如果只放一张早期结果表，可以保留：

```text
TTFT p95
TPOT p95
joint SLO
SSD read p95
SSD write backlog
waiting p95
```

## 7. Arrival rate sweep

baseline runner 跑通后，需要扫不同 arrival rate：

```text
arrival_rate = 0.25, 0.5, 1.0, 2.0 req/s
```

或者根据机器能力继续上探，直到出现明显尾延迟劣化。

这一步的目标是找到：

- 哪个 baseline 最先崩；
- 是 TTFT 崩，还是 TPOT 崩；
- 是 SSD read 暴露到前台，还是 write backlog 挤压 read；
- 是 vLLM waiting queue 堆积，还是 cache 粒度导致无效 I/O。

## 8. 根据结果决定调度优化

后续调度优化不应凭空设计，而应由 baseline 暴露的问题驱动：

| 观测现象 | 对应优化 |
|---|---|
| read p95 高 | read-first / read batching |
| write backlog 高 | write defer / write throttle |
| TTFT p95 高 | waiting-aware admission |
| TPOT p95 高 | 避免 decode 前台 restore |
| SSD bytes 高 | 细粒度 block read / 减少 chunk 放大 |
| swap/write 太多 | clean block skip-write |
| GPU bubble 高 | chunked prefill + async prefetch overlap |

## 9. 下一步执行顺序

建议后续按以下顺序推进：

1. 新建 `multiturn_baseline` runner；
2. 先支持 `vllm_continuous` 和 `vllm_chunked`；
3. 接入 `lmcache_ssd`；
4. 接入 `bam_async`；
5. 统一输出 `summary.json` 和 `RESULTS.md`；
6. 跑小规模 3-turn workload；
7. 扫 arrival rate，定位瓶颈；
8. 再把调度策略逐步接入 `vllm/core/custom_schedulers/`。

当前优先级最高的是：

> 先完成 baseline 实验闭环，再继续大改 scheduler。

## 10. 当前推进目标：Sparse Attention 驱动的细粒度 KV 调度

前面的 baseline 和 layerwise restore 实验已经给出一个更明确的方向：

```text
在长上下文 KV restore 中，如果 attention 只访问部分历史 KV block，
就不应恢复和保留完整 prefix。BaM/MDS 可以按 block 粒度读取需要的数据，
从而减少实际 SSD 访问量和物理 restore 时间。
```

当前 profiling 已经验证了这条 I/O 链路的可行性。在 1024-token 输入、
768-token prefix、28 层模型、4 层/window、7 个 MDS in-flight request
的测试中：

| 模式 | 每个 layer unit 读取 blocks | physical restore |
|---|---:|---:|
| dense | 48 | 14.72 ms |
| `tail_n=1` | 1 | 7.38 ms |
| `tail_n=2` | 2 | 12.11 ms |
| `stride=2` | 24 | 8.43 ms |

因此当前可以支持的保守结论是：

- BaM/MDS 已经能够真实执行 partial-block restore；
- sparse-style 的部分 block 访问可以降低物理 restore 时间；
- 小粒度下仍存在 submit、CQ poll、MDS handle 和 scheduler 控制面等固定开销；
- 目前验证的是 I/O profiling 收益，还不是 sparse attention 端到端
  TTFT/TPOT 收益。

### 目标系统

后续目标是把 layerwise prefetch 和 sparse-style block 访问结合起来：

```text
layer window 决定什么时候恢复；
sparse attention 访问计划决定恢复哪些 KV block；
BaM/MDS 负责 SSD -> GPU 的细粒度异步读取；
调度器在多个请求之间管理 restore、priority、in-flight 和 lead distance；
当前 layer 使用完成且后续不再访问的 KV block 可以逐出 GPU。
```

这个目标针对两个实际问题：

```text
1. 长上下文请求的完整 KV 无法长期驻留 GPU；
2. 多轮对话和多请求 batching 会放大 KV 显存压力，限制并发扩展。
```

理想的系统行为是：

```text
请求 A 正在执行当前 layer；
请求 B 的下一 layer 所需 KV 在后台从 SSD 恢复；
请求 C 只恢复 sparse attention 实际会访问的 block；
已消费且不再需要的 block 被释放，供其他请求复用。
```

最终希望形成：

> 面向长上下文与多轮对话服务，基于 BaM GPU-Initiated 直通存储路径，
> 按 sparse attention 访问计划恢复部分 KV block，结合 layer-window
> overlap、多请求细粒度调度和用后逐出，缓解 KV Cache 对 GPU 显存和
> batch 扩展性的限制。

### 还需要补齐的语义

当前 partial-block restore 仍然是 profiling-only，真正接入 sparse
attention 前需要补齐：

1. **Partial residency：** 明确哪些 KV block 已在 GPU，哪些仍在 SSD；
2. **Sparse consumer contract：** 保证 attention 只访问已经恢复的 block；
3. **Safe eviction：** 根据后续 layer 的访问计划判断 block 何时可以逐出；
4. **Multi-request scheduling：** 在多个请求之间批量提交和公平调度
   layer-window restore，避免单个长上下文请求独占 MDS slots。

因此下一步不应直接重写完整 scheduler，而应按以下顺序推进：

```text
1. 把真实 sparse attention 的 block 访问计划接入 PrefetchPlan；
2. 验证 partial residency 与 attention block table 的一致性；
3. 在单请求上验证 layerwise sparse restore 的正确性；
4. 再加入多请求 batch、in-flight 限制和用后逐出；
5. 最后与 LMCache dense restore baseline 对比端到端 TTFT/TPOT 和显存占用。
```
