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
