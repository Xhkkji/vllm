# GPU-Initiated 后续思路 20260813

## 1. 当前主线判断

当前工作不再优先强行完整复现 Tutti，而是把 BaM/GPU-Initiated direct storage 的优势落到更适合的长上下文问题上：

```text
长上下文请求的完整 KV 很难全部驻留 GPU；
多轮对话和多请求 batching 会进一步放大 KV 显存压力；
如果上层 attention 只需要部分历史 KV block，就不应恢复、保留和调度完整 prefix KV。
```

因此当前更合理的目标是：

```text
基于 BaM/MDS 的细粒度 SSD KV 后端，
按照通用 prefetch plan 描述每层/每个阶段需要的 KV block，
用 layer window 控制恢复时机，
用 sparse selector 控制恢复范围，
再通过 scheduler 管理多请求之间的 restore、evict、in-flight 和优先级。
```

这条路线既能覆盖 Tutti-style layerwise prefetch，也能继续扩展到 sparse attention 式的部分 token/block 预取。

## 2. 已经做到的部分

### 2.1 BaM MDS SSD KV 基础链路

已经打通：

- vLLM KV Connector 到 BaM MDS 后端的读写链路；
- Prefix Cache 命中、SSD 索引管理、KV 写回与 SSD 到 GPU 的恢复；
- MDS async restore、batch done、completion poll 与 scheduler event 回收；
- 与 LMCache SSD 的同口径 baseline 对比；
- MDS service lifetime 的 `resident` / `io_active` 控制，其中 `io_active` 已降级为后台轮询优化，不再作为性能差异的核心解释。

当前结论是：

```text
BaM/MDS 路径的数据正确性和 SSD restore 链路已经基本跑通；
prefix 场景下 BaM 与 LMCache 已能形成同口径 baseline；
BaM 的优势不应只靠普通 dense prefix restore 来体现，而应放到细粒度访问和重叠调度场景中。
```

### 2.2 Layerwise prefetch 进度

已经实现并验证过 layer-window restore plan：

- 把完整 prefix KV 按 layer window 切分；
- 调度器能生成 first-window-ready 和 full-restore-ready trace；
- 可以先恢复前几个 layer 的 KV，再让后续 layer KV 在后台恢复；
- model runner 侧已有 layer barrier 的基础接入，能在进入某个 layer 前等待对应 window ready。

已有 30K/2K 长上下文场景结果：

```text
Qwen2.5-7B
prefix/suffix = 30K/2K
layer-window restore + prefill overlap
相比 full-restore baseline，TTFT 下降约 16.7%
```

这个结果说明 layerwise overlap 是有收益窗口的，但需要注意：

```text
这版仍主要是 CPU 控制提交时机；
BaM/MDS 执行的是 SSD 到 GPU 的异步恢复；
还没有做到 Tutti 那种完整 GPU-side rolling activation 协议。
```

进一步看 Tutti 的原始思路，它解决的重点仍然是：

```text
何时恢复 KV
如何让恢复与计算重叠
如何减少 full restore 暴露在 TTFT 关键路径上的时间
```

它已经把 SSD-backed dense KV restore 的时序问题推进了很多，但它并没有把问题进一步改写成：

```text
attention 实际需要哪些 KV block
哪些 block 可以不恢复
哪些 block 只在某些 layer / window 中可见
```

所以更准确地说，Tutti 证明的是 **layer-wise restore 需要 overlap**，但还没有把 **restore 粒度** 推到 **访问感知的 sparse block level**。这正是我们后面引入 sparse attention 的切入点：  
前者优化“什么时候恢复”，后者优化“恢复哪些”。

### 2.3 Sparse-style fine-grained restore profiling

当前已经把 layerwise prefetch 的组织方式进一步收束成更通用的 `PrefetchPlan` / `PrefetchUnit`：

- `PrefetchUnit` 描述一个恢复单元；
- `block_indices` 表示这个 unit 需要从 SSD 恢复的 logical prefix blocks；
- `consumer_blocks_by_layer` 表示每个 layer 实际可消费的 block 集合；
- dense layerwise prefetch 是默认情况；
- sparse selector 可以只让某些 layer 消费部分 block。

新增的 `SparseKVAccessPlan` 语义是：

```text
layer_index -> logical prefix block indices
```

其中 block index 是相对完整 prefix block table 的逻辑编号，不是 SSD restore mapping 内部偏移。每个 layer 可以有不同 block 集合，每个 layer window 恢复这些 layer 所需 block 的并集。

当前 profiling 结果：

```text
total tokens: 1024
prefix/suffix: 768/256
layers: 28
window layers: 4
MDS max in-flight: 7
repetitions: 1
```

| 模式 | 每个 unit 读取 blocks | physical restore |
|---|---:|---:|
| dense | 48 | 14.72 ms |
| tail_n=1 | 1 | 7.38 ms |
| tail_n=2 | 2 | 12.11 ms |
| stride=2 | 24 | 8.43 ms |

可以支持的保守结论：

- BaM/MDS 已经能真实发起 partial-block SSD 到 GPU restore；
- sparse-style 访问计划能减少真实读取 block 数；
- 在部分 block 访问模式下，物理 restore 时间确实下降；
- 小粒度场景下仍存在 submit、CQ poll、MDS handle 和 scheduler 控制面的固定开销；
- 当前仍是 I/O profiling，不是端到端 sparse attention TTFT/TPOT 收益。

### 2.4 这一页 PPT 的组织方式

这一页建议只讲一个核心判断：

```text
Tutti-style layerwise prefetch 解决了“什么时候恢复”，
但没有解决“恢复哪些 KV block”。
```

可以按下面三段展开：

| 现有层级预取已经解决 | 仍然存在的问题 | 引出的新思路 |
|---|---|---|
| full restore 的尾部可以被部分隐藏 | 仍然按 dense prefix / dense layer 恢复，读了很多当前不需要的 KV | sparse attention |
| restore 可以与 prefill / forward overlap | 只回答“何时恢复”，没有回答“恢复哪些” | 访问感知的 block 选择 |
| 长上下文的 TTFT 能被改善 | 完整 prefix KV 仍可能给 HBM residency 和 I/O 带来压力 | partial residency / eviction |

这一页最后可以落到一句话：

```text
层级预取优化的是时序，sparse attention 优化的是访问集合；
当前问题是两者还没有统一到同一个细粒度 KV I/O 接口里。
```

## 3. 当前代码组织

当前相关逻辑主要收束在：

```text
vllm/core/custom_schedulers/hierarchical_io/
```

核心文件职责：

- `plan.py`：定义通用 prefetch plan、layer window、sparse block selector；
- `runtime.py`：负责根据 plan 组织 MDS restore units；
- `barrier.py`：负责 forward 期间的 layer-window wait 和 active sparse block 暴露；
- `residency.py`：维护 worker-local 的 partial residency 状态；
- `async_kv_transfer.py`：承接 scheduler 与 MDS handle/event 的交互；
- `async_kv_scheduler.py`：在调度层接入 prefix restore admission；
- `worker.py` / `qwen2.py`：把 layer barrier 接到模型执行路径。

当前新增的 residency 语义是：

```text
QUEUED -> PENDING -> READY
```

它只描述“哪些 logical block 对当前 layer 已经安全可见”，不代表已经修改了 vLLM 原生 allocator 的物理 block 生命周期。

## 4. 当前实现能保证什么

当前可以保证：

- dense layerwise prefetch 的正确逻辑仍然保留；
- sparse selector 可以生成 partial-block restore plan；
- MDS restore 可以按 unit 只恢复部分 logical blocks；
- worker 能记录哪些 block 已经 READY；
- model forward 到某层时，可以查询当前 layer 允许消费的 sparse block 集合；
- request 结束后，residency 状态会随 finished request 清理。

当前不能声称：

- 已经实现真正 sparse attention；
- attention kernel 已经只访问 restored blocks；
- 未恢复的 dense prefix block 已经可以安全缺失；
- GPU KV block 已经可以按 layer 用完后物理释放；
- 已经解决多请求下的完整 prefetch/evict 调度。

## 5. 当前主要限制

### 5.1 Sparse attention consumer 还没接入

现在 `get_active_sparse_kv_blocks()` 已经能把当前 layer 的 block 集合暴露出来，但 Qwen2 的 attention 仍然是 dense block table 语义。

所以当前 partial restore 只能用于 profiling，不能直接进入正常生成路径。

### 5.2 还没有 sparse block table

真正运行 sparse attention 时，attention 后端不能继续拿完整 prefix block table，而应该拿每层对应的 sparse block table。

否则模型可能访问尚未恢复的 block。

### 5.3 物理 eviction 还不能打开

当前可以在逻辑上判断某个 unit 是否已经不再被后续 layer 需要，但还不能马上释放物理 GPU KV block。

必须等到：

- attention 确认只访问 sparse block table；
- allocator ownership 清楚；
- 后续 layer 不会再访问这些 block；
- release 不会破坏原生 dense block table 假设。

### 5.4 GPU-side rolling activation 还没完成

当前每次 submit 仍主要由 CPU/scheduler 组织。更接近 Tutti 的长期版本应该是：

```text
scheduler 先提交前 N 个 window；
model forward 到 layer 边界；
GPU/worker 发出安全推进信号；
scheduler 或 runtime 激活后续 window；
始终保持固定 lead distance。
```

这需要一个明确的安全协议，不能在 worker 里临时硬做。

## 6. 下一步实现顺序

建议先做最小闭环，不急着把所有机制一次塞进去。

### Step 1：接入真实 sparse attention policy

先不改 kernel，先确定 policy 如何生成 `SparseKVAccessPlan`：

```text
输入：request、prefix blocks、layer index、attention policy
输出：每个 layer 需要访问的 logical block indices
```

第一版可以只支持简单策略：

- tail-n blocks；
- stride blocks；
- fixed ratio blocks；
- 后续再接真实 sparse attention 的 mask 或检索结果。

目标是让上层 sparse 访问语义和底层 BaM restore plan 对齐。

### Step 2：让 attention backend 消费 sparse block set

把 `get_active_sparse_kv_blocks()` 接到 attention 输入侧，生成每层真正使用的 sparse block table。

判断标准：

```text
attention 只访问 READY blocks；
未恢复 blocks 不会被 kernel 读到；
dense 模式仍然保持原有行为。
```

这是从 profiling 走向端到端正确性的关键一步。

### Step 3：验证单请求 layerwise sparse restore

先用并发 1 跑通：

```text
long prefix + short suffix
layer window restore
sparse block selector
output 1-8 tokens
```

要观察：

- physical restore bytes 是否下降；
- restore tail 是否能与 suffix prefill overlap；
- TTFT 是否相比 dense full restore 下降；
- 输出是否稳定，不出现访问未恢复 block 的错误。

### Step 4：加入安全 eviction

当某些 block 在未来 layer 不再被访问时，才允许释放 GPU KV residency。

第一版建议只做保守 eviction：

```text
只释放 sparse plan 明确声明后续不会再消费的 blocks；
只在 layer boundary 执行；
只在 request-local residency 中先验证；
再逐步接 allocator 的物理释放。
```

目标是开始体现：

```text
每层用完的 KV block 可以逐出；
长上下文不需要完整 prefix 常驻 GPU；
多请求 batching 的 KV 显存压力下降。
```

### Step 5：扩展到多请求调度

单请求正确后，再做 scheduler 策略：

- 多请求 prefetch queue；
- MDS in-flight slot 控制；
- read-first 或 decode-safe priority；
- lead distance；
- restore/evict 协同；
- waiting 请求 admission。

这里才是后续打多轮对话和 batch 扩展性的主战场。

## 7. 后续评测计划

### 7.1 I/O profiling

继续保留当前 sparse profiling，用来回答：

```text
访问更少 block 时，BaM/MDS 的物理 restore 时间是否下降？
固定开销占比有多大？
什么粒度下细粒度 I/O 最划算？
```

重点补多次重复：

- dense；
- tail_n=1；
- tail_n=2；
- stride=2；
- 不同 prefix length；
- 不同 window size。

### 7.2 单请求端到端

优先继续用已经确定更容易出收益的点：

```text
30K prefix / 2K suffix
31K prefix / 1K suffix
```

对比：

- dense full restore；
- dense layerwise prefetch；
- sparse layerwise prefetch；
- LMCache dense SSD baseline。

指标：

- TTFT；
- physical restore latency；
- SSD read bytes；
- peak GPU KV residency；
- layer wait time。

### 7.3 多请求和多轮对话

在单请求 sparse 正确之后，再回到多轮对话：

- 多 session；
- 多 turn prefix reuse；
- read/write 混合；
- limited GPU KV capacity；
- active eviction；
- continuous batching。

核心指标：

- TTFT p95 / p99；
- TPOT p95 / p99；
- SLO 达成率；
- SSD read/write backlog；
- GPU KV peak residency；
- batch size under memory pressure。

## 8. 预期贡献表述

当前最适合整理成的系统贡献是：

```text
面向长上下文和多轮对话推理，设计基于 GPU-Initiated direct storage 的
细粒度 SSD KV Cache 后端。系统通过通用 prefetch plan 组织 layer-window
和 sparse block 级 KV restore，使 SSD KV 能按 attention 访问需求恢复到
GPU，并为后续 layer-by-layer overlap、KV eviction 和多请求调度提供统一接口。
```

更短的版本：

```text
基于 BaM 的 GPU-Initiated SSD KV 后端，将长上下文 KV Cache 从“整段恢复”
推进到“按 layer 和 sparse block 细粒度恢复”，初步验证 partial-block restore
能降低物理 I/O 时间，并为后续 sparse attention 与多请求 KV 调度奠定基础。
```

## 9. 当前结论

当前已经可以支持的结论是：

```text
在长上下文问题中，如果上层 attention 以 sparse block 方式访问部分历史 KV，
BaM/MDS 的细粒度 I/O 链路确实可以减少真实 SSD 读取范围，并降低物理 restore 时间。
```

下一步不是继续堆新的调度分支，而是把已有 layerwise 逻辑收束成真正的 sparse attention 消费闭环：

```text
SparseKVAccessPlan
  -> partial-block MDS restore
  -> per-layer sparse block table
  -> attention kernel 只读 READY blocks
  -> layer boundary safe eviction
  -> multi-request prefetch/evict scheduler
```

这样才能最终回答：

```text
BaM 细粒度 GPU-Initiated SSD I/O 是否能在长上下文 sparse attention 和多轮对话场景下，
同时降低 restore stall、降低 GPU KV 常驻压力，并提升 batching 扩展性。
```

## 10. SolidAttention 和相关 sparse attention 参考

### 10.1 SolidAttention 的 sparse attention 机制

SolidAttention 的核心不是单纯把 attention 做成随机稀疏，而是把 **动态 sparse attention** 和 **SSD-backed KV 管理** 一起设计。

它的基本访问形态可以理解成三部分：

```text
init blocks    -> 开头少量固定 block，保留 attention sink
local blocks   -> 最近 token 的局部窗口
selected blocks -> 根据当前 query 动态挑选的历史 block
```

更具体地说：

- KV cache 被按 block 切分；
- 每个 block 用一个代表向量做粗粒度打分；
- 当前 query 对这些 block 估计相关性；
- 只加载 top-k 的 selected blocks，再加上固定的 init/local blocks；
- attention 只在这些 block 上展开。

SolidAttention 的重点不只是“少看一些 token”，而是让 sparse 访问和 SSD 传输能一起工作：

- 通过 KV Consolidator 把 K/V 更紧凑地组织到 SSD 友好的布局里；
- 通过 speculative prefetch 预测下一层或下一轮可能要读的 block；
- 通过 SSD-aware scheduler 把 q-proj、select、kv-proj、load、attention、store 拆成可调度 microtask。

它给当前工作的启发是：

```text
sparse attention 的关键不只是 selector，
而是 selector + KV layout + prefetch + scheduler 必须一起对齐。
```

### 10.2 值得重点看的相关工作

下面这些工作和当前 BaM sparse 路线最相关，建议按优先级看：

| 工作 | 核心点 | 和当前工作的关系 |
|---|---|---|
| [SolidAttention](https://www.usenix.org/conference/fast26/presentation/zheng) | 动态 sparse attention + SSD-backed KV + speculative prefetch + SSD-aware scheduler | 最直接的对照对象，尤其适合看 block selection 和层级预取 |
| [SPIN](https://arxiv.org/abs/2604.26837) | sparse attention + hierarchical memory，统一多种 sparse 算法 | 很适合看“通用 sparse serving 框架”怎么做 |
| [SparseServe](https://arxiv.org/abs/2509.24626) | 动态 sparse attention 下的 HBM-DRAM 分层管理、working-set-aware batching、layer-segmented prefill | 很适合参考多请求调度和容量压力下的策略 |
| [LServe](https://arxiv.org/abs/2502.14866) | unified sparse attention，prefill/decode 一体化稀疏 serving | 适合看 serving 指标和 sparse page selection |
| [Quest](https://arxiv.org/abs/2406.10774) | query-aware sparse page/block selection | 适合作为第一版 sparse selector 的简单 baseline |

如果想再补两类参考：

- **MInference**：更偏 prefill 侧的 sparse pattern 识别；
- **DuoAttention / Kascade / MagicPIG**：更偏 head 或 block 级策略，对 sparse policy 设计有启发。

### 10.3 sparse attention 需要重点关注的指标

#### 1) 质量指标

这类指标先决定 sparse 是否“还能答对”：

- LongBench / RULER / InfiniteBench；
- Needle-in-a-Haystack / passkey retrieval；
- exact match / F1 / accuracy；
- 与 dense baseline 的质量差距；
- 长生成下是否出现明显 drift。

如果只做系统原型，至少要保证：

```text
dense output 和 sparse output 语义一致
needle / passkey 能命中
长上下文任务不明显掉点
```

#### 2) sparse selector 指标

这类指标决定“选得准不准”：

- selected blocks / total blocks；
- top-k recall；
- attention mass recall；
- false negative rate；
- prefetch hit rate；
- misprediction rate；
- cross-layer similarity；
- selector overhead。

对当前项目最关键的是三组集合要对齐：

```text
计划恢复的 block
实际恢复成功的 block
attention 真正消费的 block
```

#### 3) I/O 指标

这是 BaM 最该体现优势的地方：

- physical read bytes；
- logical requested bytes；
- I/O amplification；
- restore latency p50/p95/p99；
- effective bandwidth；
- random/sequential ratio；
- queue depth / in-flight；
- CQ polling overhead；
- blocking restore time；
- overlap ratio。

如果 sparse selector 真的有效，应该先看到：

```text
读取 bytes 下降
physical restore 下降
blocking wait 下降
```

然后才是 TTFT / TPOT 的下降。

#### 4) GPU KV residency 指标

这类指标决定“能不能省显存”：

- peak GPU KV memory；
- resident blocks per layer；
- restored-but-unused blocks；
- evicted blocks；
- eviction correctness；
- allocator fragmentation；
- max batch size under memory cap。

这部分是 sparse attention 在 serving 场景里很重要的系统价值：  
不是只追求快，还要让 GPU 上能同时容纳更多请求。

#### 5) 端到端 serving 指标

最后才看用户可见指标：

- TTFT p50 / p95 / p99；
- TPOT p50 / p95 / p99；
- throughput；
- request latency；
- SLO attainment；
- waiting time；
- batch size；
- tail latency under load。

### 10.4 对当前 BaM 路线的建议

当前不建议直接跳到完整 SolidAttention 复现，而是先做一个更轻的闭环：

```text
1. 先用一个简单 sparse policy 生成 SparseKVAccessPlan
2. 让 attention backend 真的消费 sparse block set
3. 在 30K/2K、31K/1K 上验证 partial restore
4. 再补安全 eviction
5. 最后再做多请求调度和 lead distance
```

这样可以先回答一个最关键的问题：

```text
如果 attention 只看部分 block，BaM/MDS 的细粒度 SSD I/O 是否真的能减少物理访问时间，并进一步降低显存压力？
```
