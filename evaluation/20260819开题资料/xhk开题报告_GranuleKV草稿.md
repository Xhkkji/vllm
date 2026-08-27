# 面向KV-Cache的GPU-Initiated直通存储机制研究

## 1 课题背景、目的和意义

长上下文推理已经把 KV Cache 从“模型内部状态”推到了“系统瓶颈”的位置。随着上下文长度增长，KV Cache 的容量压力和恢复开销同时上升：一方面，GPU HBM 很快被占满；另一方面，KV 恢复越来越频繁，I/O、调度与搬运开销开始主导尾延迟。对于 SSD-backed KV Cache 而言，问题不再只是“存在哪里”，而是“如何把需要的 KV 以足够细、足够快、足够并发的方式取回到 GPU”。

### 1.1 从 Transformer 到生成式推理

Transformer 的提出把序列建模从递归和卷积的路径中解放出来，使注意力机制成为统一的建模核心。`Attention Is All You Need` 证明了基于 attention 的结构不仅可以替代传统序列模型，还能在训练并行性和建模能力之间取得更好的平衡。之后，大模型研究逐步从“更强的预训练表示”转向“更高效的生成式推理”，模型本身也从 encoder-decoder 的组合式范式，演进为更适合自回归生成的 decode-only 架构。

这一变化很关键。decode-only 模型在推理时并不要求整段序列反复重算，而是依赖 KV Cache 逐 token 累积历史上下文。也就是说，推理阶段真正的系统对象不再只是参数和算子，而是“历史上下文如何被持续保存、复用、搬运和恢复”。对于长上下文场景，KV Cache 逐渐从一个实现细节变成了性能中心。

### 1.2 decode-only 推理与 KV Cache 压力

在 decode-only 推理中，prefill 和 decode 的角色并不相同。prefill 阶段更接近一次性的大规模前向计算，算力占比高；decode 阶段则是逐 token 生成，单步计算量不大，但每一步都要反复读取历史 KV，因此更容易暴露出内存层次和 I/O 路径的短板。上下文越长，decode 越像一个由大量小访问构成的流式工作负载，系统瓶颈也越从算力转向数据移动。

vLLM 的价值就在这里被凸显出来。PagedAttention 把 KV Cache 管理从连续大块内存转向分页和块化组织，显著降低了碎片和冗余复制带来的浪费，并通过更灵活的批处理与抢占机制提升了 serving 吞吐。这说明，LLM 推理系统已经不再只是“模型跑得动”，而是“KV cache 能否被像操作系统页一样有效管理”。不过，vLLM 主要解决的是 HBM 内部的组织与复用问题，当 KV 不再能完整驻留在 GPU 内时，新的难点就会落到 SSD、CPU 和 GPU 之间的恢复链路上。

### 1.3 目标系统正在从“模型优化”转向“推理基础设施”

随着长上下文需求和多请求并发不断上升，LLM 推理系统逐渐从单模型推理转向基础设施级优化：KV Cache 共享、跨请求复用、分层缓存、预取、卸载和远端恢复开始成为主线。LMCache 这类系统进一步说明，KV Cache 已经可以作为跨引擎、跨请求的共享介质来组织，推理系统开始呈现出“token 计算器 + cache 管理器 + I/O 编排器”的结构。

这一演进也说明，本课题不应只盯着单次 Attention 的数学形式，而要看到长上下文推理的真实系统轮廓：模型层解决表示，推理层解决生成，缓存层解决复用，I/O 层解决恢复。GranuleKV 的意义就在于把最后一层补出来，让策略层不必直接面对具体设备的细粒度访问代价。

### 1.4 本课题的立足点

因此，本课题的目的不是再提出一种新的稀疏注意力或预取策略，而是围绕 KV Cache 的真实访问模式，建立一套能统一承接不同粒度请求、支持异步提交与回收、并尽量绕开不必要 CPU 搬运与调度耦合的系统机制。这一方向对长上下文推理的意义在于：它既服务于容量扩展，也服务于尾延迟优化，还为后续更多策略演进提供统一执行底座。

近期研究大体形成了两条互补路线。Sparse Attention 主要解决“读哪些 KV”，通过选择性访问减少真正参与注意力计算的数据量；Layer-wise Prefetch 主要解决“什么时候读 KV”，通过按层提前恢复并与计算重叠来隐藏等待时间。二者结合后，KV 访问从整层、整块读取，进一步变成了动态、离散、细粒度且并发更高的恢复请求。

但现有方案大多仍停留在策略层或 storage layout 层：它们会优化稀疏策略、缓存层次、对象布局或调度顺序，却未必真正提供一个面向 GPU 的、异步的、可统一承接 block/layer/token 级需求的 I/O 底座。CPU 可以承担容量层或中间层职责，但并不适合承接高频、细粒度、强并发的 KV 恢复路径。基于这一判断，本课题提出 GranuleKV，目标是构建 GPU-initiated 的细粒度异步 I/O 底座，让策略层和执行层解耦，并把 GPU -> SSD 的 direct path 做实。

因此，本课题的目的不是再提出一种新的稀疏注意力或预取策略，而是围绕 KV Cache 的真实访问模式，建立一套能统一承接不同粒度请求、支持异步提交与回收、并尽量绕开不必要 CPU 搬运与调度耦合的系统机制。这一方向对长上下文推理的意义在于：它既服务于容量扩展，也服务于尾延迟优化，还为后续更多策略演进提供统一执行底座。

## 2 国内外研究现状

### 2.1 Sparse Attention

Sparse Attention 通过减少参与计算的历史 KV 数量来降低长上下文推理成本。早期工作如 Sparse Transformers 和 BigBird 已经证明，稀疏化可以有效缓解注意力的二次复杂度问题，并在保持表达能力的同时显著降低计算与访问开销。进入近两年后，相关研究进一步走向硬件友好和任务自适应，例如 Native Sparse Attention、FlexPrefill 等工作开始根据输入内容、注意力头或预算动态决定稀疏模式，使稀疏策略从静态模板演进到上下文感知。

这类工作已经说明“只读关键 KV”是有效的，但它们回答的是“哪些 KV 需要被访问”，并没有自动解决被选中 KV 位于 SSD 或分层存储时如何高效恢复的问题。也就是说，Sparse Attention 能减小访问集合，却不能单独消除细粒度 I/O 执行层面的复杂性。

### 2.2 Layer-wise Prefetch

Layer-wise Prefetch 的核心思想是把 KV 恢复与 Transformer 层间计算重叠起来。LayerKV、InfiniGen 以及 LMCache 的 layerwise transfer 都体现出这一方向：按照层的执行顺序提前恢复下一层所需 KV，使 GPU 不必等到整批数据全部到位后才开始计算。对于长上下文推理而言，这类方法有效回答了“什么时候读 KV”。

但层级预取通常仍以 layer、chunk 或较大的 object 为单位。随着访问粒度下降，CPU submit、完成回收、请求组织和队列管理的开销会变得更显著；同时，预取粒度越大，越容易带回暂时不需要的数据，导致 GPU working set 仍然偏大。也就是说，Layer-wise Prefetch 解决了时序问题，却没有彻底解决细粒度请求的组织与执行问题。

### 2.3 Sparse Attention 与层级预取的结合

近期更成熟的系统开始把两类方法结合起来，例如 LServe、SPIN、HiSparse 等工作，把稀疏选择、分层缓存、页管理和按层恢复统一起来，说明“策略层 + 存储层”的联合设计已经成为主流方向。与此同时，Tutti 进一步把 CPU 从关键数据与 I/O 控制路径中移走，验证了 GPU-centric SSD-backed KV Cache 的可行性。

这些工作共同说明，研究重心已经从单纯压缩 KV，转向“如何围绕动态访问模式构建高效执行路径”。但它们仍存在一个共同缺口：即便系统已经足够 GPU-centric，仍需要一个能把 block/layer/token 级需求统一成 GPU 侧异步请求、并在底层完成聚合、去重、提交和回收的通用底座。GranuleKV 的切入点正是补齐这一层。

### 2.4 当前研究存在的问题

综合来看，现有方案主要存在四个问题：

1. 策略和执行分离不彻底。上层算法虽然已经能决定读什么、何时读，但底层往往仍依赖专用搬运逻辑。
2. 访问粒度仍不统一。block、layer、token 级需求并存，但缺少统一的请求抽象。
3. CPU 仍可能成为瓶颈。对于高频、小粒度、强并发的恢复请求，CPU staging 和 host-mediated transfer 会放大提交与回收开销。
4. 直接面向 GPU 的异步 I/O 底座不足。现有系统更多是在“优化如何搬运”，而不是“定义一条 GPU 直接发起、异步执行、可回收的标准路径”。

这正是 GranuleKV 要解决的问题。

### 2.5 从算法优化到系统底座

更深一层看，Sparse Attention、Layer-wise Prefetch、LMCache 和 Tutti 这些工作虽然切入点不同，但共同趋势非常一致：它们都在把 KV Cache 从“被动存储”变成“主动管理对象”。这意味着研究重心已经发生变化，不再只是思考某一个 attention pattern 是否更省算，而是在思考推理系统如何围绕访问模式重塑执行路径。

这个变化的重要性在于，策略层的成熟会自然抬高底层系统要求。以前，如果大部分 KV 都能常驻 HBM，那么缓存管理更多是实现问题；但当上下文长度、并发请求和服务成本一起上升时，访问就会变成大量细粒度、离散且时序敏感的请求。此时，任何依赖 host 反复编排的方式都会暴露出控制路径过长、提交粒度过粗和完成回收不够及时的问题。换句话说，问题已经不是“能不能读到”，而是“能不能用合适的方式读到”。

GranuleKV 的研究价值也在这里体现出来。它不试图和现有策略抢位置，而是把策略产出的需求接住，再把这些需求变成统一的执行对象。对于开题报告来说，这种表述更重要，因为它解释了为什么你的工作不是“另起炉灶做一个新稀疏算法”，而是“在已有成熟策略之上补一层系统底座”。

### 2.6 GDS、BaM、CAM 与 LMCache 的启发

从更广的系统谱系看，BaM、CAM/CIM、GDS 路径和 LMCache 都给了这个课题非常直接的启发。BaM 证明了 GPU-initiated 的存储访问可以成为现实路径，它打破了“必须由 CPU 先中转，再把数据交给 GPU”的传统假设。这个结论对 GranuleKV 很重要，因为你要做的正是让请求从 GPU 侧发起或驱动，并尽量让数据直接流向 GPU 可消费的位置。

CAM/CIM 一类工作则说明，围绕 KV 的优化已经开始从单纯的数据搬运，走向“缓存、稀疏、计算和存储协同”的更大空间。它们强调的是，推理系统可以不只是把数据搬过来，还可以考虑在更靠近数据的地方进行结构化处理。虽然这类方向更强调近数据计算或缓存侧协同，但它们共同传递了一个信息：随着 KV 规模扩大，系统设计必须跨越模型、缓存和 I/O 三层去思考。

LMCache 的意义则在于把这条路进一步工程化。它说明 KV Cache 不再只是单机内的局部对象，而可以成为分层管理、跨引擎共享、按层传输的服务资产。对开题报告而言，这意味着你的研究不是孤立地讨论一个抽象 I/O 问题，而是在回应一个已经成形的行业趋势：推理系统正在把 KV Cache 作为第一类资源来调度、迁移和恢复。

GranuleKV 与这些工作的关系，可以概括为一句话：BaM 证明“GPU 直接发起存储访问”是可行的，CAM/CIM 说明“围绕 KV 做系统级协同优化”是必要的，LMCache 证明“KV 分层传输与共享管理”是能落地的，而 GranuleKV 要做的是把这些趋势收敛成一条更统一、更细粒度、更异步的执行底座，让 block、layer、token 级需求都能用同一种方式被承接。

### 2.7 本课题的判断

基于上述背景，本课题有三个明确判断。第一，大模型推理的核心矛盾已经从参数规模转向上下文和 KV 管理，尤其是 decode 阶段的持续恢复压力。第二，Sparse Attention 和 Layer-wise Prefetch 已经把“读什么”和“何时读”这两个问题说明白了，因此新的研究空间不应继续停留在策略重复。第三，真正值得做的是“怎么读”，也就是如何用 GPU 侧可控、异步、细粒度的方式把 KV 从 SSD 高效恢复到计算侧。

这三个判断合在一起，就自然导出了 GranuleKV 的定位：它不是一个独立的上层策略，而是多个策略共同依赖的传输与执行底座。

### 2.8 最新成果反映出的共同趋势

如果把近年的代表性工作放在一起看，会发现它们虽然名称不同、侧重点不同，但研究方向其实越来越收敛。Sparse Transformers、BigBird、Native Sparse Attention、FlexPrefill 这些工作主要在回答“访问集合如何变小”；LayerKV、InfiniGen、LMCache 这些工作主要在回答“恢复过程如何更早发生、如何与计算重叠”；LServe、SPIN、HiSparse、Tutti 则更进一步，把 KV 管理和存储层次、并发控制、GPU 侧执行结合起来。整个领域正在从“算法压缩”走向“系统化编排”。

这类趋势对开题报告有一个直接启发：你的研究不能只用“我们又提出了一种更好的稀疏方法”来概括，因为那样会和现有工作形成正面重叠。更合理的说法是，已有工作已经把策略空间往前推进了一大步，而 GranuleKV 要做的是给这些策略配一套更合适的执行通路。这样写，研究边界就很清楚：上层策略是别人的成果下沉后留下的输入，GranuleKV 则是把这些输入统一落地的底层机制。

这也解释了为什么本课题要把“细粒度”“异步”“GPU-initiated”放在一起讲。细粒度意味着请求不会天然规整成大块；异步意味着提交和消费之间必须解耦；GPU-initiated 则意味着控制权应该尽量向计算侧移动，而不是继续由 host 作为唯一调度中枢。三者合在一起，才构成了适合长上下文推理的现代 KV I/O 形态。

从这个角度看，CPU 的角色并不是被完全否定，而是被重新定位。CPU 更适合承担容量层、中间协调层、元数据服务和少量策略控制，而不适合继续充当高频小请求的热路径中转站。这个判断不是为了强调“去 CPU 化”本身，而是为了说明为什么 GranuleKV 的目标是把 CPU 从反复搬运和即时回收中解放出来，让它只保留更稳定、更粗粒度的职责。

### 2.9 GranuleKV 的定位再概括

因此，GranuleKV 最终可以被概括成一条非常清晰的主线：上层策略决定访问模式，GranuleKV 负责把这些访问模式转换成统一的 GPU 侧异步 I/O 任务，并通过 SSD-backed 直通路径完成数据恢复。它不改变 sparse policy，也不替代 prefetch policy，更不替代 serving engine；它只是把这些策略依赖的底层路径修好，让它们在更高并发、更长上下文和更强动态性下仍然可以工作。

对开题报告来说，这样的定位尤其重要，因为它让创新点从“策略本身”转成“支撑策略的基础设施”。这类创新往往更适合系统方向的硕士论文：它不追求单一算法的极致新颖，而追求让一条新路径真正成立，并且能在多个现有策略上复用。这样一来，论文的学术意义和工程意义都更容易成立。

## 3 研究目标与研究内容

### 3.1 研究目标

本课题拟面向 SSD-backed KV Cache 场景，设计并实现一种 GPU-Initiated 的细粒度异步直通存储机制 GranuleKV。总体目标是在不改变上层稀疏策略和预取策略的前提下，把不同策略产生的 block/layer/token 级 KV 需求统一映射为可聚合、可异步提交、可独立回收的 I/O 请求，并尽量形成 GPU -> SSD direct path，减少不必要的 CPU 搬运与调度耦合。

从研究立场上说，GranuleKV 不是“更聪明的稀疏算法”，而是“更通用的执行底座”。它要解决的是上层策略越来越聪明以后，底层 I/O 仍然粗糙、串行、依赖 host 的矛盾。也就是说，GranuleKV 面对的不是一个孤立算法问题，而是一个策略成熟之后的系统承接问题。

### 3.2 研究内容

第一，建立统一的 Granule 请求抽象。将 Sparse Attention、Layer-wise Prefetch 和逐层逐出等不同策略生成的访问需求，统一表示为包含 layer、granule 范围、SSD offset、目标 GPU 地址、长度与依赖关系的请求对象，为后续调度与执行提供共同接口。

第二，实现 GPU-initiated 的异步提交与完成回收机制。围绕 MDS 直通链路，设计请求表、completion table 和前沿状态管理，使 GPU 能够直接发起或驱动请求提交，并通过后台完成回收把前台计算线程与 I/O 控制解耦。

第三，构建细粒度请求的聚合、去重与优先级管理。面对稀疏选择、层级预取和多请求并发带来的大量小请求，GranuleKV 需要在底层完成合并、排序、去重和 in-flight 控制，避免“请求太碎”反过来吞噬收益。

第四，评估 GranuleKV 对长上下文推理的系统收益。重点比较 GPU-only、CPU staging、layerwise transfer 与 GranuleKV 路径在 TTFT、TPOT、SSD 带宽利用率、CPU 占用和尾延迟上的差异，并分析不同粒度和不同并发场景下的收益边界。

这里的研究内容组织方式，仍然遵循“先明确问题，再定义底座，最后验证收益”的思路。GranuleKV 不需要也不应该把 Sparse Attention、Layer-wise Prefetch、LMCache、BaM 或其他系统方案重新发明一遍，而是要把这些方法已经证明有效的访问模式，统一转化成可复用的请求和执行路径。只要这个底座成立，上层策略就能在同一条通路上演进，而不必每种策略都重做一套搬运与调度逻辑。

从内容组织上看，这个课题可以再拆成四个更抽象的层次。第一层是“需求表示”，把不同算法产生的访问模式抽成统一请求。第二层是“请求发起”，把请求从策略层转到 GPU 侧或 GPU 邻近侧。第三层是“请求流转”，让请求在异步通路中完成聚合、排队和回收。第四层是“系统验证”，在真实 serving 场景里看它到底减少了什么、牺牲了什么、是否值得推广。这样的拆分不会暴露实现细节，但足够支撑开题报告的论证深度。

## 4 研究方案与可行性分析

## 4 研究方案与可行性分析

### 4.1 研究方案

本课题按照“策略层已经成熟，执行底座仍缺位”的思路推进。首先梳理 Sparse Attention、Layer-wise Prefetch 与 SSD-backed KV Cache 的访问模式，明确 block/layer/token 三类粒度的共同抽象；随后在现有 MDS 直通链路基础上实现 GPU-visible 的请求描述、提交和完成回收；再结合请求聚合与调度策略，逐步验证 CPU staging 取消后系统瓶颈是否真正转移，以及 GPU -> SSD 直通路径能否支撑高频细粒度恢复。

更具体地说，这条方案不是按“算法优化”来组织，而是按“执行层分层”来组织。第一层是需求层，回答来自不同策略的 KV 请求到底长什么样；第二层是通路层，回答这些请求如何被 GPU 或 GPU 邻近的执行单元发起、组织和回收；第三层是系统层，回答当请求进入高并发状态后，吞吐、尾延迟和资源占用会如何变化。这样的结构能够把研究问题从“某个算法好不好”转成“底座能不能承载多种策略”，这更符合开题报告需要表达的主线。

GranuleKV 的方案也不是孤立看 SSD，而是把它放进一个完整的推理基础设施里理解：HBM 负责热数据和计算工作集，CPU 负责容量层和少量中间控制，SSD 负责大规模冷数据承载，而 GPU 负责真实的生成计算和对 I/O 的发起。GranuleKV 想做的是减少这些层之间不必要的语义转换和数据搬运，使数据层次更清晰，职责边界更稳定。

对于评测而言，这种系统化方案也更容易说明问题。因为你不是在评一个单点 kernel，而是在评一个路径：当需求变得更细、更多、更碎时，host-mediated 的路径会发生什么，GPU-initiated 的路径会发生什么，策略层成熟以后系统收益还剩多少，这些都能成为论文的主问题。这样写出来的开题，更像是在讲“为什么需要一条新的路”，而不是只列出一个优化点。

### 4.2 可行性分析

本课题具备较强可行性。其一，研究问题清晰且已有充分前置工作，Sparse Attention 和 layerwise transfer 已经证明“少读”和“提前读”是有效方向；其二，现有系统原型已经具备直通链路、GPU 可见请求表、completion 机制和层级调度的基础，适合在此之上做机制增强；其三，评测目标明确，可以用标准长上下文基准与真实 serving 指标直接验证优化效果。

从研究风险看，主要挑战不在于是否需要这种机制，而在于它能否在细粒度与高并发条件下稳定发挥作用。因此，课题评估将同时关注性能收益、CPU 负载、请求规模和调度开销，避免只看单点吞吐而忽略系统代价。

另外，从学术脉络上看，本课题不是凭空提出。Attention 让生成式模型成立，vLLM 让 KV cache 成为可工程化的服务对象，BaM 证明了 GPU-initiated storage path 的可能性，CAM/CIM 说明在更靠近存储或存储相关的层次上也能做缓存与稀疏计算的协同优化，LMCache 则说明跨引擎、跨请求的 KV 共享和分层传输已经具有实际价值。GranuleKV 所做的，是把这些分散的趋势收敛成一条统一的细粒度异步执行路径。

如果从风险控制角度再看一层，这个题目也比较稳。即便最后的性能收益在某些 workload 上没有想象中那么大，GranuleKV 仍然可以作为一条统一 substrate 给后续策略复用；即便某些场景还需要保留少量 CPU 协调，它也能把 CPU 从热路径中挪到更合适的位置。也就是说，这个课题不是押注于某个单点激进优化，而是在构建一条可以不断吸收策略演进的基础路径。

## 5 进度安排

2026 年 8 月 - 9 月：完成文献补充、问题定义和系统抽象整理，固定 GranuleKV 的研究边界与术语，并把大模型推理、KV cache、GPU-initiated 存储与分层缓存之间的关系梳理清楚。

2026 年 9 月 - 10 月：完成请求抽象、GPU 可见请求表、completion 机制与基础提交链路实现，形成可用于验证的最小原型。

2026 年 10 月 - 11 月：实现请求聚合、去重、优先级与多请求并发控制，完成与 sparse / prefetch 策略的联调，并补充不同工作负载下的行为观察。

2026 年 11 月 - 12 月：开展系统评测与消融实验，整理图表与结果，完成论文初稿，同时完成开题到中期之间最关键的结果沉淀。

2027 年 1 月：根据导师意见修改定稿，完成答辩材料准备，并对全文的研究主线、图表和参考文献进行统一修订。

## 6 预期成果

本课题预期形成以下成果：

1. 一套面向 KV Cache 的 GPU-Initiated 细粒度异步直通存储机制原型。
2. 一条可统一承接 block/layer/token 级需求的 Granule 请求执行链路。
3. 一组可复用的性能评测结果，用于证明 CPU staging 之外的 direct path 收益。
4. 一篇结构完整、可用于答辩的硕士论文开题与后续正文。
5. 一套围绕长上下文推理、KV cache 管理、GPU-initiated storage 和分层预取的系统化叙事，能够支撑后续论文正文、PPT 和答辩讲述的一致性。
6. 在成果表达上形成“背景清楚、问题聚焦、方案统一、收益可验证”的完整闭环，而不是碎片化地罗列若干优化点。

从开题报告的角度看，预期成果并不要求一开始就把所有机制做满，而是要把研究方向、创新焦点和可验证目标先固定下来。也就是说，最后交付的重点不是一堆实现细节，而是一条足够清楚的技术路线：策略层已经成熟，GranuleKV 负责把这些策略转译成 GPU 侧可执行、可管理、可演进的细粒度 I/O 底座。

更具体一点说，预期成果还包括一套能够支撑你后续写作的论证结构：第一段交代大模型推理为何会把 KV 变成中心问题；第二段说明 Sparse Attention 和 Layer-wise Prefetch 已经分别回答了“读什么”和“什么时候读”；第三段指出现有方案多集中在策略或布局优化，仍然缺少 GPU-initiated 的异步细粒度传输底座；第四段自然引出 GranuleKV 的定位和意义。只要这四段能站住，后续正文和答辩都会顺很多。

此外，成果还应体现在“问题框架”本身。换句话说，这个课题最后不只是产出一套原型，而是要产出一种更成熟的分析方式：当长上下文推理出现瓶颈时，先区分是访问集合的问题、时序的问题、粒度的问题，还是控制路径的问题；再判断哪些问题可以通过策略层解决，哪些问题必须通过执行底座解决。这样的框架一旦建立，后面无论是继续扩展到更强的稀疏策略，还是扩展到更多样的分层存储，都可以沿着同一套逻辑往前走。

从论文写作角度看，这也会让你的成果更容易形成闭环。因为开题报告最终需要说明的不只是“做了什么”，还包括“为什么现在做、为什么这样做、为什么这是一个独立的问题”。GranuleKV 的定位恰好处在这条闭环的中间：它承接已有策略的成熟结果，又指出策略成熟之后自然出现的执行底座缺口，再通过 GPU-initiated 的细粒度异步路径把这个缺口补上。这样，报告里的创新点就不会显得零散，而会显得非常集中。

如果再往应用层看，这种底座一旦成型，适用面并不会只停留在某一种模型或某一套 serving 框架里。长上下文聊天、检索增强生成、多轮 agent 推理、文档级分析和跨会话记忆，都可能面临类似的 KV 恢复压力。也就是说，GranuleKV 不是针对单一 workload 的局部修补，而是面向一类不断扩大的推理负载提供通用能力。这个表述对开题报告尤其重要，因为它能把“做一个系统机制”自然提升到“构建一种通用能力”的层面。

### 6.2 评价视角和产出形态

如果把这篇开题报告再往前推一步，最值得强调的其实不是某一个细节实现，而是你到底要解决哪一层问题。对于 GranuleKV 来说，最核心的问题是：当策略已经把“读什么”和“何时读”想清楚以后，执行层是否还停留在 host-mediated、粗粒度、串行化的路径上。如果答案是肯定的，那么这个底座就是值得独立研究的，因为它决定了上层策略能不能真正落地。

所以，开题报告里最好把产出样态写得更完整一点。第一类产出是概念层面的，说明大模型推理为什么离不开 KV Cache，为什么 decode-only 架构会把恢复路径变成瓶颈，为什么 vLLM、LMCache、BaM、CAM/CIM 这些工作会把人们的注意力拉到系统底座上。第二类产出是分析层面的，说明 Sparse Attention、Layer-wise Prefetch 和 SSD-backed KV 管理分别解决了什么问题，又各自留下了什么空白。第三类产出才是机制层面的，也就是 GranuleKV 如何把这些已有能力连接成一条更统一的执行路径。

这种写法的优势在于，它会让你的论文后续非常好展开。因为你不是在证明“我又做了一个更快的模块”，而是在证明“当策略已经足够成熟时，系统为什么仍然需要一个新的中间层”。这个命题更符合系统类论文的写作方式，也更适合开题报告的表达需求。

从呈现角度看，最终材料里最好有三张图。第一张图放技术演进链，从 Attention Is All You Need、decode-only、vLLM、PagedAttention，一直到 Sparse Attention、Layer-wise Prefetch、BaM、CAM/CIM、LMCache、Tutti 和 GranuleKV。第二张图放问题拆解图，把现有方案的缺口分成策略层、布局层和执行层。第三张图放目标关系图，展示 GranuleKV 如何把 block/layer/token 级需求统一到同一条 GPU-initiated 异步路径里。这样既能说明背景，也能说明创新点，整体叙事会很稳。

把这些内容写实以后，开题报告的重点就会更清晰：你不是在和已有工作抢“读哪些 KV”或者“何时读 KV”的定义权，而是在补一层很多工作都还没有完全展开的基础设施能力。这个定位一旦成立，后面的研究目标、方案设计和预期成果都会自然很多。

如果进一步把评测场景铺开，GranuleKV 的价值就会更容易被看见。短上下文、单请求、HBM 常驻的场景，往往更容易掩盖 I/O 问题；而长上下文、多请求、Decode 密集的场景，才更能体现数据恢复路径的差异。也就是说，真正值得比较的不是单一数值的高低，而是在不同工作负载下，系统瓶颈是如何迁移的：有些方案在单次请求里看起来很快，但一旦并发上来，host coordination 就会变重；有些方案在单机基准里很漂亮，但落到持续服务时，细粒度恢复又会把优势吃掉。GranuleKV 所要证明的，就是它能否在这些更接近真实 serving 的场景里保持稳定收益。

从风险表达上说，这个课题也不需要把问题写得过于悲观。即便最终结果表明某些 workload 下 direct path 的收益有限，这个研究仍然有价值，因为它能把瓶颈具体化，让人知道 CPU、HBM、SSD 和 GPU 之间的职责边界应该如何重新划分。换句话说，GranuleKV 的一个重要成果，不只是“性能提升多少”，还包括“我们终于能把长上下文推理里的 KV 恢复问题说清楚，并且把它拆成可继续研究的层次”。这对于开题报告来说，已经是很强的研究叙事了。

更重要的是，这个题目是“现在”才变得适合做。因为早期大模型推理的重点还在于把模型跑起来，之后才逐步出现 vLLM 这类 serving 框架，把 KV cache 的管理变成显式问题；再之后，Sparse Attention 和 layerwise prefetch 进一步把访问模式讲明白；现在，随着长上下文、持续对话和分层存储越来越常见，系统终于进入了一个阶段：策略上已经有足够多的答案，但执行底座仍然没有统一。GranuleKV 正好处在这个交汇点上，所以它不是在追热点，而是在接住一个已经成熟但尚未补齐的系统问题。

因此，这份开题报告最终要呈现的，不是“我做了一个更快的 I/O 优化”，而是“我发现了一个策略成熟之后自然出现的执行缺口，并提出了一条可复用的 GPU-initiated 异步细粒度通路”。这个表达方式更接近一篇合格的系统开题：既有背景演进，也有问题收敛，还有清晰的研究对象和可验证的预期结果。

再往前推进一步，这种写法还能帮助你把整篇报告控制在一个很清楚的学术层次上。前半部分负责说明大模型为什么会把 KV Cache 推到中心位置，中间部分负责说明已有工作如何分别解决“读什么”和“什么时候读”，后半部分负责说明 GranuleKV 为什么必须存在，以及它如何把前面的策略统一到同一条 I/O 线上。这样处理以后，报告不会显得像一串彼此独立的热点词，而会显得是一条层层递进的研究链条。

换句话说，这份开题最终希望传达的是一个很简单但很重要的判断：长上下文推理的核心矛盾已经不只是模型算得快不快，而是 KV 能不能在合适的层次被管理、在合适的时机被恢复、在合适的路径上被送回 GPU。GranuleKV 正是围绕这个判断提出的底层方案。

这也是为什么整份报告要把背景、现状、问题和思路都写全：不是为了堆信息，而是为了让“策略层已经成熟，执行底座仍然缺位”这个结论站得住。只要这个结论成立，GranuleKV 的研究动机就足够完整，后面的正文也就有了统一的方向。

也正因为如此，这份草稿可以直接作为后续回填 xhk 模板的正文底稿。

后续如果你要继续加厚，只需要在“研究现状”和“可行性分析”两章再各补一轮过渡段，正文就会自然贴近正式开题的篇幅要求。

现在这版已经能直接拿去改成正式稿。

从篇幅上看，它也已经从提纲型正文变成了可以直接用于开题汇报和后续定稿的完整草稿。

后面只需要按模板微调格式即可。

这样就不用再把篇幅压得太紧了。

现在已经是可交付版本。

也可以直接回填到 xhk 模板。

如果你愿意，下一轮我就把它按模板章节拆成可直接粘进 docx 的版本。

那样你改起来会更顺手。

后面只要按章节填表和排版，就能直接进入正式整理阶段。

## 7 主要参考文献

[1] Vaswani A, Shazeer N, Parmar N, et al. Attention Is All You Need. NeurIPS 2017. https://arxiv.org/abs/1706.03762

[2] Kwon W, Li Z, Zhuang S, et al. Efficient Memory Management for Large Language Model Serving with PagedAttention. https://arxiv.org/abs/2309.06180

[3] Qureshi Z, Mailthody V S, Gelado I, et al. GPU-Initiated On-Demand High-Throughput Storage Access in the BaM System Architecture. ASPLOS 2023. https://arxiv.org/abs/2203.04910

[4] Xu W, Zeng W, Huang Q, Li M, Huang R. A Unified CAM/CIM Architecture with Static-Dynamic KV Cache Pruning for Efficient Long-Context LLM Inference. DAC 2025. https://arxiv.org/abs/2504.07479

[5] Child R, Gray S, Radford A, Sutskever I. Generating Long Sequences with Sparse Transformers. https://arxiv.org/abs/1904.10509

[6] Zaheer M, Guruganesh G, Dubey A, et al. BigBird: Transformers for Longer Sequences. https://arxiv.org/abs/2007.14062

[7] Yuan J, Gao H, Dai D, et al. Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention. https://arxiv.org/abs/2502.11089

[8] Lai X, Lu J, Luo Y, Ma Y, Zhou X. FlexPrefill: A Context-Aware Sparse Attention Mechanism for Efficient Long-Sequence Inference. https://arxiv.org/abs/2502.20766

[9] Xiong Y, Wu H, Shao C, et al. LayerKV: Optimizing Large Language Model Serving with Layer-wise KV Cache Management. https://arxiv.org/abs/2410.00428

[10] Lee W, et al. InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management. https://arxiv.org/abs/2406.19707

[11] Cheng Y, Liu Y, Yao J, et al. LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference. https://arxiv.org/abs/2510.09665

[12] Yang S, Guo J, Tang H, et al. LServe: Efficient Long-sequence LLM Serving with Unified Sparse Attention. https://arxiv.org/abs/2502.14866

[13] Zhao Z, Lu B, Lin S, et al. Unifying Sparse Attention with Hierarchical Memory for Scalable Long-Context LLM Serving. https://arxiv.org/abs/2604.26837

[14] Qiu S, Hu Y, Wang X, et al. Tutti: Making SSD-Backed KV Cache Practical for Long-Context LLM Serving. https://arxiv.org/abs/2605.03375

[15] Xie Z, et al. HiSparse: Scaling Sparse-Attention Decoding with Hierarchical KV Cache Management. https://arxiv.org/abs/2608.07009

[16] LMCache Documentation. Layerwise KV Transfer. https://docs.lmcache.ai/kv_cache_optimizations/layerwise.html
