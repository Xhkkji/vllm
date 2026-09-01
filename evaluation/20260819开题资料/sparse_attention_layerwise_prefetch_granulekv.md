# Sparse Attention、层级预取与 GranuleKV 研究现状

> 更新时间：2026-08-20
>
> 本文用于开题 PPT 的研究现状、问题分析和创新点组织。核心目标不是把系统描述成某一种 Sparse Attention 实现，而是将 GranuleKV 定位为面向多种 KV 调度策略的统一细粒度异步 I/O 底座。

## 一、核心判断

长上下文推理中的 KV Cache 优化大致形成了三条技术路线：

```text
Sparse Attention
    └─ 减少真正参与计算的 KV 数量

Layer-wise Prefetch
    └─ 将 KV 恢复与 Transformer 层间计算重叠

Sparse Attention + Layer-wise Prefetch
    └─ 只恢复重要 KV，并根据层间访问顺序提前恢复下一批 KV
```

这些工作已经证明了“减少 KV 访问量”和“隐藏 KV 恢复开销”的有效性。因此，不能再将创新点简单表述为“首次结合 Sparse Attention 与层级预取”，也不宜仅表述为“将 BaM 迁移到 KV Cache”。

更合适的系统定位是：

> **GranuleKV：面向 SSD-backed KV Cache 的策略无关细粒度异步 I/O 底座。**

GranuleKV 不绑定具体的稀疏算法或预取算法，而是将不同策略生成的 KV 访问需求统一转换为细粒度、可聚合、可异步提交和可独立回收的 SSD-GPU I/O 请求。

## 二、PPT 推荐组织方式

建议将这一页组织成“四个框”：

```text
┌────────────────┐  ┌────────────────┐  ┌────────────────────┐  ┌────────────────┐
│ Sparse Attention│  │ Layer Prefetch │  │ Sparse + Prefetch  │  │ GranuleKV      │
│ 减少访问量      │  │ 隐藏恢复开销    │  │ 联合优化工作集      │  │ 统一细粒度底座  │
└────────────────┘  └────────────────┘  └────────────────────┘  └────────────────┘
```

前三个框介绍已有研究方向，第四个框说明当前工作的切入点：已有方法主要解决“访问哪些 KV”或“何时恢复 KV”，GranuleKV 进一步解决“如何将这些动态访问高效地组织成 SSD I/O”。

## 三、框一：Sparse Attention

### 3.1 核心内容

Sparse Attention 只对当前 Query 需要的部分历史 KV 进行 Attention 计算，从而减少计算量、显存工作集和 KV 读取量。

典型流程为：

```text
历史 KV Cache
      ↓
重要性评估 / Token 选择 / Block 选择
      ↓
只对选中的 KV 执行 Attention
```

### 3.2 代表性成果

- **Sparse Transformers**：较早系统化探索稀疏注意力模式，通过局部、跨步或固定结构减少 Attention 的全连接访问。
- **BigBird**：结合局部窗口、随机连接和全局 Token，建立具有理论表达能力的稀疏注意力结构。
- **Native Sparse Attention，ACL 2025**：结合粗粒度压缩和细粒度选择，形成可训练的原生稀疏注意力机制。
- **FlexPrefill，ICLR 2025**：根据输入和注意力头动态确定稀疏模式与计算预算。
- **HieraSparse，2026**：利用层次化 KV 压缩和半结构化稀疏计算降低 KV 规模与 Attention 开销。

### 3.3 现有不足

Sparse Attention 主要回答的是：

> **哪些 KV 需要参与计算？**

但它不自动解决：

1. 被选择的 KV 位于 SSD 时，如何以细粒度方式恢复；
2. 动态、离散的 Token/Block 选择如何合并成高效 SSD 请求；
3. 选择结果如何与 HBM 工作集、层间计算和多请求调度协同；
4. 如果 KV 仍然全量驻留 HBM，长上下文的容量问题仍然存在。

因此，Sparse Attention 可以减少“需要读取的数据量”，但还需要一个能够高效执行这些不规则访问的 I/O 底座。

## 四、框二：Layer-wise Prefetch

### 4.1 核心内容

层级预取按照 Transformer 层的执行顺序组织 KV 恢复：计算第 `i` 层时，提前提交第 `i+1` 层的 KV 读取请求。

```text
计算 Layer i       ────────────────┐
                                    ├─ 计算与 I/O 重叠
预取 Layer i+1  ────────────────┘
                                    ↓
                         Layer i 完成后直接使用
```

理想情况下，总时间从：

```text
IO + Compute
```

降低为：

```text
max(IO, Compute) + 未隐藏的等待时间
```

### 4.2 代表性成果

- **LayerKV，2024**：以 Layer 为单位管理、分配和卸载 KV Cache。
- **InfiniGen，OSDI 2024**：预测下一阶段真正重要的 KV，并提前预取，减少不必要的数据移动。
- **LMCache Layerwise Transfer**：按层传输 KV，使下一层计算不必等待全部 KV 恢复。
- **Tutti，2026**：面向 SSD-backed KV Cache，通过 GPU-centric I/O 和按层调度重叠 SSD 恢复与模型计算。

### 4.3 现有不足

层级预取主要回答的是：

> **什么时候恢复 KV？**

但仍可能存在以下问题：

1. 数据组织通常以 Layer、Chunk 或较大的 KV Object 为单位；
2. 预取一个层或一个 Chunk 时，可能读入当前层中暂时不需要的 KV；
3. 单请求的 GPU working set 仍可能较大，限制多请求并发；
4. 当 SSD I/O 粒度进一步缩小时，CPU submit、完成回收和请求管理开销会变得突出；
5. 单请求的 overlap 收益受到 `max(IO, Compute)` 上限约束，不能单独解决所有尾延迟问题。

因此，层级预取解决了“何时读取”的问题，但还没有完全解决“以多小的粒度读取”和“如何高效管理大量小请求”的问题。

## 五、框三：Sparse Attention + Layer-wise Prefetch

### 5.1 核心内容

这一方向将两个思路结合起来：

```text
Sparse Attention：决定当前层需要哪些 KV
Layer-wise Prefetch：根据层间执行顺序提前恢复下一层 KV
```

完整流程可以表示为：

```text
Layer i 当前计算
      ↓
预测 Layer i+1 的重要 KV
      ↓
只预取选中的 Token/Block
      ↓
当前层结束后逐出已使用 KV
      ↓
GPU 维持有限大小的滑动工作集
```

### 5.2 代表性成果

- **SolidAttention，FAST 2026**：联合动态稀疏注意力与 SSD 存储管理，使用 KV Block 和 speculative prefetch，面向 SSD-backed KV Cache 减少恢复开销。
- **SPIN，2026**：统一不同稀疏粒度的 KV 分区，并结合 HBM-CPU 分层缓存和工作集管理。
- **HiSparse，2026**：在 GPU 上维护有限 KV 工作集，执行选择、替换和恢复，并利用跨层选择相似性执行 Layer-wise Prefetch。
- **ECHO，OSDI 2026**：面向原生 Sparse Attention，支持动态 KV 淘汰、恢复以及 intra-query/inter-query lossless prefetch。

### 5.3 现有不足与可区分空间

这些工作说明“Sparse Attention + Layer-wise Prefetch”已经成为重要发展方向，但从系统切入角度仍可区分出以下问题：

1. **存储层次不同**：部分工作主要面向 HBM-DRAM 或 HBM-Host Memory，不能直接代表 SSD 级随机 I/O 已经解决。
2. **物理粒度仍可能偏粗**：面向 SSD 的方案通常仍然围绕 KV Block、Chunk 或较大的存储对象组织。
3. **策略与后端耦合**：稀疏选择、预取、驱逐和存储搬运往往在同一个执行框架中绑定实现，难以复用同一底层 I/O 机制。
4. **动态不规则请求管理不足**：多个 Layer、多个请求和多个稀疏选择结果同时到达时，需要请求聚合、去重、限流、完成回收和优先级调度。
5. **SSD 控制面开销仍然存在**：即使数据面支持直达 GPU，CPU 发起大量细粒度请求时仍可能受到 submit、SQ/CQ 管理和同步等待影响。

因此，GranuleKV 的切入点不是重新提出一种 Sparse Attention，而是补齐：

> **从动态 KV 选择结果到 SSD-GPU 细粒度异步 I/O 执行之间的系统抽象和执行机制。**

## 六、GranuleKV 的核心定位

### 6.1 统一抽象

上层策略不直接操作 SSD，而是统一生成 KV granule 请求：

```text
GranuleRequest = {
    request_id,
    layer_id,
    block/token range,
    SSD offset,
    destination KV address,
    length,
    dependency / priority
}
```

不同策略可以复用同一接口：

```text
Layer Prefetch Scheduler  ─┐
Sparse Attention Scheduler ├─> GranuleKV submit()
Layer Eviction Scheduler  ┤
Multi-request Scheduler    ─┘
```

底层统一负责：

```text
请求聚合与去重
SSD offset 映射
in-flight 请求控制
后台 CQ 轮询
完成状态发布
直接写入 GPU KV Cache
```

### 6.2 当前系统已经具备的基础

当前 BaM direct KV 路径已经形成了 GranuleKV 的底层雏形：

- GPU-visible request table：请求描述可以由 GPU tensor 表达；
- completion table：按请求或按 chunk 发布完成状态；
- frontier table：表达提交、读取、缓存和可消费等阶段前沿；
- `kv_worker_submit`：为 KV 专用 worker 提供统一提交入口；
- 后台 CQ service：将完成回收从前台计算线程中解耦；
- direct KV path：SSD 数据直接恢复到 vLLM GPU KV Cache，避免 CPU payload staging；
- layerwise scheduler：已经能够验证按 Layer 组织预取并与计算重叠。

当前实现的准确定位是：**GranuleKV 的接口与执行底座已经建立，但完整的任意 Token/Block 稀疏请求、多请求 QoS 和真正 GPU-side request generation 仍需要继续验证。**

## 七、GranuleKV 可以进一步解决的问题

### 7.1 统一不同 KV 策略的数据通路

现有方法通常针对某一种算法设计专用的 cache 管理和数据搬运逻辑。GranuleKV 将 Layer Prefetch、Sparse Attention、逐层逐出和多请求调度统一成同一种请求格式，使上层策略可以独立演进。

### 7.2 支持比 Chunk/Layer 更细的 SSD 访问

针对 Sparse Attention 产生的非连续选择结果，GranuleKV 可以进一步支持：

```text
连续 Block 合并
离散 Block 聚合
重复请求去重
按 Layer 分组
按请求优先级调度
部分 Block 或 Token 范围恢复
```

### 7.3 降低细粒度 I/O 的控制面开销

大量小请求不能简单依赖前台线程逐个 submit 和等待。GranuleKV 的重点是将：

```text
submit 聚合
in-flight 管理
CQ 轮询
完成回收
地址映射
```

收束到后台异步执行路径中，从而减少线程与数据、请求与完成之间的强耦合。

### 7.4 支持长上下文下的 GPU 滑动工作集

对于超过 GPU KV Cache 容量的上下文，可以只在 HBM 中保留有限的 Layer/Block 工作集：

```text
计算当前层
    ↓
预取下一层需要的 KV
    ↓
当前层计算完成
    ↓
逐出已经使用的 KV
    ↓
继续处理下一层
```

这为“单条上下文整体无法驻留 HBM，但仍然能够完成推理”提供系统基础。

### 7.5 支持多请求细粒度调度

在多轮对话和多用户场景中，不同请求的 KV 工作集会同时竞争 HBM、SSD 带宽和 I/O 队列。GranuleKV 可以进一步加入：

- 请求级优先级；
- Layer/block 级 residency 管理；
- SSD 带宽和 in-flight 配额；
- 热 KV 保留与冷 KV 逐出；
- 多请求之间的 I/O 合并和公平调度。

## 八、当前创新点建议

### 8.1 论文/开题版

> **面向 SSD-backed KV Cache 的策略无关细粒度异步 I/O 系统。针对层级预取、Sparse Attention 和多请求 residency 管理产生的非连续 KV 访问，设计统一的 KV granule 描述与异步提交接口，并通过请求聚合、后台 CQ 回收和 GPU 直接恢复，实现不同 KV 调度策略对同一条高并发 SSD-GPU I/O 链路的复用。**

### 8.2 PPT 精简版

> **GranuleKV 将层级预取、Sparse Attention 和逐层逐出统一转换为细粒度 KV I/O 请求，使不同 KV 调度策略共享同一条高并发 SSD-GPU 异步数据通路。**

### 8.3 避免过度承诺的版本

> **现有系统已完成面向 Layer/Block 的细粒度异步 KV I/O 接口和层级预取验证，后续将扩展到稀疏选择结果的请求聚合、多请求 residency 管理和 Token/Block 子集恢复。**

## 九、建议的验证路线

为了证明 GranuleKV 不是简单的后端迁移，建议按三层实验推进：

### 第一层：I/O 机制验证

固定同一组 KV 访问请求，对比：

- HBM 常驻；
- 原生 Chunk/Full restore；
- GranuleKV 细粒度 restore；
- 不同请求数量和不同 in-flight 深度。

关注指标：

- Restore p50/p95；
- 有效 SSD 带宽；
- 实际读取量与理论读取量；
- submit/cq 回收开销；
- 请求完成顺序和正确性。

### 第二层：策略复用验证

让不同上层策略生成请求，但使用同一底层接口：

```text
Layer Prefetch  → Layer granule requests
Sparse Attention → Selected block/token requests
Layer Eviction  → Release / residency requests
```

证明三种策略不需要分别实现一套 SSD 读路径。

### 第三层：真实推理验证

在长上下文和多用户场景中比较：

- Full restore；
- Layerwise Prefetch；
- Sparse restore；
- Sparse + Layerwise Prefetch；
- 多请求滑动工作集。

关注指标：

- TTFT；
- TPOT p95；
- decode stall；
- HBM 峰值占用；
- 同时服务请求数；
- SSD 读取放大；
- 端到端吞吐。

## 十、参考成果

- Sparse Transformers，2019。
- BigBird，NeurIPS 2020。
- LayerKV，2024，arXiv:2410.00428。
- InfiniGen，OSDI 2024。
- Native Sparse Attention，ACL 2025。
- FlexPrefill，ICLR 2025。
- LMCache Layerwise Transfer，LMCache Documentation。
- Tutti，2026，arXiv:2605.03375。
- SolidAttention，FAST 2026。
- SPIN，2026，arXiv:2604.26837。
- HiSparse，2026，arXiv:2608.07009。
- ECHO，OSDI 2026。

