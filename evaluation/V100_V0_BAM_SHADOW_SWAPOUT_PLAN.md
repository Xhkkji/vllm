# V100 + vLLM V0 + BaM Shadow Swap-Out 第一版方案

本文档整理 `BaM_IOStack` 接入 `vLLM V0` 的第一版落地方案。该方案的目标不是立刻替换 `vLLM` 当前的 `CPU swap`，而是在 **不破坏现有运行语义** 的前提下，先打通 `GPU -> SSD(BaM)` 的旁路写出通路。

## 当前状态

这条第一版路线已经在本地跑通，且已经确认发生了真实的 `swap_out -> BaM shadow write`。

当前最小成功配置：

- 模型：`/home/xhk/model/Qwen3-0.6B`
- `num_prompts=24`
- `prompt_len=6144`
- `max_tokens=1024`
- `temperature=0.8`
- `best_of=4`
- `max_model_len=8192`
- `gpu_memory_utilization=0.16`
- `swap_space=16`
- `preemption_mode=swap`
- `max_num_seqs=8`

对应成功日志：

- [v0_swap_trace_Qwen3-0.6B_20260610_202648.log](/home/xhk/llm-inference/vllm/evaluation/logs/v0_swap_trace_Qwen3-0.6B_20260610_202648.log:1)

从这份日志中已经能看到完整链路：

- `Scheduler` 触发 `op=preempt`
- `BlockManager` 触发 `op=swap_out`
- `Worker.execute` 执行 `swap_out`
- `BaM shadow writer` 记录 `[BAM_SHADOW] swap_out_shadow`

例如第一轮事件就已经完整出现：

- [Scheduler preempt](/home/xhk/llm-inference/vllm/evaluation/logs/v0_swap_trace_Qwen3-0.6B_20260610_202648.log:660)
- [BlockManager swap_out](/home/xhk/llm-inference/vllm/evaluation/logs/v0_swap_trace_Qwen3-0.6B_20260610_202648.log:661)
- [Worker execute swap_out](/home/xhk/llm-inference/vllm/evaluation/logs/v0_swap_trace_Qwen3-0.6B_20260610_202648.log:664)
- [BaM shadow write](/home/xhk/llm-inference/vllm/evaluation/logs/v0_swap_trace_Qwen3-0.6B_20260610_202648.log:666)

当前成功日志汇总结果：

- `swap_out_shadow` 次数：`24`
- 累计 `mappings`：`9835`
- 累计写入量：`18,047,303,680 bytes`
- 累计写入量约：`16.81 GiB`
- 单次平均写入量约：`0.70 GiB`
- 单次平均 `mappings` 约：`409.79`
- 单次平均写入耗时约：`556.52 ms`
- 单次平均写入带宽约：`1.26 GiB/s`

一个重要经验是：仅靠增大并发请求数，很多时候只会形成 waiting queue；要更稳定地制造 `swap_out`，需要让单个请求组内部形成多分支运行态，因此这次成功配置里的 `temperature=0.8 + best_of=4` 是关键。

## 方案定位

第一版采用 `shadow swap-out` 思路：

- 保留 `vLLM` 原生 `GPU -> CPU` 的 `swap_out`
- 在同一次 `swap_out` 之后，额外将同批 KV block 再写一份到 `SSD(BaM)`
- `swap_in` 暂时完全不改，仍然走原来的 `CPU -> GPU`

因此，这一版的 `BaM` 角色不是新的主数据源，而是：

- 一个影子写出后端
- 一个可观测、可测量的 `GPU -> SSD` 数据通路原型

## 为什么第一版要这样做

当前 `vLLM V0` 的 `swap` 不只是“搬数据”，还同时包含：

- `Scheduler` 决定何时 preempt
- `BlockManager` 修改 block 的逻辑归属设备
- `Worker` 下发本轮 `swap_in / swap_out` 映射
- `CacheEngine` 真正执行搬运

尤其是 `BlockManager.swap_out()` 当前语义是：

- 逻辑上把 block 从 `Device.GPU` 换到 `Device.CPU`
- 更新 block allocator 和 block table

如果第一步就直接改成“只写 SSD，不落 CPU”，那么需要同时修改：

- block allocator 的设备模型
- `can_swap_in / can_swap_out`
- `SWAPPED` block 的来源设备语义
- `swap_in` 的恢复路径
- block table 中 physical block id 的含义

这会让第一步跨度过大，调试成本也会明显升高。

因此更稳妥的顺序是：

1. 先保留原始 `CPU swap` 行为
2. 额外接入 `GPU -> SSD(BaM)` 影子写出
3. 先解决 block 布局、地址映射、吞吐测量
4. 后续再考虑真正的 `SSD -> GPU swap_in`

## 第一版的明确目标

第一版只验证下面四件事：

1. `GPU -> SSD(BaM)` 写出通路能稳定跑通
2. SSD 上的 block 布局和映射关系清晰、可追踪
3. 可以记录每批 `swap_out` 的写出耗时和带宽
4. 不影响当前 `vLLM` 基于 `CPU swap` 的正常推理结果

第一版 **不要求**：

- 替换当前 `CPU swap`
- 从 `SSD` 执行 `swap_in`
- 实现异步预取
- 改动 scheduler 策略
- 做端到端加速结论

## 当前 vLLM 中对应的主链路

当前 `swap_out` 主链路可以概括为：

`Scheduler -> BlockManager -> Worker.prepare -> Worker.execute -> CacheEngine.swap_out`

第一版建议只在这条链路的最末端增加一个旁路动作：

`CacheEngine.swap_out(GPU->CPU)` 结束后，再触发一次 `BaM shadow write`

换句话说，主链路不变，只是在 `CacheEngine` 内部追加一个“把本轮换出的 block 再写到 SSD”的动作。

## 推荐的数据通路形态

第一版建议使用下面的逻辑：

1. `BlockManager` 仍返回当前已有的 `GPU physical block id -> CPU physical block id` 映射
2. `Worker.execute()` 仍按原逻辑调用 `cache_engine.swap_out(...)`
3. `CacheEngine.swap_out(...)` 在完成原生 `GPU -> CPU` 搬运后，调用一个新的 `BaM shadow writer`
4. `BaM shadow writer` 读取本轮换出的 block 内容，并写入 SSD
5. 同时记录：
   - 本批 `mappings`
   - `block_bytes`
   - `total_bytes`
   - `elapsed_ms`
   - `GiB/s`

## 第一版推荐的模块边界

为了让后续真正接 `BaM` 更顺，建议先在设计上把新增逻辑拆成三类角色。

### 1. `BaMShadowConfig`

职责：

- 控制是否开启 shadow 写出
- 配置 SSD 目标路径或目标设备
- 配置 block 对齐大小
- 配置是否同步写出
- 配置 batch 写入参数

第一版它只是一组配置，不参与调度决策。

### 2. `BaMBlockStore`

职责：

- 管理 `vLLM block -> SSD offset` 映射
- 为新写出的 block 分配 SSD 空间
- 维护 block 的落盘元数据
- 后续为 `swap_in` 扩展 `read` 接口

第一版只需要“写路径”，因此它至少应支持：

- 为一批 block 分配连续或可追踪的 offset
- 记录每个 block 当前对应的 SSD 位置

第一版不需要复杂回收，可以先采用简单的 append-only 方式。

### 3. `BaMShadowWriter`

职责：

- 接收一次 `swap_out` 的 block 批次
- 将这些 block 组织成可提交给 `BaM` 的写入请求
- 执行写出
- 统计并记录耗时、吞吐、映射范围

它是第一版真正的数据面入口。

## SSD 上的最小数据单位

第一版建议直接以 **单个 vLLM physical block** 作为最小写入单位。

原因：

- 当前测得 `block_bytes ≈ 1.75 MiB`
- 这个粒度已经比较适合 SSD 大块写入
- 与现有 `ms/block` 基线口径天然一致
- 不需要第一版就引入额外的 page packing 复杂度

因此第一版可以直接约定：

- 一个 `vLLM block` 对应 SSD 上一段定长区域
- 一次 `swap_out` 的 `mappings` 作为一个批量写请求

## SSD 地址分配建议

第一版建议用最简单的 append-only 布局：

- 第 0 个 block 写到 `offset = 0`
- 第 1 个 block 写到 `offset = block_bytes`
- 第 2 个 block 写到 `offset = 2 * block_bytes`
- 以此类推

优点：

- 简单、稳定、便于调试
- 容易做日志核对
- 不需要第一版就实现 free list 或块复用

缺点：

- 长时间运行会增长文件或地址空间

但这对第一版不是问题，因为第一版的目标本来就不是长期在线运行，而是验证通路。

## 第一版的插入点建议

从代码结构上看，最合适的插入点是在 `CacheEngine.swap_out(...)` 内部。

原因：

- `Worker.execute()` 已经知道本轮有多少 `mappings`
- `CacheEngine` 才持有真正的 `gpu_cache / cpu_cache`
- block 数据视图的组织在这里最自然

因此逻辑上建议是：

1. `Worker.execute()` 按原样调用 `cache_engine.swap_out(...)`
2. `CacheEngine.swap_out(...)` 先完成当前 `GPU -> CPU`
3. 若开启 `bam shadow`，则立刻调用 `BaMShadowWriter.on_swap_out(...)`

这样上层调度和 block 状态完全不需要知道 `BaM` 的存在。

## 第一版的数据来源选择

这里有一个重要设计点：影子写 SSD 的数据，到底从哪里取。

### 路线 A：从 `CPU cache` 取数据再写 SSD

优点：

- 更容易做
- 可以先把 SSD 布局、映射、日志链路打通
- 不会第一步就卡在 GPU 侧直写接口细节

缺点：

- 这更像“CPU 中转后写 SSD”
- 不能完整体现 `BaM` 的研究目标

### 路线 B：直接从 `GPU cache` 写 SSD

优点：

- 更符合 `BaM` 的设计方向
- 更接近未来真正的 `GPU -> SSD` 直达通路

缺点：

- 接口和调试复杂度更高

因此第一版文档建议采用分阶段思路：

- `v1a`：允许先从 `CPU cache` 做 shadow 写，验证元数据和 SSD 通路
- `v1b`：切换到真正的 `GPU -> SSD(BaM)` 数据源

如果你已经确认 BaM 接口足够清晰，也可以直接从 `v1b` 起步。

## 日志设计建议

第一版建议新增一个独立日志前缀，例如：

- `[BAM_SHADOW] op=swap_out_submit`
- `[BAM_SHADOW] op=swap_out_done`

每次至少记录：

- `mappings`
- `block_bytes`
- `total_bytes`
- `elapsed_ms`
- `gib_per_sec`
- `first_block_id`
- `last_block_id`
- `ssd_offset_begin`
- `ssd_offset_end`

这样后续可以直接与当前 CPU 基线对齐比较。

## 第一版成功标准

第一版建议用下面的标准判断是否“做成了”：

1. 开启 `bam shadow` 后，`vLLM` 推理结果与当前版本一致
2. 在触发 `swap_out` 的 workload 下，日志中能稳定看到 `BaM shadow write` 事件
3. SSD 侧能确认写出了对应 block 数据
4. 可以统计每轮写出的 `ms/block` 和 `GiB/s`

只要满足这四条，就可以认为第一版已经完成了“可运行的 `GPU -> SSD` 影子写出原型”。

## 与当前 CPU swap 基线的关系

当前已经测得 `CPU swap` 的关键参考值：

- 大批量单向 `swap_out`：约 `0.42 ms/block`
- 大批量单向 `swap_in`：约 `0.42 ms/block`
- 往返 `round_trip`：约 `0.84 ms/block`
- 有效带宽：约 `4.0 GiB/s`

第一版 `BaM shadow swap-out` 跑通后，最先要比较的就是：

- `GPU -> SSD` 的单向 `ms/block`
- `GPU -> SSD` 的有效写带宽
- 批量写出时的稳定性

这一步先不比较端到端吞吐，只比较“写出这条通路本身”。

## 第一版之后的自然下一步

当 shadow `swap_out` 跑通后，后续最自然的两条路是：

### 路线 1：先做写出基线完善

- 系统化测量不同 batch 大小下的 `GPU -> SSD` 写出成本
- 与当前 `CPU swap_out` 基线做对照

### 路线 2：开始设计 `SSD -> GPU swap_in`

- 先做同步 `swap_in`
- 后续再做异步预取和调度重叠

建议顺序是：

1. 先稳住 `swap_out`
2. 再做 `swap_in`
3. 最后才碰异步和 scheduler

## 本文档对应的总建议

一句话总结第一版方案：

**不要在第一步改掉 `vLLM` 当前的 `CPU swap` 语义，也不要一开始就重写 scheduler；先在 `CacheEngine.swap_out()` 后面增加一个 `BaM shadow writer`，把每批被换出的 KV block 额外写一份到 SSD，并把这条路径的 `ms/block` 与 `GiB/s` 测出来。**

这就是当前阶段最稳妥、最清晰、最容易验证的第一版接入路线。
