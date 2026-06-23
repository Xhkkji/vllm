# V100 + vLLM V0 + BaM Shadow Swap-Out / Swap-In 方案

本文档整理 `BaM_IOStack` 接入 `vLLM V0` 的当前落地方案。该方案的目标不是立刻替换 `vLLM` 当前的整套 `CPU swap` 机制，而是在 **不破坏现有运行语义** 的前提下，先打通：

- `GPU -> SSD(BaM)` 的 shadow `swap_out`
- `SSD(BaM) -> GPU` 的实验性 `swap_in`

## 当前状态

这条路线当前已经在本地跑通真实 vLLM V0 调度闭环，并且完成了 `swap_in` 后的全量 byte-level 正确性校验。

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

早期 shadow write 成功日志：

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

早期 shadow write 日志汇总结果：

- `swap_out_shadow` 次数：`24`
- 累计 `mappings`：`9835`
- 累计写入量：`18,047,303,680 bytes`
- 累计写入量约：`16.81 GiB`
- 单次平均写入量约：`0.70 GiB`
- 单次平均 `mappings` 约：`409.79`
- 单次平均写入耗时约：`556.52 ms`
- 单次平均写入带宽约：`1.26 GiB/s`

一个重要经验是：仅靠增大并发请求数，很多时候只会形成 waiting queue；要更稳定地制造 `swap_out`，需要让单个请求组内部形成多分支运行态，因此这次成功配置里的 `temperature=0.8 + best_of=4` 是关键。

## 最新代码进展

在 `2026-06-15` 的最新代码里，已经补上 `BaM swap_in` 读回路径、共享 BaM 后端、全量正确性校验，并完成真实 vLLM V0 调度路径下的端到端闭环验证。

当前最新成功日志：

- [v100_v0_bam_swap_roundtrip_20260615_183422.log](/home/xhk/llm-inference/vllm/evaluation/logs/v100_v0_bam_swap_roundtrip_20260615_183422.log:1)

当前新增内容包括：

- `BaMRowStore.load_rows()`：为 `BaM` 行存补齐按行读回能力
- `BaMRowStore` 使用 `int64 chunk` 方式存储原始字节，避免 `uint8` 路径的不确定性
- `BaMBlockStore`：共享同一个 BaM row-store 后端，避免写路径和读路径各自初始化控制器
- `BaMSwapReader`：独立的 `SSD -> GPU` block 读回模块
- `VLLM_BAM_SWAPIN_ENABLE=1`：在 `CacheEngine.swap_in()` 中切换到 `BaM` 读回
- `VLLM_BAM_SWAPIN_VERIFY=1`：在 `swap_in` 后把恢复出的 GPU block 与 `cpu_cache` 参考 block 做显式校验
- `VLLM_BAM_SWAPIN_VERIFY_BLOCKS`：支持全量校验或按当前 batch 前 N 个映射抽样校验
- `GIDS_FORCE_SYNC_READ=1`：第一轮实验建议走同步读，优先验证正确性
- `VLLM_BAM_CACHE_SIZE_MB=1024`：当前 roundtrip/shadow 脚本固定使用已验证通过的 BaM page cache 配置

当前设计仍然保持“尽量解耦”：

- `swap_out` 写路径仍由 `BaMShadowWriter` 负责
- `swap_in` 读路径单独放在 `BaMSwapReader`
- `CacheEngine` 只负责在原生路径和 `BaM` 路径之间分发

最新真实 vLLM 闭环实验已经确认：

- `swap_out` 触发 `24` 次
- `swap_in` 触发 `24` 次
- `BaM shadow write` 触发 `24` 次
- `BaM swap_in readback` 触发 `24` 次
- 端到端推理正常结束，日志中存在完整 `Run summary`
- `BAM_SWAPIN_VERIFY` 全量校验通过 `24` 次
- 全量校验 `9835` 个 mapping，对应 `275380` 个 layer-block
- 日志中没有 `mismatch detected` 或 `Traceback`

最新闭环实验的平均表现：

- `swap_out` 平均 `409.79` mappings/次
- `swap_out` 累计写出 `16.81 GiB`
- `swap_out` 平均写出 `0.700 GiB`/次
- `swap_out` 平均耗时 `396.80 ms`
- `swap_out` 平均带宽 `1.86 GiB/s`
- `swap_in` 累计读回 `16.81 GiB`
- `swap_in` 平均读回 `0.700 GiB`/次
- `swap_in` 平均耗时 `230.65 ms`
- `swap_in` 平均带宽 `3.08 GiB/s`

## 当前正确性结论

截至 `2026-06-15` 的最新实验，可以确认真实 vLLM V0 调度路径下的数据正确性：

- `Scheduler -> swap_out -> BaM write -> swap_in -> BaM read -> GPU cache restore` 这条链路已经真实发生
- `swap_out` 和 `swap_in` 的次数一一对应
- 每次 `swap_in` 后，读回的 GPU block 都与 `cpu_cache` 对应 block 做了 byte-level `torch.equal` 校验
- 最新日志中 `24` 次 `[BAM_SWAPIN_VERIFY] mode=full ... exact=1`
- 读回后程序完整生成了最终输出

因此当前可以正式表述为：

- “BaM swap roundtrip 已在真实 vLLM V0 调度路径中跑通”
- “换出与换入链路已真实触发并完成”
- “BaM swap_in 读回数据正确性已通过全量 byte-level 校验”

需要同时保留的限制是：当前正确性结论是在 `GIDS_FORCE_SYNC_READ=1` 和 `VLLM_BAM_CACHE_SIZE_MB=1024` 下得到的。默认 `64MB` BaM page cache 会触发 page-cache 容量边界问题，后续需要单独修 BaM 小 cache 替换/回写路径。

## 最新校验实现

当前代码已经补上 `swap_in` 后的显式正确性校验逻辑，校验位置放在 `BaMSwapReader` 内部，保持和 `CacheEngine`、`BaMShadowWriter` 解耦：

- 先按 `cpu_block_id -> row_id` 从 `BaM` 读回 block
- 回填到 `gpu_cache`
- 再把恢复后的 GPU block 与 `cpu_cache` 中对应 block 做 `torch.equal`
- 若 `torch.equal` 失败，再补充 `torch.allclose` 与 `max_abs_diff` 作为诊断信息
- 若存在不一致，直接抛出 `RuntimeError`

当前支持两种模式：

- 全量校验：`VLLM_BAM_SWAPIN_VERIFY=1` 且 `VLLM_BAM_SWAPIN_VERIFY_BLOCKS=0`
- 抽样校验：`VLLM_BAM_SWAPIN_VERIFY=1` 且 `VLLM_BAM_SWAPIN_VERIFY_BLOCKS=N`

当前抽样策略采用“当前 batch 前 N 个映射”，目的是让日志可重复、可复现。

校验通过时，日志中会打印：

- `[BAM_SWAPIN_VERIFY] mode=full checked_mappings=... exact=1`
- 或 `[BAM_SWAPIN_VERIFY] mode=sample checked_mappings=... exact=1`

最新真实 vLLM 日志已经实际看到这条校验成功记录，因此当前结论已经升级为：

- “BaM swap_in 读回数据正确性已验证”

## BaM Page Cache 配置说明

当前实验脚本固定使用：

- `VLLM_BAM_CACHE_SIZE_MB=1024`

原因是当前一次真实 vLLM `swap_out` batch 大约包含：

- `mappings ~= 390-422`
- `num_layers=28`
- 每个 layer block 为 `65536 bytes`

也就是一次 batch 需要约 `11000-11816` 个 64KB page。默认 `64MB` BaM cache 只有：

- `64MB / 64KB = 1024 pages`

之前的失败点出现在 `first_bad_cpu_block=37`，对应 `37 * 28 = 1036 pages`，刚好越过 1024 page 边界。因此当前先固定 `1024MB`：

- `1024MB / 64KB = 16384 pages`

这个容量可以覆盖当前实验中的单次真实 swap batch，避免触发 BaM page-cache 替换路径的不稳定问题。

## 方案定位

当前阶段采用“保守闭环”思路：

- 保留 `vLLM` 原生 `GPU -> CPU` 的 `swap_out`
- 在同一次 `swap_out` 之后，额外将同批 KV block 再写一份到 `SSD(BaM)`
- `swap_in` 第一轮先改成“可切换”
  - 默认仍可走原来的 `CPU -> GPU`
  - 开启 `VLLM_BAM_SWAPIN_ENABLE=1` 时，优先从 `BaM` 读回到 `GPU cache`

因此，这一阶段的 `BaM` 角色还不是完全替代 `CPU swap` 的主后端，而是：

- 一个影子写出后端
- 一个可观测、可测量的 `GPU -> SSD -> GPU` 数据通路原型

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
4. 先补最小 `SSD -> GPU swap_in`
5. 再做更强的正确性与性能验证

## 第一版的明确目标

当前阶段优先验证下面五件事：

1. `GPU -> SSD(BaM)` 写出通路能稳定跑通
2. SSD 上的 block 布局和映射关系清晰、可追踪
3. 可以记录每批 `swap_out` 的写出耗时和带宽
4. `SSD(BaM) -> GPU` 读回路径能按 block 正确恢复 KV cache
5. 不影响当前 `vLLM` 正常推理结果

当前阶段 **仍然不要求**：

- 替换当前 `CPU swap`
- 实现异步预取
- 改动 scheduler 策略
- 做端到端加速结论

## 当前 vLLM 中对应的主链路

当前 `swap_out` 主链路可以概括为：

`Scheduler -> BlockManager -> Worker.prepare -> Worker.execute -> CacheEngine.swap_out`

当前 `swap_in` 主链路则是：

`Scheduler -> BlockManager -> Worker.prepare -> Worker.execute -> CacheEngine.swap_in`

第一版建议只在这条链路的最末端增加一个旁路动作：

`CacheEngine.swap_out(GPU->CPU)` 结束后，再触发一次 `BaM shadow write`

换句话说，主链路不变，只是在 `CacheEngine` 内部追加一个“把本轮换出的 block 再写到 SSD”的动作。

## 当前推荐的数据通路形态

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
6. 当后续同一批 block 需要 `swap_in` 时：
   - 若未开启 `VLLM_BAM_SWAPIN_ENABLE`，则继续走原生 `CPU -> GPU`
   - 若开启 `VLLM_BAM_SWAPIN_ENABLE`，则由 `BaMSwapReader` 按 `cpu_block_id -> row_id` 从 `BaM` 读回，并回填到 `gpu_cache`

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

### 4. `BaMSwapReader`

职责：

- 接收一次 `swap_in` 的 block 批次
- 根据 `cpu_block_id` 生成 `BaM row id`
- 从 `BaM` 读取对应 rows
- 恢复成 `vLLM` 当前 KV block 布局
- 把 block 写回 `gpu_cache`

这一层与 `BaMShadowWriter` 分离，目的是保持读写逻辑解耦，避免把 `CacheEngine` 变成一个臃肿的大类。

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

## 当前执行命令

推荐先跑“同步读 + 最小闭环”版本：

```bash
cd /home/xhk/llm-inference/vllm
bash evaluation/run_v100_v0_bam_swap_roundtrip.sh /home/xhk/model/Qwen3-0.6B
```

这个脚本默认会打开：

- `VLLM_BAM_SHADOW_ENABLE=1`
- `VLLM_BAM_SWAPIN_ENABLE=1`
- `VLLM_BAM_SWAPIN_VERIFY=1`
- `VLLM_BAM_SWAPIN_VERIFY_BLOCKS=0`
- `VLLM_BAM_CACHE_SIZE_MB=1024`
- `GIDS_FORCE_SYNC_READ=1`
- `VLLM_V0_SWAP_TRACE=1`

如果只想复现之前已经成功验证过的 shadow `swap_out` 写入路径，则继续使用：

```bash
cd /home/xhk/llm-inference/vllm
bash evaluation/run_v100_v0_bam_shadow_swapout.sh /home/xhk/model/Qwen3-0.6B
```

## 当前数据来源选择

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

## 当前已达成标准

截至 `2026-06-15`，第一版“可运行的 `GPU -> SSD -> GPU` 原型”已经达到下面标准：

1. 开启 `BaM shadow` 后，真实 vLLM V0 推理可以正常结束
2. 在触发 `swap_out` 的 workload 下，日志中稳定出现 `BaM shadow write`
3. `swap_in` 可以从 `BaM` 读回并恢复到 `gpu_cache`
4. `swap_in` 后已经对恢复 block 和 `cpu_cache` 参考 block 做全量 byte-level 校验
5. 可以统计每轮 `swap_out_shadow / swap_in` 的 `ms/event` 和 `GiB/s`

最新成功日志对应的数据：

- `swap_out_shadow`：`24` 次，累计 `16.81 GiB`，平均 `396.80 ms/event`，平均 `1.86 GiB/s`
- `swap_in`：`24` 次，累计 `16.81 GiB`，平均 `230.65 ms/event`，平均 `3.08 GiB/s`
- 正确性：`24` 次全量校验通过，共 `9835` 个 mapping、`275380` 个 layer-block

## 与当前 CPU swap 基线的关系

当前已经测得 `CPU swap` 的关键参考值：

- 大批量单向 `swap_out`：约 `0.42 ms/block`
- 大批量单向 `swap_in`：约 `0.42 ms/block`
- 往返 `round_trip`：约 `0.84 ms/block`
- 有效带宽：约 `4.0 GiB/s`

这次 BaM roundtrip 日志里的平均 mapping 数约为 `409.79`，对应粗略单 block 成本：

- BaM `swap_out_shadow`：约 `0.968 ms/block`
- BaM `swap_in`：约 `0.563 ms/block`
- BaM 同步读写合计：约 `1.531 ms/block`

这个比较只能作为当前同步路径的工程参考，不能直接得出最终 BaM 一定慢于 CPU swap 的结论。原因是当前实现仍保留原生 `GPU -> CPU` swap，并且 BaM 写入是 shadow 额外动作；读路径也固定为 `GIDS_FORCE_SYNC_READ=1`，还没有启用异步读、队列并发和调度重叠。

## 接下来建议

当前最值得优先推进的是把这条线从“正确性闭环”推进到“性能基线清晰”：

1. 做同 workload 下的 CPU swap 与 BaM roundtrip 对照实验
2. 把 `VLLM_BAM_SWAPIN_VERIFY` 从全量校验切到抽样或关闭，测去掉校验开销后的 BaM 读写成本
3. 单独修 `64MB` BaM page cache 下的替换/回写问题，避免长期依赖 `1024MB`
4. 在正确性和同步性能基线稳定后，再评估 BaM 异步读路径和预取重叠

暂时不建议马上改 scheduler 或完全移除 CPU swap。当前最稳的路线是先保留 vLLM 原生语义，把 BaM 作为可切换、可验证、可测量的数据面后端逐步做实。

## 本文档对应的总建议

一句话总结当前阶段：

**BaM 的真实 vLLM V0 换出换入闭环已经跑通，读回数据也通过全量 byte-level 校验；下一步不要急着重写 scheduler，而是先做同 workload 的 CPU/BaM 性能对照、降低校验开销、修小 page cache，再进入异步读和调度重叠。**
