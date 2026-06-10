# V100 + vLLM V0 CPU Swap Baseline

本文档整理当前阶段已经完成的 `V100 + vLLM V0 + CPU swap` 基线测量结果，作为后续对接 `BaM_IOStack` 前的对照参考。

## 实验目标

- 验证 `V100` 上 `vLLM V0` 的 `swap_out / swap_in` 路径确实可触发
- 量化当前原生 `CPU <-> GPU` KV block 搬运成本
- 为后续 `BaM` 同步 / 异步接入提供对照基线

## 实验环境

- GPU: `Tesla V100S-PCIE-32GB`
- vLLM branch: `xhk/bam-sync-swap-v100`
- 模型: `/home/xhk/model/Qwen3-0.6B`
- attention backend: `XFormers`
- vLLM engine: `V0`
- dtype: `float16`
- `gpu_memory_utilization=0.6`
- `swap_space=8 GiB`

## 使用的脚本

- 压测脚本: [v0_swap_trace_eval.py](/home/xhk/llm-inference/vllm/evaluation/v0_swap_trace_eval.py)
- 日志统计脚本: [v0_swap_baseline.py](/home/xhk/llm-inference/vllm/evaluation/v0_swap_baseline.py)
- 纯搬运 microbenchmark: [v0_swap_microbench.py](/home/xhk/llm-inference/vllm/evaluation/v0_swap_microbench.py)

## 关键日志

- swap trace 日志:
  [v0_swap_trace_Qwen3-0.6B_20260609_163742.log](/home/xhk/llm-inference/vllm/evaluation/logs/v0_swap_trace_Qwen3-0.6B_20260609_163742.log)

该日志已经明确包含：

- `Scheduler op=preempt`
- `BlockManager op=swap_out`
- `Worker.execute op=swap_out`
- `CacheEngine op=swap_out`
- `BlockManager op=swap_in`
- `Worker.execute op=swap_in`
- `CacheEngine op=swap_in`

这说明 `V0` 的真实换入换出链路已经打通：

`Scheduler -> BlockManager -> Worker -> CacheEngine`

## 基本块信息

从日志和脚本统计可得：

- `block_size_tokens = 16`
- `block_bytes = 1835008`，约 `1.75 MiB`
- `num_attention_layers = 28`
- `num_gpu_blocks = 9616`
- `num_cpu_blocks = 4681`

## 一、Trace 日志基线

这是基于 `CacheEngine` trace 日志解析得到的结果，对应命令：

```bash
python evaluation/v0_swap_baseline.py \
  /home/xhk/llm-inference/vllm/evaluation/logs/v0_swap_trace_Qwen3-0.6B_20260609_163742.log
```

### 汇总结果

`swap_out`

- `events = 125`
- `total_mappings = 129253`
- `total_bytes = 220.89 GiB`
- `weighted_ms_per_block = 0.259702`
- `weighted_gib_per_sec = 6.581`

`swap_in`

- `events = 33`
- `total_mappings = 129253`
- `total_bytes = 220.89 GiB`
- `weighted_ms_per_block = 0.261078`
- `weighted_gib_per_sec = 6.546`

`round_trip_hint`

- `swap_round_trip_ms_per_block = 0.520781`

### 常见 batch size

`swap_out` 常见 `mappings`

- `258`, `260`, `514`, `257`, `259`
- 更大的批次也很常见，例如 `1285`, `1542`

`swap_in` 常见 `mappings`

- 主要集中在大批量：`3848 ~ 4147`
- 这说明真实运行时 `swap_in` 往往是大块批量回填，而不是零碎单块恢复

### 口径说明

这组数字来自 `CacheEngine` 内部 trace：
[cache_engine.py](/home/xhk/llm-inference/vllm/vllm/worker/cache_engine.py:133)

它更接近“在 vLLM 真正 workload 里观察到的批量搬运开销”，但因为内部计时点没有显式 `cuda.synchronize()`，所以这组数据更适合视为：

- 偏乐观
- 偏提交时间
- 适合反映真实批量形态

而不一定等于“严格同步完成时间”。

## 二、Microbenchmark 基线

这是直接对底层 `CacheEngine.swap_in/swap_out` 做同步计时得到的结果，对应命令：

```bash
python evaluation/v0_swap_microbench.py \
  /home/xhk/model/Qwen3-0.6B \
  --batch-sizes 64,256,1024,2048 \
  --warmup-iters 1 \
  --repeat-iters 3
```

脚本在每轮前后都做了 `torch.cuda.synchronize()`，因此更接近“真实阻塞成本”。

### `swap_out`

- `64 mappings`: `0.520344 ms/block`, `3.284 GiB/s`
- `256 mappings`: `0.420637 ms/block`, `4.063 GiB/s`
- `1024 mappings`: `0.420345 ms/block`, `4.066 GiB/s`
- `2048 mappings`: `0.419533 ms/block`, `4.074 GiB/s`

### `swap_in`

- `64 mappings`: `0.443126 ms/block`, `3.857 GiB/s`
- `256 mappings`: `0.428563 ms/block`, `3.988 GiB/s`
- `1024 mappings`: `0.424384 ms/block`, `4.027 GiB/s`
- `2048 mappings`: `0.423781 ms/block`, `4.033 GiB/s`

### `round_trip`

- `64 mappings`: `0.857244 ms/block`, `3.987 GiB/s`
- `256 mappings`: `0.849799 ms/block`, `4.022 GiB/s`
- `1024 mappings`: `0.840914 ms/block`, `4.065 GiB/s`
- `2048 mappings`: `0.841403 ms/block`, `4.062 GiB/s`

### 观察

- 当 `mappings >= 256` 后，带宽基本进入平台区
- 大批量下：
  - `swap_out ≈ 0.42 ms/block`
  - `swap_in ≈ 0.424 ms/block`
  - `round_trip ≈ 0.84 ms/block`
- 小批量 `64 mappings` 明显更差，说明固定开销不可忽略

## 三、两套基线为什么不一样

目前有两套数字：

- trace 基线：约 `0.26 ms/block`，约 `6.5 GiB/s`
- microbench 基线：约 `0.42 ms/block`，约 `4.0 GiB/s`

这不是互相矛盾，而是测量口径不同：

- trace 基线更像“真实 workload 中的批量提交成本”
- microbench 基线更像“严格同步完成后的真实阻塞成本”

当前更建议把 `microbench` 结果作为后续 `BaM` 对比时的主基线，原因是它更保守，也更接近 `swap_in` 对 decode 造成的真实 stall。

## 四、当前阶段结论

1. `V100 + vLLM V0 + CPU swap` 路径已经可稳定复现。
2. 当前 KV block 大小约为 `1.75 MiB/block`。
3. 原生 CPU swap 的真实同步搬运成本，大批量下大致为：
   - 单向 `~0.42 ms/block`
   - 往返 `~0.84 ms/block`
   - 有效带宽 `~4.0 GiB/s`
4. 真实运行日志表明，`swap_in` 常常是大批量恢复，这对后续 `BaM` 设计很关键。

## 五、对接 BaM 前的建议

后续评估 `BaM` 时，至少建议和当前基线对齐比较下面几项：

- 单向 `swap_out` 的 `ms/block`
- 单向 `swap_in` 的 `ms/block`
- `round_trip` 的 `ms/block`
- 大批量 `mappings` 下的有效带宽
- `swap_in` 对 decode stall 的影响

更具体地说：

- 如果 `BaM` 优势主要体现在吞吐，应重点看大批量 `mappings`
- 如果 `BaM` 优势主要体现在异步预取，应重点看 `swap_in` stall 是否下降

