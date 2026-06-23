# vLLM-BaM Evaluation

这个目录现在包含两类实验：

- `LMCache` 单卡 baseline
- `vLLM V0 + BaM swap` 观测与闭环验证

其中当前和后续对比 `BaM` 最相关的 baseline 是：

- [SINGLE_GPU_LMCACHE_SSD_BASELINE_QWEN25.md](/home/xhk/llm-inference/vllm-bam/evaluation/SINGLE_GPU_LMCACHE_SSD_BASELINE_QWEN25.md)
- [SINGLE_GPU_LMCACHE_BAM_SHADOW_BASELINE_QWEN25.md](/home/xhk/llm-inference/vllm-bam/evaluation/SINGLE_GPU_LMCACHE_BAM_SHADOW_BASELINE_QWEN25.md)

`BaM swap` 这部分用于复现当前阶段的第一步实验目标：

- 强制走 `vLLM V0`
- 尽量稳定触发 `swap_out / swap_in`
- 配合 `VLLM_V0_SWAP_TRACE=1` 观察 scheduler、block manager、worker、cache engine 的日志

## 文件说明

- [v0_swap_trace_eval.py](/home/xhk/llm-inference/vllm/evaluation/v0_swap_trace_eval.py)
  生成固定长度的 token prompts，并发调用 `LLM.generate()`，用于制造 KV cache 压力。
- [v0_swap_baseline.py](/home/xhk/llm-inference/vllm/evaluation/v0_swap_baseline.py)
  解析 `V0_SWAP_TRACE` 日志中的 `CacheEngine` 事件，统计当前 CPU swap 的每 block 成本和批量吞吐基线。
- [v0_swap_microbench.py](/home/xhk/llm-inference/vllm/evaluation/v0_swap_microbench.py)
  直接初始化 vLLM 并调用底层 `CacheEngine.swap_in/swap_out`，测不同 mappings 大小下的纯搬运成本。
- [run_v0_swap_trace.sh](/home/xhk/llm-inference/vllm/evaluation/run_v0_swap_trace.sh)
  设置关键环境变量并保存日志，便于直接复现实验。
- [run_v100_v0_bam_shadow_swapout.sh](/home/xhk/llm-inference/vllm/evaluation/run_v100_v0_bam_shadow_swapout.sh)
  当前已经验证可触发 `swap_out` 和 `BaM shadow write` 的最小复现脚本。
- [run_v100_v0_bam_swap_roundtrip.sh](/home/xhk/llm-inference/vllm/evaluation/run_v100_v0_bam_swap_roundtrip.sh)
  当前用于验证 `BaM shadow swap_out + BaM swap_in` 最小闭环的执行脚本，默认同步读并开启全量正确性校验。
- [run_single_gpu_no_prefix_reuse_baseline_qwen25.sh](/home/xhk/llm-inference/vllm-bam/evaluation/run_single_gpu_no_prefix_reuse_baseline_qwen25.sh)
  单卡原生 vLLM baseline，不带 LMCache，不测 prefix reuse。
- [run_single_gpu_lmcache_no_prefix_reuse_qwen25.sh](/home/xhk/llm-inference/vllm-bam/evaluation/run_single_gpu_lmcache_no_prefix_reuse_qwen25.sh)
  单卡 LMCache baseline 的通用脚本，可通过环境变量切换 `CPU` 或 `SSD` 后端。
- [run_single_gpu_lmcache_ssd_no_prefix_reuse_qwen25.sh](/home/xhk/llm-inference/vllm-bam/evaluation/run_single_gpu_lmcache_ssd_no_prefix_reuse_qwen25.sh)
  当前推荐的单卡 `LMCache SSD-only` baseline 脚本。
- [SINGLE_GPU_LMCACHE_SSD_BASELINE_QWEN25.md](/home/xhk/llm-inference/vllm-bam/evaluation/SINGLE_GPU_LMCACHE_SSD_BASELINE_QWEN25.md)
  汇总当前单卡 `Qwen2.5-7B-Instruct` 的原生 baseline 与 `LMCache SSD-only baseline`，并说明它和后续 `BaM` 接入的关系。
- [SINGLE_GPU_LMCACHE_BAM_SHADOW_BASELINE_QWEN25.md](/home/xhk/llm-inference/vllm-bam/evaluation/SINGLE_GPU_LMCACHE_BAM_SHADOW_BASELINE_QWEN25.md)
  汇总当前单卡 `Qwen2.5-7B-Instruct` 的 `LMCache + BaM shadow write` 基线，并说明它和后续 `Prefer BaM load` 的关系。
- [SINGLE_GPU_LMCACHE_BAM_PREFER_LOAD_QWEN25.md](/home/xhk/llm-inference/vllm-bam/evaluation/SINGLE_GPU_LMCACHE_BAM_PREFER_LOAD_QWEN25.md)
  汇总当前已经跑通的 `LMCache + BaM prefer-load` 结果：第二个请求优先从 `BaM` 读回 KV，并通过正确性校验。
- [V100_V0_CPU_SWAP_BASELINE.md](/home/xhk/llm-inference/vllm/evaluation/V100_V0_CPU_SWAP_BASELINE.md)
  汇总当前 `V100 + vLLM V0 + CPU swap` 的 trace 基线与 microbenchmark 基线。
- [V100_V0_BAM_SHADOW_SWAPOUT_PLAN.md](/home/xhk/llm-inference/vllm/evaluation/V100_V0_BAM_SHADOW_SWAPOUT_PLAN.md)
  记录 `BaM` 当前接入方案：先保留原始 `CPU swap_out`，再逐步补齐 `GPU -> SSD shadow swap_out` 和 `SSD -> GPU swap_in`。
- [VLLM_KV_AND_BAM_SSD_LAYOUT.md](/home/xhk/llm-inference/vllm/evaluation/VLLM_KV_AND_BAM_SSD_LAYOUT.md)
  总结当前 `vLLM` KV block 的实际数据组织、大小，以及本地 `BaM_IOStack` 在 SSD 侧更合适的 KV 布局方式。

## 典型用法

## LMCache 单卡 baseline

如果当前目标是先拿到 `LMCache SSD` baseline，而不是直接压 `BaM swap`，建议先进入仓库根目录：

```bash
cd /home/xhk/llm-inference/vllm-bam
```

原生 no-prefix-reuse baseline：

```bash
bash evaluation/run_single_gpu_no_prefix_reuse_baseline_qwen25.sh
```

当前推荐的 `LMCache SSD-only baseline`：

```bash
bash evaluation/run_single_gpu_lmcache_ssd_no_prefix_reuse_qwen25.sh
```

这条 baseline 的说明、日志位置、当前结果和限制，统一写在：

- [SINGLE_GPU_LMCACHE_SSD_BASELINE_QWEN25.md](/home/xhk/llm-inference/vllm-bam/evaluation/SINGLE_GPU_LMCACHE_SSD_BASELINE_QWEN25.md)

如果当前目标是验证 `LMCache -> BaM shadow write` 已经稳定跑通，可以运行：

```bash
VLLM_BAM_LMCACHE_SHADOW_ENABLE=1 \
PYTHON_BIN=/home/xhk/miniconda3/envs/pytorch-vllm/bin/python \
bash evaluation/run_single_gpu_lmcache_no_prefix_reuse_qwen25.sh
```

这条 shadow baseline 的说明、日志位置和当前结果，统一写在：

- [SINGLE_GPU_LMCACHE_BAM_SHADOW_BASELINE_QWEN25.md](/home/xhk/llm-inference/vllm-bam/evaluation/SINGLE_GPU_LMCACHE_BAM_SHADOW_BASELINE_QWEN25.md)

当前这条 `LMCache + BaM shadow` 路径已经从旧的 `1KB token-row/page`
更新成固定 `256-token` 槽位下的 `128KB page` 布局。
也就是说，一个满 chunk：

- `[2, 28, 256, 512]`

现在会被切成：

- `112` 个 `128KB` page

最新 `128KB page` 版本日志：

- [run.log](/home/xhk/llm-inference/vllm-bam/evaluation/logs/single_gpu_lmcache_no_prefix_reuse_qwen25/20260622_233047/run.log)

这轮结果已经确认：

- `page_bytes=131072`
- `pages_per_chunk=112`
- `warmup bam_shadow_elapsed_s=2.2156`
- `request_1_elapsed_s=2.9673`
- `request_2_elapsed_s=2.7374`
- 稳态 `BaM shadow write` 带宽约 `3.2 ~ 3.6 GiB/s`

一页式结果摘要见：

- [LMCACHE_BAM_128KB_PAGE_RESULT_SUMMARY_20260622.md](/home/xhk/llm-inference/vllm-bam/evaluation/LMCACHE_BAM_128KB_PAGE_RESULT_SUMMARY_20260622.md)

如果要复现当前已经跑通的 `LMCache + BaM prefer-load`：

```bash
cd /home/xhk/llm-inference/vllm-bam
sudo -v
VLLM_BAM_LMCACHE_SHADOW_ENABLE=1 \
VLLM_BAM_LMCACHE_PREFER_LOAD_ENABLE=1 \
MAX_MODEL_LEN=8192 \
PROMPT_REPEAT=100 \
PYTHON_BIN=/home/xhk/miniconda3/envs/pytorch-vllm/bin/python \
bash evaluation/run_single_gpu_lmcache_no_prefix_reuse_qwen25.sh
```

最新 prefer-load 日志：

- [run.log](/home/xhk/llm-inference/vllm-bam/evaluation/logs/single_gpu_lmcache_no_prefix_reuse_qwen25/20260623_174138/run.log)

这轮已经确认：

- `request_1_elapsed_s=1.8220`
- `request_2_elapsed_s=1.6120`
- BaM read 带宽约 `2.95 ~ 3.68 GiB/s`
- `LMCACHE_BAM_VERIFY exact_equal=True`
- `LMCACHE_REBUILD` 和 `XFORMERS_PREFIX_FALLBACK` 都有执行
- 第二个请求输出正常

注意：这轮是 `PROMPT_REPEAT=100`，旧的 `LMCache SSD-only` 对照是
`PROMPT_REPEAT=400`，暂时只能证明链路正确和可运行；要做最终性能对比，
还需要统一 prompt 长度重跑。

如果要单独复核 `BaM` 顺序写新槽位的首块 / 第二块 / 稳态块开销，可以运行：

```bash
bash evaluation/run_lmcache_bam_write_microbench.sh
```

最新顺序写 microbench 日志：

- [lmcache_bam_write_microbench_20260622_235727.log](/home/xhk/llm-inference/vllm-bam/evaluation/logs/lmcache_bam_write_microbench_20260622_235727.log)

## BaM Swap 实验

先进入仓库根目录：

```bash
cd /home/xhk/llm-inference/vllm-bam
```

然后运行：

```bash
bash evaluation/run_v0_swap_trace.sh <model>
```

例如：

```bash
bash evaluation/run_v0_swap_trace.sh facebook/opt-125m
```

如果你已经有本地模型路径，也可以直接传本地目录：

```bash
bash evaluation/run_v0_swap_trace.sh /path/to/your/model
```

如果要直接复现当前已经验证成功的 `V100 + V0 + BaM shadow swap_out`：

```bash
bash evaluation/run_v100_v0_bam_shadow_swapout.sh /home/xhk/model/Qwen3-0.6B
```

如果要验证当前最新的 `BaM swap_in` 最小闭环，直接运行：

```bash
bash evaluation/run_v100_v0_bam_swap_roundtrip.sh /home/xhk/model/Qwen3-0.6B
```

它默认会打开：

- `VLLM_BAM_SHADOW_ENABLE=1`
- `VLLM_BAM_SWAPIN_ENABLE=1`
- `VLLM_BAM_SWAPIN_VERIFY=1`
- `VLLM_BAM_SWAPIN_VERIFY_BLOCKS=0`
- `VLLM_BAM_CACHE_SIZE_MB=1024`
- `GIDS_FORCE_SYNC_READ=1`

当前最新成功闭环日志：

- [v100_v0_bam_swap_roundtrip_20260615_183422.log](/home/xhk/llm-inference/vllm/evaluation/logs/v100_v0_bam_swap_roundtrip_20260615_183422.log:1)

早期 shadow write 成功样例日志：

- [v0_swap_trace_Qwen3-0.6B_20260610_202648.log](/home/xhk/llm-inference/vllm/evaluation/logs/v0_swap_trace_Qwen3-0.6B_20260610_202648.log:1)

早期 shadow write 实验已经确认：

- `Scheduler` 发生了多次 `op=preempt` 和 `op=swap_out`
- `Worker.execute` 真正执行了 `swap_out`
- `BaM shadow writer` 发生了多次 `[BAM_SHADOW] swap_out_shadow`
- 该日志中共触发 `24` 次 `swap_out_shadow`，累计写入约 `16.81 GiB`

当前代码状态已经补上实验性 `BaM swap_in` 路径：

- 当 `VLLM_BAM_SWAPIN_ENABLE=1` 时，`CacheEngine.swap_in()` 会优先从 `BaM` 读回 block
- 默认脚本配合 `GIDS_FORCE_SYNC_READ=1` 和全量校验，优先验证读回正确性
- 当前重点是确认 `swap_out -> BaM write -> swap_in -> BaM read -> GPU cache restore` 链路正确，不先追求性能最优

`2026-06-15` 这轮真实 vLLM `roundtrip` 实验已经确认：

- `swap_out` 触发 `24` 次
- `swap_in` 触发 `24` 次
- `BaM shadow write` 触发 `24` 次
- `BaM swap_in readback` 触发 `24` 次
- 端到端推理正常结束
- 全量校验 `9835` 个 mapping，对应 `275380` 个 layer-block
- 日志中出现 `24` 次 `[BAM_SWAPIN_VERIFY] mode=full ... exact=1`
- 日志中没有 `mismatch detected` 或 `Traceback`

这轮实验数据如下：

- BaM 累计写入：`18,047,303,680 bytes`，约 `16.81 GiB`
- BaM 累计读回：`18,047,303,680 bytes`，约 `16.81 GiB`
- `swap_out_shadow` 平均耗时：`396.80 ms/event`
- `swap_out_shadow` 平均带宽：`1.86 GiB/s`
- `swap_in` 平均耗时：`230.65 ms/event`
- `swap_in` 平均带宽：`3.08 GiB/s`
- 端到端生成吞吐：`prompt_tokens_per_sec=585.02`，`generated_tokens_per_sec=97.50`

因此现在可以正式表述为：

- `BaM swap roundtrip 已在真实 vLLM V0 调度路径中跑通`
- `换出与换入链路已真实触发并完成`
- `BaM swap_in 读回数据正确性已通过全量 byte-level 校验`

现在代码里已经补上了这套显式校验逻辑：

- `VLLM_BAM_SWAPIN_VERIFY=1`：开启 `swap_in` 后的正确性校验
- `VLLM_BAM_SWAPIN_VERIFY_BLOCKS=0`：全量校验当前 batch 的全部映射
- `VLLM_BAM_SWAPIN_VERIFY_BLOCKS=N`：只校验当前 batch 前 `N` 个映射，便于做抽样验证

校验通过时，日志中会出现：

- `[BAM_SWAPIN_VERIFY] mode=full ... exact=1`
- 或 `[BAM_SWAPIN_VERIFY] mode=sample ... exact=1`

如果校验失败，会直接抛出 `[BAM_SWAPIN_VERIFY] mismatch detected`，并给出：

- `layer`
- `checked_mappings`
- `first_bad_cpu_block`
- `first_bad_gpu_block`
- `allclose`
- `max_abs_diff`

需要注意的是，当前正确性结论基于 `GIDS_FORCE_SYNC_READ=1` 和 `VLLM_BAM_CACHE_SIZE_MB=1024`。默认 `64MB` BaM page cache 只有 `1024` 个 64KB page，而当前单次真实 swap batch 约需要 `11000-11816` 个 page，所以脚本先固定为 `1024MB`，避免触发小 cache 替换路径的不稳定问题。

生成日志后，可以直接统计 swap 基线：

```bash
python evaluation/v0_swap_baseline.py
```

如果要指定某一份日志：

```bash
python evaluation/v0_swap_baseline.py /path/to/v0_swap_trace_xxx.log
```

如果要直接测底层搬运 microbenchmark：

```bash
python evaluation/v0_swap_microbench.py /path/to/your/model \
  --batch-sizes 64,256,1024,2048 \
  --warmup-iters 1 \
  --repeat-iters 3
```

## 常用调参

可以通过环境变量覆盖默认参数：

```bash
NUM_PROMPTS=48 \
PROMPT_LEN=3072 \
MAX_TOKENS=32 \
GPU_MEMORY_UTILIZATION=0.55 \
SWAP_SPACE=16 \
bash evaluation/run_v0_swap_trace.sh facebook/opt-125m
```

含义如下：

- `NUM_PROMPTS`: 并发请求数，越大越容易触发 swap
- `PROMPT_LEN`: 每个请求的 prompt token 数，越长越容易触发 swap
- `MAX_TOKENS`: 每个请求继续生成的 token 数
- `GPU_MEMORY_UTILIZATION`: 给 KV cache 的 GPU 显存比例，调小更容易触发 swap
- `SWAP_SPACE`: 每张 GPU 可用的 CPU swap 空间，单位 GiB
- `MAX_MODEL_LEN`: 传给 vLLM 的 `max_model_len`
- `DTYPE`: 例如 `half`
- `ENFORCE_EAGER`: 设为 `1` 时强制 eager，便于先排除图捕获因素
- `TEMPERATURE`: 采样温度；`>0` 配合 `BEST_OF>1` 更容易制造多序列压力
- `BEST_OF`: 每个请求内部保留的候选数；`>1` 时更容易触发 `swap_out`
- `BAM_SWAPIN_ENABLE`: 设为 `1` 时，`swap_in` 优先从 `BaM` 读回
- `BAM_SWAPIN_VERIFY`: 设为 `1` 时，`swap_in` 后把恢复出的 GPU block 与 `cpu_cache` 参考 block 做显式校验
- `BAM_SWAPIN_VERIFY_BLOCKS`: `0` 表示全量校验，正整数表示仅校验当前 batch 前 N 个映射
- `VLLM_BAM_CACHE_SIZE_MB`: BaM page cache 大小；当前 roundtrip/shadow 脚本固定为 `1024`
- `GIDS_FORCE_SYNC_READ`: 设为 `1` 时，BaM 读路径使用同步基线实现，便于先验证正确性

## 关键日志前缀

打开 `VLLM_V0_SWAP_TRACE=1` 后，重点看这些日志：

- `[V0_SWAP_TRACE][Scheduler]`
- `[V0_SWAP_TRACE][BlockManager]`
- `[V0_SWAP_TRACE][Worker.prepare]`
- `[V0_SWAP_TRACE][Worker.execute]`
- `[V0_SWAP_TRACE][CacheEngine]`
- `[BAM_SHADOW]`
- `[BAM_SWAPIN]`

## 当前最小成功配置

当前本地已经验证可稳定打出 `swap_out` 和 `BaM` 写入的关键参数组合是：

- `NUM_PROMPTS=24`
- `PROMPT_LEN=6144`
- `MAX_TOKENS=1024`
- `TEMPERATURE=0.8`
- `BEST_OF=4`
- `GPU_MEMORY_UTILIZATION=0.16`
- `PREEMPTION_MODE=swap`
- `MAX_NUM_SEQS=8`

一个关键经验是：单纯堆高 `NUM_PROMPTS` 往往只会增加 waiting queue；真正更容易把系统打进 `swap_out` 的，是 `TEMPERATURE>0` 配合 `BEST_OF>1` 形成多分支运行态。

对于 `BaM swap_in` 的闭环验证，建议沿用这组参数，并额外打开：

- `VLLM_BAM_SWAPIN_ENABLE=1`
- `VLLM_BAM_SWAPIN_VERIFY=1`
- `VLLM_BAM_SWAPIN_VERIFY_BLOCKS=0`
- `VLLM_BAM_CACHE_SIZE_MB=1024`
- `GIDS_FORCE_SYNC_READ=1`

推荐最小执行命令：

```bash
cd /home/xhk/llm-inference/vllm
bash evaluation/run_v100_v0_bam_swap_roundtrip.sh /home/xhk/model/Qwen3-0.6B
```

如果只需要当前最准确的结论，可以写成：

- `BaM swap roundtrip 已在真实 vLLM V0 调度路径中跑通`
- `换出与换入链路已真实触发并完成`
- `BaM swap_in 读回数据正确性已通过全量 byte-level 校验`

## 一个容易误判的现象

在 `BEST_OF>1` 时，vLLM 的进度条可能显示类似：

```text
Processed prompts: 25%|24/96
```

这通常不是只完成了四分之一，而是进度条总数按内部 `parallel_sample` 请求口径统计了。判断是否真正跑完，优先看最终 `Run summary` 和日志中的 `swap_out_shadow` 事件。

## 一个实用判断

如果没有触发 swap，通常优先从下面几项里调大压力：

1. 提高 `NUM_PROMPTS`
2. 提高 `PROMPT_LEN`
3. 降低 `GPU_MEMORY_UTILIZATION`
4. 换更大的模型

如果出现：

```text
Aborted due to the lack of CPU swap space
```

说明 CPU swap 空间不够，可以适当增大 `SWAP_SPACE`。
