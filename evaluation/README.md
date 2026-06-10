# V100 + vLLM V0 Swap 观测实验

这个目录用于复现当前阶段的第一步实验目标：

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
- [V100_V0_CPU_SWAP_BASELINE.md](/home/xhk/llm-inference/vllm/evaluation/V100_V0_CPU_SWAP_BASELINE.md)
  汇总当前 `V100 + vLLM V0 + CPU swap` 的 trace 基线与 microbenchmark 基线。
- [V100_V0_BAM_SHADOW_SWAPOUT_PLAN.md](/home/xhk/llm-inference/vllm/evaluation/V100_V0_BAM_SHADOW_SWAPOUT_PLAN.md)
  记录 `BaM` 第一版接入方案：保留原始 `CPU swap`，额外实现 `GPU -> SSD` 的 shadow `swap_out`。
- [VLLM_KV_AND_BAM_SSD_LAYOUT.md](/home/xhk/llm-inference/vllm/evaluation/VLLM_KV_AND_BAM_SSD_LAYOUT.md)
  总结当前 `vLLM` KV block 的实际数据组织、大小，以及本地 `BaM_IOStack` 在 SSD 侧更合适的 KV 布局方式。

## 典型用法

先进入仓库根目录：

```bash
cd /home/xhk/llm-inference/vllm
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

当前成功样例对应日志：

- [v0_swap_trace_Qwen3-0.6B_20260610_202648.log](/home/xhk/llm-inference/vllm/evaluation/logs/v0_swap_trace_Qwen3-0.6B_20260610_202648.log:1)

这次实验已经确认：

- `Scheduler` 发生了多次 `op=preempt` 和 `op=swap_out`
- `Worker.execute` 真正执行了 `swap_out`
- `BaM shadow writer` 发生了多次 `[BAM_SHADOW] swap_out_shadow`
- 最新成功日志中共触发 `24` 次 `swap_out_shadow`，累计写入约 `16.81 GiB`

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

## 关键日志前缀

打开 `VLLM_V0_SWAP_TRACE=1` 后，重点看这些日志：

- `[V0_SWAP_TRACE][Scheduler]`
- `[V0_SWAP_TRACE][BlockManager]`
- `[V0_SWAP_TRACE][Worker.prepare]`
- `[V0_SWAP_TRACE][Worker.execute]`
- `[V0_SWAP_TRACE][CacheEngine]`
- `[BAM_SHADOW]`

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
