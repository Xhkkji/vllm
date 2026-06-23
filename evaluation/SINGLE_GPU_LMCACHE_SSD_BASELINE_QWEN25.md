# 单卡 LMCache SSD Baseline

这份文档整理当前在 `vllm-bam` 中已经跑通的单卡 baseline，目标是先固定一个后续可和 `BaM` 对比的 `SSD baseline`。

## 结论

当前可以暂时把下面两条作为单卡对照基线：

- 原生 `vLLM V0` no-prefix-reuse baseline
- `LMCache SSD-only` no-prefix-reuse baseline

它们现在的定位是：

- 用来确认单卡 `Qwen2.5-7B-Instruct` 在当前 `V100` 环境上可以稳定跑通
- 用来确认 `LMCache + SSD` 这条链路已经可用
- 用来作为后续 `BaM` 接入时的暂定 baseline

它们现在**不是**：

- prefix reuse 命中收益结论
- disaggregated prefill/decode 结论
- 最终论文级性能结论

## 当前环境

- GPU: `Tesla V100S-PCIE-32GB`
- Compute capability: `7.0`
- `vLLM V1` 不能用，当前统一走 `V0`
- attention backend: `XFormers`
- model: `/home/xhk/llm-inference/models/Qwen2.5-7B-Instruct`
- repo: `/home/xhk/llm-inference/vllm-bam`
- LMCache repo: `/home/xhk/llm-inference/LMCache-v0-torch26`

## 为什么先固定 no-prefix-reuse

当前这台机器上的 `shared prefix reuse` 路径不稳定，纯 vLLM 第二次共享长前缀请求会触发底层崩溃，因此当前 baseline 统一先固定为：

- `enable_prefix_caching=false`
- `enable_chunked_prefill=false`
- `enforce_eager=true`
- `dtype=half`

现阶段先回答一个更简单的问题：

- 原生路径能稳定跑多快
- `LMCache SSD` 路径能稳定跑通到哪一步

## 当前脚本

原生 baseline：

- [run_single_gpu_no_prefix_reuse_baseline_qwen25.sh](/home/xhk/llm-inference/vllm-bam/evaluation/run_single_gpu_no_prefix_reuse_baseline_qwen25.sh)

LMCache 通用 baseline：

- [run_single_gpu_lmcache_no_prefix_reuse_qwen25.sh](/home/xhk/llm-inference/vllm-bam/evaluation/run_single_gpu_lmcache_no_prefix_reuse_qwen25.sh)

当前推荐的 `LMCache SSD-only baseline`：

- [run_single_gpu_lmcache_ssd_no_prefix_reuse_qwen25.sh](/home/xhk/llm-inference/vllm-bam/evaluation/run_single_gpu_lmcache_ssd_no_prefix_reuse_qwen25.sh)

当前 `LMCache + BaM shadow` 也复用同一个通用脚本，只是额外打开：

- `VLLM_BAM_LMCACHE_SHADOW_ENABLE=1`

并且脚本现在会先发一个单独的 `warmup` 请求，用来提前完成 BaM 初始化。
这个 `warmup` 的耗时不会算入后面的 `request_1/request_2` 正式结果。

## 当前执行命令

先进入仓库：

```bash
cd /home/xhk/llm-inference/vllm-bam
```

原生 no-prefix-reuse baseline：

```bash
bash evaluation/run_single_gpu_no_prefix_reuse_baseline_qwen25.sh
```

LMCache SSD-only no-prefix-reuse baseline：

```bash
bash evaluation/run_single_gpu_lmcache_ssd_no_prefix_reuse_qwen25.sh
```

## 当前结果

原生 baseline 日志：

- `/tmp/vllm-bam-single-gpu-no-prefix-reuse-qwen25.log`

关键结果：

- `request_1_elapsed_s=2.1716`
- `request_2_elapsed_s=2.2151`

LMCache SSD-only baseline 日志：

- `/tmp/vllm-bam-single-gpu-lmcache-no-prefix-reuse-qwen25.log`

关键结果：

- `request_1_elapsed_s=2.0224`
- `request_2_elapsed_s=1.6895`

### 最新复跑结果

为了和后续 `BaM shadow` 这轮实验保持同样的 prompt 长度，本轮又用：

- `PROMPT_REPEAT=400`
- `MAX_MODEL_LEN=8192`

重新跑了一次 `LMCache SSD-only` baseline。

最新日志：

- [run.log](/home/xhk/llm-inference/vllm-bam/evaluation/logs/single_gpu_lmcache_no_prefix_reuse_qwen25/20260622_225752/run.log)

关键结果：

- `request_1_elapsed_s=3.0629`
- `request_2_elapsed_s=2.6346`

因此，和 `BaM shadow` 做当前阶段对比时，应该优先使用这组最新 baseline，
而不是上面那组更早、prompt 更短的结果。

LMCache SSD-only 这次还额外确认了两件事：

- 日志中 `local_cpu=False`
- 日志中 `local_disk='/home/xhk/llm-inference/lmcache_local_disk/'`

并且本地 SSD 目录已经实际落盘：

- `/home/xhk/llm-inference/lmcache_local_disk/`

当前目录下已有一批 `.pt` 文件，运行后目录占用约 `161M`。

## 如何理解这组 baseline

目前更稳妥的表述是：

- 原生 baseline 已稳定
- `LMCache SSD-only` baseline 已稳定
- `LMCache + SSD` 链路已跑通
- 在 `PROMPT_REPEAT=400`、`MAX_MODEL_LEN=8192` 下，
  `LMCache SSD-only` 最新基线约为 `3.06s / 2.63s`

但这组数字暂时不要直接解读成：

- “LMCache 明显优于原生”
- “SSD 命中已经带来显著收益”

因为当前 workload 是：

- 单卡
- 两次独立请求
- no-prefix-reuse

所以它更多是在回答“链路是否稳定、路径是否可跑、是否真的落盘”，而不是回答“缓存命中收益有多大”。

## 这份 baseline 和 BaM 的关系

后续更合理的主线不是继续围绕 Mooncake，而是：

1. 保留 `LMCache` 作为 control plane / connector baseline
2. 把 `BaM` 作为 `SSD data plane backend`
3. 用 `LMCache SSD-only baseline` 作为对照组

也就是说，后面的公平比较目标应该是：

- `LMCache + local SSD backend`
- `LMCache + BaM SSD backend`

而不是：

- `LMCache`
- `vLLM V0 internal BaM swap`

后者可以继续保留做底层数据通路验证，但不应该作为主 baseline 对比口径。

## 下一步 BaM 路径

当前建议的接入顺序是：

1. `BaM shadow store under LMCache`
   先在 `LMCache save` 生命周期里，把同一份 KV 额外写进 `BaM`，但读取仍走原始 `LMCache SSD`。

2. `Prefer BaM load`
   在 `LMCache load` 生命周期里优先从 `BaM` 读回；若失败或校验不通过，再回退到原始 `LMCache SSD`。

3. `BaM-only backend mode`
   等 shadow store 和 prefer-load 都稳定后，再考虑把 `LMCache` 的本地 SSD 后端真正替换成 `BaM backend`。

这条路线的核心思想是：

- `LMCache` 负责“何时存、何时取、哪些 token/chunk 需要处理”
- `BaM` 负责“KV payload 如何高效写入 SSD、如何从 SSD 读回”

对应的整体方案文档在：

- [lmcache_bam_tutti_plan.md](/home/xhk/llm-inference/vllm-bam/docs/lmcache_bam_tutti_plan.md)

如果只看当前工程状态，可以把下一步具体工作理解成：

- 不是继续扩 `worker/cache_engine.py` 里的 `V0 swap` 钩子
- 而是把已有的 `BaM row/block` 能力重新组织成 `LMCache` 可调用的 backend adapter

## 当前一句话定位

截至现在，可以把这条 baseline 写成：

- `vllm-bam` 上的单卡 `LMCache SSD-only baseline` 已跑通，可作为后续 `LMCache + BaM` 集成前的暂定对照组。

对应的下一阶段中间基线现在也已经有了：

- [SINGLE_GPU_LMCACHE_BAM_SHADOW_BASELINE_QWEN25.md](/home/xhk/llm-inference/vllm-bam/evaluation/SINGLE_GPU_LMCACHE_BAM_SHADOW_BASELINE_QWEN25.md)

它表示：

- `LMCache` 原始 `SSD` 路径保持不变
- `BaM` 对同一份 KV chunk 额外做 `shadow write`
- `warmup` 与正式请求耗时分开统计
