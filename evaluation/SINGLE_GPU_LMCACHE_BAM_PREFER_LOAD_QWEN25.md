# 单卡 LMCache + BaM Prefer-Load 结果

这份文档整理当前 `vllm-bam` 中已经跑通的 `LMCache + BaM prefer-load`
结果。它和 `shadow write` 的区别是：第二个请求已经优先从 `BaM` 读回
KV chunk，然后交回 LMCache/vLLM 继续执行。

## 当前结论

截至 `2026-06-23 17:41:38` 这轮日志，主链路已经跑通：

- `LMCache` 仍负责上层 chunk/key 语义和 fallback。
- `BaM` 在 `put` 阶段保存同一份 KV payload。
- 第二个请求命中共享前缀时，`load` 阶段优先从 `BaM` 读取。
- BaM 读回数据与原始 LMCache 数据校验一致。
- vLLM 侧成功进入 prefix continuation 路径，最终输出正常文本。

这说明当前已经不只是“影子写入能跑”，而是 `LMCache -> BaM -> LMCache -> vLLM`
这条读写闭环已经在真实 vLLM 请求里跑通。

## 环境与口径

- repo: `/home/xhk/llm-inference/vllm-bam`
- LMCache repo: `/home/xhk/llm-inference/LMCache-v0-torch26`
- model: `/home/xhk/llm-inference/models/Qwen2.5-7B-Instruct`
- GPU: `Tesla V100S-PCIE-32GB`
- vLLM: `V0`
- dtype: `float16`
- `MAX_MODEL_LEN=8192`
- `PROMPT_REPEAT=100`
- `enable_chunked_prefill=false`
- `enforce_eager=true`
- `LMCache chunk_size=256`
- `LMCache local_disk=/home/xhk/llm-inference/lmcache_local_disk/`
- `VLLM_BAM_LMCACHE_SHADOW_ENABLE=1`
- `VLLM_BAM_LMCACHE_PREFER_LOAD_ENABLE=1`

## 日志

- [run.log](/home/xhk/llm-inference/vllm-bam/evaluation/logs/single_gpu_lmcache_no_prefix_reuse_qwen25/20260623_174138/run.log)

## 关键结果

- warmup: `bam_shadow_elapsed_s=2.1491`
- request 1: `request_1_elapsed_s=1.8220`
- request 2: `request_2_elapsed_s=1.6120`

BaM 写入：

- warmup 尾块：`57.353 ms`，`0.238 GiB/s`
- request 1 满块：`8.742 / 8.130 / 8.325 / 8.102 ms`
- request 1 后续块：`5.136 / 5.110 ms`
- 对应写入带宽约 `1.56 ~ 2.68 GiB/s`

BaM 读取：

- `4.629 ms`，`2.953 GiB/s`
- `4.373 ms`，`3.126 GiB/s`
- `3.717 ms`，`3.678 GiB/s`
- `3.868 ms`，`3.535 GiB/s`

正确性：

- `LMCACHE_BAM_VERIFY exact_equal=True`
- `max_abs_diff=0.000000`
- `mean_abs_diff=0.000000`
- 校验形状：`(2, 28, 256, 512)`
- dtype: `torch.float16`

vLLM 路径：

- 日志中出现 `prefer-load hit`
- 日志中出现 `LMCACHE_REBUILD`
- 日志中出现 `XFORMERS_PREFIX_FALLBACK`
- 第二个请求输出正常，没有乱码

## 和 SSD-only baseline 的关系

当前可直接引用的 `LMCache SSD-only` 对照仍是：

- [run.log](/home/xhk/llm-inference/vllm-bam/evaluation/logs/single_gpu_lmcache_no_prefix_reuse_qwen25/20260622_225752/run.log)
- `PROMPT_REPEAT=400`
- `request_1_elapsed_s=3.0629`
- `request_2_elapsed_s=2.6346`

需要注意：这组 SSD-only baseline 和最新 prefer-load 日志的 prompt 长度不同，
所以现在不能直接把 `2.6346s` 和 `1.6120s` 当作公平性能对比。

更准确的阶段性结论是：

- `LMCache SSD-only` 基线已经跑通。
- `LMCache + BaM shadow write` 已经跑通。
- `LMCache + BaM prefer-load` 已经跑通并通过正确性校验。
- 下一步需要在同一组 `PROMPT_REPEAT` / token 长度下重复跑 SSD-only 和
  prefer-load，形成公平表格。

## 已修复的问题

此前第二个请求输出异常，根因不在 BaM 数据本身，而在 LMCache V0
GPU connector 对 vLLM V0 KV cache 页容量的解释上。

修复位置：

- `/home/xhk/llm-inference/LMCache-v0-torch26/lmcache/experimental/gpu_connector.py`

核心点：

- vLLM V0 的 KV cache 应按 `[2, num_blocks * block_size, hidden_dim]`
  理解。
- `page_buffer_size` 应该是 token slot 数，即 `num_blocks * block_size`。
- 不能把 flattened page width 当成 page buffer 的 token 数。

修复后：

- 无 BaM 的 LMCache retrieve 输出恢复正常。
- BaM prefer-load 输出也恢复正常。
- BaM 读回校验为 exact match。

## 当前一句话定位

`LMCache + BaM prefer-load` 已经在真实 vLLM 单卡请求中跑通：BaM 能写入、
能读回、能通过正确性校验，并且读回后的 KV 能被 vLLM prefix continuation
路径正常消费。
