# Single-GPU LMCache SSD Baseline Logs

这个目录保存 `2026-06-22` 这轮单卡 baseline 的归档日志。

当前归档的日志包括：

- `vllm-bam-single-gpu-no-prefix-reuse-qwen25.log`
  原生 `vLLM V0` no-prefix-reuse baseline。
- `vllm-bam-single-gpu-lmcache-no-prefix-reuse-qwen25.log`
  `LMCache SSD-only` no-prefix-reuse baseline。

这两份日志对应的整理说明见：

- [SINGLE_GPU_LMCACHE_SSD_BASELINE_QWEN25.md](/home/xhk/llm-inference/vllm-bam/evaluation/SINGLE_GPU_LMCACHE_SSD_BASELINE_QWEN25.md)
