# vLLM Evaluation

当前评测目录只维护两条运行主线：

- 原生 LMCache：作为 CPU/SSD storage baseline；
- GranuleKV：由 vLLM 负责调度和层级预取，BaM_IOStack 提供独立的 GPU-initiated KV I/O。

## 原生 LMCache

固定 prompt 的单卡 SSD baseline：

```bash
cd /home/xhk/llm-inference/vllm-bam
bash evaluation/run_single_gpu_lmcache_ssd_no_prefix_reuse_qwen25.sh
```

LongBench-TriviaQA 的原生 SSD baseline：

```bash
bash evaluation/lmcache_ssd_read_paths_baseline/run_longbench_triviaqa_ssd_read_paths_qwen25.sh ssd_cpu_gpu
```

冷 page-cache 版本：

```bash
bash evaluation/lmcache_ssd_read_paths_baseline/run_longbench_triviaqa_lmcache_ssd_cold_cgroup_qwen25.sh
```

相关脚本只使用 LMCache 原生 `LMCacheConnector` 和 `local_disk`，数据路径为：

```text
SSD -> CPU MemoryObj -> vLLM paged KV cache
```

## GranuleKV

GranuleKV 的控制面位于 `vllm/granulekv/`，IO 数据面位于
`BaM_IOStack/gids_module/granulekv/`。构建 native 扩展：

```bash
cd /home/xhk/llm-inference/BaM_IOStack
conda run -n pytorch-vllm bash gids_module/granulekv/native/build.sh
cd /home/xhk/llm-inference/vllm-bam
conda run -n pytorch-vllm bash vllm/granulekv/native/build.sh
```

GranuleKV 当前保留：

- GPU/host region 注册；
- SSD -> GPU 和 SSD -> CPU direct transfer；
- descriptor pool、persistent CQ 和 GPU polling；
- MPS service 生命周期；
- vLLM 层级预取、异步提交、完成回收和容量管理。

核心回归测试：

```bash
conda run -n pytorch-vllm pytest -q \
  tests/test_kv_receive_runtime.py \
  tests/core/test_hierarchical_io.py \
  tests/test_xformers_prefix_fallback.py
```

旧 BaM sync、LMCache-BaM shadow、one-copy、GDS wrapper 和 Mooncake row-store
脚本已从当前执行入口移除。历史结果和分析文档仍保留在各自归档目录中，不能
作为当前运行命令使用。
