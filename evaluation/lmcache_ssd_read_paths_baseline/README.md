# LMCache SSD Read Path Baseline

这个目录维护原生 LMCache 从 SSD 读回 KV chunk 的 baseline。当前只保留：

```text
SSD -> CPU MemoryObj -> multi_layer_kv_transfer -> vLLM paged KV cache
```

## 固定 Prompt

```bash
cd /home/xhk/llm-inference/vllm-bam
bash evaluation/lmcache_ssd_read_paths_baseline/run_single_gpu_lmcache_ssd_read_paths_qwen25.sh ssd_cpu_gpu
```

## LongBench-TriviaQA

```bash
cd /home/xhk/llm-inference/vllm-bam
bash evaluation/lmcache_ssd_read_paths_baseline/run_longbench_triviaqa_ssd_read_paths_qwen25.sh ssd_cpu_gpu
```

默认使用 Qwen2.5 token 数分桶后的 `lt4k.jsonl`，每条样本先执行一次 write，
再执行一次或多次 read。可通过 `MANIFEST_PATH`、`NUM_SAMPLES`、`REPEAT_READ`
和 `MAX_MODEL_LEN` 调整规模。

冷读版本会在每次 read 前执行 `sync` 和 `drop_caches`，并可用 cgroup 限制
CPU/page cache 影响：

```bash
bash evaluation/lmcache_ssd_read_paths_baseline/run_longbench_triviaqa_lmcache_ssd_cold_cgroup_qwen25.sh
```

日志和 `metrics.jsonl` 默认写入：

```text
evaluation/lmcache_ssd_read_paths_baseline/logs_longbench_triviaqa/ssd_cpu_gpu/<timestamp>/
```

本目录不再提供 BaM、GDS wrapper 或 one-copy runner。相关旧结果文件保留为
历史记录，不属于当前 baseline 入口。
