# LMCache SSD Read Path Baseline

这个目录专门放 LMCache 从 SSD 读回 KV chunk 的通路 baseline 对比脚本。

当前保留两条链路：

- `ssd_cpu_gpu`：原生 LMCache V0 local_disk，数据路径是 `SSD -> CPU MemoryObj -> multi_layer_kv_transfer -> vLLM paged KV cache`。
- `gds_gpu`：LMCache-style GDS wrapper，数据路径是 `SSD/cufile -> CUDA chunk tensor -> LMCache MemoryObj -> multi_layer_kv_transfer -> vLLM paged KV cache`。

默认日志目录：

```text
evaluation/lmcache_ssd_read_paths_baseline/logs/<mode>/<timestamp>/run.log
```

运行命令：

```bash
cd /home/xhk/llm-inference/vllm-bam
bash evaluation/lmcache_ssd_read_paths_baseline/run_single_gpu_lmcache_ssd_read_paths_qwen25.sh ssd_cpu_gpu
```

```bash
cd /home/xhk/llm-inference/vllm-bam
bash evaluation/lmcache_ssd_read_paths_baseline/run_single_gpu_lmcache_ssd_read_paths_qwen25.sh gds_gpu
```

如果当前机器没有可用 cuFile/GDS 环境，可以先用 POSIX fallback 验证 wrapper 通路：

```bash
cd /home/xhk/llm-inference/vllm-bam
VLLM_GDS_LMCACHE_USE_GDS=0 bash evaluation/lmcache_ssd_read_paths_baseline/run_single_gpu_lmcache_ssd_read_paths_qwen25.sh gds_gpu
```

## LongBench-TriviaQA Baseline

LongBench-TriviaQA 用于把固定 prompt baseline 扩展到真实长上下文 QA 样本。
当前保留三条链路：

- `ssd_cpu_gpu`：原生 LMCache local_disk。
- `gds_gpu`：LMCache-style GDS wrapper。
- `bam_one_copy`：当前 cta=4 BaM one-copy 稳定基线。

每条样本默认连续跑两次：

- `request_1`：同一条样本第一次生成，用来写入或建立可复用 KV 数据。
- `request_2`：同一条样本第二次生成，用来触发对应 backend 的 KV 读回。

具体来说：

- `ssd_cpu_gpu`：request_2 从 LMCache local_disk 读回。
- `gds_gpu`：request_2 从 GDS wrapper 读回。
- `bam_one_copy`：request_2 走 BaM prefer-load + one-copy direct placement。

默认使用：

```text
/home/xhk/llm-inference/datasets/longbench/organized/triviaqa/full/buckets/lt4k.jsonl
```

原因是当前 Qwen2.5-7B 单卡 baseline 默认 `MAX_MODEL_LEN=4096`。如果要测更长
bucket，需要显式提高 `MAX_MODEL_LEN`，不要让脚本自动改变显存压力。

运行原生 LMCache SSD：

```bash
cd /home/xhk/llm-inference/vllm-bam
bash evaluation/lmcache_ssd_read_paths_baseline/run_longbench_triviaqa_ssd_read_paths_qwen25.sh ssd_cpu_gpu
```

运行 LMCache-style GDS：

```bash
cd /home/xhk/llm-inference/vllm-bam
bash evaluation/lmcache_ssd_read_paths_baseline/run_longbench_triviaqa_ssd_read_paths_qwen25.sh gds_gpu
```

运行 BaM one-copy：

```bash
cd /home/xhk/llm-inference/vllm-bam
bash evaluation/lmcache_ssd_read_paths_baseline/run_longbench_triviaqa_bam_one_copy_qwen25.sh
```

换成 `4k_8k` bucket 的例子：

```bash
cd /home/xhk/llm-inference/vllm-bam
MANIFEST_PATH=/home/xhk/llm-inference/datasets/longbench/organized/triviaqa/full/buckets/4k_8k.jsonl \
MAX_MODEL_LEN=8192 \
NUM_SAMPLES=4 \
bash evaluation/lmcache_ssd_read_paths_baseline/run_longbench_triviaqa_ssd_read_paths_qwen25.sh ssd_cpu_gpu
```

输出目录：

```text
evaluation/lmcache_ssd_read_paths_baseline/logs_longbench_triviaqa/<mode>/<timestamp>/
  run.log
  metrics.jsonl
```

其中 BaM one-copy 的 `<mode>` 固定为 `bam_one_copy`。
