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
- `bam_one_copy`：默认 1 service CTA + 4 mover CTA 的 BaM one-copy 主线。

每条样本默认连续跑两次：

- `request_1`：同一条样本第一次生成，用来写入或建立可复用 KV 数据。
- `request_2`：同一条样本第二次生成，用来触发对应 backend 的 KV 读回。

具体来说：

- `ssd_cpu_gpu`：request_2 从 LMCache local_disk 读回。
- `gds_gpu`：request_2 从 GDS wrapper 读回。
- `bam_one_copy`：request_2 走 BaM prefer-load + one-copy direct placement。

默认使用：

```text
/home/xhk/llm-inference/datasets/longbench/organized/triviaqa/qwen25/full/buckets/lt4k.jsonl
```

这个 manifest 已经按 Qwen2.5 tokenizer 的真实 token 数重新分桶，并默认为
`MAX_TOKENS=32` 预留输出空间。旧的 raw-length `full/` 和 `input/` manifest
已经弃用并删除，避免后续误用 LongBench 原始 `length` 字段导致运行中超过
`MAX_MODEL_LEN`。

当前默认样本量是 `NUM_SAMPLES=25`，也就是完整跑 Qwen-tokenized `lt4k`
bucket。这个 bucket 是当前默认配置下最稳的规模，适合先看通路性能和释放
行为。

如果要跑全量 LongBench-TriviaQA 200 条：

```bash
MANIFEST_PATH=/home/xhk/llm-inference/datasets/longbench/organized/triviaqa/qwen25/full/all.jsonl \
NUM_SAMPLES=0 \
MAX_MODEL_LEN=24576 \
bash evaluation/lmcache_ssd_read_paths_baseline/run_longbench_triviaqa_bam_one_copy_qwen25.sh
```

BaM one-copy 只保留两个 topology 开关：

- `GIDS_KV_GPU_WORKER_MOVER_CTAS`：并行搬运 CTA 数，默认 `4`；设为 `1`
  时回归原始单 CTA 路径。
- `GIDS_KV_GPU_WORKER_MODE`：多 CTA 拓扑，默认 `dedicated`（1 service + N
  mover）；`mixed` 保留为稳定正确性对照。

上层链路统一通过 `VLLM_BAM_KV_BRANCH` 选择，底层 runtime、persistent 和
one-copy 开关由启动脚本派生，不再使用旧底层开关组合反推分支。

这里 `NUM_SAMPLES=0` 表示不截断 manifest，直接按全量跑。

默认日志模式只保留性能相关输出：

- runner 配置和逐请求 `elapsed_s`；
- 末尾 `longbench-triviaqa-summary` 汇总；
- BaM cache 命中统计；
- direct placement / read 的性能摘要；
- warning 和 error。

如果需要恢复底层完整调试日志：

```bash
LONGBENCH_DEBUG_LOG=1 bash evaluation/lmcache_ssd_read_paths_baseline/run_longbench_triviaqa_bam_one_copy_qwen25.sh
```

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
MANIFEST_PATH=/home/xhk/llm-inference/datasets/longbench/organized/triviaqa/qwen25/full/buckets/4k_8k.jsonl \
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
