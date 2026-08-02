# 20260802 Baseline

本目录用于真实 LongBench KV swap 趋势验证，当前只保留两条路径：

1. CPU/POSIX SSD cold+cgroup；
2. vLLM BaM direct KVStore。

结果统一写入：

```text
evaluation/20260802baseline/result/
```

## 口径

- `LMCACHE_CHUNK_SIZE=1` 是 1 token chunk，不是严格的 1 个 vLLM physical
  KV block；它只能作为短期趋势验证，不能替代正式 block-level GDS baseline。
- CPU 路径必须使用 cold+cgroup wrapper：每次 read 前 drop page cache，并用
  memory cgroup 限制 CPU/page cache 影响。
- BaM 路径读取真实 LongBench `prompt`，tokenize 后固定到 2048 tokens，
  使用当前已验证的 direct serial `io_active` 生命周期。
- 这里不修改 direct KVStore、LMCache 或 BaM one-copy 主逻辑。

## 脚本

```bash
bash evaluation/20260802baseline/run_cpu_longbench_cold_cgroup.sh
bash evaluation/20260802baseline/run_bam_direct_longbench.sh
```

默认 smoke 口径：

```text
NUM_SAMPLES=25
REPEAT_READ=1
MAX_TOKENS=128
PROMPT_LEN=2048
BEST_OF=4
MAX_NUM_SEQS=8
NUM_GPU_BLOCKS_OVERRIDE=260
```

需要改变样本数时从环境变量覆盖，例如：

```bash
NUM_SAMPLES=50 REPEAT_READ=1 \
  bash evaluation/20260802baseline/run_cpu_longbench_cold_cgroup.sh

NUM_SAMPLES=50 PROMPT_LEN=2048 MAX_TOKENS=128 \
  bash evaluation/20260802baseline/run_bam_direct_longbench.sh
```
