# 20260802 Baseline

本目录保留 CPU/POSIX SSD cold-read 的历史趋势验证脚本和结果。当前可运行的
路径只有原生 LMCache：

```text
SSD -> POSIX read -> CPU buffer -> GPU
```

运行：

```bash
cd /home/xhk/llm-inference/vllm-bam
bash evaluation/20260802baseline/run_cpu_longbench_cold_cgroup.sh
```

结果统一写入：

```text
evaluation/20260802baseline/result/
```

旧 BaM direct KVStore runner 已移除。`result/` 中的旧 BaM 数据仅作历史归档，
不再对应当前代码入口。
