# 20260802 Baseline

本目录用于短期趋势验证：在 LMCache 路径上把 `LMCACHE_CHUNK_SIZE=1`，
观察更细粒度 chunk 访问下传统 GDS 与 CPU SSD cold+cgroup 路径的表现。

结果统一写入：

```text
evaluation/20260802baseline/result/
```

## 口径

- `LMCACHE_CHUNK_SIZE=1` 是 1 token chunk，不是严格的 1 个 vLLM physical
  KV block；它只能作为短期趋势验证，不能替代正式 block-level GDS baseline。
- GDS 路径复用现有 `lmcache_ssd_read_paths_baseline` 的 LMCache-style GDS
  wrapper。
- CPU 路径必须使用 cold+cgroup wrapper：每次 read 前 drop page cache，并用
  memory cgroup 限制 CPU/page cache 影响。
- 这里不修改 direct KVStore、LMCache、GDS 或旧 BaM one-copy 主逻辑。

## 脚本

```bash
bash evaluation/20260802baseline/run_lmcache_gds_chunk1.sh
sudo -n bash evaluation/20260802baseline/run_lmcache_cpu_cgroup_chunk1.sh
bash evaluation/20260802baseline/run_bam_direct_serial_smoke.sh
bash evaluation/20260802baseline/run_gds_block_serial_smoke.sh
```

默认 smoke 口径：

```text
NUM_SAMPLES=2
REPEAT_READ=1
MAX_TOKENS=16
LMCACHE_CHUNK_SIZE=1
```

需要扩大样本时从环境变量覆盖，例如：

```bash
NUM_SAMPLES=25 REPEAT_READ=1 bash evaluation/20260802baseline/run_lmcache_gds_chunk1.sh
sudo -n NUM_SAMPLES=25 REPEAT_READ=1 bash evaluation/20260802baseline/run_lmcache_cpu_cgroup_chunk1.sh
```
