# 20260802 Baseline 结果

## 测试范围

本目录记录一次短期趋势验证：在 LMCache 路径上使用
`LMCACHE_CHUNK_SIZE=1`，观察更细粒度 chunk 访问是否会按预期放大传统
GDS / CPU SSD 路径的压力。这个验证用于决定是否值得继续做严格的 vLLM
block-level GDS baseline。

重要限制：`LMCACHE_CHUNK_SIZE=1` 表示每个 LMCache chunk 包含 1 个 token，
不是 1 个 vLLM physical KV block。因此本结果只能作为趋势探针，不能作为严格的
BaM direct KVStore 对照 baseline。

## 已完成测试

### LMCache GDS，chunk size 1，smoke

命令：

```bash
bash evaluation/20260802baseline/run_lmcache_gds_chunk1.sh
```

运行目录：

```text
evaluation/20260802baseline/result/lmcache_gds_chunk1/20260802_001912
```

summary：

```text
requests=4 samples=2 repeat_read=1 total_elapsed_s=11.9122 avg_request_s=2.9781 write_avg_s=3.8526 read_avg_s=2.1035
```

解释：

- LMCache-style GDS wrapper 在 `chunk_size=1` 的 smoke 口径下可以跑通。
- 这个结果可以作为 sanity check / 趋势点，但不能作为严格 block-level
  baseline 去对比 BaM direct KVStore。

### LMCache CPU SSD cold+cgroup，chunk size 1，smoke

命令：

```bash
sudo -n /usr/bin/bash /home/xhk/llm-inference/vllm-bam/evaluation/20260802baseline/run_lmcache_cpu_cgroup_chunk1.sh
```

运行目录：

```text
evaluation/20260802baseline/result/lmcache_cpu_cgroup_chunk1/20260802_002902
```

summary：

```text
requests=4 samples=2 repeat_read=1 total_elapsed_s=8.5725 avg_request_s=2.1431 write_avg_s=2.6985 read_avg_s=1.5878
```

cold+cgroup 证据：

```text
drop_caches_before_read=True
iter=2 action=sync_drop_caches
iter=4 action=sync_drop_caches
memory_limit_bytes=17179869184
memory.max_usage_in_bytes=17179869184
total_pgmajfault=13200
```

解释：

- CPU SSD 路径在 16GB memory cgroup 下完成。
- read request 前明确执行了 Linux page cache 清理。
- 在同样 `LMCACHE_CHUNK_SIZE=1`、`samples=2`、`repeat_read=1` 配置下，
  这个 smoke 结果比 LMCache-style GDS 更快。考虑到样本数很小，并且
  LMCache metadata / chunk 组织开销会被 `chunk_size=1` 放大，这个结果应被视为
  一个警告：LMCache chunk-size-1 GDS 不是严格 vLLM block-level GDS 的干净替代。

## LMCache chunk=1 proxy 对比

| Backend | Requests | Samples | repeat_read | avg_request_s | write_avg_s | read_avg_s |
|---|---:|---:|---:|---:|---:|---:|
| LMCache GDS chunk=1 | 4 | 2 | 1 | 2.9781 | 3.8526 | 2.1035 |
| LMCache CPU SSD cold+cgroup chunk=1 | 4 | 2 | 1 | 2.1431 | 2.6985 | 1.5878 |

本次 smoke 观察到的趋势：

```text
GDS / CPU read_avg = 2.1035 / 1.5878 = 1.325x slower
```

这个结果不否定 BaM direct KVStore。它说明的是：LMCache chunk-size-1 这个近似
方案引入了足够多的额外开销和形态偏差，只适合作为快速压力信号，不适合作为最终
BaM-vs-GDS 证据。

## 本轮有效结果总览

| 路径 | 口径 | 有效结果 |
|---|---|---|
| LMCache GDS chunk=1 | LMCache proxy / 2 samples | read_avg_s=2.1035 |
| LMCache CPU SSD cold+cgroup chunk=1 | LMCache proxy / 2 samples / drop cache / 16GB cgroup | read_avg_s=1.5878 |
| BaM direct KVStore | vLLM preemption swap / Qwen2.5-7B / io_active | read avg_ms=42.169, write avg_ms=56.306 |
| GDS block KVStore | vLLM preemption swap / Qwen2.5-7B / cuFile sync compat | read avg_ms=221.781, write avg_ms=191.986 |

注意：前三者不是严格同口径横向对比。LMCache 两条路径是 `chunk_size=1`
proxy，用于判断这种 proxy 是否可靠；BaM direct 是真实 vLLM physical block
swap/preemption 链路，用于验证当前串行链路是否能跑通并得到 direct IO 内部延迟。

## Block-level GDS baseline

### 实现口径

新增最小 block-level GDS baseline：

```text
vllm/bam/gds_block_store.py
```

接入方式：

```text
CacheEngine.swap_out(src_to_dst)
  -> GDSBlockKVStore.swap_out
  -> pack 一个 vLLM physical block 的全部 layer/K/V fragments 到 CUDA staging
  -> cuFileWrite 到单 slab 文件

CacheEngine.swap_in(src_to_dst)
  -> GDSBlockKVStore.swap_in
  -> cuFileRead 到 CUDA staging
  -> scatter 回 vLLM physical KV block
```

新增开关：

```text
VLLM_GDS_BLOCK_KVSTORE_ENABLE=1
VLLM_GDS_BLOCK_SLAB_PATH=<path>/kv_slab.bin
VLLM_GDS_BLOCK_SLAB_GB=0
VLLM_GDS_BLOCK_USE_DIRECT_IO=1
```

测试脚本：

```text
evaluation/20260802baseline/run_gds_block_serial_smoke.sh
```

第一版有意保持最简化：

- 单 slab 文件；
- 每个 vLLM storage block 一次 cuFile transfer；
- CUDA staging buffer 复用；
- 串行同步完成，不做 queue depth、并发或 IO 合并优化；
- 与 BaM direct 使用同一个 `src_to_dst` swap mapping 和同一个 vLLM preemption workload。

### GDS 环境结论

当前机器 `gdscheck -p` 显示：

```text
NVMe : Unsupported
properties.use_compat_mode : true
GPU Tesla V100S supports GDS
IOMMU: disabled
```

因此当前 block-level GDS baseline 实际是 cuFile API 的 compat/sync 路径，不是
NVMe 硬件 GDS fast path。最初使用 `cuFileReadAsync/cuFileWriteAsync` 时，独立
probe 和 vLLM workload 都返回：

```text
GDS write transferred -5 bytes
```

随后改为同步 `cuFileRead/cuFileWrite`，probe 和完整 workload 均可跑通。这个结果
仍是传统 CPU/cuFile submission baseline，但需要在报告中标注为 `cuFile compat`
而不是硬件 NVMe GDS。

### GDS block 串行链路结果

命令：

```bash
bash evaluation/20260802baseline/run_gds_block_serial_smoke.sh
```

运行目录：

```text
evaluation/20260802baseline/result/gds_block_serial/20260802_094948
```

trace log：

```text
evaluation/20260802baseline/result/gds_block_serial/20260802_094948/v0_swap_trace_Qwen2.5-7B-Instruct_20260802_094950.log
```

关键事件计数：

```text
op=swap_out                              124
op=swap_in                               124
[GDS_BLOCK_KVSTORE] op=write phase=done  31
[GDS_BLOCK_KVSTORE] op=read phase=done   31
Run summary                              1
```

GDS block 内部 IO 汇总：

```text
write count=31 avg_blocks=129.52 avg_ms=191.986 p50_ms=189.253 p95_ms=206.809
read  count=31 avg_blocks=129.52 avg_ms=221.781 p50_ms=223.851 p95_ms=239.976
```

Run summary：

```text
elapsed=108.173s
total_prompt_tokens=16384
total_generated_tokens=1024
prompt_tokens_per_sec=151.46
generated_tokens_per_sec=9.47
```

### BaM direct vs block-level GDS

同 workload、同模型、同 `num_gpu_blocks_override=260`、同 preemption/swap 事件
计数下：

| Path | write avg_ms | write p95_ms | read avg_ms | read p95_ms | elapsed_s | generated_tokens_per_sec |
|---|---:|---:|---:|---:|---:|---:|
| BaM direct KVStore | 56.306 | 60.234 | 42.169 | 43.423 | 98.230 | 10.42 |
| GDS block KVStore cuFile compat | 191.986 | 206.809 | 221.781 | 239.976 | 108.173 | 9.47 |

相对结果：

```text
write: GDS / BaM = 191.986 / 56.306 = 3.41x slower
read : GDS / BaM = 221.781 / 42.169 = 5.26x slower
end-to-end elapsed: GDS / BaM = 108.173 / 98.230 = 1.10x slower
decode throughput: BaM / GDS = 10.42 / 9.47 = 1.10x faster
```

解释：

- 这组结果比 LMCache `chunk_size=1` proxy 更干净：GDS block 与 BaM direct 都吃
  vLLM scheduler 生成的 physical block swap mapping。
- 当前 GDS 数字是 cuFile compat/sync 路径，不是 NVMe hardware GDS fast path；因此
  结论应表述为“在当前机器可用的传统 cuFile compat block-level baseline 下，
  BaM direct 明显更快”。
- BaM direct 的优势主要体现在 swap IO 内部延迟；端到端只快约 10%，因为 request
  总时间还包含 forward/decode/scheduler 等非 IO 部分。

## BaM direct 串行链路尝试

本轮已补充 BaM direct KVStore 串行 smoke wrapper：

```text
evaluation/20260802baseline/run_bam_direct_serial_smoke.sh
```

该脚本使用当前 direct KVStore 的 `io_active` 生命周期，目标是验证真实 vLLM
preemption/swap 链路：

```text
swap_out -> BaM direct write DONE -> forward -> swap_in -> BaM direct read DONE -> Run summary
```

命令：

```bash
bash evaluation/20260802baseline/run_bam_direct_serial_smoke.sh
```

运行目录：

```text
evaluation/20260802baseline/result/bam_direct_serial/20260802_005218
```

状态：未完成。失败发生在 BaM/GIDS controller 初始化阶段，尚未进入 request loop。

失败信息：

```text
RuntimeError: Failed to open descriptor: Permission denied
```

定位：

```text
SSD index: 0
/dev/libnvm0 -> crw------- root root
```

解释：

- direct KVStore 会按 `VLLM_BAM_SSD_LIST=0` 打开 `/dev/libnvm0`。
- 当前设备节点权限是 `root:root 0600`，普通用户 `xhk` 无法 `O_RDWR` 打开。
- 这是设备权限问题，不是 BaM direct 数据面或 `io_active` 生命周期的性能结果。

下一步需要用户手动执行最小授权，然后再重跑 BaM direct smoke：

```bash
sudo chgrp xhk /dev/libnvm0
sudo chmod g+rw /dev/libnvm0
```

重跑命令：

```bash
cd /home/xhk/llm-inference/vllm-bam
bash evaluation/20260802baseline/run_bam_direct_serial_smoke.sh
```

注意：这是临时设备节点权限；如果 `/dev/libnvm0` 由模块重载或 udev 重新创建，权限可能需要重新设置。

### BaM direct 串行链路第二次尝试

手动添加 `/dev/libnvm0` 权限后重跑：

```bash
bash evaluation/20260802baseline/run_bam_direct_serial_smoke.sh
```

运行目录：

```text
evaluation/20260802baseline/result/bam_direct_serial/20260802_005649
```

状态：仍未完成。此次已经越过 `/dev/libnvm0` descriptor open，但在
`cudaHostRegisterIoMemory` 注册 controller MMIO/IO memory 时失败：

```text
RuntimeError: Unexpected error while mapping IO memory (cudaHostRegister): operation not permitted
```

解释：

- `/dev/libnvm0` 文件权限已经不是当前阻塞点。
- BaM controller 初始化会对 NVMe controller memory 调用
  `cudaHostRegister(..., cudaHostRegisterIoMemory)`。
- 该操作需要比普通用户文件读写权限更高的进程权限；按照 BaM 原始 README 的运行方式，
  这类应用通常需要 `sudo` 执行。

下一步需要用户明确授权 root 方式运行 BaM direct workload。建议命令：

```bash
cd /home/xhk/llm-inference/vllm-bam
sudo -n /usr/bin/bash evaluation/20260802baseline/run_bam_direct_serial_smoke.sh
```

如果 `sudo -n` 不可用，可手动执行：

```bash
cd /home/xhk/llm-inference/vllm-bam
sudo /usr/bin/bash evaluation/20260802baseline/run_bam_direct_serial_smoke.sh
```

### BaM direct wrapper sudo 化

参考已有 `/usr/local/sbin/run-bam-one-copy-qwen25` 和
`run_longbench_triviaqa_bam_one_copy_qwen25.sh`，当前
`run_bam_direct_serial_smoke.sh` 已加入 sudo re-entry：

```text
non-root shell -> sudo -n env ... /usr/bin/bash run_bam_direct_serial_smoke.sh
```

该改动只影响测试 wrapper，不修改 direct KVStore 实现逻辑。

历史阻塞：

```text
sudo: a password is required
```

说明当时 sudoers 还没有授权这条新 wrapper 的 root 执行入口。后续用户添加权限后，
脚本已进入 root 路径。

### BaM direct 串行链路第三次尝试：完成

修复 wrapper 中空 `CUDA_VISIBLE_DEVICES` 透传问题后，BaM direct 串行链路完成
真实 Qwen2.5-7B-Instruct vLLM preemption/swap workload。

命令：

```bash
bash evaluation/20260802baseline/run_bam_direct_serial_smoke.sh
```

运行目录：

```text
evaluation/20260802baseline/result/bam_direct_serial/20260802_010720
```

trace log：

```text
evaluation/20260802baseline/result/bam_direct_serial/20260802_010720/v0_swap_trace_Qwen2.5-7B-Instruct_20260802_010723.log
```

关键事件计数：

```text
op=swap_out                              124
op=swap_in                               124
[BAM_DIRECT_KVSTORE] op=write phase=done 31
[BAM_DIRECT_KVSTORE] op=read phase=done  31
Run summary                              1
```

BaM direct 内部 IO 汇总：

```text
write count=31 avg_blocks=129.52 avg_ms=56.306 p50_ms=56.048 p95_ms=60.234
read  count=31 avg_blocks=129.52 avg_ms=42.169 p50_ms=41.776 p95_ms=43.423
```

Run summary：

```text
elapsed=98.230s
total_prompt_tokens=16384
total_generated_tokens=1024
prompt_tokens_per_sec=166.79
generated_tokens_per_sec=10.42
```

解释：

- 当前 `io_active` 生命周期下，BaM direct KVStore 串行链路可以跑通真实 vLLM
  swap/preemption workload。
- 链路中同时出现了 vLLM scheduler 的 `swap_out/swap_in` 和 direct KVStore 的
  `write/read DONE`，说明不是只跑了普通 decode。
- 本次每个 direct IO batch 平均约 129.5 个 vLLM KV blocks；read 平均约
  42.2 ms，write 平均约 56.3 ms。
- Python 进程在 `Run summary` 后返回了 exit code 120，并打印
  `Exception ignored in sys.unraisablehook`；但所需关键事件和 summary 已经完整记录。
  wrapper 已改为先记录 Python exit status，再依据关键 trace events 判定 smoke 是否
  完成，避免将这种 shutdown 异常误判为链路失败。

## 之前失败的尝试

### LMCache CPU SSD cold+cgroup，chunk size 1

状态：sudo 权限添加后第一次尝试执行，但在进入 request loop 前失败。原因是 root
cgroup wrapper 无法连接当时仍在运行的 MPS daemon。

运行目录：

```text
evaluation/20260802baseline/result/lmcache_cpu_cgroup_chunk1/20260802_002430
```

失败信息：

```text
RuntimeError: cudaGetDeviceCount() failed with Error 805: MPS client failed to connect to the MPS control daemon or the MPS server
```

失败 run 的 cgroup 清理统计：

```text
memory_limit_bytes=17179869184
memory.max_usage_in_bytes=1515429888
total_pgmajfault=3828
```

解释：

- CPU cgroup wrapper 必须以 root 身份运行，因为它要创建 memory cgroup 并清理
  page cache。
- 当时机器上还有用户 `xhk` 启动的 MPS daemon；root CUDA client 在该 MPS 状态下
  初始化 CUDA 失败。
- 这是启动环境问题，不是 CPU SSD baseline 的有效结果。

成功重跑时采用的恢复方式：先停止 user-owned MPS，再运行 root cgroup workload。
CPU 路径不需要 MPS。

```bash
echo quit | nvidia-cuda-mps-control
sudo -n /usr/bin/bash /home/xhk/llm-inference/vllm-bam/evaluation/20260802baseline/run_lmcache_cpu_cgroup_chunk1.sh
```

sudoers 更新前的权限状态：

```text
sudo -n true -> sudo: a password is required
```

CPU run 复用了已有 cold+cgroup wrapper：read request 前 drop page cache，并在
memory cgroup 下运行。成功结果写入：

```text
evaluation/20260802baseline/result/lmcache_cpu_cgroup_chunk1/20260802_002902
```
