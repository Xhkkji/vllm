# 20260802 Baseline Results

本文件只保留当前有用结果、复现实验命令，以及必要的失败原因记录。

## 性能结果总览

### 最新有效结果

| 路径 | Workload / 口径 | 样本 / 请求 | swap / BaM 事件 | write avg | read avg | elapsed | 吞吐 / 请求延迟 |
|---|---|---:|---:|---:|---:|---:|---:|
| CPU SSD cold+cgroup chunk=1 | LMCache CPU/POSIX SSD cold path，LongBench TriviaQA，16GB memory cgroup，max_tokens=128 | 25 samples / 50 requests | N/A | 6.2504 s | 4.4428 s | 267.3304 s | 5.3466 s/request |
| BaM direct KVStore real2048 mt128 n8 | vLLM direct KVStore，LongBench TriviaQA 前 25 条，2048 tokens，max_tokens=128 | 25 prompts | swap_out=99, swap_in=99, write_done=99, read_done=99 | 57.151 ms | 53.401 ms | 308.066 s | 10.39 tok/s |

说明：

- 这张表只放最新一轮真正可用的结果：CPU cold+cgroup baseline 和 BaM
  direct KVStore 有效路径。
- CPU 路径是 LMCache chunk=1 proxy，单位是单次 request 秒级延迟。
- BaM direct KVStore 是真实 vLLM swap/preemption 链路，单位是 BaM
  `write/read DONE` 内部毫秒级延迟。
- 本轮 BaM I/O p95：`write=60.081ms`，`read=63.610ms`。
- 两者不是严格同口径横向 speedup 表；当前用途是确认 BaM direct 串行链路在
  vLLM 细粒度 KV swap 场景下可以稳定跑通，并给出可复现实测延迟。

### 当前有效结果

| 路径 | Workload / 口径 | 样本 / 请求 | swap 事件 | write avg | read avg | elapsed | 生成吞吐 / 请求延迟 |
|---|---|---:|---:|---:|---:|---:|---:|
| LMCache CPU SSD cold+cgroup chunk=1 | LongBench TriviaQA / CPU cold path / 16GB cgroup / max_tokens=128 | 25 samples / 50 requests | N/A | 6.2504 s | 4.4428 s | 267.3304 s | 5.3466 s/request |
| BaM direct serial smoke | 合成 2048-token prompts / Qwen2.5-7B / io_active | 8 prompts | swap_out=31, swap_in=31 | 56.306 ms | 42.169 ms | 98.230 s | 10.42 tok/s |
| BaM direct real2048 mt128 n8 | LongBench TriviaQA 前 8 条 / 2048 tokens / max_tokens=128 | 8 prompts | swap_out=124, swap_in=124 | 56.718 ms | 42.274 ms | 98.190 s | 10.43 tok/s |
| BaM direct LongBench real25 | LongBench TriviaQA 前 25 条 / 2048 tokens / max_tokens=128 | 25 prompts | swap_out=99, swap_in=99 | 57.151 ms | 53.401 ms | 308.066 s | 10.39 tok/s |

说明：

- `BaM direct real2048 mt128 n8` 是当前“真实数据内容 + BaM direct 串行
  swap 链路”的主要有效结果。
- `BaM direct serial smoke` 保留为稳定性/链路可用性 baseline。
- CPU 行是 LMCache chunk=1 proxy 下的 CPU/POSIX SSD cold baseline；它与 BaM
  direct 不是严格同口径，只用于当前阶段趋势参考。

### 历史 baseline 表格保留

#### LMCache chunk=1 proxy 对比

| Backend | Requests | Samples | repeat_read | avg_request_s | write_avg_s | read_avg_s |
|---|---:|---:|---:|---:|---:|---:|
| LMCache GDS chunk=1 | 4 | 2 | 1 | 2.9781 | 3.8526 | 2.1035 |
| LMCache CPU SSD cold+cgroup chunk=1 | 4 | 2 | 1 | 2.1431 | 2.6985 | 1.5878 |

该表只作为 LMCache proxy 趋势记录。`chunk_size=1` 会放大 LMCache metadata
和组织开销，不能作为严格 vLLM block-level GDS baseline。

#### BaM direct vs block-level GDS compat

同 workload、同模型、同 `num_gpu_blocks_override=260`、同 preemption/swap 事件
计数下：

| Path | write avg_ms | write p95_ms | read avg_ms | read p95_ms | elapsed_s | generated_tokens_per_sec |
|---|---:|---:|---:|---:|---:|---:|
| BaM direct KVStore | 56.306 | 60.234 | 42.169 | 43.423 | 98.230 | 10.42 |
| GDS block KVStore cuFile compat | 191.986 | 206.809 | 221.781 | 239.976 | 108.173 | 9.47 |

相对结果：

```text
write: GDS compat / BaM = 191.986 / 56.306 = 3.41x slower
read : GDS compat / BaM = 221.781 / 42.169 = 5.26x slower
end-to-end elapsed: GDS compat / BaM = 108.173 / 98.230 = 1.10x slower
decode throughput: BaM / GDS compat = 10.42 / 9.47 = 1.10x faster
```

注意：这里的 `GDS block KVStore cuFile compat` 不是 true NVMe GDS fast path。
当前机器 `gdscheck` 显示 `NVMe : Unsupported` 且
`properties.use_compat_mode : true`，因此这组 GDS 数字只能作为 compat path
参考，不能作为最终 hardware GDS baseline。

## 有效结果

### 1. LMCache CPU SSD cold+cgroup，LongBench TriviaQA

目的：作为 CPU/POSIX SSD 路径 baseline。read request 前 drop page cache，并用
memory cgroup 限制 CPU/page cache 影响。

命令：

```bash
cd /home/xhk/llm-inference/vllm-bam
NUM_SAMPLES=25 REPEAT_READ=1 MAX_TOKENS=128 \
  bash evaluation/20260802baseline/run_cpu_longbench_cold_cgroup.sh
```

数据集：

```text
/home/xhk/llm-inference/datasets/longbench/organized/triviaqa/qwen25/full/buckets/lt4k.jsonl
```

结果目录：

```text
evaluation/20260802baseline/result/cpu_longbench_cgroup/20260802_152748
```

结果：

```text
requests=50
samples=25
repeat_read=1
total_elapsed_s=267.3304
avg_request_s=5.3466
write_avg_s=6.2504
read_avg_s=4.4428
memory_limit_bytes=17179869184
memory.max_usage_in_bytes=17179869184
```

说明：

- 这是当前 CPU 路径可用结果。
- LMCache chunk size 为 1，用于短期趋势验证；它不是严格的 vLLM physical
  block 粒度。

### 2. BaM direct serial smoke，vLLM swap/preemption

目的：验证 BaM direct KVStore 串行链路可以触发真实 vLLM `swap_out/swap_in`，
并完成 BaM direct `write/read DONE`。

命令：

```bash
cd /home/xhk/llm-inference/vllm-bam
bash evaluation/20260802baseline/run_bam_direct_serial_smoke.sh
```

结果目录：

```text
evaluation/20260802baseline/result/bam_direct_serial/20260802_141253
```

结果：

```text
PASS
swap_out=31
swap_in=31
python_exit_status=120

elapsed=98.230s
total_prompt_tokens=16384
total_generated_tokens=1024
prompt_tokens_per_sec=166.79
generated_tokens_per_sec=10.42

BAM write_done=31
BAM read_done=31
BAM write_avg_ms≈56.3
BAM read_avg_ms≈42.2
```

说明：

- `python_exit_status=120` 出现在 `Run summary` 和关键 trace events 之后，
  属于 shutdown/finalization 阶段异常；本轮依据关键链路事件判定为 PASS。
- 该结果使用固定合成 long-context prompts，不是真实 LongBench prompt。

### 3. BaM direct，真实 LongBench 25 samples / real2048 / max_tokens=128

目的：使用 LongBench TriviaQA 的真实 prompt 内容，同时维持能触发 vLLM
swap/preemption 的 BaM direct 压力口径。

口径：

```text
samples=25
prompt_len=2048
prompt source=LongBench TriviaQA lt4k 前 25 条，tokenize 后截断/填充到 2048 tokens
max_tokens=128
best_of=4
num_gpu_blocks_override=260
preemption_mode=swap
max_num_seqs=8
```

运行环境要点：

```bash
cd /home/xhk/llm-inference/vllm-bam
NUM_SAMPLES=25 PROMPT_LEN=2048 MAX_TOKENS=128 \
  BEST_OF=4 MAX_NUM_SEQS=8 NUM_GPU_BLOCKS_OVERRIDE=260 \
  bash evaluation/20260802baseline/run_bam_direct_longbench.sh
```

该脚本内部使用以下 BaM direct 环境：

```bash
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_V0_SWAP_TRACE=1
export VLLM_BAM_DIRECT_KVSTORE_ENABLE=1
export VLLM_BAM_DIRECT_SERVICE_LIFETIME=io_active
export VLLM_BAM_SHADOW_ENABLE=0
export VLLM_BAM_SWAPIN_ENABLE=0
export VLLM_BAM_IMPORT_PATH=/home/xhk/llm-inference/BaM_IOStack/gids_module
export VLLM_BAM_SSD_LIST=0
export PYTHONPATH=/home/xhk/llm-inference/vllm-bam:/home/xhk/llm-inference/BaM_IOStack/gids_module:/home/xhk/llm-inference/BaM_IOStack/gids_module/build:${PYTHONPATH}
export LD_LIBRARY_PATH=/home/xhk/llm-inference/BaM_IOStack/bam/build/lib:${LD_LIBRARY_PATH}
```

结果目录：

```text
evaluation/20260802baseline/result/bam_direct_longbench/20260802_153424
```

结果：

```text
elapsed=308.066s
total_prompt_tokens=51200
total_generated_tokens=3200
prompt_tokens_per_sec=166.20
generated_tokens_per_sec=10.39

swap_out=99
swap_in=99
BAM write_done=99
BAM read_done=99
BAM write_avg_ms=57.151
BAM read_avg_ms=53.401
BAM write_p95_ms=60.081
BAM read_p95_ms=63.610
```

说明：

- 这是当前“真实数据内容 + BaM direct 串行 swap 链路”的有效结果。
- `python_exit_status=120` 出现在关键 trace 和 `Run summary` 之后，按现有
  smoke 判定规则记为 PASS。
- 一个 vLLM physical block 对应一个 logical row，当前 row 大小为
  `917504 bytes = 896 KiB`。
- row 内按 layer/K/V fragment 组织 I/O；当前每个 fragment 约 16 KiB，
  小于 BaM 单 request 128 KiB 上限。

## 失败原因记录

### GDS fast path 未作为有效 baseline

原因：

```text
gdscheck:
NVMe : Unsupported
properties.use_compat_mode : true
```

当前系统 `nvme.ko` 缺少 GDS/NVFS 所需符号：

```text
nvme_v1_register_nvfs_dma_ops
```

因此当前 cuFile 只能走 compat path，不能作为 true NVMe GDS fast path baseline。

### BaM 初始化失败的根因

早期 BaM direct 初始化曾失败：

```text
[ioctl_map] Page mapping kernel request failed: Invalid argument
RuntimeError: Failed to map device memory
```

原因是当时 `libnvm.ko` 被构建成非 CUDA/P2P 版：

```text
ccflags-y 没有 -D_CUDA
KBUILD_EXTRA_SYMBOLS 为空
nm -u libnvm.ko 中没有 nvidia_p2p_* 符号
```

修复方式：

```bash
cd /usr/src/nvidia-535.230.02
sudo make

cd /home/xhk/llm-inference/BaM_IOStack/bam/build
cmake .. -DNVIDIA=/usr/src/nvidia-535.230.02

cd /home/xhk/llm-inference/BaM_IOStack/bam/build/module
make clean
make
sudo make load
```

修复后必须确认：

```bash
grep -nE '_CUDA|KBUILD_EXTRA_SYMBOLS|nvidia|Module.symvers' \
  /home/xhk/llm-inference/BaM_IOStack/bam/build/module/Makefile
nm -u /home/xhk/llm-inference/BaM_IOStack/bam/build/module/libnvm.ko | grep nvidia_p2p
```

期望看到：

```text
-D_CUDA
KBUILD_EXTRA_SYMBOLS := /usr/src/nvidia-535.230.02/Module.symvers
nvidia_p2p_get_pages
nvidia_p2p_dma_map_pages
```

### 普通用户运行 BaM direct 失败

原因：

```text
RuntimeError: Unexpected error while mapping IO memory (cudaHostRegister): operation not permitted
```

BaM direct/vLLM smoke 脚本会自动使用 `sudo -n` 进入 root 环境运行。手写
BaM direct runner 时也需要使用等价的 `sudo -n env ... python ...` 口径。

### raw LongBench 25 prompts 没有触发 BaM I/O

raw LongBench 前 25 条 prompt 直接生成时可以完成：

```text
elapsed=90.185s
total_prompt_tokens=72618
total_generated_tokens=400
```

但 worker trace 中：

```text
swap_in=0
swap_out=0
```

没有 BaM direct `write/read DONE`，因此不能作为 BaM I/O 性能点。

### num_gpu_blocks_override=128 不可用于 max_model_len=4096

原因：

```text
ValueError: The model's max seq len (4096) is larger than the maximum number
of tokens that can be stored in KV cache (2048).
```

`num_gpu_blocks_override=128` 时 KV cache 只能容纳 `128 * 16 = 2048`
tokens，不满足 `max_model_len=4096`。
