# BaM vs Native GDS Trace Replay Plan

这条线用于验证 `BaM` 和原生 `GDS/cuFile` 数据面的差距，不改变当前已经
跑通的 `LMCache + BaM` 和 vLLM V0 主路径。

## 当前代码组织

- `vllm/bam/gds_baseline/chunk_store_base.py`
  定义 BaM/GDS 共用的 chunk store 接口。
- `vllm/bam/gds_baseline/cufile_context.py`
  从 LMCache V1 GDS 路径抽出的最小 cuFile slab 读写逻辑。
- `vllm/bam/gds_baseline/gds_chunk_store.py`
  用原生 GDS/cuFile 实现 chunk 级 put/get。
- `vllm/bam/gds_baseline/lmcache_style_gds_store.py`
  更贴近 LMCache V1 `GdsBackend` 的文件组织：两级 hash 目录、
  `.kvcache.safetensors` 数据文件、4KB metadata header、`.metadata`
  元数据文件。
- `vllm/bam/lmcache_gds_storage.py`
  LMCache V0 可选 wrapper，通过开关把 LMCache-style GDS 接到 V0
  `storage_manager` 的 put/get 生命周期里。
- `vllm/bam/gds_baseline/trace_schema.py`
  定义后续真实 LMCache chunk trace 的 JSONL 格式。
- `evaluation/kv_chunk_trace_replay.py`
  用同一批 chunk 负载 replay BaM 和 GDS。
- `evaluation/run_bam_vs_gds_trace_replay.sh`
  一键运行脚本。

## 复用范围

这条 baseline 复用的是 LMCache V1 GDS 的核心数据面思想：

- slab file
- `O_DIRECT`
- cuFile handle
- CUDA buffer registration
- `cuFileReadAsync` / `cuFileWriteAsync`

它没有迁移 LMCache V1 的完整 StorageManager、LRU、目录分片和 vLLM V1
connector，因为当前 V100 不能跑 vLLM V1。

## 运行命令

```bash
cd /home/xhk/llm-inference/vllm-bam
sudo -v
bash evaluation/run_bam_vs_gds_trace_replay.sh
```

只跑 BaM：

```bash
BACKEND=bam bash evaluation/run_bam_vs_gds_trace_replay.sh
```

只跑 GDS：

```bash
BACKEND=gds bash evaluation/run_bam_vs_gds_trace_replay.sh
```

只跑 LMCache-style GDS 文件组织：

```bash
BACKEND=lmcache_gds bash evaluation/run_bam_vs_gds_trace_replay.sh
```

如果当前环境还没有 `cufile` Python bindings，可以先用 POSIX fallback
只验证文件组织和接口：

```bash
BACKEND=lmcache_gds LMCACHE_GDS_USE_POSIX=1 bash evaluation/run_bam_vs_gds_trace_replay.sh
```

## 默认负载

默认合成负载和当前 Qwen2.5-7B LMCache chunk 对齐：

- shape: `[2, 28, 256, 512]`
- dtype: `float16`
- chunk size: `14 MiB`
- chunk 数: `8`

## 注意

如果当前环境没有安装 `cufile` Python bindings 或没有可用的 `libcufile.so`，
GDS backend 会在初始化时报错。这不是 BaM 路径问题，而是原生 GDS 环境
还没有准备好。

## 端到端开关

当前 V0 LMCache connector 里已经预留了可选 wrapper。默认全关，不影响
原生 LMCache SSD，也不影响 BaM：

```bash
VLLM_GDS_LMCACHE_SHADOW_ENABLE=1
VLLM_GDS_LMCACHE_PREFER_LOAD_ENABLE=1
VLLM_GDS_LMCACHE_PATH=/tmp/vllm-bam-lmcache-gds
VLLM_GDS_LMCACHE_USE_GDS=1
VLLM_GDS_LMCACHE_USE_DIRECT_IO=1
```

调试文件组织时也可以先关闭 cuFile：

```bash
VLLM_GDS_LMCACHE_USE_GDS=0
```
