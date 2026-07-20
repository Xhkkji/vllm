#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# 用和当前 BaM one-copy 基线相同的单请求样例，单独测试 LMCache 从 SSD
# 读回 KV chunk 的两条链路：
#
# 1. ssd_cpu_gpu
#    原生 LMCache V0 local_disk:
#      SSD -> CPU MemoryObj -> multi_layer_kv_transfer -> vLLM paged KV cache
#
# 2. gds_gpu
#    vLLM-BaM 仓库里整理出的 LMCache-style GDS wrapper:
#      SSD/cufile -> CUDA chunk tensor -> LMCache MemoryObj -> vLLM paged KV cache
#
# 注意：当前 LMCache V0 本体没有直接把 GDS backend 接成 StorageManager 的
# 原生后端；这里的 GDS 路径是我们包在 storage_manager 外层的对照 wrapper。

MODE="${1:-${LMCACHE_SSD_READ_PATH:-ssd_cpu_gpu}}"
case "${MODE}" in
  ssd|ssd_cpu|ssd_cpu_gpu)
    MODE="ssd_cpu_gpu"
    GDS_SHADOW_ENABLE="0"
    GDS_PREFER_LOAD_ENABLE="0"
    ;;
  gds|gds_gpu|ssd_gds_gpu)
    MODE="gds_gpu"
    GDS_SHADOW_ENABLE="1"
    GDS_PREFER_LOAD_ENABLE="1"
    ;;
  *)
    echo "[lmcache-ssd-read-paths] unsupported mode=${MODE}" >&2
    echo "[lmcache-ssd-read-paths] valid modes: ssd_cpu_gpu, gds_gpu" >&2
    exit 2
    ;;
esac

LOG_ROOT_VALUE="${LOG_ROOT:-${SCRIPT_DIR}/logs/${MODE}}"
RUN_DIR_VALUE="${RUN_DIR:-${LOG_ROOT_VALUE}/${TIMESTAMP}}"
LOG_FILE_VALUE="${LOG_FILE:-${RUN_DIR_VALUE}/run.log}"
TEST_ROOT_VALUE="${LMCACHE_SSD_TEST_ROOT:-${RUN_DIR_VALUE}}"

# 每轮测试默认使用独立目录，避免命中历史 chunk 后看不清 request_1 写入、
# request_2 从 SSD/GDS 读回的实际链路。需要复用历史数据时，可以外部显式
# 覆盖 LMCACHE_LOCAL_DISK 或 VLLM_GDS_LMCACHE_PATH。
LMCACHE_LOCAL_DISK_VALUE="${LMCACHE_LOCAL_DISK:-${TEST_ROOT_VALUE}/lmcache_local_disk/}"
VLLM_GDS_LMCACHE_PATH_VALUE="${VLLM_GDS_LMCACHE_PATH:-${TEST_ROOT_VALUE}/lmcache_gds/}"

mkdir -p "${LMCACHE_LOCAL_DISK_VALUE}"
if [[ "${MODE}" == "gds_gpu" ]]; then
  mkdir -p "${VLLM_GDS_LMCACHE_PATH_VALUE}"
fi

echo "[lmcache-ssd-read-paths] mode=${MODE}"
echo "[lmcache-ssd-read-paths] run_dir=${RUN_DIR_VALUE}"
echo "[lmcache-ssd-read-paths] lmcache_local_disk=${LMCACHE_LOCAL_DISK_VALUE}"
if [[ "${MODE}" == "gds_gpu" ]]; then
  echo "[lmcache-ssd-read-paths] gds_path=${VLLM_GDS_LMCACHE_PATH_VALUE}"
fi

LMCACHE_REPO_PATH="${LMCACHE_REPO_PATH:-/home/xhk/llm-inference/LMCache-v0-torch26}" \
PYTHON_BIN="${PYTHON_BIN:-/home/xhk/miniconda3/envs/pytorch-vllm/bin/python}" \
LOG_ROOT="${LOG_ROOT_VALUE}" \
RUN_DIR="${RUN_DIR_VALUE}" \
LOG_FILE="${LOG_FILE_VALUE}" \
LMCACHE_USE_EXPERIMENTAL="${LMCACHE_USE_EXPERIMENTAL:-True}" \
LMCACHE_LOCAL_CPU="${LMCACHE_LOCAL_CPU:-False}" \
LMCACHE_LOCAL_DISK="${LMCACHE_LOCAL_DISK_VALUE}" \
LMCACHE_MAX_LOCAL_DISK_SIZE="${LMCACHE_MAX_LOCAL_DISK_SIZE:-20}" \
PROMPT_REUSE_REQUEST1="${PROMPT_REUSE_REQUEST1:-1}" \
DIRECT_RETRIEVE_PREWARM_AFTER_REQUEST1="${DIRECT_RETRIEVE_PREWARM_AFTER_REQUEST1:-0}" \
VLLM_BAM_LMCACHE_SHADOW_ENABLE=0 \
VLLM_BAM_LMCACHE_PREFER_LOAD_ENABLE=0 \
VLLM_BAM_KV_FAST_PATH=0 \
VLLM_BAM_DIRECT_PLACEMENT=0 \
VLLM_BAM_DIRECT_PLACEMENT_DEFER_RUNTIME=0 \
VLLM_BAM_KV_BRANCH=rowctx_baseline \
GIDS_KV_GPU_WORKER_RUNTIME_ENABLE=0 \
GIDS_KV_GPU_WORKER_PERSISTENT_ENABLE=0 \
GIDS_KV_GPU_WORKER_MOVER_CTAS=0 \
VLLM_GDS_LMCACHE_SHADOW_ENABLE="${GDS_SHADOW_ENABLE}" \
VLLM_GDS_LMCACHE_PREFER_LOAD_ENABLE="${GDS_PREFER_LOAD_ENABLE}" \
VLLM_GDS_LMCACHE_PATH="${VLLM_GDS_LMCACHE_PATH_VALUE}" \
VLLM_GDS_LMCACHE_USE_GDS="${VLLM_GDS_LMCACHE_USE_GDS:-1}" \
VLLM_GDS_LMCACHE_USE_DIRECT_IO="${VLLM_GDS_LMCACHE_USE_DIRECT_IO:-1}" \
VLLM_GDS_LMCACHE_FMT="${VLLM_GDS_LMCACHE_FMT:-KV_2LTD}" \
VLLM_GDS_LMCACHE_USE_REGISTERED_BUFFER="${VLLM_GDS_LMCACHE_USE_REGISTERED_BUFFER:-0}" \
VLLM_GDS_LMCACHE_REGISTERED_BUFFER_MB="${VLLM_GDS_LMCACHE_REGISTERED_BUFFER_MB:-0}" \
VLLM_GDS_LMCACHE_VERIFY_BUDGET="${VLLM_GDS_LMCACHE_VERIFY_BUDGET:-0}" \
bash "${SCRIPT_DIR}/../run_single_gpu_lmcache_no_prefix_reuse_qwen25.sh"
