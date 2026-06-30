#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT_DIR}/evaluation/logs/bam_vs_gds_trace_replay/${TIMESTAMP}"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/run.log"

PYTHON_BIN="${PYTHON_BIN:-/home/xhk/miniconda3/envs/pytorch-vllm/bin/python}"
BAM_LIB_DIR="${BAM_LIB_DIR:-/home/xhk/llm-inference/BaM_IOStack/bam/build/lib}"
GIDS_MODULE_DIR="${GIDS_MODULE_DIR:-/home/xhk/llm-inference/BaM_IOStack/gids_module}"
BACKEND="${BACKEND:-all}"
NUM_CHUNKS="${NUM_CHUNKS:-8}"
NUM_LAYERS="${NUM_LAYERS:-28}"
SLOT_NUM_TOKENS="${SLOT_NUM_TOKENS:-256}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
DTYPE="${DTYPE:-float16}"
DEVICE="${DEVICE:-cuda:0}"
GDS_SLAB_PATH="${GDS_SLAB_PATH:-${LOG_DIR}/gds_slab/lmcache_gds_slab.bin}"
GDS_SLAB_GB="${GDS_SLAB_GB:-4}"
LMCACHE_GDS_PATH="${LMCACHE_GDS_PATH:-${LOG_DIR}/lmcache_gds}"
LMCACHE_GDS_USE_POSIX="${LMCACHE_GDS_USE_POSIX:-0}"
LMCACHE_GDS_USE_DIRECT_IO="${LMCACHE_GDS_USE_DIRECT_IO:-1}"
LMCACHE_GDS_FMT="${LMCACHE_GDS_FMT:-KV_2LTD}"
LMCACHE_GDS_USE_REGISTERED_BUFFER="${LMCACHE_GDS_USE_REGISTERED_BUFFER:-0}"
LMCACHE_GDS_REGISTERED_BUFFER_MB="${LMCACHE_GDS_REGISTERED_BUFFER_MB:-0}"
CUFILE_LD_PRELOAD="${CUFILE_LD_PRELOAD:-/usr/lib/x86_64-linux-gnu/librt.so.1}"
SUMMARY_WARMUP_SAMPLES="${SUMMARY_WARMUP_SAMPLES:-1}"
BATCH_PREFETCH_WARMUP="${BATCH_PREFETCH_WARMUP:-1}"
BAM_LMCACHE_READ_MODE="${BAM_LMCACHE_READ_MODE:-}"
if [[ -z "${BAM_LMCACHE_READ_MODE}" ]]; then
  if [[ "${BACKEND}" == "bam_prefetch" || "${BACKEND}" == "bam_prefetch_batch" || \
        "${BACKEND}" == "bam_kv_fast_path" || "${BACKEND}" == "bam_kv_fast_path_batch" ]]; then
    # replay 的 prefetch/KV backend 都需要 rowctx，不能强制 GIDS 同步读。
    BAM_LMCACHE_READ_MODE="prefetch"
  else
    BAM_LMCACHE_READ_MODE="sync"
  fi
fi
if [[ "${BACKEND}" == "bam_kv_fast_path" || "${BACKEND}" == "bam_kv_fast_path_batch" ]]; then
  VLLM_BAM_KV_FAST_PATH_VALUE="${VLLM_BAM_KV_FAST_PATH:-1}"
else
  VLLM_BAM_KV_FAST_PATH_VALUE="${VLLM_BAM_KV_FAST_PATH:-0}"
fi
VLLM_BAM_KV_EXECUTOR_VALUE="${VLLM_BAM_KV_EXECUTOR:-rowctx}"
TRACE_JSONL="${TRACE_JSONL:-}"

# 只有 BaM 需要访问 /dev/libnvm0，GDS/LMCache-GDS baseline 不需要 sudo。
if [[ "${BACKEND}" == "bam" || "${BACKEND}" == "bam_prefetch" || \
      "${BACKEND}" == "bam_prefetch_batch" || \
      "${BACKEND}" == "bam_kv_fast_path" || \
      "${BACKEND}" == "bam_kv_fast_path_batch" || \
      "${BACKEND}" == "bam_cold_read" || "${BACKEND}" == "all" ]]; then
  USE_SUDO="${USE_SUDO:-1}"
else
  USE_SUDO="${USE_SUDO:-0}"
fi

case "${DTYPE}" in
  float16) DTYPE_BYTES=2 ;;
  float32) DTYPE_BYTES=4 ;;
  *)
    echo "Unsupported DTYPE for BaM cache sizing: ${DTYPE}" >&2
    exit 1
    ;;
esac

PAGE_BYTES=131072
KV_LAYER_BYTES=$((SLOT_NUM_TOKENS * HIDDEN_DIM * DTYPE_BYTES))
PAGES_PER_KV_LAYER=$(((KV_LAYER_BYTES + PAGE_BYTES - 1) / PAGE_BYTES))
PAGES_PER_CHUNK=$((2 * NUM_LAYERS * PAGES_PER_KV_LAYER))
REPLAY_CACHE_MB=$(((NUM_CHUNKS * PAGES_PER_CHUNK * PAGE_BYTES + 1024 * 1024 - 1) / (1024 * 1024)))
AUTO_BAM_CACHE_SIZE_MB=$((((REPLAY_CACHE_MB + 63) / 64) * 64))
if (( AUTO_BAM_CACHE_SIZE_MB < 64 )); then
  # 只按本次 replay 真实需要的 chunk/page 数给 cache。
  # 之前默认强行 1GB 会让 BaM flush_cache 扫很多未使用 page，
  # 在 128KB row 路径上更容易触发底层 page-cache 状态机问题。
  AUTO_BAM_CACHE_SIZE_MB=64
fi
BAM_CACHE_SIZE_MB="${VLLM_BAM_CACHE_SIZE_MB:-${AUTO_BAM_CACHE_SIZE_MB}}"

GIDS_FORCE_SYNC_READ_VALUE="${GIDS_FORCE_SYNC_READ:-}"
if [[ -z "${GIDS_FORCE_SYNC_READ_VALUE}" && "${BAM_LMCACHE_READ_MODE}" == "sync" ]]; then
  # 同步读路径用于先验证正确性，避免 BaM rowctx/prefetch 实验路径影响主线对比。
  GIDS_FORCE_SYNC_READ_VALUE="1"
fi

CMD=(env
  "LD_PRELOAD=${CUFILE_LD_PRELOAD}${LD_PRELOAD:+:${LD_PRELOAD}}"
  "LD_LIBRARY_PATH=${BAM_LIB_DIR}:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
  "PYTHONPATH=${ROOT_DIR}:${PYTHONPATH:-}"
  "VLLM_BAM_IMPORT_PATH=${GIDS_MODULE_DIR}"
  "VLLM_BAM_CACHE_SIZE_MB=${BAM_CACHE_SIZE_MB}"
  "VLLM_BAM_NUM_SSD=${VLLM_BAM_NUM_SSD:-1}"
  "VLLM_BAM_SSD_LIST=${VLLM_BAM_SSD_LIST:-0}"
  "VLLM_BAM_CTRL_IDX=${VLLM_BAM_CTRL_IDX:-0}"
  "VLLM_BAM_LMCACHE_SHADOW_CHUNKS=${VLLM_BAM_LMCACHE_SHADOW_CHUNKS:-1024}"
  "VLLM_BAM_LMCACHE_READ_MODE=${BAM_LMCACHE_READ_MODE}"
  "VLLM_BAM_KV_FAST_PATH=${VLLM_BAM_KV_FAST_PATH_VALUE}"
  "VLLM_BAM_KV_EXECUTOR=${VLLM_BAM_KV_EXECUTOR_VALUE}"
  "GIDS_FORCE_SYNC_READ=${GIDS_FORCE_SYNC_READ_VALUE}"
  "LMCACHE_CHUNK_SIZE=${SLOT_NUM_TOKENS}"
  "${PYTHON_BIN}"
  "${ROOT_DIR}/evaluation/kv_chunk_trace_replay.py"
  --backend "${BACKEND}"
  --num-chunks "${NUM_CHUNKS}"
  --num-layers "${NUM_LAYERS}"
  --slot-num-tokens "${SLOT_NUM_TOKENS}"
  --hidden-dim "${HIDDEN_DIM}"
  --dtype "${DTYPE}"
  --device "${DEVICE}"
  --gds-slab-path "${GDS_SLAB_PATH}"
  --gds-slab-gb "${GDS_SLAB_GB}"
  --lmcache-gds-path "${LMCACHE_GDS_PATH}"
  --lmcache-gds-fmt "${LMCACHE_GDS_FMT}"
  --lmcache-gds-registered-buffer-mb "${LMCACHE_GDS_REGISTERED_BUFFER_MB}"
  --summary-warmup-samples "${SUMMARY_WARMUP_SAMPLES}"
)

if [[ "${USE_SUDO}" == "1" ]]; then
  CMD=(sudo "${CMD[@]}")
fi

if [[ -n "${TRACE_JSONL}" ]]; then
  CMD+=(--trace-jsonl "${TRACE_JSONL}")
fi

if [[ "${LMCACHE_GDS_USE_POSIX}" == "1" ]]; then
  CMD+=(--lmcache-gds-use-posix)
fi

if [[ "${LMCACHE_GDS_USE_DIRECT_IO}" == "0" ]]; then
  CMD+=(--no-lmcache-gds-use-direct-io)
fi

if [[ "${LMCACHE_GDS_USE_REGISTERED_BUFFER}" == "1" ]]; then
  CMD+=(--lmcache-gds-use-registered-buffer)
fi

if [[ "${BATCH_PREFETCH_WARMUP}" == "0" ]]; then
  CMD+=(--no-batch-prefetch-warmup)
fi

# 允许临时把 Python 参数透传进 replay，比如 --no-verify。
CMD+=("$@")

echo "Log will be written to: ${LOG_PATH}"
echo "Running command:"
printf '  %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}" 2>&1 | tee "${LOG_PATH}"
