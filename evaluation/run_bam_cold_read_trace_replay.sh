#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT_DIR}/evaluation/logs/bam_cold_read_trace_replay/${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/xhk/miniconda3/envs/pytorch-vllm/bin/python}"
BAM_LIB_DIR="${BAM_LIB_DIR:-/home/xhk/llm-inference/BaM_IOStack/bam/build/lib}"
GIDS_MODULE_DIR="${GIDS_MODULE_DIR:-/home/xhk/llm-inference/BaM_IOStack/gids_module}"
NUM_CHUNKS="${NUM_CHUNKS:-8}"
NUM_LAYERS="${NUM_LAYERS:-28}"
SLOT_NUM_TOKENS="${SLOT_NUM_TOKENS:-256}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
DTYPE="${DTYPE:-float16}"
DEVICE="${DEVICE:-cuda:0}"
CUFILE_LD_PRELOAD="${CUFILE_LD_PRELOAD:-/usr/lib/x86_64-linux-gnu/librt.so.1}"
BAM_CACHE_SIZE_MB="${VLLM_BAM_CACHE_SIZE_MB:-1024}"
BAM_BASE_ROW_OFFSET="${VLLM_BAM_LMCACHE_BASE_ROW_OFFSET:-0}"
BAM_LMCACHE_READ_MODE="${BAM_LMCACHE_READ_MODE:-sync}"
GIDS_FORCE_SYNC_READ_VALUE="${GIDS_FORCE_SYNC_READ:-1}"
BAM_COLD_VERIFY="${BAM_COLD_VERIFY:-0}"
MANIFEST_PATH="${BAM_COLD_MANIFEST:-${LOG_DIR}/bam_cold_manifest.json}"
LOG_PATH="${LOG_DIR}/run.log"

COMMON_ENV=(
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
  "VLLM_BAM_LMCACHE_BASE_ROW_OFFSET=${BAM_BASE_ROW_OFFSET}"
  "GIDS_FORCE_SYNC_READ=${GIDS_FORCE_SYNC_READ_VALUE}"
  "LMCACHE_CHUNK_SIZE=${SLOT_NUM_TOKENS}"
)

COMMON_ARGS=(
  "${ROOT_DIR}/evaluation/kv_chunk_trace_replay.py"
  --num-chunks "${NUM_CHUNKS}"
  --num-layers "${NUM_LAYERS}"
  --slot-num-tokens "${SLOT_NUM_TOKENS}"
  --hidden-dim "${HIDDEN_DIM}"
  --dtype "${DTYPE}"
  --device "${DEVICE}"
  --bam-cold-manifest "${MANIFEST_PATH}"
)

READ_CMD=(sudo env "${COMMON_ENV[@]}" "${PYTHON_BIN}" "${COMMON_ARGS[@]}" --backend bam_cold_read)
if [[ "${BAM_COLD_VERIFY}" != "1" ]]; then
  READ_CMD+=(--no-verify)
fi

if [[ ! -f "${MANIFEST_PATH}" ]]; then
  cat >&2 <<EOF
Missing BaM cold-read manifest: ${MANIFEST_PATH}

请先用稳定的 BaM 热路径生成 manifest，例如：
  BACKEND=bam bash ${ROOT_DIR}/evaluation/run_bam_vs_gds_trace_replay.sh --bam-cold-manifest ${MANIFEST_PATH}

这个脚本现在只做 read-only 冷读，不再运行不稳定的独立写进程。
EOF
  exit 1
fi

{
  echo "Log will be written to: ${LOG_PATH}"
  echo "Manifest: ${MANIFEST_PATH}"
  echo "BaM base row offset: ${BAM_BASE_ROW_OFFSET}"
  echo "BaM cold verify: ${BAM_COLD_VERIFY}"
  echo "Read command:"
  printf '  %q' "${READ_CMD[@]}"
  printf '\n'
  echo
  echo "===== BaM cold read phase ====="
  "${READ_CMD[@]}"
} 2>&1 | tee "${LOG_PATH}"
