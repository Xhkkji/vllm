#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/evaluation/logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${LOG_DIR}/lmcache_bam_write_microbench_${TIMESTAMP}.log"

PYTHON_BIN="${PYTHON_BIN:-/home/xhk/miniconda3/envs/pytorch-vllm/bin/python}"
BAM_LIB_DIR="${BAM_LIB_DIR:-/home/xhk/llm-inference/BaM_IOStack/bam/build/lib}"
GIDS_MODULE_DIR="${GIDS_MODULE_DIR:-/home/xhk/llm-inference/BaM_IOStack/gids_module}"
NUM_ITERS="${NUM_ITERS:-12}"
NUM_LAYERS="${NUM_LAYERS:-28}"
SLOT_NUM_TOKENS="${SLOT_NUM_TOKENS:-256}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
DTYPE="${DTYPE:-float16}"
CTRL_IDX="${CTRL_IDX:-0}"

CMD=(
  sudo env
  "LD_LIBRARY_PATH=${BAM_LIB_DIR}:${LD_LIBRARY_PATH:-}"
  "PYTHONPATH=${ROOT_DIR}:${PYTHONPATH:-}"
  "VLLM_BAM_IMPORT_PATH=${GIDS_MODULE_DIR}"
  "VLLM_BAM_CACHE_SIZE_MB=64"
  "VLLM_BAM_NUM_SSD=1"
  "VLLM_BAM_SSD_LIST=0"
  "VLLM_BAM_CTRL_IDX=${CTRL_IDX}"
  "LMCACHE_CHUNK_SIZE=${SLOT_NUM_TOKENS}"
  "${PYTHON_BIN}"
  "${ROOT_DIR}/evaluation/lmcache_bam_write_microbench.py"
  --num-iters "${NUM_ITERS}"
  --num-layers "${NUM_LAYERS}"
  --slot-num-tokens "${SLOT_NUM_TOKENS}"
  --hidden-dim "${HIDDEN_DIM}"
  --dtype "${DTYPE}"
  --ctrl-idx "${CTRL_IDX}"
)

echo "Log will be written to: ${LOG_PATH}"
echo "Running command:"
printf '  %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}" 2>&1 | tee "${LOG_PATH}"
