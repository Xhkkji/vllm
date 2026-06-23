#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash evaluation/run_v100_v0_bam_swap_roundtrip.sh <model-or-local-path>"
  exit 1
fi

MODEL="$1"
shift || true

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/evaluation/logs"
mkdir -p "${LOG_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/xhk/miniconda3/envs/pytorch-vllm/bin/python}"
BAM_LIB_DIR="${BAM_LIB_DIR:-/home/xhk/llm-inference/BaM_IOStack/bam/build/lib}"
GIDS_MODULE_DIR="${GIDS_MODULE_DIR:-/home/xhk/llm-inference/BaM_IOStack/gids_module}"

NUM_PROMPTS="${NUM_PROMPTS:-24}"
PROMPT_LEN="${PROMPT_LEN:-6144}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
TEMPERATURE="${TEMPERATURE:-0.8}"
BEST_OF="${BEST_OF:-4}"
N="${N:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.16}"
SWAP_SPACE="${SWAP_SPACE:-16}"
DTYPE="${DTYPE:-half}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
PREEMPTION_MODE="${PREEMPTION_MODE:-swap}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
SEED="${SEED:-1234}"
BAM_SWAPIN_ENABLE="${BAM_SWAPIN_ENABLE:-1}"
GIDS_FORCE_SYNC_READ="${GIDS_FORCE_SYNC_READ:-1}"
BAM_SWAPIN_VERIFY="${BAM_SWAPIN_VERIFY:-1}"
BAM_SWAPIN_VERIFY_BLOCKS="${BAM_SWAPIN_VERIFY_BLOCKS:-0}"
# 固定使用已验证通过的 BaM page cache 配置。
BAM_CACHE_SIZE_MB="1024"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG_PATH="${LOG_DIR}/v100_v0_bam_swap_roundtrip_${TIMESTAMP}.log"

SUDO_CMD=(
  sudo env
  "LD_LIBRARY_PATH=${BAM_LIB_DIR}:${LD_LIBRARY_PATH:-}"
  "PYTHONPATH=${GIDS_MODULE_DIR}:${PYTHONPATH:-}"
  "VLLM_USE_V1=0"
  "VLLM_V0_SWAP_TRACE=1"
  "VLLM_BAM_SHADOW_ENABLE=1"
  "VLLM_BAM_SWAPIN_ENABLE=${BAM_SWAPIN_ENABLE}"
  "VLLM_BAM_SWAPIN_VERIFY=${BAM_SWAPIN_VERIFY}"
  "VLLM_BAM_SWAPIN_VERIFY_BLOCKS=${BAM_SWAPIN_VERIFY_BLOCKS}"
  "VLLM_BAM_IMPORT_PATH=${GIDS_MODULE_DIR}"
  "VLLM_BAM_CACHE_SIZE_MB=${BAM_CACHE_SIZE_MB}"
  "VLLM_ATTENTION_BACKEND=XFORMERS"
  "GIDS_FORCE_SYNC_READ=${GIDS_FORCE_SYNC_READ}"
  "${PYTHON_BIN}"
  "${ROOT_DIR}/evaluation/v0_swap_trace_eval.py"
  "${MODEL}"
  --num-prompts "${NUM_PROMPTS}"
  --prompt-len "${PROMPT_LEN}"
  --max-tokens "${MAX_TOKENS}"
  --temperature "${TEMPERATURE}"
  --best-of "${BEST_OF}"
  --n "${N}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --swap-space "${SWAP_SPACE}"
  --dtype "${DTYPE}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --preemption-mode "${PREEMPTION_MODE}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --seed "${SEED}"
)

echo "Wrapper log will be written to: ${RUN_LOG_PATH}"
echo "Running command:"
printf '  %q' "${SUDO_CMD[@]}"
printf '\n'

"${SUDO_CMD[@]}" "$@" 2>&1 | tee "${RUN_LOG_PATH}"
