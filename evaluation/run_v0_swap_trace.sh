#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash evaluation/run_v0_swap_trace.sh <model-or-local-path>"
  exit 1
fi

MODEL="$1"
shift || true

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/evaluation/logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${LOG_DIR}/v0_swap_trace_${TIMESTAMP}.log"

export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export VLLM_V0_SWAP_TRACE="${VLLM_V0_SWAP_TRACE:-1}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"

NUM_PROMPTS="${NUM_PROMPTS:-32}"
PROMPT_LEN="${PROMPT_LEN:-2048}"
MAX_TOKENS="${MAX_TOKENS:-16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.6}"
SWAP_SPACE="${SWAP_SPACE:-8}"
DTYPE="${DTYPE:-auto}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
SEED="${SEED:-1234}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"

CMD=(
  python "${ROOT_DIR}/evaluation/v0_swap_trace_eval.py"
  "${MODEL}"
  --num-prompts "${NUM_PROMPTS}"
  --prompt-len "${PROMPT_LEN}"
  --max-tokens "${MAX_TOKENS}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --swap-space "${SWAP_SPACE}"
  --dtype "${DTYPE}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --seed "${SEED}"
)

if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  CMD+=(--trust-remote-code)
fi

if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  CMD+=(--enforce-eager)
fi

echo "Logs will be written to: ${LOG_PATH}"
echo "Running command:"
printf '  %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}" "$@" 2>&1 | tee "${LOG_PATH}"
