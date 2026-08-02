#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BAM_ROOT="${BAM_ROOT:-/home/xhk/llm-inference/BaM_IOStack}"
PYTHON_BIN="${PYTHON_BIN:-/home/xhk/miniconda3/envs/pytorch-vllm/bin/python}"
MODEL="${MODEL:-/home/xhk/llm-inference/models/Qwen2.5-7B-Instruct}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

LOG_ROOT="${LOG_ROOT:-${SCRIPT_DIR}/result/bam_direct_serial}"
LOG_DIR="${RUN_DIR:-${LOG_ROOT}/${TIMESTAMP}}"
CONSOLE_LOG="${LOG_DIR}/console.log"
mkdir -p "${LOG_DIR}"

REQUESTED_GPU_BLOCKS="${NUM_GPU_BLOCKS_OVERRIDE:-260}"
if (( REQUESTED_GPU_BLOCKS > 260 )); then
  EFFECTIVE_GPU_BLOCKS=260
else
  EFFECTIVE_GPU_BLOCKS="${REQUESTED_GPU_BLOCKS}"
fi

if [[ "${EUID}" -ne 0 ]]; then
  exec sudo -n env \
    "BAM_ROOT=${BAM_ROOT}" \
    "PYTHON_BIN=${PYTHON_BIN}" \
    "MODEL=${MODEL}" \
    "LOG_ROOT=${LOG_ROOT}" \
    "RUN_DIR=${LOG_DIR}" \
    "NUM_GPU_BLOCKS_OVERRIDE=${EFFECTIVE_GPU_BLOCKS}" \
    "NUM_PROMPTS=${NUM_PROMPTS:-8}" \
    "PROMPT_LEN=${PROMPT_LEN:-2048}" \
    "MAX_TOKENS=${MAX_TOKENS:-128}" \
    "TEMPERATURE=${TEMPERATURE:-0.8}" \
    "BEST_OF=${BEST_OF:-4}" \
    "MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}" \
    "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.60}" \
    "SWAP_SPACE=${SWAP_SPACE:-4}" \
    "MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}" \
    "CUDA_DEVICE=${CUDA_DEVICE:-0}" \
    "VLLM_BAM_DIRECT_SERVICE_LIFETIME=${VLLM_BAM_DIRECT_SERVICE_LIFETIME:-io_active}" \
    "VLLM_BAM_SSD_LIST=${VLLM_BAM_SSD_LIST:-0}" \
    "PYTHONPATH=${PYTHONPATH:-}" \
    "LD_LIBRARY_PATH=${BAM_ROOT}/bam/build/lib:${LD_LIBRARY_PATH:-}" \
    /usr/bin/bash "$0" "$@"
fi

export PYTHONPATH="${ROOT_DIR}:${BAM_ROOT}/gids_module:${BAM_ROOT}/gids_module/build${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${BAM_ROOT}/bam/build/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_V0_SWAP_TRACE=1
export VLLM_BAM_DIRECT_KVSTORE_ENABLE=1
export VLLM_BAM_DIRECT_SERVICE_LIFETIME="${VLLM_BAM_DIRECT_SERVICE_LIFETIME:-io_active}"
export VLLM_BAM_SHADOW_ENABLE=0
export VLLM_BAM_SWAPIN_ENABLE=0
export VLLM_BAM_IMPORT_PATH="${BAM_ROOT}/gids_module"
export VLLM_BAM_SSD_LIST="${VLLM_BAM_SSD_LIST:-0}"

echo "[20260802baseline] backend=bam_direct_serial"
echo "[20260802baseline] model=${MODEL}"
echo "[20260802baseline] service_lifetime=${VLLM_BAM_DIRECT_SERVICE_LIFETIME}"
echo "[20260802baseline] gpu_blocks=${EFFECTIVE_GPU_BLOCKS}"
echo "[20260802baseline] log_dir=${LOG_DIR}"

set +e
"${PYTHON_BIN}" "${ROOT_DIR}/evaluation/v0_swap_trace_eval.py" \
  "${MODEL}" \
  --num-prompts "${NUM_PROMPTS:-8}" \
  --prompt-len "${PROMPT_LEN:-2048}" \
  --max-tokens "${MAX_TOKENS:-128}" \
  --temperature "${TEMPERATURE:-0.8}" \
  --best-of "${BEST_OF:-4}" \
  --n 1 \
  --max-model-len "${MAX_MODEL_LEN:-4096}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.60}" \
  --swap-space "${SWAP_SPACE:-4}" \
  --dtype half \
  --tensor-parallel-size 1 \
  --preemption-mode swap \
  --max-num-seqs "${MAX_NUM_SEQS:-8}" \
  --num-gpu-blocks-override "${EFFECTIVE_GPU_BLOCKS}" \
  --enforce-eager \
  --log-dir "${LOG_DIR}" \
  >"${CONSOLE_LOG}" 2>&1
PY_STATUS=$?
set -e

LOG_PATH="$(find "${LOG_DIR}" -maxdepth 1 -type f -name 'v0_swap_trace_*.log' -print -quit)"
if [[ -z "${LOG_PATH}" ]]; then
  echo "[20260802baseline] FAIL: trace log was not created" >&2
  exit 1
fi

for pattern in \
  'op=swap_out' \
  'op=swap_in' \
  '[BAM_DIRECT_KVSTORE] op=write phase=done' \
  '[BAM_DIRECT_KVSTORE] op=read phase=done' \
  'Run summary'; do
  if ! grep -Fq "${pattern}" "${LOG_PATH}"; then
    echo "[20260802baseline] FAIL: missing ${pattern}" >&2
    echo "[20260802baseline] log=${LOG_PATH}" >&2
    exit 1
  fi
done

SWAP_OUT_COUNT="$(grep -Fc '[BAM_DIRECT_KVSTORE] op=write phase=done' "${LOG_PATH}")"
SWAP_IN_COUNT="$(grep -Fc '[BAM_DIRECT_KVSTORE] op=read phase=done' "${LOG_PATH}")"
echo "[20260802baseline] log=${LOG_PATH}"
echo "[20260802baseline] python_exit_status=${PY_STATUS}"
echo "[20260802baseline] swap_out=${SWAP_OUT_COUNT} swap_in=${SWAP_IN_COUNT}"
if [[ "${PY_STATUS}" -ne 0 ]]; then
  echo "[20260802baseline] WARN: python exited with ${PY_STATUS} after required trace events were recorded"
fi
echo "[20260802baseline] PASS"
