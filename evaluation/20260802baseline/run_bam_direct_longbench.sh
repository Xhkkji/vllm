#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BAM_ROOT="${BAM_ROOT:-/home/xhk/llm-inference/BaM_IOStack}"
PYTHON_BIN="${PYTHON_BIN:-/home/xhk/miniconda3/envs/pytorch-vllm/bin/python}"
MODEL="${MODEL:-/home/xhk/llm-inference/models/Qwen2.5-7B-Instruct}"
MANIFEST="${MANIFEST:-/home/xhk/llm-inference/datasets/longbench/organized/triviaqa/qwen25/full/buckets/lt4k.jsonl}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${LOG_ROOT:-${SCRIPT_DIR}/result/bam_direct_longbench}"
LOG_DIR="${RUN_DIR:-${LOG_ROOT}/${TIMESTAMP}}"

NUM_SAMPLES="${NUM_SAMPLES:-25}"
PROMPT_LEN="${PROMPT_LEN:-2048}"
MAX_TOKENS="${MAX_TOKENS:-128}"
BEST_OF="${BEST_OF:-4}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
NUM_GPU_BLOCKS_OVERRIDE="${NUM_GPU_BLOCKS_OVERRIDE:-260}"

if [[ "${EUID}" -ne 0 ]]; then
  exec sudo -n env \
    "BAM_ROOT=${BAM_ROOT}" \
    "PYTHON_BIN=${PYTHON_BIN}" \
    "MODEL=${MODEL}" \
    "MANIFEST=${MANIFEST}" \
    "LOG_ROOT=${LOG_ROOT}" \
    "RUN_DIR=${LOG_DIR}" \
    "NUM_SAMPLES=${NUM_SAMPLES}" \
    "PROMPT_LEN=${PROMPT_LEN}" \
    "MAX_TOKENS=${MAX_TOKENS}" \
    "BEST_OF=${BEST_OF}" \
    "MAX_NUM_SEQS=${MAX_NUM_SEQS}" \
    "NUM_GPU_BLOCKS_OVERRIDE=${NUM_GPU_BLOCKS_OVERRIDE}" \
    "VLLM_BAM_SERVICE_LIFETIME=${VLLM_BAM_SERVICE_LIFETIME:-io_active}" \
    bash "$0" "$@"
fi

mkdir -p "${LOG_DIR}"
export PYTHONPATH="${ROOT_DIR}:${BAM_ROOT}/gids_module:${BAM_ROOT}/gids_module/build${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${BAM_ROOT}/bam/build/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_V0_SWAP_TRACE=1
export VLLM_BAM_DIRECT_KVSTORE_ENABLE=1
export VLLM_BAM_DIRECT_SERVICE_LIFETIME="${VLLM_BAM_SERVICE_LIFETIME:-io_active}"
export VLLM_BAM_SHADOW_ENABLE=0
export VLLM_BAM_SWAPIN_ENABLE=0
export VLLM_BAM_IMPORT_PATH="${BAM_ROOT}/gids_module"
export VLLM_BAM_SSD_LIST="${VLLM_BAM_SSD_LIST:-0}"

CONSOLE_LOG="${LOG_DIR}/console.log"
set +e
"${PYTHON_BIN}" "${SCRIPT_DIR}/bam_direct_longbench_eval.py" \
  "${MODEL}" \
  --manifest "${MANIFEST}" \
  --num-samples "${NUM_SAMPLES}" \
  --prompt-len "${PROMPT_LEN}" \
  --max-tokens "${MAX_TOKENS}" \
  --best-of "${BEST_OF}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --num-gpu-blocks-override "${NUM_GPU_BLOCKS_OVERRIDE}" \
  --log-dir "${LOG_DIR}" \
  >"${CONSOLE_LOG}" 2>&1
PY_STATUS=$?
set -e

LOG_PATH="$(find "${LOG_DIR}" -maxdepth 1 -type f -name 'v0_swap_trace_*.log' -print -quit)"
if [[ -z "${LOG_PATH}" ]]; then
  echo "[20260802baseline] FAIL: BaM trace log was not created" >&2
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

echo "[20260802baseline] backend=bam_direct_longbench"
echo "[20260802baseline] log=${LOG_PATH}"
echo "[20260802baseline] python_exit_status=${PY_STATUS}"
echo "[20260802baseline] write_done=$(grep -Fc '[BAM_DIRECT_KVSTORE] op=write phase=done' "${LOG_PATH}")"
echo "[20260802baseline] read_done=$(grep -Fc '[BAM_DIRECT_KVSTORE] op=read phase=done' "${LOG_PATH}")"
if [[ "${PY_STATUS}" -ne 0 ]]; then
  echo "[20260802baseline] WARN: python exited with ${PY_STATUS} after trace events"
fi
echo "[20260802baseline] PASS"
