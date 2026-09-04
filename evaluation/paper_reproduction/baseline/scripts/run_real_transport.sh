#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VLLM_ROOT="$(cd -- "${BASELINE_ROOT}/../../.." && pwd)"
GRANULEKV_ROOT="$(cd -- "${VLLM_ROOT}/../GranuleKV" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xhk/miniconda3/envs/pytorch-vllm/bin/python}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RESULT_DIR="${RUN_DIR:-${BASELINE_ROOT}/results/${RUN_ID}}"
MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/granulekv-mps-pipe}"
MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/tmp/granulekv-mps-log}"
CUDA_IPC_LIBRARY="${CUDA_IPC_LIBRARY:-${GRANULEKV_ROOT}/gids_module/build/libgranulekv_cuda_ipc.so}"
TORCH_BRIDGE_DIR="${TORCH_BRIDGE_DIR:-${VLLM_ROOT}/vllm/granulekv/build/torch_bridge}"

NUM_BLOCKS="${NUM_BLOCKS:-128}"
GPU_BLOCKS="${GPU_BLOCKS:-128}"
STORAGE_BLOCKS="${STORAGE_BLOCKS:-18724}"
NUM_LAYERS="${NUM_LAYERS:-28}"
WINDOW_LAYERS="${WINDOW_LAYERS:-4}"
LEAD_WINDOWS="${LEAD_WINDOWS:-2}"
COMPUTE_REPEATS="${COMPUTE_REPEATS:-30}"
COMPUTE_MATRIX="${COMPUTE_MATRIX:-1024}"
MAX_IN_FLIGHT="${MAX_IN_FLIGHT:-4}"

if [[ "${EUID}" -ne 0 ]]; then
  exec sudo -n env \
    "RUN_ID=${RUN_ID}" "RUN_DIR=${RESULT_DIR}" \
    "PYTHON_BIN=${PYTHON_BIN}" "MPS_PIPE_DIRECTORY=${MPS_PIPE_DIRECTORY}" \
    "MPS_LOG_DIRECTORY=${MPS_LOG_DIRECTORY}" \
    "NUM_BLOCKS=${NUM_BLOCKS}" "GPU_BLOCKS=${GPU_BLOCKS}" \
    "STORAGE_BLOCKS=${STORAGE_BLOCKS}" "NUM_LAYERS=${NUM_LAYERS}" \
    "WINDOW_LAYERS=${WINDOW_LAYERS}" "LEAD_WINDOWS=${LEAD_WINDOWS}" \
    "COMPUTE_REPEATS=${COMPUTE_REPEATS}" \
    "COMPUTE_MATRIX=${COMPUTE_MATRIX}" "MAX_IN_FLIGHT=${MAX_IN_FLIGHT}" \
    "CUDA_IPC_LIBRARY=${CUDA_IPC_LIBRARY}" \
    "TORCH_BRIDGE_DIR=${TORCH_BRIDGE_DIR}" \
    /usr/bin/bash "$0" "$@"
fi

if [[ -e "${RESULT_DIR}" ]]; then
  echo "result directory already exists: ${RESULT_DIR}" >&2
  exit 2
fi
mkdir -p "${RESULT_DIR}/sync" "${RESULT_DIR}/async"

export CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIRECTORY}"
export CUDA_MPS_LOG_DIRECTORY="${MPS_LOG_DIRECTORY}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${VLLM_ROOT}:${GRANULEKV_ROOT}/gids_module:${GRANULEKV_ROOT}/gids_module/build${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${GRANULEKV_ROOT}/gids_module/build:${GRANULEKV_ROOT}/bam/build/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

if [[ ! -f "${CUDA_IPC_LIBRARY}" ]]; then
  echo "missing GranuleKV CUDA library: ${CUDA_IPC_LIBRARY}" >&2
  exit 1
fi
if ! compgen -G "${TORCH_BRIDGE_DIR}/granulekv_torch_bridge*.so" >/dev/null; then
  echo "missing GranuleKV torch bridge: ${TORCH_BRIDGE_DIR}" >&2
  exit 1
fi

# Reuse the canonical workflow. This is idempotent for the matching root MPS
# instance and never stops an MPS instance owned outside this experiment.
bash "${GRANULEKV_ROOT}/gids_module/start_granulekv_mps.sh"
bash "${GRANULEKV_ROOT}/gids_module/check_granulekv_mps.sh"

DAEMON_PID=""
start_daemon() {
  local backend="$1"
  local backend_dir="${RESULT_DIR}/${backend}"
  local control_dir="${backend_dir}/control"
  mkdir -p "${control_dir}"
  VLLM_GRANULEKV_SERVICE_LIFETIME=resident \
    VLLM_GRANULEKV_IDLE_STOP_DELAY_MS=0 \
    "${PYTHON_BIN}" -m granulekv.daemon \
      --control-dir "${control_dir}" \
      --cuda-ipc-library "${CUDA_IPC_LIBRARY}" \
      --ssd-index "${VLLM_GRANULEKV_SSD_INDEX:-0}" \
      --max-in-flight "${MAX_IN_FLIGHT}" \
      >"${backend_dir}/daemon.log" 2>&1 &
  DAEMON_PID=$!
  for _ in $(seq 1 300); do
    [[ -f "${control_dir}/control.slot" ]] && return 0
    if ! kill -0 "${DAEMON_PID}" 2>/dev/null; then
      echo "GranuleKV daemon exited during ${backend} startup" >&2
      return 1
    fi
    sleep 0.1
  done
  echo "timed out waiting for ${backend} GranuleKV daemon" >&2
  return 1
}

stop_daemon() {
  if [[ -n "${DAEMON_PID}" ]] && kill -0 "${DAEMON_PID}" 2>/dev/null; then
    kill -TERM "${DAEMON_PID}" || true
    wait "${DAEMON_PID}" || true
  fi
  DAEMON_PID=""
}
trap stop_daemon EXIT

run_backend() {
  local backend="$1"
  local control_dir="${RESULT_DIR}/${backend}/control"
  start_daemon "${backend}"
  if [[ "${backend}" == "sync" ]]; then
    env \
      VLLM_GRANULEKV_ENABLE=1 \
      VLLM_GRANULEKV_IOSTACK_ROOT="${GRANULEKV_ROOT}" \
      VLLM_GRANULEKV_CONTROL_DIR="${control_dir}" \
      VLLM_GRANULEKV_CUDA_IPC_LIBRARY="${CUDA_IPC_LIBRARY}" \
      VLLM_GRANULEKV_TORCH_BRIDGE_DIR="${TORCH_BRIDGE_DIR}" \
      VLLM_GRANULEKV_MAX_IN_FLIGHT="${MAX_IN_FLIGHT}" \
      VLLM_GRANULEKV_TIMEOUT_SECONDS=300 \
      VLLM_GRANULEKV_LAYER_WORKING_SET_ENABLE=0 \
      "${PYTHON_BIN}" "${BASELINE_ROOT}/synthetic_transport_eval.py" \
        --backend sync --output "${RESULT_DIR}/sync/summary.json" \
        --num-blocks "${NUM_BLOCKS}" --gpu-blocks "${GPU_BLOCKS}" \
        --storage-blocks "${STORAGE_BLOCKS}" --num-layers "${NUM_LAYERS}" \
        --window-layers "${WINDOW_LAYERS}" \
        --compute-repeats "${COMPUTE_REPEATS}" \
        --compute-matrix "${COMPUTE_MATRIX}" \
        >"${RESULT_DIR}/sync/console.log" 2>&1
  else
    env \
      VLLM_GRANULEKV_ENABLE=1 \
      VLLM_GRANULEKV_IOSTACK_ROOT="${GRANULEKV_ROOT}" \
      VLLM_GRANULEKV_CONTROL_DIR="${control_dir}" \
      VLLM_GRANULEKV_CUDA_IPC_LIBRARY="${CUDA_IPC_LIBRARY}" \
      VLLM_GRANULEKV_TORCH_BRIDGE_DIR="${TORCH_BRIDGE_DIR}" \
      VLLM_GRANULEKV_MAX_IN_FLIGHT="${MAX_IN_FLIGHT}" \
      VLLM_GRANULEKV_TIMEOUT_SECONDS=300 \
      VLLM_GRANULEKV_LAYER_WORKING_SET_ENABLE=0 \
      VLLM_GRANULEKV_HIERARCHICAL_IO_ENABLE=1 \
      VLLM_GRANULEKV_HIERARCHICAL_LAYER_BARRIER=1 \
      VLLM_GRANULEKV_HIERARCHICAL_ROLLING_ENABLE=1 \
      "${PYTHON_BIN}" "${BASELINE_ROOT}/synthetic_transport_eval.py" \
        --backend async --output "${RESULT_DIR}/async/summary.json" \
        --num-blocks "${NUM_BLOCKS}" --gpu-blocks "${GPU_BLOCKS}" \
        --storage-blocks "${STORAGE_BLOCKS}" --num-layers "${NUM_LAYERS}" \
        --window-layers "${WINDOW_LAYERS}" --lead-windows "${LEAD_WINDOWS}" \
        --compute-repeats "${COMPUTE_REPEATS}" \
        --compute-matrix "${COMPUTE_MATRIX}" \
        >"${RESULT_DIR}/async/console.log" 2>&1
  fi
  stop_daemon
}

run_backend sync
run_backend async
"${PYTHON_BIN}" "${BASELINE_ROOT}/aggregate_synthetic.py" "${RESULT_DIR}"
