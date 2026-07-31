#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# LongBench-TriviaQA 版本的 LMCache SSD read-path baseline。
#
# 与固定 prompt 脚本不同，这里从 manifest 读取真实长上下文 QA 样本。
# 每条样本默认连续跑两次：
#   request_1: 写入/建立 LMCache SSD 数据
#   request_2: 复用同一 prompt，触发 SSD/GDS 读回
#
# 默认 manifest 选择 qwen25/full/lt4k 小桶。这个小桶按 Qwen2.5 tokenizer
# 的真实 prompt token 数分桶，并为 MAX_TOKENS=32 预留输出空间，所以能直接
# 搭配默认 MAX_MODEL_LEN=4096。需要测更长 bucket 时，外部显式传
# MANIFEST_PATH 和 MAX_MODEL_LEN，避免脚本静默改变显存压力。

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
    echo "[longbench-triviaqa-ssd] unsupported mode=${MODE}" >&2
    echo "[longbench-triviaqa-ssd] valid modes: ssd_cpu_gpu, gds_gpu" >&2
    exit 2
    ;;
esac

MANIFEST_PATH_VALUE="${MANIFEST_PATH:-/home/xhk/llm-inference/datasets/longbench/organized/triviaqa/qwen25/full/buckets/lt4k.jsonl}"
MODEL_PATH_VALUE="${MODEL_PATH:-/home/xhk/llm-inference/models/Qwen2.5-7B-Instruct}"
PYTHON_BIN_VALUE="${PYTHON_BIN:-/home/xhk/miniconda3/envs/pytorch-vllm/bin/python}"
LMCACHE_REPO_PATH_VALUE="${LMCACHE_REPO_PATH:-/home/xhk/llm-inference/LMCache-v0-torch26}"

LOG_ROOT_VALUE="${LOG_ROOT:-${SCRIPT_DIR}/logs_longbench_triviaqa/${MODE}}"
RUN_DIR_VALUE="${RUN_DIR:-${LOG_ROOT_VALUE}/${TIMESTAMP}}"
LOG_FILE_VALUE="${LOG_FILE:-${RUN_DIR_VALUE}/run.log}"
METRICS_JSONL_VALUE="${METRICS_JSONL:-${RUN_DIR_VALUE}/metrics.jsonl}"
TEST_ROOT_VALUE="${LMCACHE_SSD_TEST_ROOT:-${RUN_DIR_VALUE}}"

LMCACHE_LOCAL_DISK_VALUE="${LMCACHE_LOCAL_DISK:-${TEST_ROOT_VALUE}/lmcache_local_disk/}"
VLLM_GDS_LMCACHE_PATH_VALUE="${VLLM_GDS_LMCACHE_PATH:-${TEST_ROOT_VALUE}/lmcache_gds/}"

MAX_MODEL_LEN_VALUE="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION_VALUE="${GPU_MEMORY_UTILIZATION:-0.60}"
MAX_TOKENS_VALUE="${MAX_TOKENS:-32}"
SWAP_SPACE_VALUE="${SWAP_SPACE:-4}"
NUM_SAMPLES_VALUE="${NUM_SAMPLES:-25}"
REPEAT_READ_VALUE="${REPEAT_READ:-1}"
BATCH_SIZE_VALUE="${BATCH_SIZE:-1}"
DTYPE_VALUE="${DTYPE:-half}"
ENFORCE_EAGER_VALUE="${ENFORCE_EAGER:-true}"
ENABLE_CHUNKED_PREFILL_VALUE="${ENABLE_CHUNKED_PREFILL:-false}"
LONGBENCH_DEBUG_LOG_VALUE="${LONGBENCH_DEBUG_LOG:-0}"

mkdir -p "${RUN_DIR_VALUE}"
mkdir -p "${LMCACHE_LOCAL_DISK_VALUE}"
if [[ "${MODE}" == "gds_gpu" ]]; then
  mkdir -p "${VLLM_GDS_LMCACHE_PATH_VALUE}"
fi

echo "[longbench-triviaqa-ssd] mode=${MODE}"
echo "[longbench-triviaqa-ssd] manifest=${MANIFEST_PATH_VALUE}"
echo "[longbench-triviaqa-ssd] run_dir=${RUN_DIR_VALUE}"
echo "[longbench-triviaqa-ssd] log_file=${LOG_FILE_VALUE}"
echo "[longbench-triviaqa-ssd] metrics_jsonl=${METRICS_JSONL_VALUE}"
echo "[longbench-triviaqa-ssd] lmcache_local_disk=${LMCACHE_LOCAL_DISK_VALUE}"
echo "[longbench-triviaqa-ssd] debug_log=${LONGBENCH_DEBUG_LOG_VALUE}"
if [[ "${MODE}" == "gds_gpu" ]]; then
  echo "[longbench-triviaqa-ssd] gds_path=${VLLM_GDS_LMCACHE_PATH_VALUE}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-0}"
export VLLM_USE_V1=0
export PYTHONPATH="${LMCACHE_REPO_PATH_VALUE}:${PYTHONPATH:-}"
export LMCACHE_USE_EXPERIMENTAL="${LMCACHE_USE_EXPERIMENTAL:-True}"
export LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-256}"
export LMCACHE_LOCAL_CPU="${LMCACHE_LOCAL_CPU:-False}"
export LMCACHE_MAX_LOCAL_CPU_SIZE="${LMCACHE_MAX_LOCAL_CPU_SIZE:-5.0}"
export LMCACHE_LOCAL_DISK="${LMCACHE_LOCAL_DISK_VALUE}"
export LMCACHE_MAX_LOCAL_DISK_SIZE="${LMCACHE_MAX_LOCAL_DISK_SIZE:-20}"
export LONGBENCH_DEBUG_LOG="${LONGBENCH_DEBUG_LOG_VALUE}"

# 关闭 BaM KV 路径，确保这里测的是 LMCache SSD/GDS baseline，不是 BaM one-copy。
export VLLM_BAM_LMCACHE_SHADOW_ENABLE=0
export VLLM_BAM_LMCACHE_PREFER_LOAD_ENABLE=0
export VLLM_BAM_KV_FAST_PATH=0
export VLLM_BAM_DIRECT_PLACEMENT=0
export VLLM_BAM_DIRECT_PLACEMENT_DEFER_RUNTIME=0
export VLLM_BAM_KV_BRANCH=rowctx_baseline
export GIDS_KV_GPU_WORKER_RUNTIME_ENABLE=0
export GIDS_KV_GPU_WORKER_PERSISTENT_ENABLE=0
export GIDS_KV_GPU_WORKER_MOVER_CTAS=0

# GDS wrapper 只在 gds_gpu 模式打开；ssd_cpu_gpu 模式保持原生 LMCache local_disk。
export VLLM_GDS_LMCACHE_SHADOW_ENABLE="${GDS_SHADOW_ENABLE}"
export VLLM_GDS_LMCACHE_PREFER_LOAD_ENABLE="${GDS_PREFER_LOAD_ENABLE}"
export VLLM_GDS_LMCACHE_PATH="${VLLM_GDS_LMCACHE_PATH_VALUE}"
export VLLM_GDS_LMCACHE_USE_GDS="${VLLM_GDS_LMCACHE_USE_GDS:-1}"
export VLLM_GDS_LMCACHE_USE_DIRECT_IO="${VLLM_GDS_LMCACHE_USE_DIRECT_IO:-1}"
export VLLM_GDS_LMCACHE_FMT="${VLLM_GDS_LMCACHE_FMT:-KV_2LTD}"
export VLLM_GDS_LMCACHE_USE_REGISTERED_BUFFER="${VLLM_GDS_LMCACHE_USE_REGISTERED_BUFFER:-0}"
export VLLM_GDS_LMCACHE_REGISTERED_BUFFER_MB="${VLLM_GDS_LMCACHE_REGISTERED_BUFFER_MB:-0}"
export VLLM_GDS_LMCACHE_VERIFY_BUDGET="${VLLM_GDS_LMCACHE_VERIFY_BUDGET:-0}"

cd "${SCRIPT_DIR}/../.."

RUNNER_ARGS=(
  "${SCRIPT_DIR}/longbench_triviaqa_runner.py"
  --manifest "${MANIFEST_PATH_VALUE}"
  --metrics-jsonl "${METRICS_JSONL_VALUE}"
  --model "${MODEL_PATH_VALUE}"
  --max-model-len "${MAX_MODEL_LEN_VALUE}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION_VALUE}"
  --dtype "${DTYPE_VALUE}"
  --swap-space "${SWAP_SPACE_VALUE}"
  --max-tokens "${MAX_TOKENS_VALUE}"
  --num-samples "${NUM_SAMPLES_VALUE}"
  --repeat-read "${REPEAT_READ_VALUE}"
  --batch-size "${BATCH_SIZE_VALUE}"
)

if [[ "${ENFORCE_EAGER_VALUE}" == "true" ]]; then
  RUNNER_ARGS+=(--enforce-eager)
else
  RUNNER_ARGS+=(--no-enforce-eager)
fi

if [[ "${ENABLE_CHUNKED_PREFILL_VALUE}" == "true" ]]; then
  RUNNER_ARGS+=(--enable-chunked-prefill)
else
  RUNNER_ARGS+=(--no-enable-chunked-prefill)
fi

"${PYTHON_BIN_VALUE}" "${RUNNER_ARGS[@]}" 2>&1 | tee "${LOG_FILE_VALUE}"
