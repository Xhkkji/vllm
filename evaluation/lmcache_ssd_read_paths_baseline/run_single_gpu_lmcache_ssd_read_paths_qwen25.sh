#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# 用固定单请求样例测试 LMCache 从 SSD 读回 KV chunk 的原生链路：
#
# 1. ssd_cpu_gpu
#    原生 LMCache V0 local_disk:
#      SSD -> CPU MemoryObj -> multi_layer_kv_transfer -> vLLM paged KV cache
#
MODE="${1:-${LMCACHE_SSD_READ_PATH:-ssd_cpu_gpu}}"
case "${MODE}" in
  ssd|ssd_cpu|ssd_cpu_gpu)
    MODE="ssd_cpu_gpu"
    ;;
  *)
    echo "[lmcache-ssd-read-paths] only the native LMCache SSD path is retained" >&2
    echo "[lmcache-ssd-read-paths] valid mode: ssd_cpu_gpu" >&2
    exit 2
    ;;
esac

LOG_ROOT_VALUE="${LOG_ROOT:-${SCRIPT_DIR}/logs/${MODE}}"
RUN_DIR_VALUE="${RUN_DIR:-${LOG_ROOT_VALUE}/${TIMESTAMP}}"
LOG_FILE_VALUE="${LOG_FILE:-${RUN_DIR_VALUE}/run.log}"
TEST_ROOT_VALUE="${LMCACHE_SSD_TEST_ROOT:-${RUN_DIR_VALUE}}"

# 每轮测试默认使用独立目录，避免命中历史 chunk 后看不清 request_1 写入、
# request_2 从 SSD 读回的实际链路。需要复用历史数据时，可以外部显式覆盖
# LMCACHE_LOCAL_DISK。
LMCACHE_LOCAL_DISK_VALUE="${LMCACHE_LOCAL_DISK:-${TEST_ROOT_VALUE}/lmcache_local_disk/}"

mkdir -p "${LMCACHE_LOCAL_DISK_VALUE}"

echo "[lmcache-ssd-read-paths] mode=${MODE}"
echo "[lmcache-ssd-read-paths] run_dir=${RUN_DIR_VALUE}"
echo "[lmcache-ssd-read-paths] lmcache_local_disk=${LMCACHE_LOCAL_DISK_VALUE}"

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
exec env \
  LMCACHE_REPO_PATH="${LMCACHE_REPO_PATH:-/home/xhk/llm-inference/LMCache-v0-torch26}" \
  PYTHON_BIN="${PYTHON_BIN:-/home/xhk/miniconda3/envs/pytorch-vllm/bin/python}" \
  LOG_ROOT="${LOG_ROOT_VALUE}" \
  RUN_DIR="${RUN_DIR_VALUE}" \
  LOG_FILE="${LOG_FILE_VALUE}" \
  LMCACHE_LOCAL_DISK="${LMCACHE_LOCAL_DISK_VALUE}" \
  PROMPT_REUSE_REQUEST1="${PROMPT_REUSE_REQUEST1:-1}" \
  bash "${SCRIPT_DIR}/../../run_single_gpu_lmcache_baseline_qwen25.sh"
