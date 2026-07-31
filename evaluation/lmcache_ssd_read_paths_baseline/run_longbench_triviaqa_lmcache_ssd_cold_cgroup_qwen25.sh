#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# LMCache 原生 SSD cold-read + cgroup baseline。
#
# 目的：
# - request_1 正常写入 LMCache local_disk；
# - 每个 read request 计时前执行 sync + drop_caches，尽量压掉 Linux page cache；
# - 整个进程放进 memory cgroup，限制运行期间 page cache / CPU buffer 继续膨胀。
#
# 这条脚本只用于观察传统路径：
#   SSD -> Linux/POSIX read -> CPU buffer -> GPU
# 不打开 BaM/GDS wrapper，也不影响 one-copy / rowctx 分支。

CGROUP_ROOT="${CGROUP_ROOT:-/sys/fs/cgroup/memory}"
CGROUP_NAME="${CGROUP_NAME:-lmcache_ssd_cold_${TIMESTAMP}}"
CGROUP_PATH="${CGROUP_ROOT}/${CGROUP_NAME}"

# Qwen2.5-7B 进程加载模型时 host 侧也会占用内存。实测 8GB/12GB 都无法
# 进入 request loop：默认 vLLM swap_space=4GB，LMCache pinned allocator=5GB，
# 再叠加模型加载和 page cache 后会被 cgroup 压死。因此默认从 16GB 起测，
# 仍明显小于不受限 host cache 口径；需要更激进时外部传 CGROUP_MEMORY_GB。
CGROUP_MEMORY_GB="${CGROUP_MEMORY_GB:-16}"
CGROUP_MEMORY_BYTES="${CGROUP_MEMORY_BYTES:-$((CGROUP_MEMORY_GB * 1024 * 1024 * 1024))}"

NUM_SAMPLES_DEFAULT="${NUM_SAMPLES:-2}"
REPEAT_READ_DEFAULT="${REPEAT_READ:-1}"
MAX_TOKENS_DEFAULT="${MAX_TOKENS:-16}"
LOG_ROOT_DEFAULT="${LOG_ROOT:-${SCRIPT_DIR}/logs_longbench_triviaqa/lmcache_ssd_cold_cgroup_${CGROUP_MEMORY_GB}g}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-samples)
      NUM_SAMPLES_DEFAULT="$2"
      shift 2
      ;;
    --repeat-read)
      REPEAT_READ_DEFAULT="$2"
      shift 2
      ;;
    --max-tokens)
      MAX_TOKENS_DEFAULT="$2"
      shift 2
      ;;
    --log-root)
      LOG_ROOT_DEFAULT="$2"
      shift 2
      ;;
    *)
      echo "[lmcache-ssd-cold-cgroup] unsupported arg: $1" >&2
      exit 2
      ;;
  esac
done

NUM_SAMPLES_VALUE="${NUM_SAMPLES_DEFAULT}"
REPEAT_READ_VALUE="${REPEAT_READ_DEFAULT}"
MAX_TOKENS_VALUE="${MAX_TOKENS_DEFAULT}"
LOG_ROOT_VALUE="${LOG_ROOT_DEFAULT}"

if [[ "${EUID}" -ne 0 ]]; then
  exec sudo env \
    "CGROUP_ROOT=${CGROUP_ROOT}" \
    "CGROUP_NAME=${CGROUP_NAME}" \
    "CGROUP_MEMORY_GB=${CGROUP_MEMORY_GB}" \
    "CGROUP_MEMORY_BYTES=${CGROUP_MEMORY_BYTES}" \
    "NUM_SAMPLES=${NUM_SAMPLES_VALUE}" \
    "REPEAT_READ=${REPEAT_READ_VALUE}" \
    "MAX_TOKENS=${MAX_TOKENS_VALUE}" \
    "LOG_ROOT=${LOG_ROOT_VALUE}" \
    "MANIFEST_PATH=${MANIFEST_PATH:-}" \
    "MODEL_PATH=${MODEL_PATH:-}" \
    "MAX_MODEL_LEN=${MAX_MODEL_LEN:-}" \
    "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-}" \
    "CUDA_DEVICE=${CUDA_DEVICE:-0}" \
    "PYTHON_BIN=${PYTHON_BIN:-}" \
    "LMCACHE_REPO_PATH=${LMCACHE_REPO_PATH:-}" \
    "LMCACHE_CHUNK_SIZE=${LMCACHE_CHUNK_SIZE:-}" \
    "LMCACHE_MAX_LOCAL_CPU_SIZE=${LMCACHE_MAX_LOCAL_CPU_SIZE:-}" \
    "LMCACHE_MAX_LOCAL_DISK_SIZE=${LMCACHE_MAX_LOCAL_DISK_SIZE:-}" \
    "LONGBENCH_DEBUG_LOG=${LONGBENCH_DEBUG_LOG:-0}" \
    "LONGBENCH_DROP_CACHES_SETTLE_S=${LONGBENCH_DROP_CACHES_SETTLE_S:-0}" \
    "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}" \
    "PYTHONPATH=${PYTHONPATH:-}" \
    bash "$0" "$@"
fi

mkdir -p "${CGROUP_PATH}"
echo "${CGROUP_MEMORY_BYTES}" > "${CGROUP_PATH}/memory.limit_in_bytes"
echo 0 > "${CGROUP_PATH}/memory.swappiness"

finish() {
  local status=$?
  set +e
  echo "[lmcache-ssd-cold-cgroup] memory.stat"
  egrep '^(cache|rss|mapped_file|pgmajfault|total_cache|total_rss|total_mapped_file|total_pgmajfault) ' \
    "${CGROUP_PATH}/memory.stat" || true
  echo "[lmcache-ssd-cold-cgroup] memory.max_usage_in_bytes=$(cat "${CGROUP_PATH}/memory.max_usage_in_bytes" 2>/dev/null || true)"
  # 把当前 shell 移回父 cgroup 后尝试删除临时 cgroup；失败时保留，便于事后看 stat。
  echo $$ > "${CGROUP_ROOT}/tasks" 2>/dev/null || true
  rmdir "${CGROUP_PATH}" 2>/dev/null || true
  exit "${status}"
}
trap finish EXIT

echo $$ > "${CGROUP_PATH}/tasks"

echo "[lmcache-ssd-cold-cgroup] cgroup=${CGROUP_PATH}"
echo "[lmcache-ssd-cold-cgroup] memory_limit_bytes=${CGROUP_MEMORY_BYTES}"
echo "[lmcache-ssd-cold-cgroup] num_samples=${NUM_SAMPLES_VALUE} repeat_read=${REPEAT_READ_VALUE} max_tokens=${MAX_TOKENS_VALUE}"
echo "[lmcache-ssd-cold-cgroup] log_root=${LOG_ROOT_VALUE}"

cd "${SCRIPT_DIR}/../.."

env \
  NUM_SAMPLES="${NUM_SAMPLES_VALUE}" \
  REPEAT_READ="${REPEAT_READ_VALUE}" \
  MAX_TOKENS="${MAX_TOKENS_VALUE}" \
  LOG_ROOT="${LOG_ROOT_VALUE}" \
  LMCACHE_LOCAL_CPU=False \
  LMCACHE_MAX_LOCAL_CPU_SIZE="${LMCACHE_MAX_LOCAL_CPU_SIZE:-1.0}" \
  SWAP_SPACE="${SWAP_SPACE:-0}" \
  LONGBENCH_DROP_CACHES_BEFORE_READ=1 \
  bash evaluation/lmcache_ssd_read_paths_baseline/run_longbench_triviaqa_ssd_read_paths_qwen25.sh ssd_cpu_gpu
