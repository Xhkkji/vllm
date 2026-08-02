#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

export LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-1}"
export NUM_SAMPLES="${NUM_SAMPLES:-2}"
export REPEAT_READ="${REPEAT_READ:-1}"
export MAX_TOKENS="${MAX_TOKENS:-16}"
export CGROUP_MEMORY_GB="${CGROUP_MEMORY_GB:-16}"

LOG_ROOT_VALUE="${LOG_ROOT:-${SCRIPT_DIR}/result/lmcache_cpu_cgroup_chunk1}"

echo "[20260802baseline] backend=lmcache_cpu_cgroup chunk_size=${LMCACHE_CHUNK_SIZE}"
echo "[20260802baseline] log_root=${LOG_ROOT_VALUE}"
echo "[20260802baseline] cgroup_memory_gb=${CGROUP_MEMORY_GB}"

cd "${ROOT_DIR}"
exec /usr/local/sbin/run-lmcache-ssd-cold-cgroup-qwen25 \
  --num-samples "${NUM_SAMPLES}" \
  --repeat-read "${REPEAT_READ}" \
  --max-tokens "${MAX_TOKENS}" \
  --log-root "${LOG_ROOT_VALUE}"
