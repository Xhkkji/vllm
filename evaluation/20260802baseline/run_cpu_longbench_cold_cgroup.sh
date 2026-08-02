#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 真实 LongBench CPU baseline。底层仍复用已经验证过的 cold+cgroup runner。
export NUM_SAMPLES="${NUM_SAMPLES:-25}"
export REPEAT_READ="${REPEAT_READ:-1}"
export MAX_TOKENS="${MAX_TOKENS:-128}"
export CGROUP_MEMORY_GB="${CGROUP_MEMORY_GB:-16}"
export LOG_ROOT="${LOG_ROOT:-${SCRIPT_DIR}/result/cpu_longbench_cgroup}"

exec bash "${SCRIPT_DIR}/run_lmcache_cpu_cgroup_chunk1.sh"
