#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 这个脚本服务当前稳定的 `runtime one-copy + persistent service` 基线。
#
# 它只声明三条 KV 分支中的 one-copy 分支：
#
#   VLLM_BAM_KV_BRANCH=gpu_worker_persistent_one_copy
#
# 主启动脚本会据此统一派生 executor / runtime / persistent / one-copy
# 低层开关，避免这里继续手写一串容易互相冲突的布尔变量。
#
# 当前 one-copy 基线的默认值已经在主脚本中按分支派生：
#
# - `DIRECT_RETRIEVE_PREWARM_AFTER_REQUEST1=1`
# - `GIDS_KV_GPU_WORKER_MOVER_CTAS=4`
# - `GIDS_KV_DEBUG=0`
# - `VLLM_BAM_XFORMERS_PREFIX_FALLBACK_PROFILE=0`
#
# 因此这个 wrapper 只声明分支名，避免重复透传一组容易漂移的底层开关。
# 如需临时调试或回到单 CTA，可在命令前显式覆盖对应环境变量。

VLLM_BAM_KV_BRANCH="${VLLM_BAM_KV_BRANCH:-gpu_worker_persistent_one_copy}" \
bash "${SCRIPT_DIR}/run_single_gpu_lmcache_no_prefix_reuse_qwen25_gpu_worker_persistent.sh"
