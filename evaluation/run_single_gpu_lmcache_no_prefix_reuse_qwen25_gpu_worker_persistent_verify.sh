#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 这个脚本服务“runtime one-copy + persistent service 主线”的快速启动。
#
# 它只声明三条 KV 分支中的 one-copy 分支：
#
#   VLLM_BAM_KV_BRANCH=gpu_worker_persistent_one_copy
#
# 主启动脚本会据此统一派生 executor / runtime / persistent / one-copy
# 低层开关，避免这里继续手写一串容易互相冲突的布尔变量。
#
# 当前默认改成 one-copy 性能口径：
# - 打开隐藏 prewarm，让正式 request_2 进入 warmed steady-state；
# - 关闭 KV / xFormers 高频调试日志，避免日志 I/O 污染端到端延迟；
# - 默认打开 4 个 mover CTA，验证 one-copy direct placement 的搬运线程数
#   是否是当前性能瓶颈；如需回到刚刚跑通的旧单 CTA one-copy，显式设置
#   `GIDS_KV_GPU_WORKER_MOVER_CTAS=0`。
# - 若要回到 correctness 排查口径，可在命令前显式覆盖这些变量。

VLLM_BAM_KV_BRANCH="${VLLM_BAM_KV_BRANCH:-gpu_worker_persistent_one_copy}" \
DIRECT_RETRIEVE_PREWARM_AFTER_REQUEST1="${DIRECT_RETRIEVE_PREWARM_AFTER_REQUEST1:-1}" \
GIDS_KV_DEBUG="${GIDS_KV_DEBUG:-0}" \
GIDS_KV_GPU_WORKER_MOVER_CTAS="${GIDS_KV_GPU_WORKER_MOVER_CTAS:-4}" \
VLLM_BAM_XFORMERS_PREFIX_FALLBACK_PROFILE="${VLLM_BAM_XFORMERS_PREFIX_FALLBACK_PROFILE:-0}" \
bash "${SCRIPT_DIR}/run_single_gpu_lmcache_no_prefix_reuse_qwen25_gpu_worker_persistent.sh"
