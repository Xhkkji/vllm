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
# one-copy 排查脚本默认关闭隐藏 prewarm：
# - 主 benchmark 脚本为了 steady-state 默认会在 request 1 后额外跑一次隐藏
#   direct retrieve；
# - 但当前我们要定位 request 2 输出错误，隐藏 prewarm 会额外引入一次
#   request-handle / slot_mapping / scheduler 状态变化；
# - 因此这里默认 `DIRECT_RETRIEVE_PREWARM_AFTER_REQUEST1=0`，让日志里只保留
#   request 1 和真正打印的 request 2，避免把问题边界搅浑。

DIRECT_RETRIEVE_PREWARM_AFTER_REQUEST1="${DIRECT_RETRIEVE_PREWARM_AFTER_REQUEST1:-0}" \
VLLM_BAM_KV_BRANCH="${VLLM_BAM_KV_BRANCH:-gpu_worker_persistent_one_copy}" \
bash "${SCRIPT_DIR}/run_single_gpu_lmcache_no_prefix_reuse_qwen25_gpu_worker_persistent.sh"
