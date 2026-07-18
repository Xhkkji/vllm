#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 这个脚本固定到当前输出正确的 fast 路径：
#
#   gpu_worker_persistent_materialized
#
# 设计原则：
# 1. 不复制主启动脚本的全部默认参数，避免两份默认值长期漂移；
# 2. 这里只传“三条 KV 分支”里的分支名，底层开关由主脚本统一派生；
# 3. 真正的实验逻辑、日志目录、sudo/root 透传等，仍然复用主脚本。
#
# 说明：
# - 主脚本中已经默认打开 shadow / prefer-load / KV fast path / direct placement /
#   prewarm / 1GB cache，因此这里不再重复传这些值。
# - 调试日志不再作为分支默认值；需要时在命令前显式传
#   `GIDS_KV_DEBUG=1` 或 `VLLM_BAM_XFORMERS_PREFIX_FALLBACK_PROFILE=1`。
# - 如需新增调参项，优先只补“真实实验口径”，不要重新堆底层派生开关。

VLLM_BAM_KV_BRANCH="${VLLM_BAM_KV_BRANCH:-gpu_worker_persistent_materialized}" \
bash "${SCRIPT_DIR}/run_single_gpu_lmcache_no_prefix_reuse_qwen25.sh"
