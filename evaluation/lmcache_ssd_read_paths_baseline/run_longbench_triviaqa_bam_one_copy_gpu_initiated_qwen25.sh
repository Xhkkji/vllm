#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# GPU-initiated prefetch 实验入口。
#
# 这条脚本不复制 one-copy 主脚本里的长参数列表，只设置本实验分支需要的
# 差异化开关，然后复用稳定的 one-copy 启动脚本：
#
#   LMCache prefetch phase
#     -> 收集 chunk key
#     -> flush 成 native KV batch handle
#     -> direct placement start 复用 handle
#     -> GPU persistent one-copy 继续负责 poll / scatter / cleanup
#
# 默认样本数保持较小，先验证新分支不会破坏当前 cta=4 one-copy 主线。

export VLLM_BAM_GPU_INITIATED_PREFETCH="${VLLM_BAM_GPU_INITIATED_PREFETCH:-1}"
export NUM_SAMPLES="${NUM_SAMPLES:-8}"
export LOG_ROOT="${LOG_ROOT:-${SCRIPT_DIR}/logs_longbench_triviaqa/bam_one_copy_gpu_initiated}"

exec "${SCRIPT_DIR}/run_longbench_triviaqa_bam_one_copy_qwen25.sh" "$@"
