#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 这个脚本只负责集中传参；真正的 one-copy 链路配置在
# run_longbench_triviaqa_bam_one_copy_qwen25.sh 中维护。
# 调 ref_count 时改这里，执行时也只运行这个脚本。

export MANIFEST_PATH="${MANIFEST_PATH:-/home/xhk/llm-inference/datasets/longbench/organized/triviaqa/qwen25/full/buckets/4k_8k.jsonl}"
export MODEL_PATH="${MODEL_PATH:-/home/xhk/llm-inference/models/Qwen2.5-7B-Instruct}"
export PYTHON_BIN="${PYTHON_BIN:-/home/xhk/miniconda3/envs/pytorch-vllm/bin/python}"
export LMCACHE_REPO_PATH="${LMCACHE_REPO_PATH:-/home/xhk/llm-inference/LMCache-v0-torch26}"

export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.60}"
export MAX_TOKENS="${MAX_TOKENS:-32}"
export NUM_SAMPLES="${NUM_SAMPLES:-4}"
export REPEAT_READ="${REPEAT_READ:-1}"
export DTYPE="${DTYPE:-half}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-true}"
export ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-false}"
export CUDA_DEVICE="${CUDA_DEVICE:-0}"

export VLLM_BAM_CACHE_SIZE_MB="${VLLM_BAM_CACHE_SIZE_MB:-1024}"
export VLLM_BAM_CACHE_STATS_EVERY_ITER="${VLLM_BAM_CACHE_STATS_EVERY_ITER:-1}"
export VLLM_BAM_DEBUG_STOP_SERVICE_FOR_LIFECYCLE_STATS="${VLLM_BAM_DEBUG_STOP_SERVICE_FOR_LIFECYCLE_STATS:-1}"
export GIDS_KV_REF_DEBUG="${GIDS_KV_REF_DEBUG:-1}"
export GIDS_KV_GPU_WORKER_MOVER_CTAS="${GIDS_KV_GPU_WORKER_MOVER_CTAS:-4}"
export GIDS_KV_WAIT_TIMEOUT_S="${GIDS_KV_WAIT_TIMEOUT_S:-10}"
export GIDS_KV_DEBUG="${GIDS_KV_DEBUG:-0}"
export LONGBENCH_DEBUG_LOG="${LONGBENCH_DEBUG_LOG:-0}"

exec bash "${SCRIPT_DIR}/run_longbench_triviaqa_bam_one_copy_qwen25.sh" "$@"
