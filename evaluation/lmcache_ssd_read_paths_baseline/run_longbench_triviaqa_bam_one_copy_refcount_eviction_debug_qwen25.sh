#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 这个脚本只负责“压小 cache + 观察 ref_count / eviction”的传参收束。
# 正常 one-copy 链路仍然由 run_longbench_triviaqa_bam_one_copy_qwen25.sh 维护。
#
# 参数选择原则：
# - BaM page cache 不能小到单个 request 都放不下，否则会测成 request 内部
#   未 release 前强行替换，容易误判为生命周期错误；
# - cache 又必须小到多个 request 累计后一定压过容量，从而验证
#   request-scoped release 后底层 slot 能被替换；
# - 上层 shadow chunk slots 也压小，方便日志中直接看到 chunk_slot_evictions。

export MANIFEST_PATH="${MANIFEST_PATH:-/home/xhk/llm-inference/datasets/longbench/organized/triviaqa/qwen25/full/buckets/4k_8k.jsonl}"
export MODEL_PATH="${MODEL_PATH:-/home/xhk/llm-inference/models/Qwen2.5-7B-Instruct}"
export PYTHON_BIN="${PYTHON_BIN:-/home/xhk/miniconda3/envs/pytorch-vllm/bin/python}"
export LMCACHE_REPO_PATH="${LMCACHE_REPO_PATH:-/home/xhk/llm-inference/LMCache-v0-torch26}"

export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.60}"
export MAX_TOKENS="${MAX_TOKENS:-32}"
export NUM_SAMPLES="${NUM_SAMPLES:-12}"
export REPEAT_READ="${REPEAT_READ:-1}"
export DTYPE="${DTYPE:-half}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-true}"
export ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-false}"
export CUDA_DEVICE="${CUDA_DEVICE:-0}"

export VLLM_BAM_CACHE_SIZE_MB="${VLLM_BAM_CACHE_SIZE_MB:-512}"
export VLLM_BAM_LMCACHE_SHADOW_CHUNKS="${VLLM_BAM_LMCACHE_SHADOW_CHUNKS:-64}"
export VLLM_BAM_CACHE_STATS_EVERY_ITER="${VLLM_BAM_CACHE_STATS_EVERY_ITER:-1}"
export VLLM_BAM_DEBUG_STOP_SERVICE_FOR_LIFECYCLE_STATS="${VLLM_BAM_DEBUG_STOP_SERVICE_FOR_LIFECYCLE_STATS:-1}"
export GIDS_KV_REF_DEBUG="${GIDS_KV_REF_DEBUG:-1}"
export GIDS_KV_GPU_WORKER_MOVER_CTAS="${GIDS_KV_GPU_WORKER_MOVER_CTAS:-4}"
export GIDS_KV_WAIT_TIMEOUT_S="${GIDS_KV_WAIT_TIMEOUT_S:-60}"
export GIDS_KV_DEBUG="${GIDS_KV_DEBUG:-0}"
export LONGBENCH_DEBUG_LOG="${LONGBENCH_DEBUG_LOG:-0}"

exec bash "${SCRIPT_DIR}/run_longbench_triviaqa_bam_one_copy_qwen25.sh" "$@"
