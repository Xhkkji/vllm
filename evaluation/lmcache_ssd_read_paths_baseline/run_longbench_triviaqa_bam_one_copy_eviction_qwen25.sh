#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Normal persistent one-copy path with enough pressure to trigger BaM page-cache
# replacement. This is intentionally not the refcount debug path: the service
# stays resident and lifecycle/ref debug probes are disabled.

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

# 512MB = 4096 128KB pages. A single 4k_8k request is below this, while several
# consecutive requests exceed it, so eviction pressure is request-to-request.
export VLLM_BAM_CACHE_SIZE_MB="${VLLM_BAM_CACHE_SIZE_MB:-512}"
export VLLM_BAM_LMCACHE_SHADOW_CHUNKS="${VLLM_BAM_LMCACHE_SHADOW_CHUNKS:-64}"
export VLLM_BAM_CACHE_STATS_EVERY_ITER="${VLLM_BAM_CACHE_STATS_EVERY_ITER:-1}"

# Keep the production-style resident service path. Do not stop the service or
# launch lifecycle debug kernels in this script.
export VLLM_BAM_DEBUG_STOP_SERVICE_FOR_LIFECYCLE_STATS="${VLLM_BAM_DEBUG_STOP_SERVICE_FOR_LIFECYCLE_STATS:-0}"
export GIDS_KV_REF_DEBUG="${GIDS_KV_REF_DEBUG:-0}"
export GIDS_KV_DEBUG="${GIDS_KV_DEBUG:-0}"
export GIDS_KV_GPU_WORKER_MOVER_CTAS="${GIDS_KV_GPU_WORKER_MOVER_CTAS:-4}"
export GIDS_KV_WAIT_TIMEOUT_S="${GIDS_KV_WAIT_TIMEOUT_S:-60}"
export LONGBENCH_DEBUG_LOG="${LONGBENCH_DEBUG_LOG:-0}"

exec bash "${SCRIPT_DIR}/run_longbench_triviaqa_bam_one_copy_qwen25.sh" "$@"
