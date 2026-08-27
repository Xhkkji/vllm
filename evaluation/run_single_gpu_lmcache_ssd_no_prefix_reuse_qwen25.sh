#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

LMCACHE_REPO_PATH="${LMCACHE_REPO_PATH:-/home/xhk/llm-inference/LMCache-v0-torch26}" \
PYTHON_BIN="${PYTHON_BIN:-/home/xhk/miniconda3/envs/pytorch-vllm/bin/python}" \
LMCACHE_USE_EXPERIMENTAL="${LMCACHE_USE_EXPERIMENTAL:-True}" \
LMCACHE_LOCAL_CPU="${LMCACHE_LOCAL_CPU:-False}" \
LMCACHE_LOCAL_DISK="${LMCACHE_LOCAL_DISK:-/home/xhk/llm-inference/lmcache_local_disk/}" \
LMCACHE_MAX_LOCAL_DISK_SIZE="${LMCACHE_MAX_LOCAL_DISK_SIZE:-20}" \
exec bash "${SCRIPT_DIR}/run_single_gpu_lmcache_baseline_qwen25.sh"
