#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

MODEL_PATH="${MODEL_PATH:-/home/xhk/llm-inference/models/Qwen2.5-7B-Instruct}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.60}"
LMCACHE_CHUNK_SIZE_VALUE="${LMCACHE_CHUNK_SIZE_VALUE:-256}"
LMCACHE_MAX_LOCAL_CPU_SIZE_VALUE="${LMCACHE_MAX_LOCAL_CPU_SIZE_VALUE:-5.0}"
PROMPT_REPEAT="${PROMPT_REPEAT:-100}"
MAX_TOKENS="${MAX_TOKENS:-64}"
STEADY_REPEAT="${STEADY_REPEAT:-0}"
# prompt 构造模式：
# - repeat: 维持原先“按字符串重复次数构造样例”的简单口径
# - token_budget: 先用 tokenizer 估算 token 数，再自动把 prompt 拉到目标预算附近
#
# `token_budget` 模式的目的，是为像“自然 defer 探测”这种场景提供更稳定的样例：
# 不需要手工猜 `PROMPT_REPEAT=260/300/...`，而是直接按 token 预算构造。
PROMPT_BUILD_MODE="${PROMPT_BUILD_MODE:-repeat}"
PROMPT_TARGET_TOKENS="${PROMPT_TARGET_TOKENS:-0}"
# 是否让 request_2 复用 request_1 的 prompt。
#
# 默认留空时维持原先语义：
# - prefer-load 打开：request_2 复用 request_1，用来观察前缀恢复；
# - prefer-load 关闭：request_2 使用不同 prompt，用作普通 no-prefix-reuse baseline。
#
# 调试 LMCache/BaM retrieve 正确性时，可以显式设成 1：
#   PROMPT_REUSE_REQUEST1=1
# 这样即使关闭 BaM prefer-load/direct-placement，也仍然能用“同一 prompt 的
# request_1/request_2”做严格对照。
PROMPT_REUSE_REQUEST1="${PROMPT_REUSE_REQUEST1:-}"
# 当未显式指定 `PROMPT_TARGET_TOKENS` 时，会用：
#   max_model_len - max_tokens - PROMPT_TARGET_MARGIN
# 作为自动生成的 prompt token 预算。
PROMPT_TARGET_MARGIN="${PROMPT_TARGET_MARGIN:-96}"
# 当前主线默认测 steady-state，因此把 request_1 之后的隐藏 prewarm 默认打开。
# 这样正式 request_2 更接近我们真正关心的稳态口径，而不会把一次性 warmup
# 重新算回主结果。
DIRECT_RETRIEVE_PREWARM_AFTER_REQUEST1="${DIRECT_RETRIEVE_PREWARM_AFTER_REQUEST1:-1}"
LOG_ROOT="${LOG_ROOT:-${SCRIPT_DIR}/logs/single_gpu_lmcache_no_prefix_reuse_qwen25}"
RUN_DIR="${RUN_DIR:-${LOG_ROOT}/${TIMESTAMP}}"
LOG_FILE="${LOG_FILE:-${RUN_DIR}/run.log}"
LMCACHE_REPO_PATH="${LMCACHE_REPO_PATH:-/home/xhk/llm-inference/LMCache-v0-torch26}"
# 这里不要默认用裸 `python`：
# - sudo/root 环境下 PATH 往往和当前用户不同，容易出现 `python: command not found`
# - vllm-bam/LMCache 这条链路通常还依赖特定 conda 环境
# 因此优先使用我们当前实验环境里已经验证过的解释器绝对路径。
PYTHON_BIN="${PYTHON_BIN:-/home/xhk/miniconda3/envs/pytorch-vllm/bin/python}"
DTYPE="${DTYPE:-half}"
ENFORCE_EAGER="${ENFORCE_EAGER:-true}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-false}"
LMCACHE_USE_EXPERIMENTAL_VALUE="${LMCACHE_USE_EXPERIMENTAL:-True}"
LMCACHE_LOCAL_CPU_VALUE="${LMCACHE_LOCAL_CPU:-False}"
LMCACHE_LOCAL_DISK_VALUE="${LMCACHE_LOCAL_DISK:-/home/xhk/llm-inference/lmcache_local_disk/}"
LMCACHE_MAX_LOCAL_DISK_SIZE_VALUE="${LMCACHE_MAX_LOCAL_DISK_SIZE:-20}"
# 下面这组默认值固定到当前已经跑通、并持续作为主线联调口径的配置：
# - LMCache chunk 持续 shadow 到 BaM
# - request_2 优先从 BaM prefer-load 恢复 prefix
# - 打开 KV fast path
# - 打开 direct placement，并默认走当前更贴近目标形态的 fused 路径
#
# 调试/回归到其它口径时，仍然可以通过环境变量显式覆盖。
VLLM_BAM_LMCACHE_SHADOW_ENABLE_VALUE="${VLLM_BAM_LMCACHE_SHADOW_ENABLE:-1}"
VLLM_BAM_LMCACHE_PREFER_LOAD_ENABLE_VALUE="${VLLM_BAM_LMCACHE_PREFER_LOAD_ENABLE:-1}"
VLLM_BAM_LMCACHE_SHADOW_CHUNKS_VALUE="${VLLM_BAM_LMCACHE_SHADOW_CHUNKS:-1024}"
VLLM_BAM_KV_FAST_PATH_VALUE="${VLLM_BAM_KV_FAST_PATH:-1}"
VLLM_BAM_DIRECT_PLACEMENT_VALUE="${VLLM_BAM_DIRECT_PLACEMENT:-1}"
VLLM_BAM_DIRECT_PLACEMENT_IMPL_VALUE="${VLLM_BAM_DIRECT_PLACEMENT_IMPL:-fused}"
# 当前 KV 路径只保留三条工程分支。用户层只需要选择分支名，底层
# executor / runtime / persistent / one-copy 开关都从这里派生，避免出现
# “executor 是 gpu_worker，但 persistent 没开”这类半配置状态。
#
# 兼容性：若外部没有设置 `VLLM_BAM_KV_BRANCH`，仍会根据旧低层变量推断
# 分支，这样历史命令不会立刻失效；但推荐新命令只传分支名。
VLLM_BAM_KV_BRANCH_VALUE="${VLLM_BAM_KV_BRANCH:-}"
if [[ -z "${VLLM_BAM_KV_BRANCH_VALUE}" ]]; then
  LEGACY_KV_EXECUTOR="${VLLM_BAM_KV_EXECUTOR:-rowctx}"
  LEGACY_RUNTIME_ENABLE="${GIDS_KV_GPU_WORKER_RUNTIME_ENABLE:-0}"
  LEGACY_PERSISTENT_ENABLE="${GIDS_KV_GPU_WORKER_PERSISTENT_ENABLE:-0}"
  LEGACY_ONE_COPY="${VLLM_BAM_DIRECT_PLACEMENT_RUNTIME_ONE_COPY:-0}"
  LEGACY_REQUIRE_ONE_COPY="${VLLM_BAM_DIRECT_PLACEMENT_REQUIRE_RUNTIME_ONE_COPY:-0}"
  if [[ "${LEGACY_ONE_COPY}" == "1" || "${LEGACY_REQUIRE_ONE_COPY}" == "1" ]]; then
    VLLM_BAM_KV_BRANCH_VALUE="gpu_worker_persistent_one_copy"
  elif [[ "${LEGACY_KV_EXECUTOR}" == "gpu_worker" || \
          "${LEGACY_RUNTIME_ENABLE}" == "1" || \
          "${LEGACY_PERSISTENT_ENABLE}" == "1" ]]; then
    VLLM_BAM_KV_BRANCH_VALUE="gpu_worker_persistent_materialized"
  else
    VLLM_BAM_KV_BRANCH_VALUE="rowctx_baseline"
  fi
fi

case "${VLLM_BAM_KV_BRANCH_VALUE}" in
  rowctx_baseline)
    VLLM_BAM_KV_EXECUTOR_VALUE="rowctx"
    GIDS_KV_GPU_WORKER_RUNTIME_ENABLE_VALUE="0"
    GIDS_KV_GPU_WORKER_PERSISTENT_ENABLE_VALUE="0"
    VLLM_BAM_DIRECT_PLACEMENT_RUNTIME_ONE_COPY_VALUE="0"
    VLLM_BAM_DIRECT_PLACEMENT_REQUIRE_RUNTIME_ONE_COPY_VALUE="0"
    ;;
  gpu_worker_persistent_materialized)
    VLLM_BAM_KV_EXECUTOR_VALUE="gpu_worker"
    GIDS_KV_GPU_WORKER_RUNTIME_ENABLE_VALUE="1"
    GIDS_KV_GPU_WORKER_PERSISTENT_ENABLE_VALUE="1"
    VLLM_BAM_DIRECT_PLACEMENT_RUNTIME_ONE_COPY_VALUE="0"
    VLLM_BAM_DIRECT_PLACEMENT_REQUIRE_RUNTIME_ONE_COPY_VALUE="0"
    ;;
  gpu_worker_persistent_one_copy)
    VLLM_BAM_KV_EXECUTOR_VALUE="gpu_worker"
    GIDS_KV_GPU_WORKER_RUNTIME_ENABLE_VALUE="1"
    GIDS_KV_GPU_WORKER_PERSISTENT_ENABLE_VALUE="1"
    VLLM_BAM_DIRECT_PLACEMENT_RUNTIME_ONE_COPY_VALUE="1"
    VLLM_BAM_DIRECT_PLACEMENT_REQUIRE_RUNTIME_ONE_COPY_VALUE="1"
    ;;
  *)
    echo "[single-gpu-lmcache-no-prefix-reuse] unsupported VLLM_BAM_KV_BRANCH=${VLLM_BAM_KV_BRANCH_VALUE}" >&2
    echo "[single-gpu-lmcache-no-prefix-reuse] valid branches: rowctx_baseline, gpu_worker_persistent_materialized, gpu_worker_persistent_one_copy" >&2
    exit 2
    ;;
esac
# 这里不要再默认把 metadata attachment 显式压成 0。
#
# 当前 adapter 已经有更合理的主线推导：
# - 若外部显式设置了 `VLLM_BAM_RUNTIME_METADATA_ATTACHMENT_ENABLE`
#   就尊重外部值
# - 否则只要已经进入 runtime one-copy / require-runtime-one-copy 主线，
#   就自动把 metadata attachment 视为这条主线的一部分
#
# 因此脚本层只在“外部真的显式传了值”时才继续透传；未设置时留空，让代码侧
# 自己按当前主线推导，避免脚本默认值把主线语义反向盖掉。
VLLM_BAM_RUNTIME_METADATA_ATTACHMENT_ENABLE_VALUE="${VLLM_BAM_RUNTIME_METADATA_ATTACHMENT_ENABLE:-}"
# 当前主线默认已经切到 request-handle runtime 路径：
# - 打开 `DEFER_RUNTIME`，让 connector/runtime 能持有 live handle；
# - 但默认不再强制 `min defer polls`，也就是：
#   只有真实 `not_ready` 时才返回 `DEFERRED`。
#
# 这样日志里如果出现 WAIT，默认就代表真实“当前还没 ready”，而不是为了验证
# 跨 iteration 人为多插了一轮空转。
VLLM_BAM_DIRECT_PLACEMENT_DEFER_RUNTIME_VALUE="${VLLM_BAM_DIRECT_PLACEMENT_DEFER_RUNTIME:-1}"
VLLM_BAM_DIRECT_PLACEMENT_DEFER_MIN_POLLS_VALUE="${VLLM_BAM_DIRECT_PLACEMENT_DEFER_MIN_POLLS:-0}"
VLLM_BAM_XFORMERS_PREFIX_FALLBACK_PROFILE_VALUE="${VLLM_BAM_XFORMERS_PREFIX_FALLBACK_PROFILE:-0}"
VLLM_BAM_XFORMERS_PREFIX_BACKEND_VALUE="${VLLM_BAM_XFORMERS_PREFIX_BACKEND:-auto}"
# query backend 是 xFormers fallback 的薄选择开关：
# - direct_scatter：当前较快的 Triton scatter 路径
# - segment_copy：保守逐段 copy 路径，用来排除 scatter kernel / 位置映射问题
VLLM_BAM_XFORMERS_QUERY_BACKEND_VALUE="${VLLM_BAM_XFORMERS_QUERY_BACKEND:-auto}"
if [[ -n "${VLLM_BAM_LMCACHE_READ_MODE:-}" ]]; then
  VLLM_BAM_LMCACHE_READ_MODE_VALUE="${VLLM_BAM_LMCACHE_READ_MODE}"
elif [[ "${VLLM_BAM_KV_FAST_PATH_VALUE}" == "1" ]]; then
  VLLM_BAM_LMCACHE_READ_MODE_VALUE="prefetch"
else
  VLLM_BAM_LMCACHE_READ_MODE_VALUE="sync"
fi
VLLM_BAM_IMPORT_PATH_VALUE="${VLLM_BAM_IMPORT_PATH:-/home/xhk/llm-inference/BaM_IOStack/gids_module}"
BAM_LIB_DIR_VALUE="${BAM_LIB_DIR:-/home/xhk/llm-inference/BaM_IOStack/bam/build/lib}"
# 当前 KV cache 路径下，1GB cache 在最近几轮联调里更稳，减少了 cache 容量
# 偏小时的额外抖动，因此把默认值从 512MB 提到 1024MB。
VLLM_BAM_CACHE_SIZE_MB_VALUE="${VLLM_BAM_CACHE_SIZE_MB:-1024}"
VLLM_BAM_NUM_SSD_VALUE="${VLLM_BAM_NUM_SSD:-1}"
VLLM_BAM_SSD_LIST_VALUE="${VLLM_BAM_SSD_LIST:-0}"
VLLM_BAM_CTRL_IDX_VALUE="${VLLM_BAM_CTRL_IDX:-0}"
GIDS_FORCE_SYNC_READ_VALUE="${GIDS_FORCE_SYNC_READ:-1}"
GIDS_KV_DEBUG_VALUE="${GIDS_KV_DEBUG:-0}"
GIDS_KV_WAIT_TIMEOUT_S_VALUE="${GIDS_KV_WAIT_TIMEOUT_S:-10}"
if [[ -n "${GIDS_KV_GPU_WORKER_MOVER_CTAS:-}" ]]; then
  GIDS_KV_GPU_WORKER_MOVER_CTAS_VALUE="${GIDS_KV_GPU_WORKER_MOVER_CTAS}"
elif [[ "${VLLM_BAM_KV_BRANCH_VALUE}" == "gpu_worker_persistent_one_copy" ]]; then
  # 2026-07-19 后，cta=4 one-copy 是当前稳定性能基线。
  # 非 one-copy 分支保持 0，避免 materialized/rowctx 被额外 mover CTA 干扰。
  GIDS_KV_GPU_WORKER_MOVER_CTAS_VALUE="4"
else
  GIDS_KV_GPU_WORKER_MOVER_CTAS_VALUE="0"
fi
GIDS_KV_GPU_WORKER_MODE_VALUE="${GIDS_KV_GPU_WORKER_MODE:-dedicated}"
BAM_PREFLIGHT_VALUE="${BAM_PREFLIGHT:-0}"

# BaM 读写路径都可能需要更高权限。只有显式打开 BaM 相关路径时才自动提权，
# 普通 LMCache SSD baseline 不受影响。
if [[ ( "${VLLM_BAM_LMCACHE_SHADOW_ENABLE_VALUE}" == "1" || \
        "${VLLM_BAM_LMCACHE_PREFER_LOAD_ENABLE_VALUE}" == "1" ) && \
        "${EUID}" -ne 0 ]]; then
  SUDO_ENV_ARGS=(
    "MODEL_PATH=${MODEL_PATH}"
    "CUDA_DEVICE=${CUDA_DEVICE}"
    "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
    "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
    "LMCACHE_CHUNK_SIZE_VALUE=${LMCACHE_CHUNK_SIZE_VALUE}"
    "LMCACHE_MAX_LOCAL_CPU_SIZE_VALUE=${LMCACHE_MAX_LOCAL_CPU_SIZE_VALUE}"
    "PROMPT_REPEAT=${PROMPT_REPEAT}"
    "MAX_TOKENS=${MAX_TOKENS}"
    "STEADY_REPEAT=${STEADY_REPEAT}"
    "PROMPT_BUILD_MODE=${PROMPT_BUILD_MODE}"
    "PROMPT_TARGET_TOKENS=${PROMPT_TARGET_TOKENS}"
    "PROMPT_REUSE_REQUEST1=${PROMPT_REUSE_REQUEST1}"
    "PROMPT_TARGET_MARGIN=${PROMPT_TARGET_MARGIN}"
    "DIRECT_RETRIEVE_PREWARM_AFTER_REQUEST1=${DIRECT_RETRIEVE_PREWARM_AFTER_REQUEST1}"
    "LOG_ROOT=${LOG_ROOT}"
    "RUN_DIR=${RUN_DIR}"
    "LOG_FILE=${LOG_FILE}"
    "LMCACHE_REPO_PATH=${LMCACHE_REPO_PATH}"
    "PYTHON_BIN=${PYTHON_BIN}"
    "DTYPE=${DTYPE}"
    "ENFORCE_EAGER=${ENFORCE_EAGER}"
    "ENABLE_CHUNKED_PREFILL=${ENABLE_CHUNKED_PREFILL}"
    "LMCACHE_USE_EXPERIMENTAL=${LMCACHE_USE_EXPERIMENTAL_VALUE}"
    "LMCACHE_LOCAL_CPU=${LMCACHE_LOCAL_CPU_VALUE}"
    "LMCACHE_LOCAL_DISK=${LMCACHE_LOCAL_DISK_VALUE}"
    "LMCACHE_MAX_LOCAL_DISK_SIZE=${LMCACHE_MAX_LOCAL_DISK_SIZE_VALUE}"
    "VLLM_BAM_LMCACHE_SHADOW_ENABLE=${VLLM_BAM_LMCACHE_SHADOW_ENABLE_VALUE}"
    "VLLM_BAM_LMCACHE_PREFER_LOAD_ENABLE=${VLLM_BAM_LMCACHE_PREFER_LOAD_ENABLE_VALUE}"
    "VLLM_BAM_LMCACHE_SHADOW_CHUNKS=${VLLM_BAM_LMCACHE_SHADOW_CHUNKS_VALUE}"
    "VLLM_BAM_LMCACHE_READ_MODE=${VLLM_BAM_LMCACHE_READ_MODE_VALUE}"
    "VLLM_BAM_KV_FAST_PATH=${VLLM_BAM_KV_FAST_PATH_VALUE}"
    "VLLM_BAM_KV_BRANCH=${VLLM_BAM_KV_BRANCH_VALUE}"
    "VLLM_BAM_KV_EXECUTOR=${VLLM_BAM_KV_EXECUTOR_VALUE}"
    "VLLM_BAM_DIRECT_PLACEMENT=${VLLM_BAM_DIRECT_PLACEMENT_VALUE}"
    "VLLM_BAM_DIRECT_PLACEMENT_IMPL=${VLLM_BAM_DIRECT_PLACEMENT_IMPL_VALUE}"
    "VLLM_BAM_DIRECT_PLACEMENT_RUNTIME_ONE_COPY=${VLLM_BAM_DIRECT_PLACEMENT_RUNTIME_ONE_COPY_VALUE}"
    "VLLM_BAM_DIRECT_PLACEMENT_REQUIRE_RUNTIME_ONE_COPY=${VLLM_BAM_DIRECT_PLACEMENT_REQUIRE_RUNTIME_ONE_COPY_VALUE}"
    "VLLM_BAM_DIRECT_PLACEMENT_DEFER_RUNTIME=${VLLM_BAM_DIRECT_PLACEMENT_DEFER_RUNTIME_VALUE}"
    "VLLM_BAM_DIRECT_PLACEMENT_DEFER_MIN_POLLS=${VLLM_BAM_DIRECT_PLACEMENT_DEFER_MIN_POLLS_VALUE}"
    "VLLM_BAM_XFORMERS_PREFIX_FALLBACK_PROFILE=${VLLM_BAM_XFORMERS_PREFIX_FALLBACK_PROFILE_VALUE}"
    "VLLM_BAM_XFORMERS_PREFIX_BACKEND=${VLLM_BAM_XFORMERS_PREFIX_BACKEND_VALUE}"
    "VLLM_BAM_XFORMERS_QUERY_BACKEND=${VLLM_BAM_XFORMERS_QUERY_BACKEND_VALUE}"
    "VLLM_BAM_IMPORT_PATH=${VLLM_BAM_IMPORT_PATH_VALUE}"
    "VLLM_BAM_CACHE_SIZE_MB=${VLLM_BAM_CACHE_SIZE_MB_VALUE}"
    "VLLM_BAM_NUM_SSD=${VLLM_BAM_NUM_SSD_VALUE}"
    "VLLM_BAM_SSD_LIST=${VLLM_BAM_SSD_LIST_VALUE}"
    "VLLM_BAM_CTRL_IDX=${VLLM_BAM_CTRL_IDX_VALUE}"
    "GIDS_FORCE_SYNC_READ=${GIDS_FORCE_SYNC_READ_VALUE}"
    "GIDS_KV_DEBUG=${GIDS_KV_DEBUG_VALUE}"
    "GIDS_KV_WAIT_TIMEOUT_S=${GIDS_KV_WAIT_TIMEOUT_S_VALUE}"
    "GIDS_KV_GPU_WORKER_RUNTIME_ENABLE=${GIDS_KV_GPU_WORKER_RUNTIME_ENABLE_VALUE}"
    "GIDS_KV_GPU_WORKER_PERSISTENT_ENABLE=${GIDS_KV_GPU_WORKER_PERSISTENT_ENABLE_VALUE}"
    "GIDS_KV_GPU_WORKER_MOVER_CTAS=${GIDS_KV_GPU_WORKER_MOVER_CTAS_VALUE}"
    "GIDS_KV_GPU_WORKER_MODE=${GIDS_KV_GPU_WORKER_MODE_VALUE}"
    "BAM_PREFLIGHT=${BAM_PREFLIGHT_VALUE}"
    "BAM_LIB_DIR=${BAM_LIB_DIR_VALUE}"
    "LD_LIBRARY_PATH=${BAM_LIB_DIR_VALUE}:${LD_LIBRARY_PATH:-}"
    "PYTHONPATH=${PYTHONPATH:-}"
  )
  if [[ -n "${VLLM_BAM_RUNTIME_METADATA_ATTACHMENT_ENABLE_VALUE}" ]]; then
    SUDO_ENV_ARGS+=(
      "VLLM_BAM_RUNTIME_METADATA_ATTACHMENT_ENABLE=${VLLM_BAM_RUNTIME_METADATA_ATTACHMENT_ENABLE_VALUE}"
    )
  fi
  # 注意：这里必须把 LOG_ROOT / RUN_DIR / LOG_FILE 一起透传给 sudo 后的
  # root 进程，避免出现“外层脚本先算出一个时间戳目录，内层脚本又重新按
  # 新时间戳计算 RUN_DIR，但 LOG_FILE 仍指向旧目录”的分裂状态。
  #
  # 否则会出现两个现象：
  # 1. evaluation/logs 下多出一个空目录；
  # 2. run_dir 与 log_file 指向不同目录，后续排查日志时容易误判。
  exec sudo env "${SUDO_ENV_ARGS[@]}" bash "$0" "$@"
fi

if [[ "${LMCACHE_LOCAL_DISK_VALUE}" == file://* ]]; then
  LMCACHE_LOCAL_DISK_VALUE="${LMCACHE_LOCAL_DISK_VALUE#file://}"
fi

mkdir -p "${LMCACHE_LOCAL_DISK_VALUE}"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export VLLM_USE_V1=0
export LMCACHE_USE_EXPERIMENTAL="${LMCACHE_USE_EXPERIMENTAL_VALUE}"
export LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE_VALUE}"
export LMCACHE_LOCAL_CPU="${LMCACHE_LOCAL_CPU_VALUE}"
export LMCACHE_MAX_LOCAL_CPU_SIZE="${LMCACHE_MAX_LOCAL_CPU_SIZE_VALUE}"
export LMCACHE_MAX_LOCAL_DISK_SIZE="${LMCACHE_MAX_LOCAL_DISK_SIZE_VALUE}"
export VLLM_BAM_LMCACHE_SHADOW_ENABLE="${VLLM_BAM_LMCACHE_SHADOW_ENABLE_VALUE}"
export VLLM_BAM_LMCACHE_PREFER_LOAD_ENABLE="${VLLM_BAM_LMCACHE_PREFER_LOAD_ENABLE_VALUE}"
export VLLM_BAM_LMCACHE_SHADOW_CHUNKS="${VLLM_BAM_LMCACHE_SHADOW_CHUNKS_VALUE}"
export VLLM_BAM_LMCACHE_READ_MODE="${VLLM_BAM_LMCACHE_READ_MODE_VALUE}"
export VLLM_BAM_KV_FAST_PATH="${VLLM_BAM_KV_FAST_PATH_VALUE}"
export VLLM_BAM_KV_BRANCH="${VLLM_BAM_KV_BRANCH_VALUE}"
export VLLM_BAM_KV_EXECUTOR="${VLLM_BAM_KV_EXECUTOR_VALUE}"
export VLLM_BAM_DIRECT_PLACEMENT="${VLLM_BAM_DIRECT_PLACEMENT_VALUE}"
export VLLM_BAM_DIRECT_PLACEMENT_IMPL="${VLLM_BAM_DIRECT_PLACEMENT_IMPL_VALUE}"
export VLLM_BAM_DIRECT_PLACEMENT_RUNTIME_ONE_COPY="${VLLM_BAM_DIRECT_PLACEMENT_RUNTIME_ONE_COPY_VALUE}"
export VLLM_BAM_DIRECT_PLACEMENT_REQUIRE_RUNTIME_ONE_COPY="${VLLM_BAM_DIRECT_PLACEMENT_REQUIRE_RUNTIME_ONE_COPY_VALUE}"
export VLLM_BAM_DIRECT_PLACEMENT_DEFER_RUNTIME="${VLLM_BAM_DIRECT_PLACEMENT_DEFER_RUNTIME_VALUE}"
export VLLM_BAM_DIRECT_PLACEMENT_DEFER_MIN_POLLS="${VLLM_BAM_DIRECT_PLACEMENT_DEFER_MIN_POLLS_VALUE}"
export VLLM_BAM_XFORMERS_PREFIX_FALLBACK_PROFILE="${VLLM_BAM_XFORMERS_PREFIX_FALLBACK_PROFILE_VALUE}"
export VLLM_BAM_XFORMERS_PREFIX_BACKEND="${VLLM_BAM_XFORMERS_PREFIX_BACKEND_VALUE}"
export VLLM_BAM_XFORMERS_QUERY_BACKEND="${VLLM_BAM_XFORMERS_QUERY_BACKEND_VALUE}"
export VLLM_BAM_IMPORT_PATH="${VLLM_BAM_IMPORT_PATH_VALUE}"
export VLLM_BAM_CACHE_SIZE_MB="${VLLM_BAM_CACHE_SIZE_MB_VALUE}"
export VLLM_BAM_NUM_SSD="${VLLM_BAM_NUM_SSD_VALUE}"
export VLLM_BAM_SSD_LIST="${VLLM_BAM_SSD_LIST_VALUE}"
export VLLM_BAM_CTRL_IDX="${VLLM_BAM_CTRL_IDX_VALUE}"
export GIDS_FORCE_SYNC_READ="${GIDS_FORCE_SYNC_READ_VALUE}"
export GIDS_KV_DEBUG="${GIDS_KV_DEBUG_VALUE}"
export GIDS_KV_WAIT_TIMEOUT_S="${GIDS_KV_WAIT_TIMEOUT_S_VALUE}"
export GIDS_KV_GPU_WORKER_RUNTIME_ENABLE="${GIDS_KV_GPU_WORKER_RUNTIME_ENABLE_VALUE}"
export GIDS_KV_GPU_WORKER_PERSISTENT_ENABLE="${GIDS_KV_GPU_WORKER_PERSISTENT_ENABLE_VALUE}"
export GIDS_KV_GPU_WORKER_MOVER_CTAS="${GIDS_KV_GPU_WORKER_MOVER_CTAS_VALUE}"
export GIDS_KV_GPU_WORKER_MODE="${GIDS_KV_GPU_WORKER_MODE_VALUE}"
export DIRECT_RETRIEVE_PREWARM_AFTER_REQUEST1="${DIRECT_RETRIEVE_PREWARM_AFTER_REQUEST1}"
export PROMPT_REUSE_REQUEST1="${PROMPT_REUSE_REQUEST1}"
export LD_LIBRARY_PATH="${BAM_LIB_DIR_VALUE}:${LD_LIBRARY_PATH:-}"
if [[ -n "${VLLM_BAM_RUNTIME_METADATA_ATTACHMENT_ENABLE_VALUE}" ]]; then
  export VLLM_BAM_RUNTIME_METADATA_ATTACHMENT_ENABLE="${VLLM_BAM_RUNTIME_METADATA_ATTACHMENT_ENABLE_VALUE}"
else
  unset VLLM_BAM_RUNTIME_METADATA_ATTACHMENT_ENABLE || true
fi

if [[ -n "${LMCACHE_LOCAL_DISK_VALUE}" ]]; then
  export LMCACHE_LOCAL_DISK="${LMCACHE_LOCAL_DISK_VALUE}"
else
  unset LMCACHE_LOCAL_DISK || true
fi

if [[ -d "${LMCACHE_REPO_PATH}/lmcache" ]]; then
  export PYTHONPATH="${LMCACHE_REPO_PATH}:${PYTHONPATH:-}"
fi

# 默认把日志放到仓库内的 evaluation/logs，按时间戳分目录留档。
# 如果外部显式传入 LOG_FILE / RUN_DIR / LOG_ROOT，会优先使用外部设置。
mkdir -p "${RUN_DIR}"
mkdir -p "$(dirname "${LOG_FILE}")"
rm -f "${LOG_FILE}"

if [[ ( "${VLLM_BAM_LMCACHE_SHADOW_ENABLE_VALUE}" == "1" || \
        "${VLLM_BAM_LMCACHE_PREFER_LOAD_ENABLE_VALUE}" == "1" ) && \
      "${BAM_PREFLIGHT_VALUE}" == "1" ]]; then
  echo "[single-gpu-lmcache-no-prefix-reuse] running BaM preflight"
  "${PYTHON_BIN}" - <<'PY'
import os
import sys

sys.path.insert(0, os.environ["VLLM_BAM_IMPORT_PATH"])
import bam_row_store

ssd_list = [
    int(item.strip()) for item in os.environ.get("VLLM_BAM_SSD_LIST", "0").split(",")
    if item.strip()
]

print("[bam-preflight] pid =", os.getpid())
print("[bam-preflight] euid =", os.geteuid())
print("[bam-preflight] executable =", sys.executable)
print("[bam-preflight] import_path =", os.environ["VLLM_BAM_IMPORT_PATH"])
print("[bam-preflight] ssd_list =", ssd_list)
print("[bam-preflight] start init")
store = bam_row_store.BaMRowStore(
    row_bytes=14336,
    num_rows=57344,
    cache_size_mb=int(os.environ.get("VLLM_BAM_CACHE_SIZE_MB", "512")),
    num_ssd=int(os.environ.get("VLLM_BAM_NUM_SSD", "1")),
    ssd_list=ssd_list,
    ctrl_idx=int(os.environ.get("VLLM_BAM_CTRL_IDX", "0")),
)
print("[bam-preflight] init ok", store.row_bytes, store.num_rows)
PY
fi

echo "[single-gpu-lmcache-no-prefix-reuse] model=${MODEL_PATH}"
echo "[single-gpu-lmcache-no-prefix-reuse] cuda_device=${CUDA_DEVICE}"
echo "[single-gpu-lmcache-no-prefix-reuse] vllm_use_v1=${VLLM_USE_V1}"
echo "[single-gpu-lmcache-no-prefix-reuse] max_model_len=${MAX_MODEL_LEN}"
echo "[single-gpu-lmcache-no-prefix-reuse] gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
echo "[single-gpu-lmcache-no-prefix-reuse] lmcache_chunk_size=${LMCACHE_CHUNK_SIZE_VALUE}"
echo "[single-gpu-lmcache-no-prefix-reuse] lmcache_max_local_cpu_size=${LMCACHE_MAX_LOCAL_CPU_SIZE_VALUE}"
echo "[single-gpu-lmcache-no-prefix-reuse] lmcache_use_experimental=${LMCACHE_USE_EXPERIMENTAL_VALUE}"
echo "[single-gpu-lmcache-no-prefix-reuse] lmcache_local_cpu=${LMCACHE_LOCAL_CPU_VALUE}"
echo "[single-gpu-lmcache-no-prefix-reuse] lmcache_local_disk=${LMCACHE_LOCAL_DISK_VALUE}"
echo "[single-gpu-lmcache-no-prefix-reuse] lmcache_max_local_disk_size=${LMCACHE_MAX_LOCAL_DISK_SIZE_VALUE}"
echo "[single-gpu-lmcache-no-prefix-reuse] vllm_bam_lmcache_shadow_enable=${VLLM_BAM_LMCACHE_SHADOW_ENABLE_VALUE}"
echo "[single-gpu-lmcache-no-prefix-reuse] vllm_bam_lmcache_prefer_load_enable=${VLLM_BAM_LMCACHE_PREFER_LOAD_ENABLE_VALUE}"
echo "[single-gpu-lmcache-no-prefix-reuse] vllm_bam_lmcache_shadow_chunks=${VLLM_BAM_LMCACHE_SHADOW_CHUNKS_VALUE}"
echo "[single-gpu-lmcache-no-prefix-reuse] vllm_bam_lmcache_read_mode=${VLLM_BAM_LMCACHE_READ_MODE_VALUE}"
echo "[single-gpu-lmcache-no-prefix-reuse] vllm_bam_kv_fast_path=${VLLM_BAM_KV_FAST_PATH_VALUE}"
echo "[single-gpu-lmcache-no-prefix-reuse] vllm_bam_kv_branch=${VLLM_BAM_KV_BRANCH_VALUE}"
echo "[single-gpu-lmcache-no-prefix-reuse] vllm_bam_kv_executor=${VLLM_BAM_KV_EXECUTOR_VALUE}"
echo "[single-gpu-lmcache-no-prefix-reuse] vllm_bam_direct_placement=${VLLM_BAM_DIRECT_PLACEMENT_VALUE}"
echo "[single-gpu-lmcache-no-prefix-reuse] vllm_bam_direct_placement_impl=${VLLM_BAM_DIRECT_PLACEMENT_IMPL_VALUE}"
echo "[single-gpu-lmcache-no-prefix-reuse] vllm_bam_direct_placement_runtime_one_copy=${VLLM_BAM_DIRECT_PLACEMENT_RUNTIME_ONE_COPY_VALUE}"
echo "[single-gpu-lmcache-no-prefix-reuse] vllm_bam_direct_placement_require_runtime_one_copy=${VLLM_BAM_DIRECT_PLACEMENT_REQUIRE_RUNTIME_ONE_COPY_VALUE}"
echo "[single-gpu-lmcache-no-prefix-reuse] vllm_bam_direct_placement_defer_runtime=${VLLM_BAM_DIRECT_PLACEMENT_DEFER_RUNTIME_VALUE}"
echo "[single-gpu-lmcache-no-prefix-reuse] vllm_bam_direct_placement_defer_min_polls=${VLLM_BAM_DIRECT_PLACEMENT_DEFER_MIN_POLLS_VALUE}"
echo "[single-gpu-lmcache-no-prefix-reuse] vllm_bam_xformers_prefix_fallback_profile=${VLLM_BAM_XFORMERS_PREFIX_FALLBACK_PROFILE_VALUE}"
echo "[single-gpu-lmcache-no-prefix-reuse] vllm_bam_xformers_prefix_backend=${VLLM_BAM_XFORMERS_PREFIX_BACKEND_VALUE}"
echo "[single-gpu-lmcache-no-prefix-reuse] vllm_bam_xformers_query_backend=${VLLM_BAM_XFORMERS_QUERY_BACKEND_VALUE}"
echo "[single-gpu-lmcache-no-prefix-reuse] vllm_bam_runtime_metadata_attachment_enable=${VLLM_BAM_RUNTIME_METADATA_ATTACHMENT_ENABLE_VALUE:-<auto>}"
echo "[single-gpu-lmcache-no-prefix-reuse] vllm_bam_import_path=${VLLM_BAM_IMPORT_PATH_VALUE}"
echo "[single-gpu-lmcache-no-prefix-reuse] vllm_bam_cache_size_mb=${VLLM_BAM_CACHE_SIZE_MB_VALUE}"
echo "[single-gpu-lmcache-no-prefix-reuse] vllm_bam_num_ssd=${VLLM_BAM_NUM_SSD_VALUE}"
echo "[single-gpu-lmcache-no-prefix-reuse] vllm_bam_ssd_list=${VLLM_BAM_SSD_LIST_VALUE}"
echo "[single-gpu-lmcache-no-prefix-reuse] vllm_bam_ctrl_idx=${VLLM_BAM_CTRL_IDX_VALUE}"
echo "[single-gpu-lmcache-no-prefix-reuse] gids_force_sync_read=${GIDS_FORCE_SYNC_READ_VALUE}"
echo "[single-gpu-lmcache-no-prefix-reuse] gids_kv_debug=${GIDS_KV_DEBUG_VALUE}"
echo "[single-gpu-lmcache-no-prefix-reuse] gids_kv_wait_timeout_s=${GIDS_KV_WAIT_TIMEOUT_S_VALUE}"
echo "[single-gpu-lmcache-no-prefix-reuse] gids_kv_gpu_worker_runtime_enable=${GIDS_KV_GPU_WORKER_RUNTIME_ENABLE_VALUE}"
echo "[single-gpu-lmcache-no-prefix-reuse] gids_kv_gpu_worker_persistent_enable=${GIDS_KV_GPU_WORKER_PERSISTENT_ENABLE_VALUE}"
echo "[single-gpu-lmcache-no-prefix-reuse] gids_kv_gpu_worker_mover_ctas=${GIDS_KV_GPU_WORKER_MOVER_CTAS_VALUE}"
echo "[single-gpu-lmcache-no-prefix-reuse] gids_kv_gpu_worker_mode=${GIDS_KV_GPU_WORKER_MODE_VALUE}"
echo "[single-gpu-lmcache-no-prefix-reuse] bam_preflight=${BAM_PREFLIGHT_VALUE}"
echo "[single-gpu-lmcache-no-prefix-reuse] bam_lib_dir=${BAM_LIB_DIR_VALUE}"
echo "[single-gpu-lmcache-no-prefix-reuse] prompt_repeat=${PROMPT_REPEAT}"
echo "[single-gpu-lmcache-no-prefix-reuse] max_tokens=${MAX_TOKENS}"
echo "[single-gpu-lmcache-no-prefix-reuse] steady_repeat=${STEADY_REPEAT}"
echo "[single-gpu-lmcache-no-prefix-reuse] prompt_build_mode=${PROMPT_BUILD_MODE}"
echo "[single-gpu-lmcache-no-prefix-reuse] prompt_target_tokens=${PROMPT_TARGET_TOKENS}"
echo "[single-gpu-lmcache-no-prefix-reuse] prompt_target_margin=${PROMPT_TARGET_MARGIN}"
echo "[single-gpu-lmcache-no-prefix-reuse] direct_retrieve_prewarm_after_request1=${DIRECT_RETRIEVE_PREWARM_AFTER_REQUEST1}"
echo "[single-gpu-lmcache-no-prefix-reuse] log_root=${LOG_ROOT}"
echo "[single-gpu-lmcache-no-prefix-reuse] run_dir=${RUN_DIR}"
echo "[single-gpu-lmcache-no-prefix-reuse] log_file=${LOG_FILE}"
echo "[single-gpu-lmcache-no-prefix-reuse] lmcache_repo_path=${LMCACHE_REPO_PATH}"
echo "[single-gpu-lmcache-no-prefix-reuse] python_bin=${PYTHON_BIN}"
echo "[single-gpu-lmcache-no-prefix-reuse] dtype=${DTYPE}"
echo "[single-gpu-lmcache-no-prefix-reuse] enforce_eager=${ENFORCE_EAGER}"
echo "[single-gpu-lmcache-no-prefix-reuse] enable_chunked_prefill=${ENABLE_CHUNKED_PREFILL}"
if [[ "${VLLM_BAM_LMCACHE_PREFER_LOAD_ENABLE_VALUE}" == "1" ]]; then
  echo "[single-gpu-lmcache-no-prefix-reuse] prefer-load mode: request_2 reuses request_1 prompt"
fi
echo "[single-gpu-lmcache-no-prefix-reuse] user=$(id -un) euid=$(id -u)"
if [[ -e "${LOG_FILE}" ]]; then
  echo "[single-gpu-lmcache-no-prefix-reuse] log_file_perm=$(stat -c '%a %U %G' "${LOG_FILE}")"
else
  echo "[single-gpu-lmcache-no-prefix-reuse] log_file_perm=<not-created-yet>"
fi

cd "${SCRIPT_DIR}/.."

MODEL_PATH="${MODEL_PATH}" \
MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
PROMPT_REPEAT="${PROMPT_REPEAT}" \
MAX_TOKENS="${MAX_TOKENS}" \
STEADY_REPEAT="${STEADY_REPEAT}" \
PROMPT_BUILD_MODE="${PROMPT_BUILD_MODE}" \
PROMPT_TARGET_TOKENS="${PROMPT_TARGET_TOKENS}" \
PROMPT_TARGET_MARGIN="${PROMPT_TARGET_MARGIN}" \
DIRECT_RETRIEVE_PREWARM_AFTER_REQUEST1="${DIRECT_RETRIEVE_PREWARM_AFTER_REQUEST1}" \
DTYPE="${DTYPE}" \
ENFORCE_EAGER="${ENFORCE_EAGER}" \
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL}" \
"${PYTHON_BIN}" - <<'PY' 2>&1 | tee "${LOG_FILE}"
import contextlib
import os
import time

from lmcache.integration.vllm.utils import ENGINE_NAME
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

try:
    from lmcache.experimental.cache_engine import LMCacheEngineBuilder
except ImportError:
    from lmcache.v1.cache_engine import LMCacheEngineBuilder

model = os.environ["MODEL_PATH"]
max_model_len = int(os.environ.get("MAX_MODEL_LEN", "4096"))
gpu_memory_utilization = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.60"))
prompt_repeat = int(os.environ.get("PROMPT_REPEAT", "100"))
max_tokens = int(os.environ.get("MAX_TOKENS", "64"))
dtype = os.environ.get("DTYPE", "half")
enforce_eager = os.environ.get("ENFORCE_EAGER", "true").lower() == "true"
enable_chunked_prefill = os.environ.get("ENABLE_CHUNKED_PREFILL",
                                        "false").lower() == "true"
steady_repeat = int(os.environ.get("STEADY_REPEAT", "0"))
prompt_build_mode = os.environ.get("PROMPT_BUILD_MODE", "repeat").strip().lower()
prompt_target_tokens = int(os.environ.get("PROMPT_TARGET_TOKENS", "0"))
prompt_target_margin = int(os.environ.get("PROMPT_TARGET_MARGIN", "96"))
prompt_reuse_request1_env = os.environ.get("PROMPT_REUSE_REQUEST1", "").strip()
direct_retrieve_prewarm_after_request1 = int(
    os.environ.get("DIRECT_RETRIEVE_PREWARM_AFTER_REQUEST1", "0"))
kv_branch = os.environ.get("VLLM_BAM_KV_BRANCH", "rowctx_baseline").strip()


@contextlib.contextmanager
def build_llm():
    ktc = KVTransferConfig.from_cli(
        '{"kv_connector":"LMCacheConnector","kv_role":"kv_both"}'
    )
    llm = LLM(
        model=model,
        kv_transfer_config=ktc,
        max_model_len=max_model_len,
        enable_chunked_prefill=enable_chunked_prefill,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
        enforce_eager=enforce_eager,
        trust_remote_code=False,
    )
    try:
        yield llm
    finally:
        LMCacheEngineBuilder.destroy(ENGINE_NAME)


sampling_params = SamplingParams(
    temperature=0.0,
    top_p=0.95,
    max_tokens=max_tokens,
)

prefer_load_enable = os.environ.get("VLLM_BAM_LMCACHE_PREFER_LOAD_ENABLE",
                                     "0") == "1"
if prompt_reuse_request1_env == "":
    # 默认保持旧语义：只有 prefer-load 实验自动复用 request_1。
    prompt_reuse_request1 = prefer_load_enable
else:
    # 显式开关用于 correctness 对照：
    # 可以关闭 BaM prefer-load/direct-placement，但仍让 request_2 使用
    # request_1 的同一份 prompt，从而单独验证 LMCache 原生 retrieve/rebuild。
    prompt_reuse_request1 = prompt_reuse_request1_env.lower() in (
        "1", "true", "yes", "on")


def _build_repeat_prompts():
    """维持原先按字符串重复构造样例的旧口径。"""
    shared_a = "介绍 LMCache 单卡基线的非复用开销。" * prompt_repeat
    shared_b = "介绍 BaM 接入前如何确认路径可重复性。" * prompt_repeat
    prompt_a_local = [shared_a + "然后请用三句话介绍 Qwen2.5-7B-Instruct。"]
    prompt_b_local = [shared_b + "然后请说明为什么当前阶段先不测共享长前缀复用。"]
    return prompt_a_local, prompt_b_local


def _build_token_budget_prompt(
    *,
    tokenizer,
    prefix_title: str,
    suffix_instruction: str,
) -> tuple[list[str], int, int]:
    """按目标 token 预算自动生成长前缀样例。

    设计目标：

    1. 尽量把 prompt 顶到接近 `max_model_len`，增加 prefix chunk 数；
    2. 保证 prompt 长度不超过预算，避免再出现“猜 `PROMPT_REPEAT` 猜过头”；
    3. 保持生成逻辑简单可读，不把复杂控制面掺进评测脚本。

    返回：
    - prompt 文本（按 vLLM `generate()` 的 list[str] 约定包装）
    - 实际 prompt token 数
    - 本次使用的目标 token 预算
    """
    # 没有显式传目标时，默认给输出 token 和少量控制余量留空间，
    # 避免 prompt 过于贴边而把本轮生成直接顶爆。
    target_budget = prompt_target_tokens
    if target_budget <= 0:
        target_budget = max(max_model_len - max_tokens - prompt_target_margin, 1)
    target_budget = min(target_budget, max(max_model_len - 1, 1))

    header = (
        "下面是一段用于 LMCache + BaM direct retrieve 探测的长前缀材料。"
        "请完整保留前缀上下文，再根据结尾指令作答。\n"
        f"主题：{prefix_title}\n"
        "说明：以下内容故意较长，用于观察 prefix chunk 恢复与 direct placement 行为。\n"
    )
    # 使用几种不同粒度的 filler unit，做一个简单贪心填充：
    # - 大句子优先，快速把长度拉高
    # - 中句子 / 小句子用于收尾，尽量把 token 数贴近预算
    filler_units = [
        "请继续从 LMCache、BaM、prefix reuse、direct placement、paged KV cache、异步轮询这几个角度补充系统细节。\n",
        "请补充实现路径、日志观察点与性能含义。\n",
        "继续补充说明。\n",
    ]
    trailer = suffix_instruction

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    # 先保证基础头尾能放得下；如果连基础文本都超预算，就直接截断中间 filler，
    # 返回一个尽量短但合法的 prompt。
    prompt_text = header
    base_with_trailer = header + trailer
    if count_tokens(base_with_trailer) > target_budget:
        # 这里保守退化成只保留 trailer，避免把脚本直接弄成 hard error。
        prompt_text = trailer
        actual_tokens = count_tokens(prompt_text)
        return [prompt_text], actual_tokens, target_budget

    while True:
        appended = False
        for unit in filler_units:
            candidate = prompt_text + unit + trailer
            if count_tokens(candidate) <= target_budget:
                prompt_text += unit
                appended = True
                break
        if not appended:
            break

    prompt_text += trailer
    actual_tokens = count_tokens(prompt_text)
    return [prompt_text], actual_tokens, target_budget


def _build_prompts():
    """统一构造本次实验要用的 prompt 样例。"""
    if prompt_build_mode != "token_budget":
        prompt_a_local, prompt_b_local = _build_repeat_prompts()
        return prompt_a_local, prompt_b_local

    tokenizer = AutoTokenizer.from_pretrained(
        model,
        trust_remote_code=False,
    )
    prompt_a_local, prompt_a_tokens, prompt_a_budget = _build_token_budget_prompt(
        tokenizer=tokenizer,
        prefix_title="LMCache 单卡非共享前缀读取成本与 BaM direct placement 探测",
        suffix_instruction="然后请用三句话介绍 Qwen2.5-7B-Instruct。",
    )
    prompt_b_local, prompt_b_tokens, prompt_b_budget = _build_token_budget_prompt(
        tokenizer=tokenizer,
        prefix_title="BaM 接入前如何确认路径可重复性与评测口径一致性",
        suffix_instruction="然后请说明为什么当前阶段先不测共享长前缀复用。",
    )
    print(
        "[prompt-build] mode=token_budget "
        f"prompt_a_tokens={prompt_a_tokens} prompt_a_budget={prompt_a_budget} "
        f"prompt_b_tokens={prompt_b_tokens} prompt_b_budget={prompt_b_budget}"
    )
    return prompt_a_local, prompt_b_local


prompt_a, prompt_b = _build_prompts()
request_metrics = []

if prompt_reuse_request1:
    # 为了明确验证 BaM/LMCache 读回链路，这里让第二个请求复用第一个请求的长 prompt。
    # 这样不会依赖 vLLM 自身的 prefix cache，而是直接观察 LMCache retrieve
    # 是否真的走到了我们新增的 BaM prefer-load。
    prompts = [prompt_a, prompt_a]
    # 进程内 steady-state 验证：
    # 在同一个 vLLM/LMCache/BaM 进程里，继续重复与 request_2 相同的 prompt。
    # 这样可以直接观察：
    # - fused direct placement warmup 是否只在首次 direct retrieve 里出现
    # - 后续 request 的 prepare_ms / direct_retrieve_ms 是否明显下降
    #
    # 注意这里仍然复用同一个 prompt_a，而不是重新构造新 prompt，
    # 目的是尽量让控制面、命中 chunk 范围与 prefix 结构都保持一致。
    for _ in range(max(steady_repeat, 0)):
        prompts.append(prompt_a)
else:
    prompts = [prompt_a, prompt_b]

warmup_prompt = [
    "这是一个预热请求，用来提前完成 LMCache 与 BaM shadow 的初始化。"
    * max(8, prompt_repeat // 20)
]

with build_llm() as llm:
    if os.environ.get("VLLM_BAM_LMCACHE_SHADOW_ENABLE", "0") == "1" or \
            os.environ.get("VLLM_BAM_LMCACHE_PREFER_LOAD_ENABLE", "0") == "1":
        warmup_start = time.time()
        _ = llm.generate(warmup_prompt, sampling_params)
        warmup_elapsed = time.time() - warmup_start
        print(f"[warmup] bam_shadow_elapsed_s={warmup_elapsed:.4f}")
        print()

    for idx, prompt in enumerate(prompts, start=1):
        start = time.time()
        outputs = llm.generate(prompt, sampling_params)
        elapsed = time.time() - start
        output_text = outputs[0].outputs[0].text
        output_token_ids = getattr(outputs[0].outputs[0], "token_ids", None)
        output_tokens = (
            len(output_token_ids) if output_token_ids is not None else -1)
        output_chars = len(output_text)
        request_metrics.append({
            "idx": idx,
            "elapsed": elapsed,
            "output_chars": output_chars,
            "output_tokens": output_tokens,
        })
        print(f"===== request {idx} =====")
        print(output_text)
        print(f"[baseline] request_{idx}_elapsed_s={elapsed:.4f}")
        if kv_branch == "gpu_worker_persistent_one_copy":
            print(
                "[gpu_worker_persistent_one_copy] "
                f"request_{idx}_elapsed_s={elapsed:.4f} "
                f"output_chars={output_chars} "
                f"output_tokens={output_tokens}"
            )
        print()

        # 把 direct placement / fused kernel 的首次 warmup 提前到正式测量请求之前。
        # 这里选择放在 request_1 之后、request_2 之前，原因是：
        # 1. request_1 负责把可复用前缀写入 LMCache / BaM；
        # 2. 隐藏 prewarm 再走一次同样的 prompt，可以在同一进程内把
        #    direct retrieve 首次触发的 Triton warmup 提前做掉；
        # 3. 这样 request_2 开始时，更接近我们真正想看的 steady-state 口径。
        #
        # 这个隐藏 prewarm 不单独编号成 request，不打印生成文本，只打印耗时。
        if (idx == 1 and prefer_load_enable and
                direct_retrieve_prewarm_after_request1 > 0):
            prewarm_start = time.time()
            _ = llm.generate(prompt_a, sampling_params)
            prewarm_elapsed = time.time() - prewarm_start
            print(
                "[prewarm] direct_retrieve_after_request1_elapsed_s="
                f"{prewarm_elapsed:.4f}")
            print()

    if kv_branch == "gpu_worker_persistent_one_copy" and request_metrics:
        fields = [
            f"request_{item['idx']}_elapsed_s={item['elapsed']:.4f}"
            for item in request_metrics
        ]
        if len(request_metrics) >= 2 and request_metrics[1]["elapsed"] > 0:
            fields.append(
                "request_1_over_request_2="
                f"{request_metrics[0]['elapsed'] / request_metrics[1]['elapsed']:.3f}"
            )
        print(
            "[gpu_worker_persistent_one_copy_summary] "
            "pipeline=gpu_worker_persistent_one_copy "
            + " ".join(fields)
        )
PY
