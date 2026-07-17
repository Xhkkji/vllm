#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 这个脚本固定到当前输出正确的 fast 路径：
#
#   gpu_worker_persistent_materialized
#
# 它和另外两条保留链路的关系是：
#
# - rowctx_baseline
#   直接运行 `run_single_gpu_lmcache_no_prefix_reuse_qwen25.sh`，不传
#   `VLLM_BAM_KV_EXECUTOR=gpu_worker`。
# - gpu_worker_persistent_materialized
#   运行本脚本，也是当前默认回归/性能分析口径。
# - gpu_worker_persistent_one_copy
#   在本脚本前显式覆盖：
#     VLLM_BAM_DIRECT_PLACEMENT_RUNTIME_ONE_COPY=1
#     VLLM_BAM_DIRECT_PLACEMENT_REQUIRE_RUNTIME_ONE_COPY=1
#
# 设计原则：
# 1. 不复制主启动脚本的全部默认参数，避免两份默认值长期漂移；
# 2. 这里只传“与默认脚本不一样”的参数，后续调参直接改这里即可；
# 3. 真正的实验逻辑、日志目录、sudo/root 透传等，仍然复用主脚本。
#
# 当前这层只覆盖：
# - VLLM_BAM_KV_EXECUTOR=gpu_worker
#   让 KV fast path 走 gpu_worker executor，而不是默认 rowctx。
# - GIDS_KV_GPU_WORKER_RUNTIME_ENABLE=1
#   打开 GPU-visible runtime slot，允许 host 只做轻量 submit/observe。
# - GIDS_KV_GPU_WORKER_PERSISTENT_ENABLE=1
#   打开 persistent service CTA，让 GPU 后台持续负责轮询/推进。
# - VLLM_BAM_DIRECT_PLACEMENT_RUNTIME_ONE_COPY=0
#   默认先关闭“后台 service 直接写最终 paged KV cache”的实验写端。
#   最近端到端乱码已经说明：这条 one-copy 写端即使局部 prefix verify 能过，
#   仍可能污染后续 decode 依赖的 KV cache 区域。当前默认收束为：
#     GPU persistent service 负责 poll/read/pages staging
#     host finalize 复用 rowctx 已验证正确的 BaMDirectKVPlacer 写 cache
#   这样仍能测试 GPU 后台轮询链路，同时把最终写入语义拉回正确基线。
# - VLLM_BAM_DIRECT_PLACEMENT_REQUIRE_RUNTIME_ONE_COPY=0
#   默认不强制 one-copy，允许 persistent read-service + host proven placement
#   这条 correctness-first 主线跑通。需要继续专门压测 one-copy 时，可在命令前
#   显式覆盖为 1。
# - VLLM_BAM_RUNTIME_METADATA_ATTACHMENT_ENABLE
#   默认不强制设置，让 adapter 按 one-copy 是否启用自动推导。当前默认不走
#   one-copy，因此也不需要额外挂 runtime metadata attachment。
# - VLLM_BAM_XFORMERS_PREFIX_FALLBACK_PROFILE=1
#   打开 prefix fallback 细粒度日志，便于确认 attention 端仍从 vLLM paged KV
#   cache 消费 prefix，而不是误走旧的 dense-prefix 调试旁路。
# - VLLM_BAM_DIRECT_PLACEMENT_VERIFY_OFFICIAL_WRITE=0
#   official write repair 写后读校验默认关闭；需要定位 paged KV cache 写入
#   正确性时，可在命令前覆盖为 1，或临时改这里的默认值。
# - GIDS_KV_DEBUG=1
#   默认打开 KV 路径调试日志，方便持续核对是否走到最新的 runtime/persistent
#   主线，以及快速定位卡点。
#
# 说明：
# - 主脚本中已经默认打开 shadow / prefer-load / KV fast path / direct placement /
#   prewarm / 1GB cache，因此这里不再重复传这些值。
# - one-copy 默认关闭，是为了把日常回归固定在当前输出正确的 materialized
#   fast path；需要压测 one-copy 时，在命令前显式打开下面两个覆盖项即可。
# - 需要回到 one-copy 实验时，在命令前显式传：
#   VLLM_BAM_DIRECT_PLACEMENT_RUNTIME_ONE_COPY=1
#   VLLM_BAM_DIRECT_PLACEMENT_REQUIRE_RUNTIME_ONE_COPY=1
# - 如需新增调参项，优先只在本脚本补充“覆盖项”，不要回头改主脚本默认值。

VLLM_BAM_KV_EXECUTOR="${VLLM_BAM_KV_EXECUTOR:-gpu_worker}" \
GIDS_KV_GPU_WORKER_RUNTIME_ENABLE="${GIDS_KV_GPU_WORKER_RUNTIME_ENABLE:-1}" \
GIDS_KV_GPU_WORKER_PERSISTENT_ENABLE="${GIDS_KV_GPU_WORKER_PERSISTENT_ENABLE:-1}" \
VLLM_BAM_DIRECT_PLACEMENT_RUNTIME_ONE_COPY="${VLLM_BAM_DIRECT_PLACEMENT_RUNTIME_ONE_COPY:-0}" \
VLLM_BAM_DIRECT_PLACEMENT_REQUIRE_RUNTIME_ONE_COPY="${VLLM_BAM_DIRECT_PLACEMENT_REQUIRE_RUNTIME_ONE_COPY:-0}" \
VLLM_BAM_XFORMERS_PREFIX_FALLBACK_PROFILE="${VLLM_BAM_XFORMERS_PREFIX_FALLBACK_PROFILE:-1}" \
VLLM_BAM_DIRECT_PLACEMENT_VERIFY_OFFICIAL_WRITE="${VLLM_BAM_DIRECT_PLACEMENT_VERIFY_OFFICIAL_WRITE:-0}" \
GIDS_KV_DEBUG="${GIDS_KV_DEBUG:-1}" \
bash "${SCRIPT_DIR}/run_single_gpu_lmcache_no_prefix_reuse_qwen25.sh"
