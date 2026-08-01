#!/usr/bin/env bash
set -euo pipefail

# 【BaM KVStore 直通调用链】真实 vLLM V0 scheduler/preemption smoke。
#
# 本脚本只打开独立 direct backend，不打开旧 BaM page cache、LMCache 或 GDS。
# num_gpu_blocks_override 用来稳定制造 KV 空间竞争；scheduler 仍负责选择请求、
# 生成 GPU<->storage block mapping，Worker/CacheEngine 再把 mapping 交给 BaM。

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xhk/miniconda3/envs/pytorch-vllm/bin/python}"
BAM_ROOT="${BAM_ROOT:-/home/xhk/llm-inference/BaM_IOStack}"
MODEL="${MODEL:-/home/xhk/model/Qwen3-0.6B}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT_DIR}/evaluation/logs/direct_kvstore_preemption/${TIMESTAMP}"
mkdir -p "${LOG_DIR}"
CONSOLE_LOG="${LOG_DIR}/console.log"

# 当前默认 workload 同时运行两个 best_of=4 请求组，峰值约需 320 blocks。
# smoke 的目标是稳定触发 swap，因此把外部给出的上限视为“最多可用 blocks”，
# 并收紧到 260；两个 prompt 可以完成 prefill，但进入多分支 decode 后会立即
# 竞争新增 block，从而由真实 scheduler 选择 swap victim。
REQUESTED_GPU_BLOCKS="${NUM_GPU_BLOCKS_OVERRIDE:-260}"
if (( REQUESTED_GPU_BLOCKS > 260 )); then
  EFFECTIVE_GPU_BLOCKS=260
else
  EFFECTIVE_GPU_BLOCKS="${REQUESTED_GPU_BLOCKS}"
fi

export PYTHONPATH="${ROOT_DIR}:${BAM_ROOT}/gids_module:${BAM_ROOT}/gids_module/build${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${BAM_ROOT}/bam/build/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_V0_SWAP_TRACE=1
export VLLM_BAM_DIRECT_KVSTORE_ENABLE=1
export VLLM_BAM_SHADOW_ENABLE=0
export VLLM_BAM_SWAPIN_ENABLE=0
export VLLM_BAM_IMPORT_PATH="${BAM_ROOT}/gids_module"
export VLLM_BAM_SSD_LIST="${VLLM_BAM_SSD_LIST:-0}"

echo "[DIRECT_KVSTORE_PREEMPTION] model=${MODEL} gpu_blocks=${EFFECTIVE_GPU_BLOCKS}"
echo "[DIRECT_KVSTORE_PREEMPTION] console_log=${CONSOLE_LOG}"

"${PYTHON_BIN}" "${ROOT_DIR}/evaluation/v0_swap_trace_eval.py" \
  "${MODEL}" \
  --num-prompts "${NUM_PROMPTS:-8}" \
  --prompt-len "${PROMPT_LEN:-2048}" \
  --max-tokens "${MAX_TOKENS:-128}" \
  --temperature "${TEMPERATURE:-0.8}" \
  --best-of "${BEST_OF:-4}" \
  --n 1 \
  --max-model-len "${MAX_MODEL_LEN:-4096}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.16}" \
  --swap-space "${SWAP_SPACE:-4}" \
  --dtype half \
  --tensor-parallel-size 1 \
  --preemption-mode swap \
  --max-num-seqs "${MAX_NUM_SEQS:-8}" \
  --num-gpu-blocks-override "${EFFECTIVE_GPU_BLOCKS}" \
  --enforce-eager \
  --log-dir "${LOG_DIR}" \
  >"${CONSOLE_LOG}" 2>&1

LOG_PATH="$(find "${LOG_DIR}" -maxdepth 1 -type f -name 'v0_swap_trace_*.log' -print -quit)"
if [[ -z "${LOG_PATH}" ]]; then
  echo "[DIRECT_KVSTORE_PREEMPTION] FAIL: trace log was not created" >&2
  exit 1
fi

# 完整闭环缺少任一事件都视为失败，避免“模型能返回但没有真正经过 SSD”的误判。
for pattern in \
  'op=swap_out' \
  'op=swap_in' \
  '[BAM_DIRECT_KVSTORE] op=write phase=done' \
  '[BAM_DIRECT_KVSTORE] op=read phase=done' \
  'Run summary'; do
  if ! grep -Fq "${pattern}" "${LOG_PATH}"; then
    echo "[DIRECT_KVSTORE_PREEMPTION] FAIL: missing ${pattern}" >&2
    exit 1
  fi
done

SWAP_OUT_COUNT="$(grep -Fc '[BAM_DIRECT_KVSTORE] op=write phase=done' "${LOG_PATH}")"
SWAP_IN_COUNT="$(grep -Fc '[BAM_DIRECT_KVSTORE] op=read phase=done' "${LOG_PATH}")"
echo "[DIRECT_KVSTORE_PREEMPTION] log=${LOG_PATH}"
echo "[DIRECT_KVSTORE_PREEMPTION] swap_out=${SWAP_OUT_COUNT} swap_in=${SWAP_IN_COUNT}"
echo "[DIRECT_KVSTORE_PREEMPTION] PASS"
