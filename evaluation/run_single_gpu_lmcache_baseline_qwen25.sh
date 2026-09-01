#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/xhk/llm-inference/models/Qwen2.5-7B-Instruct}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.60}"
LMCACHE_CHUNK_SIZE_VALUE="${LMCACHE_CHUNK_SIZE_VALUE:-256}"
LMCACHE_MAX_LOCAL_CPU_SIZE_VALUE="${LMCACHE_MAX_LOCAL_CPU_SIZE_VALUE:-5.0}"
PROMPT_REPEAT="${PROMPT_REPEAT:-280}"
MAX_TOKENS="${MAX_TOKENS:-64}"
LOG_FILE="${LOG_FILE:-/tmp/vllm-bam-single-gpu-lmcache-v0-qwen25.log}"
LMCACHE_REPO_PATH="${LMCACHE_REPO_PATH:-/home/xhk/llm-inference/LMCache-v0-torch26}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DTYPE="${DTYPE:-half}"
ENFORCE_EAGER="${ENFORCE_EAGER:-true}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-false}"
LMCACHE_USE_EXPERIMENTAL_VALUE="${LMCACHE_USE_EXPERIMENTAL:-True}"
LMCACHE_LOCAL_CPU_VALUE="${LMCACHE_LOCAL_CPU:-False}"
LMCACHE_LOCAL_DISK_VALUE="${LMCACHE_LOCAL_DISK:-/home/xhk/llm-inference/lmcache_local_disk/}"
LMCACHE_MAX_LOCAL_DISK_SIZE_VALUE="${LMCACHE_MAX_LOCAL_DISK_SIZE:-20}"
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
if [[ -n "${LMCACHE_LOCAL_DISK_VALUE}" ]]; then
  export LMCACHE_LOCAL_DISK="${LMCACHE_LOCAL_DISK_VALUE}"
else
  unset LMCACHE_LOCAL_DISK || true
fi

if [[ -d "${LMCACHE_REPO_PATH}/lmcache" ]]; then
  export PYTHONPATH="${LMCACHE_REPO_PATH}:${PYTHONPATH:-}"
fi

echo "[single-gpu-lmcache-baseline] model=${MODEL_PATH}"
echo "[single-gpu-lmcache-baseline] cuda_device=${CUDA_DEVICE}"
echo "[single-gpu-lmcache-baseline] vllm_use_v1=${VLLM_USE_V1}"
echo "[single-gpu-lmcache-baseline] max_model_len=${MAX_MODEL_LEN}"
echo "[single-gpu-lmcache-baseline] gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
echo "[single-gpu-lmcache-baseline] lmcache_chunk_size=${LMCACHE_CHUNK_SIZE_VALUE}"
echo "[single-gpu-lmcache-baseline] lmcache_max_local_cpu_size=${LMCACHE_MAX_LOCAL_CPU_SIZE_VALUE}"
echo "[single-gpu-lmcache-baseline] lmcache_use_experimental=${LMCACHE_USE_EXPERIMENTAL_VALUE}"
echo "[single-gpu-lmcache-baseline] lmcache_local_cpu=${LMCACHE_LOCAL_CPU_VALUE}"
echo "[single-gpu-lmcache-baseline] lmcache_local_disk=${LMCACHE_LOCAL_DISK_VALUE}"
echo "[single-gpu-lmcache-baseline] lmcache_max_local_disk_size=${LMCACHE_MAX_LOCAL_DISK_SIZE_VALUE}"
echo "[single-gpu-lmcache-baseline] prompt_repeat=${PROMPT_REPEAT}"
echo "[single-gpu-lmcache-baseline] max_tokens=${MAX_TOKENS}"
echo "[single-gpu-lmcache-baseline] log_file=${LOG_FILE}"
echo "[single-gpu-lmcache-baseline] lmcache_repo_path=${LMCACHE_REPO_PATH}"
echo "[single-gpu-lmcache-baseline] python_bin=${PYTHON_BIN}"
echo "[single-gpu-lmcache-baseline] dtype=${DTYPE}"
echo "[single-gpu-lmcache-baseline] enforce_eager=${ENFORCE_EAGER}"
echo "[single-gpu-lmcache-baseline] enable_chunked_prefill=${ENABLE_CHUNKED_PREFILL}"

cd "$(dirname "$0")/.."

MODEL_PATH="${MODEL_PATH}" \
MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
PROMPT_REPEAT="${PROMPT_REPEAT}" \
MAX_TOKENS="${MAX_TOKENS}" \
DTYPE="${DTYPE}" \
ENFORCE_EAGER="${ENFORCE_EAGER}" \
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL}" \
"${PYTHON_BIN}" - <<'PY' 2>&1 | tee "${LOG_FILE}"
import contextlib
import os
import time

from lmcache.integration.vllm.utils import ENGINE_NAME
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

try:
    from lmcache.experimental.cache_engine import LMCacheEngineBuilder
except ImportError:
    from lmcache.v1.cache_engine import LMCacheEngineBuilder

model = os.environ["MODEL_PATH"] if "MODEL_PATH" in os.environ else "/home/xhk/llm-inference/models/Qwen2.5-7B-Instruct"
max_model_len = int(os.environ.get("MAX_MODEL_LEN", "4096"))
gpu_memory_utilization = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.60"))
prompt_repeat = int(os.environ.get("PROMPT_REPEAT", "280"))
max_tokens = int(os.environ.get("MAX_TOKENS", "64"))
dtype = os.environ.get("DTYPE", "half")
enforce_eager = os.environ.get("ENFORCE_EAGER", "true").lower() == "true"
enable_chunked_prefill = os.environ.get("ENABLE_CHUNKED_PREFILL",
                                        "false").lower() == "true"


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


shared_prompt = "请介绍一下 LMCache、KV cache 和单卡 baseline 的作用。" * prompt_repeat
first_prompt = [shared_prompt + "然后请用三句话介绍 Qwen2.5-7B-Instruct。"]
second_prompt = [shared_prompt + "然后请说明原生 LMCache SSD baseline 的读回作用。"]

sampling_params = SamplingParams(
    temperature=0.0,
    top_p=0.95,
    max_tokens=max_tokens,
)

with build_llm() as llm:
    start = time.time()
    outputs1 = llm.generate(first_prompt, sampling_params)
    t1 = time.time() - start
    print("===== first =====")
    print(outputs1[0].outputs[0].text)
    print(f"[baseline] first_request_elapsed_s={t1:.4f}")
    print()

    start = time.time()
    outputs2 = llm.generate(second_prompt, sampling_params)
    t2 = time.time() - start
    print("===== second =====")
    print(outputs2[0].outputs[0].text)
    print(f"[baseline] second_request_elapsed_s={t2:.4f}")
    print()
PY
