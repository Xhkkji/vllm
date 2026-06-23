#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/xhk/llm-inference/models/Qwen2.5-7B-Instruct}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.60}"
PROMPT_REPEAT="${PROMPT_REPEAT:-220}"
MAX_TOKENS="${MAX_TOKENS:-64}"
LOG_FILE="${LOG_FILE:-/tmp/vllm-bam-single-gpu-no-prefix-reuse-qwen25.log}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DTYPE="${DTYPE:-half}"
ENFORCE_EAGER="${ENFORCE_EAGER:-true}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-false}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-false}"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export VLLM_USE_V1=0

echo "[single-gpu-no-prefix-reuse] model=${MODEL_PATH}"
echo "[single-gpu-no-prefix-reuse] cuda_device=${CUDA_DEVICE}"
echo "[single-gpu-no-prefix-reuse] vllm_use_v1=${VLLM_USE_V1}"
echo "[single-gpu-no-prefix-reuse] max_model_len=${MAX_MODEL_LEN}"
echo "[single-gpu-no-prefix-reuse] gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
echo "[single-gpu-no-prefix-reuse] prompt_repeat=${PROMPT_REPEAT}"
echo "[single-gpu-no-prefix-reuse] max_tokens=${MAX_TOKENS}"
echo "[single-gpu-no-prefix-reuse] log_file=${LOG_FILE}"
echo "[single-gpu-no-prefix-reuse] python_bin=${PYTHON_BIN}"
echo "[single-gpu-no-prefix-reuse] dtype=${DTYPE}"
echo "[single-gpu-no-prefix-reuse] enforce_eager=${ENFORCE_EAGER}"
echo "[single-gpu-no-prefix-reuse] enable_chunked_prefill=${ENABLE_CHUNKED_PREFILL}"
echo "[single-gpu-no-prefix-reuse] enable_prefix_caching=${ENABLE_PREFIX_CACHING}"

cd "$(dirname "$0")/.."

MODEL_PATH="${MODEL_PATH}" \
MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
PROMPT_REPEAT="${PROMPT_REPEAT}" \
MAX_TOKENS="${MAX_TOKENS}" \
DTYPE="${DTYPE}" \
ENFORCE_EAGER="${ENFORCE_EAGER}" \
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL}" \
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING}" \
"${PYTHON_BIN}" - <<'PY' 2>&1 | tee "${LOG_FILE}"
import os
import time

from vllm import LLM, SamplingParams

model = os.environ["MODEL_PATH"]
max_model_len = int(os.environ.get("MAX_MODEL_LEN", "4096"))
gpu_memory_utilization = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.60"))
prompt_repeat = int(os.environ.get("PROMPT_REPEAT", "280"))
max_tokens = int(os.environ.get("MAX_TOKENS", "64"))
dtype = os.environ.get("DTYPE", "half")
enforce_eager = os.environ.get("ENFORCE_EAGER", "true").lower() == "true"
enable_chunked_prefill = os.environ.get("ENABLE_CHUNKED_PREFILL",
                                        "false").lower() == "true"
enable_prefix_caching = os.environ.get("ENABLE_PREFIX_CACHING",
                                       "false").lower() == "true"

llm = LLM(
    model=model,
    max_model_len=max_model_len,
    gpu_memory_utilization=gpu_memory_utilization,
    dtype=dtype,
    enforce_eager=enforce_eager,
    enable_chunked_prefill=enable_chunked_prefill,
    enable_prefix_caching=enable_prefix_caching,
    trust_remote_code=False,
)

sampling_params = SamplingParams(
    temperature=0.0,
    top_p=0.95,
    max_tokens=max_tokens,
)

shared_a = "请介绍一下单卡原生 baseline、吞吐和时延统计的方法。" * prompt_repeat
shared_b = "请介绍一下 BaM 接入前应先确认哪些稳定性和 correctness 问题。" * prompt_repeat

prompts = [
    [shared_a + "然后请用三句话介绍 Qwen2.5-7B-Instruct。"],
    [shared_b + "然后请说明为什么当前先不测 prefix reuse。"],
]

for idx, prompt in enumerate(prompts, start=1):
    start = time.time()
    outputs = llm.generate(prompt, sampling_params)
    elapsed = time.time() - start
    print(f"===== request {idx} =====")
    print(outputs[0].outputs[0].text)
    print(f"[baseline] request_{idx}_elapsed_s={elapsed:.4f}")
    print()
PY
