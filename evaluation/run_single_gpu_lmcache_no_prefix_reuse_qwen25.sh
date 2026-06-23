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
LOG_ROOT="${LOG_ROOT:-${SCRIPT_DIR}/logs/single_gpu_lmcache_no_prefix_reuse_qwen25}"
RUN_DIR="${RUN_DIR:-${LOG_ROOT}/${TIMESTAMP}}"
LOG_FILE="${LOG_FILE:-${RUN_DIR}/run.log}"
LMCACHE_REPO_PATH="${LMCACHE_REPO_PATH:-/home/xhk/llm-inference/LMCache-v0-torch26}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DTYPE="${DTYPE:-half}"
ENFORCE_EAGER="${ENFORCE_EAGER:-true}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-false}"
LMCACHE_USE_EXPERIMENTAL_VALUE="${LMCACHE_USE_EXPERIMENTAL:-True}"
LMCACHE_LOCAL_CPU_VALUE="${LMCACHE_LOCAL_CPU:-False}"
LMCACHE_LOCAL_DISK_VALUE="${LMCACHE_LOCAL_DISK:-/home/xhk/llm-inference/lmcache_local_disk/}"
LMCACHE_MAX_LOCAL_DISK_SIZE_VALUE="${LMCACHE_MAX_LOCAL_DISK_SIZE:-20}"
VLLM_BAM_LMCACHE_SHADOW_ENABLE_VALUE="${VLLM_BAM_LMCACHE_SHADOW_ENABLE:-0}"
VLLM_BAM_LMCACHE_PREFER_LOAD_ENABLE_VALUE="${VLLM_BAM_LMCACHE_PREFER_LOAD_ENABLE:-0}"
VLLM_BAM_LMCACHE_SHADOW_CHUNKS_VALUE="${VLLM_BAM_LMCACHE_SHADOW_CHUNKS:-1024}"
VLLM_BAM_IMPORT_PATH_VALUE="${VLLM_BAM_IMPORT_PATH:-/home/xhk/llm-inference/BaM_IOStack/gids_module}"
BAM_LIB_DIR_VALUE="${BAM_LIB_DIR:-/home/xhk/llm-inference/BaM_IOStack/bam/build/lib}"
VLLM_BAM_CACHE_SIZE_MB_VALUE="${VLLM_BAM_CACHE_SIZE_MB:-64}"
VLLM_BAM_NUM_SSD_VALUE="${VLLM_BAM_NUM_SSD:-1}"
VLLM_BAM_SSD_LIST_VALUE="${VLLM_BAM_SSD_LIST:-0}"
VLLM_BAM_CTRL_IDX_VALUE="${VLLM_BAM_CTRL_IDX:-0}"
GIDS_FORCE_SYNC_READ_VALUE="${GIDS_FORCE_SYNC_READ:-1}"
BAM_PREFLIGHT_VALUE="${BAM_PREFLIGHT:-0}"

# BaM 读写路径都可能需要更高权限。只有显式打开 BaM 相关路径时才自动提权，
# 普通 LMCache SSD baseline 不受影响。
if [[ ( "${VLLM_BAM_LMCACHE_SHADOW_ENABLE_VALUE}" == "1" || \
        "${VLLM_BAM_LMCACHE_PREFER_LOAD_ENABLE_VALUE}" == "1" ) && \
        "${EUID}" -ne 0 ]]; then
  exec sudo env \
    "MODEL_PATH=${MODEL_PATH}" \
    "CUDA_DEVICE=${CUDA_DEVICE}" \
    "MAX_MODEL_LEN=${MAX_MODEL_LEN}" \
    "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}" \
    "LMCACHE_CHUNK_SIZE_VALUE=${LMCACHE_CHUNK_SIZE_VALUE}" \
    "LMCACHE_MAX_LOCAL_CPU_SIZE_VALUE=${LMCACHE_MAX_LOCAL_CPU_SIZE_VALUE}" \
    "PROMPT_REPEAT=${PROMPT_REPEAT}" \
    "MAX_TOKENS=${MAX_TOKENS}" \
    "LOG_FILE=${LOG_FILE}" \
    "LMCACHE_REPO_PATH=${LMCACHE_REPO_PATH}" \
    "PYTHON_BIN=${PYTHON_BIN}" \
    "DTYPE=${DTYPE}" \
    "ENFORCE_EAGER=${ENFORCE_EAGER}" \
    "ENABLE_CHUNKED_PREFILL=${ENABLE_CHUNKED_PREFILL}" \
    "LMCACHE_USE_EXPERIMENTAL=${LMCACHE_USE_EXPERIMENTAL_VALUE}" \
    "LMCACHE_LOCAL_CPU=${LMCACHE_LOCAL_CPU_VALUE}" \
    "LMCACHE_LOCAL_DISK=${LMCACHE_LOCAL_DISK_VALUE}" \
    "LMCACHE_MAX_LOCAL_DISK_SIZE=${LMCACHE_MAX_LOCAL_DISK_SIZE_VALUE}" \
    "VLLM_BAM_LMCACHE_SHADOW_ENABLE=${VLLM_BAM_LMCACHE_SHADOW_ENABLE_VALUE}" \
    "VLLM_BAM_LMCACHE_PREFER_LOAD_ENABLE=${VLLM_BAM_LMCACHE_PREFER_LOAD_ENABLE_VALUE}" \
    "VLLM_BAM_LMCACHE_SHADOW_CHUNKS=${VLLM_BAM_LMCACHE_SHADOW_CHUNKS_VALUE}" \
    "VLLM_BAM_IMPORT_PATH=${VLLM_BAM_IMPORT_PATH_VALUE}" \
    "VLLM_BAM_CACHE_SIZE_MB=${VLLM_BAM_CACHE_SIZE_MB_VALUE}" \
    "VLLM_BAM_NUM_SSD=${VLLM_BAM_NUM_SSD_VALUE}" \
    "VLLM_BAM_SSD_LIST=${VLLM_BAM_SSD_LIST_VALUE}" \
    "VLLM_BAM_CTRL_IDX=${VLLM_BAM_CTRL_IDX_VALUE}" \
    "GIDS_FORCE_SYNC_READ=${GIDS_FORCE_SYNC_READ_VALUE}" \
    "BAM_PREFLIGHT=${BAM_PREFLIGHT_VALUE}" \
    "BAM_LIB_DIR=${BAM_LIB_DIR_VALUE}" \
    "LD_LIBRARY_PATH=${BAM_LIB_DIR_VALUE}:${LD_LIBRARY_PATH:-}" \
    "PYTHONPATH=${PYTHONPATH:-}" \
    bash "$0" "$@"
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
export VLLM_BAM_IMPORT_PATH="${VLLM_BAM_IMPORT_PATH_VALUE}"
export VLLM_BAM_CACHE_SIZE_MB="${VLLM_BAM_CACHE_SIZE_MB_VALUE}"
export VLLM_BAM_NUM_SSD="${VLLM_BAM_NUM_SSD_VALUE}"
export VLLM_BAM_SSD_LIST="${VLLM_BAM_SSD_LIST_VALUE}"
export VLLM_BAM_CTRL_IDX="${VLLM_BAM_CTRL_IDX_VALUE}"
export GIDS_FORCE_SYNC_READ="${GIDS_FORCE_SYNC_READ_VALUE}"
export LD_LIBRARY_PATH="${BAM_LIB_DIR_VALUE}:${LD_LIBRARY_PATH:-}"

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
    cache_size_mb=int(os.environ.get("VLLM_BAM_CACHE_SIZE_MB", "64")),
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
echo "[single-gpu-lmcache-no-prefix-reuse] vllm_bam_import_path=${VLLM_BAM_IMPORT_PATH_VALUE}"
echo "[single-gpu-lmcache-no-prefix-reuse] vllm_bam_cache_size_mb=${VLLM_BAM_CACHE_SIZE_MB_VALUE}"
echo "[single-gpu-lmcache-no-prefix-reuse] vllm_bam_num_ssd=${VLLM_BAM_NUM_SSD_VALUE}"
echo "[single-gpu-lmcache-no-prefix-reuse] vllm_bam_ssd_list=${VLLM_BAM_SSD_LIST_VALUE}"
echo "[single-gpu-lmcache-no-prefix-reuse] vllm_bam_ctrl_idx=${VLLM_BAM_CTRL_IDX_VALUE}"
echo "[single-gpu-lmcache-no-prefix-reuse] gids_force_sync_read=${GIDS_FORCE_SYNC_READ_VALUE}"
echo "[single-gpu-lmcache-no-prefix-reuse] bam_preflight=${BAM_PREFLIGHT_VALUE}"
echo "[single-gpu-lmcache-no-prefix-reuse] bam_lib_dir=${BAM_LIB_DIR_VALUE}"
echo "[single-gpu-lmcache-no-prefix-reuse] prompt_repeat=${PROMPT_REPEAT}"
echo "[single-gpu-lmcache-no-prefix-reuse] max_tokens=${MAX_TOKENS}"
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

model = os.environ["MODEL_PATH"]
max_model_len = int(os.environ.get("MAX_MODEL_LEN", "4096"))
gpu_memory_utilization = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.60"))
prompt_repeat = int(os.environ.get("PROMPT_REPEAT", "100"))
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


sampling_params = SamplingParams(
    temperature=0.0,
    top_p=0.95,
    max_tokens=max_tokens,
)

shared_a = "介绍 LMCache 单卡基线的非复用开销。" * prompt_repeat
shared_b = "介绍 BaM 接入前如何确认路径可重复性。" * prompt_repeat

prefer_load_enable = os.environ.get("VLLM_BAM_LMCACHE_PREFER_LOAD_ENABLE",
                                     "0") == "1"

prompt_a = [shared_a + "然后请用三句话介绍 Qwen2.5-7B-Instruct。"]
prompt_b = [shared_b + "然后请说明为什么当前阶段先不测共享长前缀复用。"]

if prefer_load_enable:
    # 为了明确验证 BaM/LMCache 读回链路，这里让第二个请求复用第一个请求的长 prompt。
    # 这样不会依赖 vLLM 自身的 prefix cache，而是直接观察 LMCache retrieve
    # 是否真的走到了我们新增的 BaM prefer-load。
    prompts = [prompt_a, prompt_a]
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
        print(f"===== request {idx} =====")
        print(outputs[0].outputs[0].text)
        print(f"[baseline] request_{idx}_elapsed_s={elapsed:.4f}")
        print()
PY
