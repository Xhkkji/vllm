# SPDX-License-Identifier: Apache-2.0

import hashlib
import os
import sys
import tempfile
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    VLLM_HOST_IP: str = ""
    VLLM_PORT: Optional[int] = None
    VLLM_RPC_BASE_PATH: str = tempfile.gettempdir()
    VLLM_USE_MODELSCOPE: bool = False
    VLLM_RINGBUFFER_WARNING_INTERVAL: int = 60
    VLLM_NCCL_SO_PATH: Optional[str] = None
    LD_LIBRARY_PATH: Optional[str] = None
    VLLM_USE_TRITON_FLASH_ATTN: bool = False
    VLLM_FLASH_ATTN_VERSION: Optional[int] = None
    LOCAL_RANK: int = 0
    CUDA_VISIBLE_DEVICES: Optional[str] = None
    VLLM_ENGINE_ITERATION_TIMEOUT_S: int = 60
    VLLM_API_KEY: Optional[str] = None
    S3_ACCESS_KEY_ID: Optional[str] = None
    S3_SECRET_ACCESS_KEY: Optional[str] = None
    S3_ENDPOINT_URL: Optional[str] = None
    VLLM_MODEL_REDIRECT_PATH: Optional[str] = None
    VLLM_CACHE_ROOT: str = os.path.expanduser("~/.cache/vllm")
    VLLM_CONFIG_ROOT: str = os.path.expanduser("~/.config/vllm")
    VLLM_USAGE_STATS_SERVER: str = "https://stats.vllm.ai"
    VLLM_NO_USAGE_STATS: bool = False
    VLLM_DO_NOT_TRACK: bool = False
    VLLM_USAGE_SOURCE: str = ""
    VLLM_CONFIGURE_LOGGING: int = 1
    VLLM_LOGGING_LEVEL: str = "INFO"
    VLLM_LOGGING_PREFIX: str = ""
    VLLM_LOGGING_CONFIG_PATH: Optional[str] = None
    VLLM_LOGITS_PROCESSOR_THREADS: Optional[int] = None
    VLLM_TRACE_FUNCTION: int = 0
    VLLM_ATTENTION_BACKEND: Optional[str] = None
    VLLM_USE_FLASHINFER_SAMPLER: Optional[bool] = None
    VLLM_FLASHINFER_FORCE_TENSOR_CORES: bool = False
    VLLM_PP_LAYER_PARTITION: Optional[str] = None
    VLLM_CPU_KVCACHE_SPACE: int = 0
    VLLM_V0_SWAP_TRACE: bool = False
    VLLM_LMCACHE_SEND_DECODE_KV: bool = False
    VLLM_BAM_SHADOW_ENABLE: bool = False
    VLLM_BAM_SWAPIN_ENABLE: bool = False
    VLLM_BAM_SWAPIN_VERIFY: bool = False
    VLLM_BAM_SWAPIN_VERIFY_BLOCKS: int = 0
    VLLM_BAM_LMCACHE_SHADOW_ENABLE: bool = False
    VLLM_BAM_LMCACHE_PREFER_LOAD_ENABLE: bool = False
    VLLM_BAM_LMCACHE_SHADOW_CHUNKS: int = 1024
    VLLM_BAM_LMCACHE_READ_MODE: str = "sync"
    VLLM_BAM_LMCACHE_BASE_ROW_OFFSET: int = 0
    VLLM_BAM_KV_FAST_PATH: bool = False
    VLLM_BAM_GPU_INITIATED_PREFETCH: bool = False
    VLLM_BAM_DIRECT_KVSTORE_ENABLE: bool = False
    VLLM_BAM_DIRECT_READ_MODE: str = "direct"
    VLLM_BAM_SYNC_CACHE_SIZE_MB: int = 64
    VLLM_BAM_MDS_ENABLE: bool = False
    VLLM_BAM_MDS_IOSTACK_ROOT: Optional[str] = None
    VLLM_BAM_MDS_CONTROL_DIR: Optional[str] = None
    VLLM_BAM_MDS_CUDA_IPC_LIBRARY: Optional[str] = None
    VLLM_BAM_MDS_TORCH_BRIDGE_DIR: Optional[str] = None
    VLLM_BAM_MDS_TIMEOUT_SECONDS: float = 120.0
    VLLM_BAM_MDS_MAX_IN_FLIGHT: int = 4
    VLLM_BAM_MDS_SERVICE_LIFETIME: str = "resident"
    VLLM_BAM_MDS_IDLE_STOP_DELAY_MS: int = 0
    VLLM_BAM_MDS_PREFIX_ENABLE: bool = False
    VLLM_BAM_MDS_HIERARCHICAL_IO_ENABLE: bool = False
    VLLM_BAM_MDS_HIERARCHICAL_NUM_LAYERS: int = 0
    VLLM_BAM_MDS_HIERARCHICAL_WINDOW_LAYERS: int = 0
    VLLM_BAM_MDS_HIERARCHICAL_LAYER_BARRIER: bool = False
    VLLM_BAM_MDS_HIERARCHICAL_ROLLING_ENABLE: bool = False
    VLLM_BAM_MDS_HIERARCHICAL_LEAD_WINDOWS: int = 1
    VLLM_BAM_MDS_HIERARCHICAL_MAX_LEAD_WINDOWS: int = 1
    VLLM_BAM_MDS_HIERARCHICAL_TARGET_SLACK_MS: float = 0.0
    VLLM_BAM_MDS_PREFETCH_BLOCK_SELECTOR: str = "dense"
    VLLM_BAM_MDS_PREFETCH_BLOCK_COUNT: int = 0
    VLLM_BAM_MDS_PREFETCH_BLOCK_STRIDE: int = 1
    VLLM_BAM_DIRECT_PLACEMENT: bool = False
    VLLM_BAM_DIRECT_PLACEMENT_IMPL: str = "lmcache"
    VLLM_BAM_DIRECT_PLACEMENT_RUNTIME_ONE_COPY: bool = False
    VLLM_BAM_DIRECT_PLACEMENT_REQUIRE_RUNTIME_ONE_COPY: bool = False
    VLLM_BAM_DIRECT_PLACEMENT_DEFER_RUNTIME: bool = False
    VLLM_BAM_DIRECT_PLACEMENT_DEFER_MIN_POLLS: int = 0
    VLLM_BAM_RUNTIME_IDLE_STOP_SECONDS: float = 0.0
    VLLM_BAM_XFORMERS_PREFIX_FALLBACK_PROFILE: bool = False
    VLLM_BAM_XFORMERS_PREFIX_BACKEND: str = "auto"
    VLLM_BAM_XFORMERS_QUERY_BACKEND: str = "auto"
    VLLM_BAM_XFORMERS_VERIFY_PREFIX_GATHER: bool = False
    VLLM_BAM_XFORMERS_VERIFY_ATTENTION_OUTPUT: bool = False
    VLLM_BAM_XFORMERS_VERIFY_ATTENTION_OUTPUT_FULL: bool = False
    VLLM_BAM_XFORMERS_VERIFY_ATTENTION_OUTPUT_FULL_LAYER: int = 0
    VLLM_BAM_LOGITS_SEMANTIC_DEBUG: bool = False
    VLLM_BAM_IMPORT_PATH: Optional[str] = None
    VLLM_BAM_CACHE_SIZE_MB: int = 512
    VLLM_BAM_NUM_SSD: int = 1
    VLLM_BAM_SSD_LIST: Optional[str] = None
    VLLM_BAM_CTRL_IDX: int = 0
    VLLM_GDS_LMCACHE_SHADOW_ENABLE: bool = False
    VLLM_GDS_LMCACHE_PREFER_LOAD_ENABLE: bool = False
    VLLM_GDS_LMCACHE_PATH: str = "/tmp/vllm-bam-lmcache-gds"
    VLLM_GDS_LMCACHE_USE_GDS: bool = True
    VLLM_GDS_LMCACHE_USE_DIRECT_IO: bool = True
    VLLM_GDS_LMCACHE_DEVICE: Optional[str] = None
    VLLM_GDS_LMCACHE_FMT: str = "KV_2LTD"
    VLLM_GDS_LMCACHE_USE_REGISTERED_BUFFER: bool = False
    VLLM_GDS_LMCACHE_REGISTERED_BUFFER_MB: int = 0
    VLLM_CPU_OMP_THREADS_BIND: str = ""
    VLLM_CPU_MOE_PREPACK: bool = True
    VLLM_XLA_CACHE_PATH: str = os.path.join(VLLM_CACHE_ROOT, "xla_cache")
    VLLM_XLA_CHECK_RECOMPILATION: bool = False
    VLLM_FUSED_MOE_CHUNK_SIZE: int = 64 * 1024
    VLLM_USE_RAY_SPMD_WORKER: bool = False
    VLLM_USE_RAY_COMPILED_DAG: bool = False
    VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE: str = "auto"
    VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM: bool = False
    VLLM_WORKER_MULTIPROC_METHOD: str = "fork"
    VLLM_ASSETS_CACHE: str = os.path.join(VLLM_CACHE_ROOT, "assets")
    VLLM_IMAGE_FETCH_TIMEOUT: int = 5
    VLLM_VIDEO_FETCH_TIMEOUT: int = 30
    VLLM_AUDIO_FETCH_TIMEOUT: int = 10
    VLLM_MM_INPUT_CACHE_GIB: int = 8
    VLLM_TARGET_DEVICE: str = "cuda"
    MAX_JOBS: Optional[str] = None
    NVCC_THREADS: Optional[str] = None
    VLLM_USE_PRECOMPILED: bool = False
    VLLM_TEST_USE_PRECOMPILED_NIGHTLY_WHEEL: bool = False
    VLLM_NO_DEPRECATION_WARNING: bool = False
    VLLM_KEEP_ALIVE_ON_ENGINE_DEATH: bool = False
    CMAKE_BUILD_TYPE: Optional[str] = None
    VERBOSE: bool = False
    VLLM_ALLOW_LONG_MAX_MODEL_LEN: bool = False
    VLLM_RPC_TIMEOUT: int = 10000  # ms
    VLLM_PLUGINS: Optional[list[str]] = None
    VLLM_TORCH_PROFILER_DIR: Optional[str] = None
    VLLM_USE_TRITON_AWQ: bool = False
    VLLM_ALLOW_RUNTIME_LORA_UPDATING: bool = False
    VLLM_SKIP_P2P_CHECK: bool = False
    VLLM_DISABLED_KERNELS: list[str] = []
    VLLM_USE_V1: bool = True
    VLLM_ROCM_USE_AITER: bool = False
    VLLM_ROCM_USE_AITER_PAGED_ATTN: bool = False
    VLLM_ROCM_USE_AITER_LINEAR: bool = True
    VLLM_ROCM_USE_AITER_MOE: bool = True
    VLLM_ROCM_USE_AITER_RMSNORM: bool = True
    VLLM_ROCM_USE_AITER_MLA: bool = True
    VLLM_ROCM_USE_SKINNY_GEMM: bool = True
    VLLM_ROCM_FP8_PADDING: bool = True
    VLLM_ROCM_MOE_PADDING: bool = True
    VLLM_ROCM_CUSTOM_PAGED_ATTN: bool = True
    VLLM_ENABLE_V1_MULTIPROCESSING: bool = True
    VLLM_LOG_BATCHSIZE_INTERVAL: float = -1
    VLLM_DISABLE_COMPILE_CACHE: bool = False
    Q_SCALE_CONSTANT: int = 200
    K_SCALE_CONSTANT: int = 200
    V_SCALE_CONSTANT: int = 100
    VLLM_SERVER_DEV_MODE: bool = False
    VLLM_V1_OUTPUT_PROC_CHUNK_SIZE: int = 128
    VLLM_MLA_DISABLE: bool = False
    VLLM_ENABLE_MOE_ALIGN_BLOCK_SIZE_TRITON: bool = False
    VLLM_RAY_PER_WORKER_GPUS: float = 1.0
    VLLM_RAY_BUNDLE_INDICES: str = ""
    VLLM_CUDART_SO_PATH: Optional[str] = None
    VLLM_USE_HPU_CONTIGUOUS_CACHE_FETCH: bool = True
    VLLM_HPU_USE_DELAYED_SAMPLING: bool = False
    VLLM_DP_RANK: int = 0
    VLLM_DP_RANK_LOCAL: int = -1
    VLLM_DP_SIZE: int = 1
    VLLM_DP_MASTER_IP: str = ""
    VLLM_DP_MASTER_PORT: int = 0
    VLLM_MARLIN_USE_ATOMIC_ADD: bool = False
    VLLM_V0_USE_OUTLINES_CACHE: bool = False
    VLLM_TPU_BUCKET_PADDING_GAP: int = 0
    VLLM_USE_DEEP_GEMM: bool = False
    VLLM_XGRAMMAR_CACHE_MB: int = 0
    VLLM_MSGPACK_ZERO_COPY_THRESHOLD: int = 256


def get_default_cache_root():
    return os.getenv(
        "XDG_CACHE_HOME",
        os.path.join(os.path.expanduser("~"), ".cache"),
    )


def get_default_config_root():
    return os.getenv(
        "XDG_CONFIG_HOME",
        os.path.join(os.path.expanduser("~"), ".config"),
    )


def maybe_convert_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    return int(value)


# The begin-* and end* here are used by the documentation generator
# to extract the used env vars.

# begin-env-vars-definition

environment_variables: dict[str, Callable[[], Any]] = {

    # ================== Installation Time Env Vars ==================

    # Target device of vLLM, supporting [cuda (by default),
    # rocm, neuron, cpu]
    "VLLM_TARGET_DEVICE":
    lambda: os.getenv("VLLM_TARGET_DEVICE", "cuda"),

    # Maximum number of compilation jobs to run in parallel.
    # By default this is the number of CPUs
    "MAX_JOBS":
    lambda: os.getenv("MAX_JOBS", None),

    # Number of threads to use for nvcc
    # By default this is 1.
    # If set, `MAX_JOBS` will be reduced to avoid oversubscribing the CPU.
    "NVCC_THREADS":
    lambda: os.getenv("NVCC_THREADS", None),

    # If set, vllm will use precompiled binaries (*.so)
    "VLLM_USE_PRECOMPILED":
    lambda: bool(os.environ.get("VLLM_USE_PRECOMPILED")) or bool(
        os.environ.get("VLLM_PRECOMPILED_WHEEL_LOCATION")),

    # Whether to force using nightly wheel in python build.
    # This is used for testing the nightly wheel in python build.
    "VLLM_TEST_USE_PRECOMPILED_NIGHTLY_WHEEL":
    lambda: bool(int(os.getenv("VLLM_TEST_USE_PRECOMPILED_NIGHTLY_WHEEL", "0"))
                 ),

    # CMake build type
    # If not set, defaults to "Debug" or "RelWithDebInfo"
    # Available options: "Debug", "Release", "RelWithDebInfo"
    "CMAKE_BUILD_TYPE":
    lambda: os.getenv("CMAKE_BUILD_TYPE"),

    # If set, vllm will print verbose logs during installation
    "VERBOSE":
    lambda: bool(int(os.getenv('VERBOSE', '0'))),

    # Root directory for vLLM configuration files
    # Defaults to `~/.config/vllm` unless `XDG_CONFIG_HOME` is set
    # Note that this not only affects how vllm finds its configuration files
    # during runtime, but also affects how vllm installs its configuration
    # files during **installation**.
    "VLLM_CONFIG_ROOT":
    lambda: os.path.expanduser(
        os.getenv(
            "VLLM_CONFIG_ROOT",
            os.path.join(get_default_config_root(), "vllm"),
        )),

    # ================== Runtime Env Vars ==================

    # Root directory for vLLM cache files
    # Defaults to `~/.cache/vllm` unless `XDG_CACHE_HOME` is set
    "VLLM_CACHE_ROOT":
    lambda: os.path.expanduser(
        os.getenv(
            "VLLM_CACHE_ROOT",
            os.path.join(get_default_cache_root(), "vllm"),
        )),

    # used in distributed environment to determine the ip address
    # of the current node, when the node has multiple network interfaces.
    # If you are using multi-node inference, you should set this differently
    # on each node.
    'VLLM_HOST_IP':
    lambda: os.getenv('VLLM_HOST_IP', ""),

    # used in distributed environment to manually set the communication port
    # Note: if VLLM_PORT is set, and some code asks for multiple ports, the
    # VLLM_PORT will be used as the first port, and the rest will be generated
    # by incrementing the VLLM_PORT value.
    # '0' is used to make mypy happy
    'VLLM_PORT':
    lambda: int(os.getenv('VLLM_PORT', '0'))
    if 'VLLM_PORT' in os.environ else None,

    # path used for ipc when the frontend api server is running in
    # multi-processing mode to communicate with the backend engine process.
    'VLLM_RPC_BASE_PATH':
    lambda: os.getenv('VLLM_RPC_BASE_PATH', tempfile.gettempdir()),

    # If true, will load models from ModelScope instead of Hugging Face Hub.
    # note that the value is true or false, not numbers
    "VLLM_USE_MODELSCOPE":
    lambda: os.environ.get("VLLM_USE_MODELSCOPE", "False").lower() == "true",

    # Interval in seconds to log a warning message when the ring buffer is full
    "VLLM_RINGBUFFER_WARNING_INTERVAL":
    lambda: int(os.environ.get("VLLM_RINGBUFFER_WARNING_INTERVAL", "60")),

    # path to cudatoolkit home directory, under which should be bin, include,
    # and lib directories.
    "CUDA_HOME":
    lambda: os.environ.get("CUDA_HOME", None),

    # Path to the NCCL library file. It is needed because nccl>=2.19 brought
    # by PyTorch contains a bug: https://github.com/NVIDIA/nccl/issues/1234
    "VLLM_NCCL_SO_PATH":
    lambda: os.environ.get("VLLM_NCCL_SO_PATH", None),

    # when `VLLM_NCCL_SO_PATH` is not set, vllm will try to find the nccl
    # library file in the locations specified by `LD_LIBRARY_PATH`
    "LD_LIBRARY_PATH":
    lambda: os.environ.get("LD_LIBRARY_PATH", None),

    # flag to control if vllm should use triton flash attention
    "VLLM_USE_TRITON_FLASH_ATTN":
    lambda: (os.environ.get("VLLM_USE_TRITON_FLASH_ATTN", "True").lower() in
             ("true", "1")),

    # Force vllm to use a specific flash-attention version (2 or 3), only valid
    # when using the flash-attention backend.
    "VLLM_FLASH_ATTN_VERSION":
    lambda: maybe_convert_int(os.environ.get("VLLM_FLASH_ATTN_VERSION", None)),

    # Internal flag to enable Dynamo fullgraph capture
    "VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE":
    lambda: bool(
        os.environ.get("VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE", "1") != "0"),

    # local rank of the process in the distributed setting, used to determine
    # the GPU device id
    "LOCAL_RANK":
    lambda: int(os.environ.get("LOCAL_RANK", "0")),

    # used to control the visible devices in the distributed setting
    "CUDA_VISIBLE_DEVICES":
    lambda: os.environ.get("CUDA_VISIBLE_DEVICES", None),

    # timeout for each iteration in the engine
    "VLLM_ENGINE_ITERATION_TIMEOUT_S":
    lambda: int(os.environ.get("VLLM_ENGINE_ITERATION_TIMEOUT_S", "60")),

    # API key for vLLM API server
    "VLLM_API_KEY":
    lambda: os.environ.get("VLLM_API_KEY", None),

    # Whether to log responses from API Server for debugging
    "VLLM_DEBUG_LOG_API_SERVER_RESPONSE":
    lambda: os.environ.get("VLLM_DEBUG_LOG_API_SERVER_RESPONSE", "False").
    lower() == "true",

    # S3 access information, used for tensorizer to load model from S3
    "S3_ACCESS_KEY_ID":
    lambda: os.environ.get("S3_ACCESS_KEY_ID", None),
    "S3_SECRET_ACCESS_KEY":
    lambda: os.environ.get("S3_SECRET_ACCESS_KEY", None),
    "S3_ENDPOINT_URL":
    lambda: os.environ.get("S3_ENDPOINT_URL", None),

    # Usage stats collection
    "VLLM_USAGE_STATS_SERVER":
    lambda: os.environ.get("VLLM_USAGE_STATS_SERVER", "https://stats.vllm.ai"),
    "VLLM_NO_USAGE_STATS":
    lambda: os.environ.get("VLLM_NO_USAGE_STATS", "0") == "1",
    "VLLM_DO_NOT_TRACK":
    lambda: (os.environ.get("VLLM_DO_NOT_TRACK", None) or os.environ.get(
        "DO_NOT_TRACK", None) or "0") == "1",
    "VLLM_USAGE_SOURCE":
    lambda: os.environ.get("VLLM_USAGE_SOURCE", "production"),

    # Logging configuration
    # If set to 0, vllm will not configure logging
    # If set to 1, vllm will configure logging using the default configuration
    #    or the configuration file specified by VLLM_LOGGING_CONFIG_PATH
    "VLLM_CONFIGURE_LOGGING":
    lambda: int(os.getenv("VLLM_CONFIGURE_LOGGING", "1")),
    "VLLM_LOGGING_CONFIG_PATH":
    lambda: os.getenv("VLLM_LOGGING_CONFIG_PATH"),

    # this is used for configuring the default logging level
    "VLLM_LOGGING_LEVEL":
    lambda: os.getenv("VLLM_LOGGING_LEVEL", "INFO").upper(),

    # if set, VLLM_LOGGING_PREFIX will be prepended to all log messages
    "VLLM_LOGGING_PREFIX":
    lambda: os.getenv("VLLM_LOGGING_PREFIX", ""),

    # if set, vllm will call logits processors in a thread pool with this many
    # threads. This is useful when using custom logits processors that either
    # (a) launch additional CUDA kernels or (b) do significant CPU-bound work
    # while not holding the python GIL, or both.
    "VLLM_LOGITS_PROCESSOR_THREADS":
    lambda: int(os.getenv("VLLM_LOGITS_PROCESSOR_THREADS", "0"))
    if "VLLM_LOGITS_PROCESSOR_THREADS" in os.environ else None,

    # Trace function calls
    # If set to 1, vllm will trace function calls
    # Useful for debugging
    "VLLM_TRACE_FUNCTION":
    lambda: int(os.getenv("VLLM_TRACE_FUNCTION", "0")),

    # Backend for attention computation
    # Available options:
    # - "TORCH_SDPA": use torch.nn.MultiheadAttention
    # - "FLASH_ATTN": use FlashAttention
    # - "XFORMERS": use XFormers
    # - "ROCM_FLASH": use ROCmFlashAttention
    # - "FLASHINFER": use flashinfer
    # - "FLASHMLA": use FlashMLA
    "VLLM_ATTENTION_BACKEND":
    lambda: os.getenv("VLLM_ATTENTION_BACKEND", None),

    # If set, vllm will use flashinfer sampler
    "VLLM_USE_FLASHINFER_SAMPLER":
    lambda: bool(int(os.environ["VLLM_USE_FLASHINFER_SAMPLER"]))
    if "VLLM_USE_FLASHINFER_SAMPLER" in os.environ else None,

    # If set, vllm will force flashinfer to use tensor cores;
    # otherwise will use heuristic based on model architecture.
    "VLLM_FLASHINFER_FORCE_TENSOR_CORES":
    lambda: bool(int(os.getenv("VLLM_FLASHINFER_FORCE_TENSOR_CORES", "0"))),

    # Pipeline stage partition strategy
    "VLLM_PP_LAYER_PARTITION":
    lambda: os.getenv("VLLM_PP_LAYER_PARTITION", None),

    # (CPU backend only) CPU key-value cache space.
    # default is 4 GiB
    "VLLM_CPU_KVCACHE_SPACE":
    lambda: int(os.getenv("VLLM_CPU_KVCACHE_SPACE", "0")),

    # 是否开启 V0 swap 路径的详细观测日志。
    # 仅用于本地实验排查，默认关闭，避免污染正常推理日志。
    "VLLM_V0_SWAP_TRACE":
    lambda: bool(int(os.getenv("VLLM_V0_SWAP_TRACE", "0"))),

    # 实验开关：允许 V0 在 decode-only step 也调用 LMCache KV send/store。
    # 默认关闭，保持原生 vLLM 只在 prefill/mixed-prefill 批次发送 KV 的语义。
    "VLLM_LMCACHE_SEND_DECODE_KV":
    lambda: bool(int(os.getenv("VLLM_LMCACHE_SEND_DECODE_KV", "0"))),

    # 是否在 V0 swap_out 后额外执行一次 GPU -> SSD(BaM) 影子写出。
    "VLLM_BAM_SHADOW_ENABLE":
    lambda: bool(int(os.getenv("VLLM_BAM_SHADOW_ENABLE", "0"))),

    # 是否在 V0 swap_in 时优先从 BaM 读回 KV block。
    "VLLM_BAM_SWAPIN_ENABLE":
    lambda: bool(int(os.getenv("VLLM_BAM_SWAPIN_ENABLE", "0"))),

    # 是否在 BaM swap_in 后校验恢复出的 block 与 cpu_cache 参考值一致。
    "VLLM_BAM_SWAPIN_VERIFY":
    lambda: bool(int(os.getenv("VLLM_BAM_SWAPIN_VERIFY", "0"))),

    # 校验 block 数。<=0 表示全量校验，>0 表示仅校验当前 batch 前 N 个映射。
    "VLLM_BAM_SWAPIN_VERIFY_BLOCKS":
    lambda: int(os.getenv("VLLM_BAM_SWAPIN_VERIFY_BLOCKS", "0")),

    # 是否在 LMCache V0 的 put 路径上额外做一份 BaM shadow store。
    "VLLM_BAM_LMCACHE_SHADOW_ENABLE":
    lambda: bool(int(os.getenv("VLLM_BAM_LMCACHE_SHADOW_ENABLE", "0"))),

    # 是否在 LMCache V0 的 get 路径上优先从 BaM 读取，失败再回退原始 LMCache。
    "VLLM_BAM_LMCACHE_PREFER_LOAD_ENABLE":
    lambda: bool(int(os.getenv("VLLM_BAM_LMCACHE_PREFER_LOAD_ENABLE", "0"))),

    # LMCache shadow store 在 BaM 里最多预留多少个 chunk 槽位。
    "VLLM_BAM_LMCACHE_SHADOW_CHUNKS":
    lambda: int(os.getenv("VLLM_BAM_LMCACHE_SHADOW_CHUNKS", "1024")),

    # LMCache BaM 读路径：
    # - sync: 稳定同步 baseline
    # - prefetch: page-level submit/poll/complete/refill 中间层，用于 GPU-initiated 路线
    "VLLM_BAM_LMCACHE_READ_MODE":
    lambda: os.getenv("VLLM_BAM_LMCACHE_READ_MODE", "sync"),

    # LMCache chunk 在 BaM 逻辑 row 空间里的起始 row。默认 0 保持原热路径行为；
    # cold-read replay 可设为 128，避开设备最前面的保留区域。
    "VLLM_BAM_LMCACHE_BASE_ROW_OFFSET":
    lambda: int(os.getenv("VLLM_BAM_LMCACHE_BASE_ROW_OFFSET", "0")),

    # 是否启用 KVCache 专用 BaM fast path。
    # 第一阶段内部仍复用 BaM rowctx，但接口已经从通用 feature path 中拆出。
    "VLLM_BAM_KV_FAST_PATH":
    lambda: bool(int(os.getenv("VLLM_BAM_KV_FAST_PATH", "0"))),

    # 是否启用 GPU-initiated demand/frontier 实验分支。
    #
    # 默认关闭，避免改变已经验证稳定的 one-copy 主线。
    # 打开后，不再在 LMCache connector 层提前 stage descriptor，也不提前
    # CPU submit。connector 只推进 retrieve/deferred 生命周期；真正的 request
    # 计划在 direct-placement start 边界由 storage 根据真实 prefix hit 现场生成：
    #
    #   LMCache retrieve 到达 demand start
    #     -> storage 收集连续 prefix hit chunks
    #     -> 生成 context-chunk frontier/metadata
    #     -> 进入统一 submit/poll/finalize request 生命周期
    #
    # 当前 frontier 只描述“哪些连续 context chunk 已 ready / consumable”，不改
    # dense attention 语义，也不改 flash-attn kernel。后续真正把 submit 下沉到
    # GPU 时，应在这个 demand-start/request frontier 边界后面接 GPU-side
    # descriptor ring，而不是恢复 connector-stage 预提交支线。
    "VLLM_BAM_GPU_INITIATED_PREFETCH":
    lambda: bool(int(os.getenv("VLLM_BAM_GPU_INITIATED_PREFETCH", "0"))),

    # 是否启用独立的 BaM KVStore vLLM-block 直通路径：
    #
    #   vLLM scheduler block mapping
    #     -> GPU submit
    #     -> SSD DMA 直接读写 vLLM paged KV cache
    #     -> GPU persistent CQ poll
    #     -> CPU 只检查 request ready 后继续 attention
    #
    # 该开关与旧 BaM page cache / LMCache direct-placement 完全隔离。默认关闭，
    # 因此不会改变 LMCache SSD、传统 GDS、普通 V0 swap 等 baseline 的行为。
    "VLLM_BAM_DIRECT_KVSTORE_ENABLE":
    lambda: bool(int(os.getenv("VLLM_BAM_DIRECT_KVSTORE_ENABLE", "0"))),
    # 仅改变 direct KVStore 的 swap-in 实现。默认 direct 完整保留当前路径；
    # bam_sync 使用原生 BaM page cache，并在单次 kernel 内同步轮询 CQ。
    "VLLM_BAM_DIRECT_READ_MODE":
    lambda: os.getenv("VLLM_BAM_DIRECT_READ_MODE", "direct"),
    "VLLM_BAM_SYNC_CACHE_SIZE_MB":
    lambda: int(os.getenv("VLLM_BAM_SYNC_CACHE_SIZE_MB", "64")),

    # MDS 把 BaM controller、DMA mapping 和 persistent CQ service 放到独立
    # process；vLLM 仅导入 daemon-owned KV allocations 并同步提交 block mapping。
    # 默认关闭，并且与本进程 direct/legacy BaM swap 路径互斥。
    "VLLM_BAM_MDS_ENABLE":
    lambda: bool(int(os.getenv("VLLM_BAM_MDS_ENABLE", "0"))),
    "VLLM_BAM_MDS_IOSTACK_ROOT":
    lambda: os.getenv("VLLM_BAM_MDS_IOSTACK_ROOT"),
    "VLLM_BAM_MDS_CONTROL_DIR":
    lambda: os.getenv("VLLM_BAM_MDS_CONTROL_DIR"),
    "VLLM_BAM_MDS_CUDA_IPC_LIBRARY":
    lambda: os.getenv("VLLM_BAM_MDS_CUDA_IPC_LIBRARY"),
    "VLLM_BAM_MDS_TORCH_BRIDGE_DIR":
    lambda: os.getenv("VLLM_BAM_MDS_TORCH_BRIDGE_DIR"),
    "VLLM_BAM_MDS_TIMEOUT_SECONDS":
    lambda: float(os.getenv("VLLM_BAM_MDS_TIMEOUT_SECONDS", "120")),
    "VLLM_BAM_MDS_MAX_IN_FLIGHT":
    lambda: int(os.getenv("VLLM_BAM_MDS_MAX_IN_FLIGHT", "4")),
    "VLLM_BAM_MDS_SERVICE_LIFETIME":
    lambda: os.getenv("VLLM_BAM_MDS_SERVICE_LIFETIME", "resident"),
    "VLLM_BAM_MDS_IDLE_STOP_DELAY_MS":
    lambda: int(os.getenv("VLLM_BAM_MDS_IDLE_STOP_DELAY_MS", "0")),
    # 只启用 BaM MDS 自己的 SSD prefix populate/restore 控制面。关闭时
    # AsyncKVScheduler 的原有 swap-only 行为完全不变。
    "VLLM_BAM_MDS_PREFIX_ENABLE":
    lambda: bool(int(os.getenv("VLLM_BAM_MDS_PREFIX_ENABLE", "0"))),
    # Step 1/2 的层级 I/O 实验开关。NUM_LAYERS 是当前 PP rank 的本地层数；
    # 关闭时不会拆分请求，也不会增加原有 prefix baseline 的 transfer 数量。
    "VLLM_BAM_MDS_HIERARCHICAL_IO_ENABLE":
    lambda: bool(
        int(os.getenv("VLLM_BAM_MDS_HIERARCHICAL_IO_ENABLE", "0"))),
    "VLLM_BAM_MDS_HIERARCHICAL_NUM_LAYERS":
    lambda: int(os.getenv("VLLM_BAM_MDS_HIERARCHICAL_NUM_LAYERS", "0")),
    "VLLM_BAM_MDS_HIERARCHICAL_WINDOW_LAYERS":
    lambda: int(os.getenv("VLLM_BAM_MDS_HIERARCHICAL_WINDOW_LAYERS", "0")),
    # Step 4：首窗 ready 后进入 forward，并在每个 layer window 前由 worker
    # 直接确认 DMA 完成。关闭时保留 Step 1/2 的 full-restore dispatch gate。
    "VLLM_BAM_MDS_HIERARCHICAL_LAYER_BARRIER":
    lambda: bool(
        int(os.getenv("VLLM_BAM_MDS_HIERARCHICAL_LAYER_BARRIER", "0"))),
    "VLLM_BAM_MDS_HIERARCHICAL_ROLLING_ENABLE":
    lambda: bool(
        int(os.getenv("VLLM_BAM_MDS_HIERARCHICAL_ROLLING_ENABLE", "0"))),
    "VLLM_BAM_MDS_HIERARCHICAL_LEAD_WINDOWS":
    lambda: int(
        os.getenv("VLLM_BAM_MDS_HIERARCHICAL_LEAD_WINDOWS", "1")),
    "VLLM_BAM_MDS_HIERARCHICAL_MAX_LEAD_WINDOWS":
    lambda: int(
        os.getenv("VLLM_BAM_MDS_HIERARCHICAL_MAX_LEAD_WINDOWS", "1")),
    "VLLM_BAM_MDS_HIERARCHICAL_TARGET_SLACK_MS":
    lambda: float(
        os.getenv("VLLM_BAM_MDS_HIERARCHICAL_TARGET_SLACK_MS", "0")),

    "VLLM_BAM_MDS_PREFETCH_BLOCK_SELECTOR":
    lambda: os.getenv("VLLM_BAM_MDS_PREFETCH_BLOCK_SELECTOR", "dense"),
    "VLLM_BAM_MDS_PREFETCH_BLOCK_COUNT":
    lambda: int(os.getenv("VLLM_BAM_MDS_PREFETCH_BLOCK_COUNT", "0")),
    "VLLM_BAM_MDS_PREFETCH_BLOCK_STRIDE":
    lambda: int(os.getenv("VLLM_BAM_MDS_PREFETCH_BLOCK_STRIDE", "1")),

    # 是否启用 Direct Placement v0。
    # 开启后 LMCache retrieve 会优先尝试：
    #   BaM pages -> vLLM paged KV cache
    # 失败则回退原 LMCache retrieve。
    "VLLM_BAM_DIRECT_PLACEMENT":
    lambda: bool(int(os.getenv("VLLM_BAM_DIRECT_PLACEMENT", "0"))),

    # Direct Placement 的具体实现：
    # - lmcache: 保正确版本，BaM pages -> LMCache tensor -> LMCache transfer
    # - fused: 实验版本，BaM pages -> vLLM flat paged KV cache
    "VLLM_BAM_DIRECT_PLACEMENT_IMPL":
    lambda: os.getenv("VLLM_BAM_DIRECT_PLACEMENT_IMPL", "lmcache"),

    # 是否启用 runtime one-copy 实验主线：
    #
    #   BaM cache -> 最终 vLLM paged KV cache
    #
    # 这条链路当前仍在低层排查中。为了保证正式主线先稳定跑通，默认先关闭；
    # 关闭后仍然保留：
    #
    # - GPU worker / persistent service 负责后台 poll/read
    # - 前台只在 pages ready 后发一跳已验证正确的 finalize kernel
    #
    # 继续调试一次搬运时，再显式打开：
    #   VLLM_BAM_DIRECT_PLACEMENT_RUNTIME_ONE_COPY=1
    "VLLM_BAM_DIRECT_PLACEMENT_RUNTIME_ONE_COPY":
    lambda: bool(
        int(os.getenv("VLLM_BAM_DIRECT_PLACEMENT_RUNTIME_ONE_COPY", "0"))),

    # 是否要求当前请求必须严格走 runtime one-copy 主线。
    #
    # 打开后，storage 不再允许悄悄回退到：
    #   results_materialized + host finalize
    #
    # 如果 runtime attach 失败，或者 one-copy 总开关没有打开，就应立即报错。
    # 这个开关主要服务 GPU worker + persistent service 专用启动脚本。
    "VLLM_BAM_DIRECT_PLACEMENT_REQUIRE_RUNTIME_ONE_COPY":
    lambda: bool(
        int(
            os.getenv(
                "VLLM_BAM_DIRECT_PLACEMENT_REQUIRE_RUNTIME_ONE_COPY",
                "0",
            ))),

    # 是否把 direct placement request handle 上提到 runtime，允许当前
    # batch 在 retrieve 未 ready 时显式返回 DEFERRED，并在下一轮继续 poll。
    #
    # 默认关闭，保持当前稳定的 blocking recv 主线不变。
    "VLLM_BAM_DIRECT_PLACEMENT_DEFER_RUNTIME":
    lambda: bool(
        int(os.getenv("VLLM_BAM_DIRECT_PLACEMENT_DEFER_RUNTIME", "0"))),

    # 运行时验证开关：direct placement request handle 至少要经历多少次
    # `recv -> poll -> DEFERRED` 轮次之后，才允许 finalize。
    #
    # 语义：
    # - 0: 不强制额外 defer；如果本轮已经 ready，就允许同轮 finalize
    # - N>0: 至少 defer N 轮后，下一轮才允许 finalize
    #
    # 这个开关主要用于验证“跨 engine iteration 的 live handle”主线是否真的
    # 跑通，默认关闭，不影响当前正式性能口径。
    "VLLM_BAM_DIRECT_PLACEMENT_DEFER_MIN_POLLS":
    lambda: int(
        os.getenv("VLLM_BAM_DIRECT_PLACEMENT_DEFER_MIN_POLLS", "0")),

    # 测试/benchmark 用的临时 idle-stop 开关。
    #
    # 默认 0 表示保持 persistent service 常驻，不改变正常服务模式。
    # 设置为 N>0 时，KV direct placement 每完成一次 IO 后会启动一个轻量
    # watchdog；如果接下来 N 秒没有新的 KV IO request，就在 runtime idle
    # 时尝试停止后台 persistent service。这样一次性跑分脚本可以在所有 iter
    # 完成后释放 resident kernel，避免进程退出时被后台 service 拖住。
    "VLLM_BAM_RUNTIME_IDLE_STOP_SECONDS":
    lambda: float(os.getenv("VLLM_BAM_RUNTIME_IDLE_STOP_SECONDS", "0")),

    # 是否为 xformers prefix fallback 打开细粒度阶段计时。
    # 默认关闭，避免在正常跑分时引入额外 synchronize 干扰口径。
    "VLLM_BAM_XFORMERS_PREFIX_FALLBACK_PROFILE":
    lambda: bool(
        int(os.getenv("VLLM_BAM_XFORMERS_PREFIX_FALLBACK_PROFILE", "0"))),

    # xFormers prefix fallback 的 prefix KV 读取 backend。
    #
    # 默认 `auto`：保持当前主线，优先使用 packed_direct_to_workspace。
    # 调试时可显式设为：
    # - packed_direct_to_workspace：强制走当前 Triton packed 直读路径
    # - gather_then_copy：强制走保守的 vLLM gather_cache + copy 路径
    #
    # 这个开关只用于定位“paged KV cache 写对了但 xformers 读错了”的问题，
    # 不改变 BaM/GPU worker 的 submit、poll、direct placement 主线。
    "VLLM_BAM_XFORMERS_PREFIX_BACKEND":
    lambda: os.getenv("VLLM_BAM_XFORMERS_PREFIX_BACKEND", "auto").strip().lower(),

    # xFormers prefix fallback 的 query KV 写入 backend。
    #
    # 默认 `auto`：沿用当前主线，能用 Triton direct scatter 就用。
    # 调试时可显式设为：
    # - direct_scatter：强制走当前 Triton query scatter 路径
    # - segment_copy：强制走保守的逐 segment copy_ 路径
    #
    # 这个开关只影响 xFormers fallback 内部如何把“本轮新算出来的 query K/V”
    # 放入连续 full-KV workspace，不改变 BaM 读、GPU persistent service、
    # direct placement 或 paged-KV 写入语义。
    "VLLM_BAM_XFORMERS_QUERY_BACKEND":
    lambda: os.getenv("VLLM_BAM_XFORMERS_QUERY_BACKEND", "auto").strip().lower(),

    # xFormers prefix fallback 的读端正确性抽样校验。
    #
    # 打开后会把 fallback 从 paged KV cache gather 出来的 prefix K/V，
    # 和 BaM live pages 解码得到的 dense chunk reference 做少量点对点比较。
    # 它只用于定位“最终输出乱”到底发生在 paged-cache/gather 读端，还是更后面
    # 的 full workspace / attention bias / xFormers 消费语义；默认关闭，避免
    # 正式性能实验里引入额外同步。
    "VLLM_BAM_XFORMERS_VERIFY_PREFIX_GATHER":
    lambda: bool(
        int(os.getenv("VLLM_BAM_XFORMERS_VERIFY_PREFIX_GATHER", "0"))),

    # xFormers prefix fallback 的最终 attention 输出抽样校验。
    #
    # 当 prefix gather 已经确认正确，但模型输出仍然异常时，打开该开关会用
    # 同一份 query/full_key/full_value 按 bottom-right causal 语义手写少量
    # PyTorch reference attention，并和 xFormers 输出比对。
    #
    # 这能继续把问题切成两段：
    # - reference 与 xFormers 不一致：查 attention bias / full workspace 语义；
    # - reference 与 xFormers 一致：查 rebuild 后 query/input position 语义。
    "VLLM_BAM_XFORMERS_VERIFY_ATTENTION_OUTPUT":
    lambda: bool(
        int(os.getenv("VLLM_BAM_XFORMERS_VERIFY_ATTENTION_OUTPUT", "0"))),

    # xFormers prefix fallback 的整层 attention 输出校验。
    #
    # `VLLM_BAM_XFORMERS_VERIFY_ATTENTION_OUTPUT` 只做少量 query/head 抽样。
    # 当抽样通过但模型输出仍然乱码时，需要进一步确认：
    #
    #   xFormers 对整层所有 query token / head / dim 的输出
    #   是否都等于同一份 query/full_key/full_value 上手写的
    #   bottom-right causal reference。
    #
    # 这个校验会构造较大的 fp32 attention scores，开销明显，因此默认关闭，
    # 并且默认只查第 0 层。若要查全部层，可把 FULL_LAYER 设为 -1，但一般
    # correctness 定位不建议这么做。
    "VLLM_BAM_XFORMERS_VERIFY_ATTENTION_OUTPUT_FULL":
    lambda: bool(
        int(os.getenv("VLLM_BAM_XFORMERS_VERIFY_ATTENTION_OUTPUT_FULL", "0"))),

    "VLLM_BAM_XFORMERS_VERIFY_ATTENTION_OUTPUT_FULL_LAYER":
    lambda: int(
        os.getenv("VLLM_BAM_XFORMERS_VERIFY_ATTENTION_OUTPUT_FULL_LAYER", "0")),

    # 模型输出边界诊断开关。
    #
    # 这条开关只做日志，不改计算。打开后会在 model_runner 里打印：
    # - selected_token_indices / seq_lens / query_lens
    # - hidden_states 的形状
    # - 选中 hidden row 的范数
    # - logits top-k token id/value
    #
    # 目标是和 rowctx 正常路径做一一对照，定位问题是否已经发生在
    # xFormers attention 之后的 logits / sampling 入口。
    "VLLM_BAM_LOGITS_SEMANTIC_DEBUG":
    lambda: bool(int(os.getenv("VLLM_BAM_LOGITS_SEMANTIC_DEBUG", "0"))),

    # 可选：显式指定 BaM Python 模块搜索路径。
    "VLLM_BAM_IMPORT_PATH":
    lambda: os.getenv("VLLM_BAM_IMPORT_PATH", None),

    # BaM page cache 大小，单位 MB。
    "VLLM_BAM_CACHE_SIZE_MB":
    lambda: int(os.getenv("VLLM_BAM_CACHE_SIZE_MB", "512")),

    # 使用的 SSD 控制器数量。
    "VLLM_BAM_NUM_SSD":
    lambda: int(os.getenv("VLLM_BAM_NUM_SSD", "1")),

    # 可选：逗号分隔的 SSD 编号列表，例如 "0,1"。
    "VLLM_BAM_SSD_LIST":
    lambda: os.getenv("VLLM_BAM_SSD_LIST", None),

    # BaM 使用的控制 GPU 编号。
    "VLLM_BAM_CTRL_IDX":
    lambda: int(os.getenv("VLLM_BAM_CTRL_IDX", "0")),

    # 是否在 LMCache V0 的 put 路径上额外写一份 LMCache-style GDS 文件。
    "VLLM_GDS_LMCACHE_SHADOW_ENABLE":
    lambda: bool(int(os.getenv("VLLM_GDS_LMCACHE_SHADOW_ENABLE", "0"))),

    # 是否在 LMCache V0 的 get 路径上优先从 LMCache-style GDS 读取。
    "VLLM_GDS_LMCACHE_PREFER_LOAD_ENABLE":
    lambda: bool(int(os.getenv("VLLM_GDS_LMCACHE_PREFER_LOAD_ENABLE", "0"))),

    # LMCache-style GDS 的根目录，内部按 chunk_hash 前缀分两级目录。
    "VLLM_GDS_LMCACHE_PATH":
    lambda: os.getenv("VLLM_GDS_LMCACHE_PATH", "/tmp/vllm-bam-lmcache-gds"),

    # 是否使用原生 cuFile/GDS。设为 0 时只作为 POSIX fallback 调试路径。
    "VLLM_GDS_LMCACHE_USE_GDS":
    lambda: bool(int(os.getenv("VLLM_GDS_LMCACHE_USE_GDS", "1"))),

    # cuFile 路径是否使用 O_DIRECT，贴近 LMCache V1 GdsBackend 的 use_direct_io。
    "VLLM_GDS_LMCACHE_USE_DIRECT_IO":
    lambda: bool(int(os.getenv("VLLM_GDS_LMCACHE_USE_DIRECT_IO", "1"))),

    # GDS staging tensor 使用的 CUDA device；为空时跟随 VLLM_BAM_CTRL_IDX。
    "VLLM_GDS_LMCACHE_DEVICE":
    lambda: os.getenv("VLLM_GDS_LMCACHE_DEVICE", None),

    # LMCache V1 metadata 中的 MemoryFormat.value；默认贴近 V1 KV_2LTD。
    "VLLM_GDS_LMCACHE_FMT":
    lambda: os.getenv("VLLM_GDS_LMCACHE_FMT", "KV_2LTD"),

    # 是否启用 V1-like 预注册 GPU staging buffer。
    "VLLM_GDS_LMCACHE_USE_REGISTERED_BUFFER":
    lambda: bool(int(os.getenv("VLLM_GDS_LMCACHE_USE_REGISTERED_BUFFER", "0"))),

    # 预注册 staging buffer 大小。0 表示按首次 chunk 大小懒分配。
    "VLLM_GDS_LMCACHE_REGISTERED_BUFFER_MB":
    lambda: int(os.getenv("VLLM_GDS_LMCACHE_REGISTERED_BUFFER_MB", "0")),

    # (CPU backend only) CPU core ids bound by OpenMP threads, e.g., "0-31",
    # "0,1,2", "0-31,33". CPU cores of different ranks are separated by '|'.
    "VLLM_CPU_OMP_THREADS_BIND":
    lambda: os.getenv("VLLM_CPU_OMP_THREADS_BIND", "all"),

    # (CPU backend only) whether to use prepack for MoE layer. This will be
    # passed to ipex.llm.modules.GatedMLPMOE. On unsupported CPUs, you might
    # need to set this to "0" (False).
    "VLLM_CPU_MOE_PREPACK":
    lambda: bool(int(os.getenv("VLLM_CPU_MOE_PREPACK", "1"))),

    # If the env var is set, then all workers will execute as separate
    # processes from the engine, and we use the same mechanism to trigger
    # execution on all workers.
    # Run vLLM with VLLM_USE_RAY_SPMD_WORKER=1 to enable it.
    "VLLM_USE_RAY_SPMD_WORKER":
    lambda: bool(int(os.getenv("VLLM_USE_RAY_SPMD_WORKER", "0"))),

    # If the env var is set, it uses the Ray's Compiled Graph
    # (previously known as ADAG) API which optimizes the
    # control plane overhead.
    # Run vLLM with VLLM_USE_RAY_COMPILED_DAG=1 to enable it.
    # Note that this variable is set to 1 in V1 by default
    # when ray distributed executor is used.
    "VLLM_USE_RAY_COMPILED_DAG":
    lambda: bool(int(os.getenv("VLLM_USE_RAY_COMPILED_DAG", "0"))),

    # If the env var is set, Ray Compiled Graph uses the specified
    # channel type to communicate between workers belonging to
    # different pipeline-parallel stages.
    # Available options:
    # - "auto": use the default channel type
    # - "nccl": use NCCL for communication
    # - "shm": use shared memory and gRPC for communication
    # This flag is ignored if VLLM_USE_RAY_COMPILED_DAG is not set.
    "VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE":
    lambda: os.getenv("VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE", "auto"),

    # If the env var is set, it enables GPU communication overlap
    # (experimental feature) in Ray's Compiled Graph. This flag is ignored if
    # VLLM_USE_RAY_COMPILED_DAG is not set.
    "VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM":
    lambda: bool(int(os.getenv("VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM", "0"))
                 ),

    # Use dedicated multiprocess context for workers.
    # Both spawn and fork work
    "VLLM_WORKER_MULTIPROC_METHOD":
    lambda: os.getenv("VLLM_WORKER_MULTIPROC_METHOD", "fork"),

    # Path to the cache for storing downloaded assets
    "VLLM_ASSETS_CACHE":
    lambda: os.path.expanduser(
        os.getenv(
            "VLLM_ASSETS_CACHE",
            os.path.join(get_default_cache_root(), "vllm", "assets"),
        )),

    # Timeout for fetching images when serving multimodal models
    # Default is 5 seconds
    "VLLM_IMAGE_FETCH_TIMEOUT":
    lambda: int(os.getenv("VLLM_IMAGE_FETCH_TIMEOUT", "5")),

    # Timeout for fetching videos when serving multimodal models
    # Default is 30 seconds
    "VLLM_VIDEO_FETCH_TIMEOUT":
    lambda: int(os.getenv("VLLM_VIDEO_FETCH_TIMEOUT", "30")),

    # Timeout for fetching audio when serving multimodal models
    # Default is 10 seconds
    "VLLM_AUDIO_FETCH_TIMEOUT":
    lambda: int(os.getenv("VLLM_AUDIO_FETCH_TIMEOUT", "10")),

    # Cache size (in GiB) for multimodal input cache
    # Default is 4 GiB
    "VLLM_MM_INPUT_CACHE_GIB":
    lambda: int(os.getenv("VLLM_MM_INPUT_CACHE_GIB", "4")),

    # Path to the XLA persistent cache directory.
    # Only used for XLA devices such as TPUs.
    "VLLM_XLA_CACHE_PATH":
    lambda: os.path.expanduser(
        os.getenv(
            "VLLM_XLA_CACHE_PATH",
            os.path.join(get_default_cache_root(), "vllm", "xla_cache"),
        )),

    # If set, assert on XLA recompilation after each execution step.
    "VLLM_XLA_CHECK_RECOMPILATION":
    lambda: bool(int(os.getenv("VLLM_XLA_CHECK_RECOMPILATION", "0"))),
    "VLLM_FUSED_MOE_CHUNK_SIZE":
    lambda: int(os.getenv("VLLM_FUSED_MOE_CHUNK_SIZE", "32768")),

    # If set, vllm will skip the deprecation warnings.
    "VLLM_NO_DEPRECATION_WARNING":
    lambda: bool(int(os.getenv("VLLM_NO_DEPRECATION_WARNING", "0"))),

    # If set, the OpenAI API server will stay alive even after the underlying
    # AsyncLLMEngine errors and stops serving requests
    "VLLM_KEEP_ALIVE_ON_ENGINE_DEATH":
    lambda: bool(os.getenv("VLLM_KEEP_ALIVE_ON_ENGINE_DEATH", 0)),

    # If the env var VLLM_ALLOW_LONG_MAX_MODEL_LEN is set, it allows
    # the user to specify a max sequence length greater than
    # the max length derived from the model's config.json.
    # To enable this, set VLLM_ALLOW_LONG_MAX_MODEL_LEN=1.
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN":
    lambda:
    (os.environ.get("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "0").strip().lower() in
     ("1", "true")),

    # If set, forces FP8 Marlin to be used for FP8 quantization regardless
    # of the hardware support for FP8 compute.
    "VLLM_TEST_FORCE_FP8_MARLIN":
    lambda:
    (os.environ.get("VLLM_TEST_FORCE_FP8_MARLIN", "0").strip().lower() in
     ("1", "true")),
    "VLLM_TEST_FORCE_LOAD_FORMAT":
    lambda: os.getenv("VLLM_TEST_FORCE_LOAD_FORMAT", "dummy"),

    # Time in ms for the zmq client to wait for a response from the backend
    # server for simple data operations
    "VLLM_RPC_TIMEOUT":
    lambda: int(os.getenv("VLLM_RPC_TIMEOUT", "10000")),

    # a list of plugin names to load, separated by commas.
    # if this is not set, it means all plugins will be loaded
    # if this is set to an empty string, no plugins will be loaded
    "VLLM_PLUGINS":
    lambda: None if "VLLM_PLUGINS" not in os.environ else os.environ[
        "VLLM_PLUGINS"].split(","),

    # Enables torch profiler if set. Path to the directory where torch profiler
    # traces are saved. Note that it must be an absolute path.
    "VLLM_TORCH_PROFILER_DIR":
    lambda: (None if os.getenv("VLLM_TORCH_PROFILER_DIR", None) is None else os
             .path.expanduser(os.getenv("VLLM_TORCH_PROFILER_DIR", "."))),

    # If set, vLLM will use Triton implementations of AWQ.
    "VLLM_USE_TRITON_AWQ":
    lambda: bool(int(os.getenv("VLLM_USE_TRITON_AWQ", "0"))),

    # If set, allow loading or unloading lora adapters in runtime,
    "VLLM_ALLOW_RUNTIME_LORA_UPDATING":
    lambda:
    (os.environ.get("VLLM_ALLOW_RUNTIME_LORA_UPDATING", "0").strip().lower() in
     ("1", "true")),

    # By default, vLLM will check the peer-to-peer capability itself,
    # in case of broken drivers. See https://github.com/vllm-project/vllm/blob/a9b15c606fea67a072416ea0ea115261a2756058/vllm/distributed/device_communicators/custom_all_reduce_utils.py#L101-L108 for details. # noqa
    # If this env var is set to 1, vLLM will skip the peer-to-peer check,
    # and trust the driver's peer-to-peer capability report.
    "VLLM_SKIP_P2P_CHECK":
    lambda: os.getenv("VLLM_SKIP_P2P_CHECK", "0") == "1",

    # List of quantization kernels that should be disabled, used for testing
    # and performance comparisons. Currently only affects MPLinearKernel
    # selection
    # (kernels: MacheteLinearKernel, MarlinLinearKernel, ExllamaLinearKernel)
    "VLLM_DISABLED_KERNELS":
    lambda: [] if "VLLM_DISABLED_KERNELS" not in os.environ else os.environ[
        "VLLM_DISABLED_KERNELS"].split(","),

    # If set, use the V1 code path.
    "VLLM_USE_V1":
    lambda: bool(int(os.getenv("VLLM_USE_V1", "1"))),

    # Disable aiter ops unless specifically enabled.
    # Acts as a parent switch to enable the rest of the other operations.
    "VLLM_ROCM_USE_AITER":
    lambda: (os.getenv("VLLM_ROCM_USE_AITER", "False").lower() in
             ("true", "1")),

    # Whether to use aiter paged attention.
    # By default is disabled.
    "VLLM_ROCM_USE_AITER_PAGED_ATTN":
    lambda: (os.getenv("VLLM_ROCM_USE_AITER_PAGED_ATTN", "False").lower() in
             ("true", "1")),

    # use aiter linear op if aiter ops are enabled
    # The following list of related ops
    # - scaled_mm (per-tensor / rowwise)
    "VLLM_ROCM_USE_AITER_LINEAR":
    lambda: (os.getenv("VLLM_ROCM_USE_AITER_LINEAR", "True").lower() in
             ("true", "1")),

    # Whether to use aiter moe ops.
    # By default is enabled.
    "VLLM_ROCM_USE_AITER_MOE":
    lambda: (os.getenv("VLLM_ROCM_USE_AITER_MOE", "True").lower() in
             ("true", "1")),

    # use aiter rms norm op if aiter ops are enabled.
    "VLLM_ROCM_USE_AITER_RMSNORM":
    lambda: (os.getenv("VLLM_ROCM_USE_AITER_RMSNORM", "True").lower() in
             ("true", "1")),

    # Whether to use aiter mla ops.
    # By default is enabled.
    "VLLM_ROCM_USE_AITER_MLA":
    lambda: (os.getenv("VLLM_ROCM_USE_AITER_MLA", "True").lower() in
             ("true", "1")),
    # use rocm skinny gemms
    "VLLM_ROCM_USE_SKINNY_GEMM":
    lambda: (os.getenv("VLLM_ROCM_USE_SKINNY_GEMM", "True").lower() in
             ("true", "1")),

    # Pad the fp8 weights to 256 bytes for ROCm
    "VLLM_ROCM_FP8_PADDING":
    lambda: bool(int(os.getenv("VLLM_ROCM_FP8_PADDING", "1"))),

    # Pad the weights for the moe kernel
    "VLLM_ROCM_MOE_PADDING":
    lambda: bool(int(os.getenv("VLLM_ROCM_MOE_PADDING", "1"))),

    # custom paged attention kernel for MI3* cards
    "VLLM_ROCM_CUSTOM_PAGED_ATTN":
    lambda: (os.getenv("VLLM_ROCM_CUSTOM_PAGED_ATTN", "True").lower() in
             ("true", "1")),

    # Divisor for dynamic query scale factor calculation for FP8 KV Cache
    "Q_SCALE_CONSTANT":
    lambda: int(os.getenv("Q_SCALE_CONSTANT", "200")),
    # Divisor for dynamic key scale factor calculation for FP8 KV Cache
    "K_SCALE_CONSTANT":
    lambda: int(os.getenv("K_SCALE_CONSTANT", "200")),
    # Divisor for dynamic value scale factor calculation for FP8 KV Cache
    "V_SCALE_CONSTANT":
    lambda: int(os.getenv("V_SCALE_CONSTANT", "100")),

    # If set, enable multiprocessing in LLM for the V1 code path.
    "VLLM_ENABLE_V1_MULTIPROCESSING":
    lambda: bool(int(os.getenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1"))),
    "VLLM_LOG_BATCHSIZE_INTERVAL":
    lambda: float(os.getenv("VLLM_LOG_BATCHSIZE_INTERVAL", "-1")),
    "VLLM_DISABLE_COMPILE_CACHE":
    lambda: bool(int(os.getenv("VLLM_DISABLE_COMPILE_CACHE", "0"))),

    # If set, vllm will run in development mode, which will enable
    # some additional endpoints for developing and debugging,
    # e.g. `/reset_prefix_cache`
    "VLLM_SERVER_DEV_MODE":
    lambda: bool(int(os.getenv("VLLM_SERVER_DEV_MODE", "0"))),

    # Controls the maximum number of requests to handle in a
    # single asyncio task when processing per-token outputs in the
    # V1 AsyncLLM interface. It is applicable when handling a high
    # concurrency of streaming requests.
    # Setting this too high can result in a higher variance of
    # inter-message latencies. Setting it too low can negatively impact
    # TTFT and overall throughput.
    "VLLM_V1_OUTPUT_PROC_CHUNK_SIZE":
    lambda: int(os.getenv("VLLM_V1_OUTPUT_PROC_CHUNK_SIZE", "128")),

    # If set, vLLM will disable the MLA attention optimizations.
    "VLLM_MLA_DISABLE":
    lambda: bool(int(os.getenv("VLLM_MLA_DISABLE", "0"))),

    # If set, vLLM will use the Triton implementation of moe_align_block_size,
    # i.e. moe_align_block_size_triton in fused_moe.py.
    "VLLM_ENABLE_MOE_ALIGN_BLOCK_SIZE_TRITON":
    lambda: bool(int(os.getenv("VLLM_ENABLE_MOE_ALIGN_BLOCK_SIZE_TRITON", "0"))
                 ),

    # Number of GPUs per worker in Ray, if it is set to be a fraction,
    # it allows ray to schedule multiple actors on a single GPU,
    # so that users can colocate other actors on the same GPUs as vLLM.
    "VLLM_RAY_PER_WORKER_GPUS":
    lambda: float(os.getenv("VLLM_RAY_PER_WORKER_GPUS", "1.0")),

    # Bundle indices for Ray, if it is set, it can control precisely
    # which indices are used for the Ray bundle, for every worker.
    # Format: comma-separated list of integers, e.g. "0,1,2,3"
    "VLLM_RAY_BUNDLE_INDICES":
    lambda: os.getenv("VLLM_RAY_BUNDLE_INDICES", ""),

    # In some system, find_loaded_library() may not work. So we allow users to
    # specify the path through environment variable VLLM_CUDART_SO_PATH.
    "VLLM_CUDART_SO_PATH":
    lambda: os.getenv("VLLM_CUDART_SO_PATH", None),

    # Contiguous cache fetching to avoid using costly gather operation on
    # Gaudi3. This is only applicable to HPU contiguous cache. If set to true,
    # contiguous cache fetch will be used.
    "VLLM_USE_HPU_CONTIGUOUS_CACHE_FETCH":
    lambda: os.environ.get("VLLM_CONTIGUOUS_PA", "true").lower() in
    ("1", "true"),

    # Use delayed sampling for HPU to reduce host cpu overhead
    # between each step.
    "VLLM_HPU_USE_DELAYED_SAMPLING":
    lambda: os.environ.get("VLLM_DELAYED_SAMPLING", "false").lower() in
    ("1", "true"),

    # Rank of the process in the data parallel setting
    "VLLM_DP_RANK":
    lambda: int(os.getenv("VLLM_DP_RANK", "0")),

    # Rank of the process in the data parallel setting.
    # Defaults to VLLM_DP_RANK when not set.
    "VLLM_DP_RANK_LOCAL":
    lambda: int(
        os.getenv("VLLM_DP_RANK_LOCAL", sys.modules[__name__].VLLM_DP_RANK)),

    # World size of the data parallel setting
    "VLLM_DP_SIZE":
    lambda: int(os.getenv("VLLM_DP_SIZE", "1")),

    # IP address of the master node in the data parallel setting
    "VLLM_DP_MASTER_IP":
    lambda: os.getenv("VLLM_DP_MASTER_IP", "127.0.0.1"),

    # Port of the master node in the data parallel setting
    "VLLM_DP_MASTER_PORT":
    lambda: int(os.getenv("VLLM_DP_MASTER_PORT", "0")),

    # Whether to use S3 path for model loading in CI via RunAI Streamer
    "VLLM_CI_USE_S3":
    lambda: os.environ.get("VLLM_CI_USE_S3", "0") == "1",

    # Use model_redirect to redirect the model name to a local folder.
    # `model_redirect` can be a json file mapping the model between
    # repo_id and local folder:
    # {"meta-llama/Llama-3.2-1B": "/tmp/Llama-3.2-1B"}
    # or a space separated values table file:
    # meta-llama/Llama-3.2-1B   /tmp/Llama-3.2-1B
    "VLLM_MODEL_REDIRECT_PATH":
    lambda: os.environ.get("VLLM_MODEL_REDIRECT_PATH", None),

    # Whether to use atomicAdd reduce in gptq/awq marlin kernel.
    "VLLM_MARLIN_USE_ATOMIC_ADD":
    lambda: os.environ.get("VLLM_MARLIN_USE_ATOMIC_ADD", "0") == "1",

    # Whether to turn on the outlines cache for V0
    # This cache is unbounded and on disk, so it's not safe to use in
    # an environment with potentially malicious users.
    "VLLM_V0_USE_OUTLINES_CACHE":
    lambda: os.environ.get("VLLM_V0_USE_OUTLINES_CACHE", "0") == "1",

    # Gap between padding buckets for the forward pass. So we have
    # 8, we will run forward pass with [16, 24, 32, ...].
    "VLLM_TPU_BUCKET_PADDING_GAP":
    lambda: int(os.environ["VLLM_TPU_BUCKET_PADDING_GAP"])
    if "VLLM_TPU_BUCKET_PADDING_GAP" in os.environ else 0,

    # Allow use of DeepGemm kernels for fused moe ops.
    "VLLM_USE_DEEP_GEMM":
    lambda: bool(int(os.getenv("VLLM_USE_DEEP_GEMM", "0"))),

    # Control the cache sized used by the xgrammar compiler. The default
    # of 512 MB should be enough for roughly 1000 JSON schemas.
    # It can be changed with this variable if needed for some reason.
    "VLLM_XGRAMMAR_CACHE_MB":
    lambda: int(os.getenv("VLLM_XGRAMMAR_CACHE_MB", "512")),

    # Control the threshold for msgspec to use 'zero copy' for
    # serialization/deserialization of tensors. Tensors below
    # this limit will be encoded into the msgpack buffer, and
    # tensors above will instead be sent via a separate message.
    # While the sending side still actually copies the tensor
    # in all cases, on the receiving side, tensors above this
    # limit will actually be zero-copy decoded.
    "VLLM_MSGPACK_ZERO_COPY_THRESHOLD":
    lambda: int(os.getenv("VLLM_MSGPACK_ZERO_COPY_THRESHOLD", "256")),
}

# end-env-vars-definition


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())


def is_set(name: str):
    """Check if an environment variable is explicitly set."""
    if name in environment_variables:
        return name in os.environ
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def set_vllm_use_v1(use_v1: bool):
    if is_set("VLLM_USE_V1"):
        raise ValueError(
            "Should not call set_vllm_use_v1() if VLLM_USE_V1 is set "
            "explicitly by the user. Please raise this as a Github "
            "Issue and explicitly set VLLM_USE_V1=0 or 1.")
    os.environ["VLLM_USE_V1"] = "1" if use_v1 else "0"


def compute_hash() -> str:
    """
    WARNING: Whenever a new key is added to this environment
    variables, ensure that it is included in the factors list if
    it affects the computation graph. For example, different values
    of VLLM_PP_LAYER_PARTITION will generate different computation
    graphs, so it is included in the factors list. The env vars that
    affect the choice of different kernels or attention backends should
    also be included in the factors list.
    """
    factors: list[Any] = []

    # summarize environment variables
    def factorize(name: str):
        if __getattr__(name):
            factors.append(__getattr__(name))
        else:
            factors.append("None")

    # The values of envs may affects the computation graph.
    # TODO(DefTruth): hash all environment variables?
    # for key in environment_variables:
    #     factorize(key)
    environment_variables_to_hash = [
        "VLLM_PP_LAYER_PARTITION",
        "VLLM_MLA_DISABLE",
        "VLLM_USE_TRITON_FLASH_ATTN",
        "VLLM_USE_TRITON_AWQ",
        "VLLM_DP_RANK",
        "VLLM_DP_SIZE",
    ]
    for key in environment_variables_to_hash:
        if key in environment_variables:
            factorize(key)

    hash_str = hashlib.md5(str(factors).encode(),
                           usedforsecurity=False).hexdigest()

    return hash_str
