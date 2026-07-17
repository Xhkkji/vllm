# SPDX-License-Identifier: Apache-2.0
"""LMCache V0 下的 BaM 存储适配层。

当前阶段只做两件事：

- `put` 时：保留 LMCache 原始路径，并额外 shadow 一份到 BaM
- `get` 时：可选优先从 BaM 读取；失败再回退 LMCache 原始路径

实现上刻意拆成三层，避免把布局、BaM 读写、LMCache 包装逻辑揉在一起：

1. `LMCacheBaMPageLayout`
   只负责 KV chunk <-> BaM page 的形状转换
2. `LMCacheBaMStore`
   只负责 chunk slot 管理和 BaM 实际读写
3. `LMCacheBaMStorageManager`
   只负责把 BaM 能力接到 LMCache storage manager 生命周期里
"""

from __future__ import annotations

import atexit
import os
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

import vllm.envs as envs
from vllm.bam.lmcache_bam_direct_placement import (
    _DIRECT_FRONTIER_COL_CACHE_READY, _DIRECT_FRONTIER_COL_CONSUMABLE,
    _DIRECT_FRONTIER_COL_ERROR, _DIRECT_FRONTIER_COL_LAUNCH,
    _DIRECT_FRONTIER_COL_READ_READY, _DIRECT_FRONTIER_COL_STATUS,
    _DIRECT_FRONTIER_COL_TOTAL, BaMDirectKVPlacer,
    BaMRuntimeAttentionMetadataAttachment, BaMRuntimeDirectPlacementAttachment,
    BaMDirectPlacementBatchDescriptor,
    BaMDirectPlacementBatchStateSnapshot, BaMDirectPlacementChunkDescriptor,
    BaMDirectPlacementExecution, BaMDirectPlacementFrontierSnapshot,
    BaMDirectPlacementStateTracker,
    prepare_bam_results_for_vllm_kvcache)
from vllm.bam.lmcache_bam_kv_fast_path import LMCacheBaMKVFastPath
from vllm.bam.lmcache_bam_prefetch import (LMCacheBaMChunkReadRequest,
                                           LMCacheBaMPagePipeline)
from vllm.bam.row_store_loader import (import_bam_row_store,
                                       parse_optional_int_list)
from vllm.attention.ops.paged_attn import PagedAttention
from vllm.logger import init_logger

logger = init_logger(__name__)


# 固定使用 128KB 物理页。
# 在当前 Qwen2.5-7B、fp16、hidden_dim=512 的场景下：
# - 每 token 向量大小 = 512 * 2B = 1024B
# - 每页可放 128 个 token
# - 一个满 chunk [2, 28, 256, 512] 会被切成 112 页
BAM_PAGE_BYTES = 128 * 1024

# KV direct retrieve 目前只保留三条有工程意义的链路。
#
# 这些名字只用于日志、文档和调试判断，不直接改变数据面行为：
# - rowctx_baseline:
#     旧的稳定 rowctx batch read + host materialized placement。
# - gpu_worker_persistent_materialized:
#     当前端到端输出正确的 fast path。GPU persistent service 负责 poll/read/stage，
#     host finalize 再用已验证正确的 materialized placement 写 vLLM paged KV cache。
# - gpu_worker_persistent_one_copy:
#     最激进 one-copy 实验线。目标是让 GPU persistent service 直接写最终
#     vLLM paged KV cache；当前仍允许带 correctness repair/verify。
_DIRECT_PIPELINE_ROWCTX_BASELINE = "rowctx_baseline"
_DIRECT_PIPELINE_GPU_WORKER_PERSISTENT_MATERIALIZED = (
    "gpu_worker_persistent_materialized")
_DIRECT_PIPELINE_GPU_WORKER_PERSISTENT_ONE_COPY = (
    "gpu_worker_persistent_one_copy")

# finalize mode 是当前函数内部的稳定分叉名；pipeline name 是面向日志和文档的
# 外部语义名。保留这层映射，可以避免后续再把“当前正确 fast path”误叫成
# legacy/materialized fallback。
_DIRECT_FINALIZE_MODE_RESULTS_MATERIALIZED = "results_materialized"
_DIRECT_FINALIZE_MODE_RUNTIME_DIRECT = "runtime_direct"


def _extract_chunk_hash(key: Any) -> str:
    chunk_hash = getattr(key, "chunk_hash", None)
    if not isinstance(chunk_hash, str) or chunk_hash == "":
        raise ValueError(f"Invalid LMCache key chunk_hash: {chunk_hash!r}")
    return chunk_hash


def _resolve_slot_tokens(observed_num_tokens: int) -> int:
    """为 BaM 槽位选择稳定的 token 容量。

    这里的“槽位”指一个 LMCache chunk 在 BaM 里预留的 token 上限。
    当前主线里我们更希望它稳定，而不是严格等于某一次输入里看到的
    token 数，因此优先取环境变量 `LMCACHE_CHUNK_SIZE`，再退回观测值。
    """
    configured_chunk_size = os.environ.get("LMCACHE_CHUNK_SIZE", "").strip()
    if configured_chunk_size:
        try:
            return max(int(configured_chunk_size), int(observed_num_tokens))
        except ValueError:
            logger.warning(
                "[LMCACHE_BAM] invalid LMCACHE_CHUNK_SIZE=%r; "
                "fall back to observed token count %d",
                configured_chunk_size,
                observed_num_tokens,
            )
    return int(observed_num_tokens)


def _env_enabled(name: str) -> bool:
    """解析本文件内部使用的轻量实验开关。"""
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() not in ("", "0", "false", "off", "no")


@dataclass(frozen=True)
class BaMChunkMetadata:
    """记录单个 LMCache chunk 在 BaM 里的落点和还原信息。

    它只保存“怎么从 chunk 找回 BaM page”的信息，不保存真实 KV 内容。

    字段含义：
    - `slot_id`: chunk 在 BaM page cache 里的槽位编号
    - `page_offset`: 该槽位的起始 page id
    - `actual_tokens`: 真实 token 数，可能小于 slot token 数
    - `shape` / `dtype`: 还原成 LMCache tensor 时需要的原始形状和类型
    """

    slot_id: int
    page_offset: int
    actual_tokens: int
    shape: torch.Size
    dtype: torch.dtype


@dataclass(frozen=True)
class _BaMChunkWriteReadVerifySample:
    """保存 shadow write 源 chunk 的轻量抽样。

    这个对象只服务 `VLLM_BAM_WRITE_READ_VERIFY=1` 调试开关：

    - 写入 BaM 时，从 LMCache 原始 chunk 中抽少量点保存到 CPU；
    - 后续 direct retrieve 从 BaM 读回并 decode 成 dense chunk 后，再按同样
      坐标抽样比对。

    这样可以直接回答一个很关键的问题：

      `LMCache 原始 chunk -> BaM page encode/store -> BaM read/decode`

    这一段数据顺序是否保持一致。它不参与正式热路径，默认不开启。
    """

    chunk_hash: str
    shape: torch.Size
    dtype: torch.dtype
    layer_indices: tuple[int, ...]
    token_indices: tuple[int, ...]
    dim_indices: tuple[int, ...]
    values: torch.Tensor


@dataclass(frozen=True)
class LMCacheBaMPageLayout:
    """描述 LMCache KV chunk 在 BaM 中的固定页布局。

    当前这条线采用固定 128KB 页：

    - LMCache chunk 原始形状通常是 `[2, num_layers, num_tokens, hidden_dim]`
    - `2` 表示 K/V 两份数据
    - `num_tokens` 是 chunk 里的 token 数
    - `hidden_dim` 是单个 token 向量宽度

    先把 chunk pad 到固定的 `slot_num_tokens`，再切成若干 128KB page。
    所以页布局本质上是一个稳定的线性映射，不依赖某次请求的实际长度。
    """

    page_bytes: int
    num_layers: int
    hidden_dim: int
    slot_num_tokens: int
    page_token_capacity: int
    pages_per_kv_layer: int
    pages_per_chunk: int
    dtype: torch.dtype

    @classmethod
    def from_kv_shape(cls, kv_shape: torch.Size,
                      dtype: torch.dtype) -> "LMCacheBaMPageLayout":
        if len(kv_shape) != 4:
            raise ValueError(
                "LMCache KV_BLOB is expected to be [2, num_layers, num_tokens, hidden], "
                f"got shape={tuple(kv_shape)}")

        if kv_shape[0] != 2:
            raise ValueError(
                "LMCache KV_BLOB first dim must be 2 for K/V pair, "
                f"got shape={tuple(kv_shape)}")

        _, num_layers, observed_num_tokens, hidden_dim = kv_shape
        slot_num_tokens = _resolve_slot_tokens(int(observed_num_tokens))
        vector_bytes = int(hidden_dim * dtype.itemsize)
        if BAM_PAGE_BYTES % vector_bytes != 0:
            raise ValueError(
                "128KB page layout requires page_bytes divisible by per-token vector bytes: "
                f"page_bytes={BAM_PAGE_BYTES}, vector_bytes={vector_bytes}")

        page_token_capacity = int(BAM_PAGE_BYTES // vector_bytes)
        pages_per_kv_layer = int((slot_num_tokens + page_token_capacity - 1) //
                                 page_token_capacity)
        pages_per_chunk = int(2 * num_layers * pages_per_kv_layer)
        return cls(
            page_bytes=BAM_PAGE_BYTES,
            num_layers=int(num_layers),
            hidden_dim=int(hidden_dim),
            slot_num_tokens=int(slot_num_tokens),
            page_token_capacity=int(page_token_capacity),
            pages_per_kv_layer=int(pages_per_kv_layer),
            pages_per_chunk=int(pages_per_chunk),
            dtype=dtype,
        )

    def validate_tensor(self, tensor: torch.Tensor) -> None:
        if tensor.dim() != 4 or tensor.shape[0] != 2:
            raise ValueError(
                "LMCache BaM tensor is expected to be [2, num_layers, num_tokens, hidden], "
                f"got shape={tuple(tensor.shape)}")
        if tensor.shape[1] != self.num_layers:
            raise ValueError(
                f"layer count mismatch: expected {self.num_layers}, got {tensor.shape[1]}")
        if tensor.shape[3] != self.hidden_dim:
            raise ValueError(
                f"hidden dim mismatch: expected {self.hidden_dim}, got {tensor.shape[3]}")
        if tensor.shape[2] > self.slot_num_tokens:
            raise ValueError(
                "token count overflow: "
                f"slot {self.slot_num_tokens}, got {tensor.shape[2]}")

    def encode_pages(self, tensor: torch.Tensor) -> torch.Tensor:
        """把 KV chunk 编成固定数量的 128KB page。

        输入:
          tensor: `[2, num_layers, actual_tokens, hidden_dim]`

        处理顺序:
          1. pad 到 `[2, num_layers, slot_num_tokens, hidden_dim]`
          2. 按 page_token_capacity 把 token 维切成 page
          3. 把每个 page 视作 `[128KB]` 字节块

        输出:
          `[pages_per_chunk, 128KB]`
        """
        self.validate_tensor(tensor)

        # 固定槽位大小：尾块不足时补 0。
        padded = torch.zeros(
            (2, self.num_layers, self.slot_num_tokens, self.hidden_dim),
            device=tensor.device,
            dtype=tensor.dtype,
        )
        padded[:, :, :tensor.shape[2], :].copy_(tensor, non_blocking=False)

        pages = padded.contiguous().view(
            2,
            self.num_layers,
            self.pages_per_kv_layer,
            self.page_token_capacity,
            self.hidden_dim,
        )
        pages = pages.view(-1, self.page_token_capacity * self.hidden_dim)
        pages = pages.view(torch.uint8).view(self.pages_per_chunk, -1)
        if pages.shape[1] != self.page_bytes:
            raise ValueError(
                f"page width mismatch: expected {self.page_bytes}, got {pages.shape[1]}")
        return pages

    def decode_pages(self, pages: torch.Tensor,
                     metadata: BaMChunkMetadata) -> torch.Tensor:
        """把固定页还原回真实长度的 KV chunk。

        输入:
          pages: `[pages_per_chunk, 128KB]`

        还原顺序与 `encode_pages` 相反：
          1. 先把字节页 view 回多维 page 结构
          2. 再合并回固定槽位 `slot_num_tokens`
          3. 最后裁掉 padding，只保留 `actual_tokens`

        输出:
          `[2, num_layers, actual_tokens, hidden_dim]`
        """
        expected_shape = (self.pages_per_chunk, self.page_bytes)
        if tuple(pages.shape) != expected_shape:
            raise ValueError(
                "loaded BaM pages shape mismatch: "
                f"expected {expected_shape}, got {tuple(pages.shape)}")

        full_tensor = pages.view(metadata.dtype).view(
            2,
            self.num_layers,
            self.pages_per_kv_layer,
            self.page_token_capacity,
            self.hidden_dim,
        )
        full_tensor = full_tensor.reshape(
            2, self.num_layers, self.slot_num_tokens, self.hidden_dim)
        return full_tensor[:, :, :metadata.actual_tokens, :].contiguous()


@dataclass(frozen=True)
class _InFlightDirectPlacementWave:
    """描述一次已经 launch、等待后续收口的 direct placement wave。

    这层对象刻意只保存 runtime 边界真正需要的最小信息：

    - `direct_execution`
      当前 wave 对应的执行句柄，后续通过它推进 ready / 等待 frontier。
    - `direct_placer`
      负责在收口完成后打印这次 wave 的 step timing。
    - `launch_*`
      当前 wave 真实提交给 placement 的 chunk 范围。
    - `return_target_chunks`
      当前 wave 对正常推理引擎准备暴露的连续 prefix 目标。

    这样 store 主流程就不需要再把“launch 细节”和“等待细节”揉在一个 helper
    里，后续如果继续往更长期存活的 execution 推进，也可以直接从这个对象
    继续演化。
    """

    direct_execution: BaMDirectPlacementExecution
    direct_placer: BaMDirectKVPlacer
    launched_batch: Any
    wave_name: str
    launch_start_chunk: int
    launch_chunks: int
    return_target_chunks: int


@dataclass
class _DirectPlacementRequestBootstrapProfile:
    """记录 direct placement request 在 launch 前后的基础 profile。

    这里刻意只保留“本次 request 已经发生、且未来 finalize 一定还会用到”的
    计时信息：

    - `*_ms`:
      已经完成的同步阶段耗时，例如 collect/read/prepare。
    - `*_start_time`:
      还在进行中的阶段起点，例如整个 request 总耗时、frontier wave 墙钟时间。

    后续如果把 request handle 再往更高层 runtime 提升，这份 profile 结构可以
    原样复用，而不需要再从散落的局部变量里反推一次。
    """

    collect_entries_ms: float
    read_submit_ms: float
    descriptor_ms: float
    tracker_init_ms: float
    direct_total_start_time: float
    # 由于当前 request-handle 主线已经改成：
    #   start = submit BaM read
    #   poll  = 观察 read-ready frontier
    #   finalize = consume + placement
    # 所以 prepare/placement 的时间戳都要延后到 finalize 后再有意义。
    prepare_ms: float = 0.0
    frontier_wave_start_time: float | None = None


@dataclass
class _InFlightDirectPlacementRequest:
    """描述一次已经 start、后续可被 poll/finalize 的 direct placement request。

    这层对象是“request 级控制面”的第一版句柄。它和 `_InFlightDirectPlacementWave`
    的区别是：

    - `Wave` 只描述单次 placement launch 的 runtime 边界
    - `Request` 描述一次完整 direct retrieve 的稳定上下文

    因此 request handle 会额外持有：

    - prefix 命中信息
    - 当前 request 对上层准备返回的 frontier 目标
    - finalize 阶段还会继续用到的 profile / tensors / tracker

    当前版本虽然仍然由 store 同步 finalize，但上层语义已经变成：

    ```text
    start request
      -> poll request ready state
      -> finalize request return semantics
    ```

    这样后续如果要把 handle 继续上提到 runtime，就不需要再拆一次 request 级边界。
    """

    tokens: torch.Tensor
    kv_caches: list[torch.Tensor]
    slot_mapping: torch.Tensor
    kv_read_handle: Any | None
    results: list[Any] | None
    state_tracker: BaMDirectPlacementStateTracker
    # 当前 request 级统一 frontier table。
    # 这张表的目标，是把同一个 direct retrieve request 的 frontier ABI 贯穿：
    #
    # 1. native read-ready 阶段
    # 2. placement frontier 阶段
    # 3. finalize 后的稳定 consumable 阶段
    #
    # 当前虽然还主要由 host 控制面在不同阶段更新它，但后续如果要把 frontier
    # 推进真正下放给 GPU runtime / persistent service，就可以直接复用这张表，
    # 不需要再跨阶段迁移 ABI。
    frontier_table: torch.Tensor | None
    direct_placer: BaMDirectKVPlacer
    kv_cache_dtype: str
    chunk_starts: list[int]
    keys: list[Any]
    prefix_hit_chunks: int
    prefix_hit_tokens: int
    first_wave_launch_chunks: int
    first_wave_return_target_chunks: int
    read_ready_frontier_chunks: int
    bootstrap_profile: _DirectPlacementRequestBootstrapProfile
    # 当前主数据面虽然还不直接依赖这两个字段，
    # 但诊断“写端 flat 语义”和“读端 packed 语义”是否一致时必须知道它们。
    num_kv_heads: int = 0
    head_size: int = 0
    runtime_direct_placement_attached: bool = False
    runtime_direct_placement_attachment: (
        BaMRuntimeDirectPlacementAttachment | None) = None
    runtime_attention_metadata_attached: bool = False
    runtime_attention_metadata_attachment: (
        BaMRuntimeAttentionMetadataAttachment | None) = None
    # 这两个字段用来向上层 adapter 显式发布一个更“硬”的 request 级语义：
    #
    # - 不是“某个中间 metadata_ready_flag 是否正好被前台读成 1”
    # - 而是“这次 request 是否已经由 runtime direct 主线完整收口，且 metadata
    #    workspace 可以和 ret_mask 一起被直接消费”
    #
    # 当前只要走到 cleanup-only runtime direct finalize，且这条请求确实挂上了
    # runtime attention metadata attachment，就把这层语义置为 True。
    #
    # 这样上层 adapter 的 fast path 判定就能统一绑定到 request completion 主线，
    # 不需要再额外跨到另一套 `metadata_ready_flag.item()` 的只读检查上。
    runtime_metadata_fast_path_authoritative: bool = False
    runtime_metadata_consumable_tokens: int = 0
    # runtime cleanup-only 主线下，`kv_read_handle` 会在 read consume 收口后清空，
    # 但 verify / profile / 调试日志仍可能需要访问这次 native batch 对应的
    # `request_table.pages`、submit_ms、poll_ms 等信息。
    #
    # 因此这里单独保留一份“cleanup 之后仍可只读观察”的句柄引用，避免后续调试支线
    # 再误回头读已经被置空的 `kv_read_handle`。
    runtime_cleanup_handle: Any | None = None
    # 基于 live request pages 还原出来的 dense prefix chunk tensors。
    #
    # 这份数据的语义刻意对齐此前已经验证可跑通的“两次搬运”旧路径：
    #
    #   BaM pages -> [2, num_layers, tokens, hidden]
    #
    # 当前它只服务新的 xformers `dense_prefix_workspace_consume` backend，
    # 让 prefix fallback 可以直接消费一份已经 materialize 正确的 dense KV，
    # 从而暂时绕开 paged-KV consume 语义仍未完全收敛的问题。
    materialized_prefix_chunk_tensors: tuple[torch.Tensor, ...] | None = None
    # 只在 native read 还活着的阶段用于“打一次 attach 日志”。
    # 一旦我们已经确认这次请求真的挂进了 GPU worker runtime slot，就不需要在
    # 后续每次 poll 都重复打印同一份上下文。
    native_runtime_context_logged: bool = False


@dataclass(frozen=True)
class _DirectPlacementFinalizeReadOutcome:
    """描述 finalize 阶段在“读请求收口”之后得到的稳定结果。

    这层对象刻意只保存 finalize 后续真正需要分叉判断的三件事：

    1. 当前请求最终落在哪条读收口主线
    2. 若是 cleanup-only，当前仍可用于读统计/校验的原始 handle 是谁
    3. 安全校验场景下，是否已经成功从 live request pages 构造出 expected_tensors

    这样 `_finalize_direct_placement_request()` 后续就不需要再把这些临时变量
    一路在大函数内部链式传递。
    """

    # 当前只保留两条明确命名的 finalize 主线：
    # - runtime_direct:  GPU 后台已完成 one-copy，前台只做 cleanup 收口
    # - results_materialized: 前台拿到已读 pages，再走 materialized placement
    #
    # 注意：results_materialized 不是“废弃旧路径”的同义词。它同时承载：
    # - rowctx_baseline
    # - 当前输出正确的 gpu_worker_persistent_materialized fast path
    # 真正对外区分三条链路时，应看 pipeline 名称，而不是只看 finalize mode。
    read_finalize_mode: str
    pipeline_name: str
    runtime_cleanup_handle: Any | None
    runtime_verify_expected_tensors: dict[str, torch.Tensor] | None


@dataclass(frozen=True)
class _DirectPlacementFinalizeBackendOutcome:
    """描述一次 finalize consume backend 的统一返回值。"""

    backend_name: str
    snapshot: BaMDirectPlacementBatchStateSnapshot
    place_stats: Any
    frontier_wave_ms: float
    cache_ready_log_ms: float
    # one-copy raw runtime write verify 必须在 official-write repair 前执行。
    # 外层 finalize 用这个标记避免 repair 之后再次跑同一个 verifier，导致“原始
    # GPU scatter 结果”和“repair 后结果”在日志里混淆。
    raw_runtime_write_verified: bool = False


class LMCacheBaMStore:
    """管理 LMCache chunk 在 BaM 中的槽位映射与读写。"""

    def __init__(self, row_store: Any, layout: LMCacheBaMPageLayout,
                 chunk_capacity: int, base_row_offset: int = 0) -> None:
        self.row_store = row_store
        self.layout = layout
        self.chunk_capacity = int(chunk_capacity)
        self.base_row_offset = int(base_row_offset)

        self._chunk_slots: "OrderedDict[str, int]" = OrderedDict()
        self._chunk_metadata: Dict[str, BaMChunkMetadata] = {}
        self._write_read_verify_refs: Dict[
            str, _BaMChunkWriteReadVerifySample] = {}
        self._slot_lock = threading.Lock()
        self._prefetch_pipeline: Optional[LMCacheBaMPagePipeline] = None
        # Direct placement 的 placer 需要跨请求复用，才能真正保留：
        # - Triton/JIT warmup 状态
        # - 已初始化的 KV cache pointer table
        #
        # 如果每次 direct retrieve 都临时 new 一个 placer，那么 warmup 状态会在
        # 每个请求里丢失，导致“一次性成本”反复落回 request_2 的热路径上。
        self._direct_kv_placer: Optional[BaMDirectKVPlacer] = None
        # 记录最近一次 direct placement 的 batch 状态。
        # 当前同步版本里它主要用于：
        # - 测试与日志直接观察“哪些 chunk/token 已经 ready”
        # - 为后续按需 consume / 更细粒度 ready 语义预留统一接口
        self._last_direct_placement_state_tracker: Optional[
            BaMDirectPlacementStateTracker] = None
        self._kv_fast_path: Optional[LMCacheBaMKVFastPath] = None
        self._prefetch_lock = threading.Lock()
        self._prefetch_requests: Dict[str, LMCacheBaMChunkReadRequest] = {}
        self._kv_batch_pending_keys: "OrderedDict[str, Any]" = OrderedDict()
        self._kv_batch_loaded_tensors: Dict[str, torch.Tensor] = {}
        self._closed = False

    def close(self) -> None:
        """收口当前 BaM store 持有的 KV runtime 资源。

        当前最重要的职责不是“清空所有 Python 映射”，而是确保：

        - 如果底层 GPU worker persistent service 仍在空转，就把它停掉
        - 避免进程退出时，CUDA context 还背着一个后台 kernel

        这里故意不去强行重置 metadata / slot map：
        - 这些都是当前进程内普通 Python 对象
        - 真正会把进程拖住的是底层 persistent service
        """
        if self._closed:
            return
        self._closed = True
        kv_fast_path = self._kv_fast_path
        if kv_fast_path is None:
            return
        try:
            stopped = bool(kv_fast_path.kv_store.native_runtime_stop_service_if_idle())
            if stopped:
                logger.info(
                    "[LMCACHE_BAM_RUNTIME_IDLE_STOP] source=store.close")
        except Exception:
            logger.exception(
                "[LMCACHE_BAM_RUNTIME_IDLE_STOP] failed during store.close")

    def _stop_kv_runtime_service_if_idle(
        self,
        *,
        source: str,
        reason: str,
    ) -> bool:
        """在 KV runtime 已经空闲时退役 persistent service。

        这条 helper 是 KV 主线的“安全同步边界”，只做一件事：

        - 如果仍有活跃请求，绝不停止后台 service；
        - 如果请求已经 cleanup 到 active_count=0，则停止空转的 persistent CTA；
        - 停止后，host 侧才能安全 launch materialized placement/refill kernel。

        这样可以把两种数据面显式隔离：

        - GPU persistent service：负责 BaM page read / staging / 状态推进；
        - materialized finalize：负责把已物化 pages 写入 vLLM paged KV cache。

        两者不能在当前 V100 单卡路径里无条件并发，否则会出现 service 已经把
        read 推到 CONSUMED，但后续 refill kernel 卡在 launch/schedule 的问题。
        """
        kv_fast_path = self._kv_fast_path
        if kv_fast_path is None:
            return False
        try:
            kv_store = kv_fast_path.kv_store
            service_running = bool(kv_store.native_runtime_service_running())
            active_count = int(kv_store.native_runtime_active_count())
            stopped = bool(kv_store.native_runtime_stop_service_if_idle())
            logger.info(
                "[LMCACHE_BAM_RUNTIME_IDLE_STOP] "
                "source=%s reason=%s service_running=%s "
                "active_count=%d stopped=%s",
                source,
                reason,
                str(service_running).lower(),
                active_count,
                str(stopped).lower(),
            )
            return stopped
        except Exception:
            logger.exception(
                "[LMCACHE_BAM_RUNTIME_IDLE_STOP] failed source=%s reason=%s",
                source,
                reason,
            )
            return False

    def _stop_kv_runtime_service_if_idle_for_verify_debug(self) -> bool:
        """verify 调试路径复用通用 idle-stop 边界。

        这里保留旧 helper 名称，避免调试分支四处改动；实际语义已经收束到
        `_stop_kv_runtime_service_if_idle()`。
        """
        return self._stop_kv_runtime_service_if_idle(
            source="verify_debug",
            reason="read_final_kv_cache_for_debug",
        )

    def _stop_kv_runtime_service_if_idle_for_one_copy_dense_refill(self) -> bool:
        """one-copy correctness repair 进入 materialized refill 前的安全边界。

        当前 one-copy 仍处在 correctness-first 收敛阶段：GPU persistent service
        已负责 submit/poll/read，但最后为了和已知正确的 materialized 路径对齐，
        需要从 live request pages 还原一份 dense prefix，再用 vLLM 官方写端
        覆盖 paged KV cache。

        这份 dense prefix 现在复用 materialized 路径同一套 Triton refill kernel。
        在 V100 上，如果 persistent service 已经 idle 但仍常驻运行，前台再 launch
        refill kernel 可能被空转 service 拖住。因此这里明确只在 active_count=0
        时停止 service；如果还有活跃 request，底层会拒绝停止，主线不会误杀。
        """
        return self._stop_kv_runtime_service_if_idle(
            source="one_copy_dense_refill",
            reason="prepare_dense_prefix_with_materialized_refill",
        )

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @classmethod
    def from_kv_shape(cls, kv_shape: torch.Size,
                      dtype: torch.dtype) -> "LMCacheBaMStore":
        # 这里根据一次 chunk 的真实 shape 推导出整条 BaM 路径需要的页数。
        # 例如 Qwen2.5-7B fp16 常见形状 `[2, 28, 256, 512]`：
        #   512 * 2B = 1024B/token
        #   128KB/page => 每页 128 token
        #   256 token 需要 2 页
        #   每个 chunk 总页数 = 2(K/V) * 28(layer) * 2(page/layer) = 112
        layout = LMCacheBaMPageLayout.from_kv_shape(kv_shape, dtype)
        chunk_capacity = int(envs.VLLM_BAM_LMCACHE_SHADOW_CHUNKS)
        base_row_offset = int(envs.VLLM_BAM_LMCACHE_BASE_ROW_OFFSET)
        num_rows = int(base_row_offset + layout.pages_per_chunk * chunk_capacity)
        read_mode = envs.VLLM_BAM_LMCACHE_READ_MODE
        if read_mode == "sync":
            # 当前 baseline 需要稳定同步路径。
            # 这里显式设置 `GIDS_FORCE_SYNC_READ=1`，让 BaM row store 先走
            # 同步 read_feature 路径，避免异步 rowctx 的不稳定因素混进对比。
            os.environ["GIDS_FORCE_SYNC_READ"] = "1"
        elif read_mode == "prefetch":
            # `prefetch` 走 page-level submit/poll/complete/refill 中间层，
            # 底层复用 BaM rowctx；这里不能再强制同步 read_feature。
            os.environ.pop("GIDS_FORCE_SYNC_READ", None)
        else:
            raise ValueError(
                "VLLM_BAM_LMCACHE_READ_MODE must be sync or prefetch, "
                f"got {read_mode!r}")

        bam_row_store_cls = import_bam_row_store()
        ssd_list = parse_optional_int_list(envs.VLLM_BAM_SSD_LIST)
        row_store = bam_row_store_cls(
            row_bytes=layout.page_bytes,
            num_rows=num_rows,
            cache_size_mb=envs.VLLM_BAM_CACHE_SIZE_MB,
            num_ssd=envs.VLLM_BAM_NUM_SSD,
            ssd_list=ssd_list,
            ctrl_idx=envs.VLLM_BAM_CTRL_IDX,
        )

        logger.info(
            "[LMCACHE_BAM] initialized page_bytes=%d num_rows=%d "
            "pages_per_chunk=%d chunk_capacity=%d num_layers=%d "
            "slot_num_tokens=%d hidden_dim=%d page_token_capacity=%d "
            "pages_per_kv_layer=%d cache_size_mb=%d num_ssd=%d ssd_list=%s "
            "ctrl_idx=%d read_mode=%s gids_force_sync_read=%s "
            "base_row_offset=%d kv_fast_path=%s",
            layout.page_bytes,
            num_rows,
            layout.pages_per_chunk,
            chunk_capacity,
            layout.num_layers,
            layout.slot_num_tokens,
            layout.hidden_dim,
            layout.page_token_capacity,
            layout.pages_per_kv_layer,
            envs.VLLM_BAM_CACHE_SIZE_MB,
            envs.VLLM_BAM_NUM_SSD,
            ssd_list,
            envs.VLLM_BAM_CTRL_IDX,
            read_mode,
            os.environ.get("GIDS_FORCE_SYNC_READ", ""),
            base_row_offset,
            envs.VLLM_BAM_KV_FAST_PATH,
        )
        return cls(row_store=row_store,
                   layout=layout,
                   chunk_capacity=chunk_capacity,
                   base_row_offset=base_row_offset)

    def _get_or_assign_slot(self, chunk_hash: str) -> int:
        with self._slot_lock:
            slot_id = self._chunk_slots.get(chunk_hash)
            if slot_id is not None:
                self._chunk_slots.move_to_end(chunk_hash)
                return slot_id

            if len(self._chunk_slots) >= self.chunk_capacity:
                evicted_chunk_hash, evicted_slot_id = self._chunk_slots.popitem(
                    last=False)
                self._chunk_metadata.pop(evicted_chunk_hash, None)
                self._write_read_verify_refs.pop(evicted_chunk_hash, None)
                logger.info(
                    "[LMCACHE_BAM] evict oldest chunk slot chunk_hash=%s slot=%d",
                    evicted_chunk_hash[:16],
                    evicted_slot_id,
                )
                slot_id = evicted_slot_id
            else:
                slot_id = len(self._chunk_slots)

            self._chunk_slots[chunk_hash] = slot_id
            return slot_id

    def _lookup_metadata(self, chunk_hash: str) -> Optional[BaMChunkMetadata]:
        with self._slot_lock:
            metadata = self._chunk_metadata.get(chunk_hash)
            if metadata is not None and chunk_hash in self._chunk_slots:
                self._chunk_slots.move_to_end(chunk_hash)
            return metadata

    def get_chunk_metadata(self, key: Any) -> Optional[BaMChunkMetadata]:
        return self._lookup_metadata(_extract_chunk_hash(key))

    @staticmethod
    def _verify_sample_indices(length: int, candidates: tuple[int, ...],
                               *, fallback_count: int) -> tuple[int, ...]:
        """生成稳定抽样下标，覆盖开头、page 边界和尾部。

        这里不用随机采样，是为了让同一次问题在多轮运行里能稳定复现。
        """
        valid: list[int] = []
        for index in candidates:
            if 0 <= int(index) < int(length) and int(index) not in valid:
                valid.append(int(index))
        cursor = 0
        while len(valid) < fallback_count and cursor < int(length):
            if cursor not in valid:
                valid.append(cursor)
            cursor += 1
        return tuple(valid)

    def _build_write_read_verify_sample(
        self,
        *,
        chunk_hash: str,
        tensor: torch.Tensor,
    ) -> _BaMChunkWriteReadVerifySample:
        """从 LMCache 原始 chunk 中抽取一小块 CPU reference。

        抽样点特意包含 token 127/128/255 一类边界位置。对于当前 128KB page
        布局，Qwen2.5-7B fp16 的一个 page 正好容纳 128 个 token，所以这些
        位置可以帮助我们快速识别“page 顺序错位”。
        """
        if tensor.dim() != 4:
            raise ValueError(
                "write/read verify expects LMCache tensor shape "
                f"[2, layers, tokens, hidden], got {tuple(tensor.shape)}")

        _, num_layers, actual_tokens, hidden_dim = tensor.shape
        layer_indices = self._verify_sample_indices(
            int(num_layers),
            (0, 1, int(num_layers) - 1),
            fallback_count=min(2, int(num_layers)),
        )
        token_indices = self._verify_sample_indices(
            int(actual_tokens),
            (0, 1, 2, 127, 128, int(actual_tokens) - 2,
             int(actual_tokens) - 1),
            fallback_count=min(4, int(actual_tokens)),
        )
        dim_indices = self._verify_sample_indices(
            int(hidden_dim),
            (0, 1, 2, 3, 7, 15, 63, 127, int(hidden_dim) - 1),
            fallback_count=min(8, int(hidden_dim)),
        )

        layer_index_tensor = torch.tensor(
            layer_indices, device=tensor.device, dtype=torch.long)
        token_index_tensor = torch.tensor(
            token_indices, device=tensor.device, dtype=torch.long)
        dim_index_tensor = torch.tensor(
            dim_indices, device=tensor.device, dtype=torch.long)

        # 只保留很小的 CPU 样本，不把完整 KV chunk 留在 Python 侧。
        values = tensor.index_select(1, layer_index_tensor).index_select(
            2, token_index_tensor).index_select(3, dim_index_tensor)
        return _BaMChunkWriteReadVerifySample(
            chunk_hash=chunk_hash,
            shape=torch.Size(tensor.shape),
            dtype=tensor.dtype,
            layer_indices=layer_indices,
            token_indices=token_indices,
            dim_indices=dim_indices,
            values=values.detach().cpu().clone(),
        )

    def _record_write_read_verify_reference(self, chunk_hash: str,
                                            tensor: torch.Tensor) -> None:
        """记录写入 BaM 前的源 chunk 抽样。"""
        try:
            sample = self._build_write_read_verify_sample(
                chunk_hash=chunk_hash,
                tensor=tensor,
            )
            with self._slot_lock:
                self._write_read_verify_refs[chunk_hash] = sample
            logger.info(
                "[LMCACHE_BAM_WRITE_READ_VERIFY_REF] chunk_hash=%s "
                "shape=%s layers=%s tokens=%s dims=%s",
                chunk_hash[:16],
                tuple(sample.shape),
                sample.layer_indices,
                sample.token_indices,
                sample.dim_indices,
            )
        except Exception:
            logger.exception(
                "[LMCACHE_BAM_WRITE_READ_VERIFY_REF_FAIL] chunk_hash=%s",
                chunk_hash[:16],
            )

    def _verify_decoded_chunk_against_write_reference(
        self,
        *,
        chunk_hash: str,
        decoded_tensor: torch.Tensor,
    ) -> bool:
        """核对 BaM 读回 decode 后的 chunk 是否等于 shadow write 源数据。

        这一步是当前排查里最关键的分界线：

        - 如果这里失败，说明问题在 BaM 写入顺序、row offset、slot 复用或读回
          decode 上，后面的 paged KV / xformers 都只是消费了错误源数据。
        - 如果这里通过，说明 BaM 写读闭环基本正确，问题应继续往
          official write slot_mapping 或 xformers gather 方向查。
        """
        with self._slot_lock:
            sample = self._write_read_verify_refs.get(chunk_hash)
        if sample is None:
            logger.warning(
                "[LMCACHE_BAM_WRITE_READ_VERIFY_SKIP] chunk_hash=%s "
                "reason=no_write_reference",
                chunk_hash[:16],
            )
            return True

        if tuple(decoded_tensor.shape) != tuple(sample.shape):
            logger.error(
                "[LMCACHE_BAM_WRITE_READ_VERIFY_FAIL] chunk_hash=%s "
                "reason=shape_mismatch expected=%s actual=%s",
                chunk_hash[:16],
                tuple(sample.shape),
                tuple(decoded_tensor.shape),
            )
            return False

        if decoded_tensor.is_cuda and not _env_enabled(
                "VLLM_BAM_WRITE_READ_VERIFY_SYNC_COMPARE"):
            # 这里不能默认做 `decoded_tensor -> CPU` 的逐值比对。
            #
            # 原因是当前 GPU worker persistent service 仍保持常驻运行；
            # 对 CUDA tensor 调 `.cpu()` 会引入一次隐式同步。在这条路径里，
            # 这个同步点可能和后台 service / 当前 stream 的生命周期互相等待，
            # 表现为“主线已经 consumable，但前台卡在 verify”。
            #
            # 因此默认只记录写端 reference 和读端 decode 边界。真要做强同步
            # 数据值对比时，再显式打开二级调试开关：
            #
            #   VLLM_BAM_WRITE_READ_VERIFY_SYNC_COMPARE=1
            logger.warning(
                "[LMCACHE_BAM_WRITE_READ_VERIFY_SKIP] chunk_hash=%s "
                "reason=gpu_tensor_sync_compare_disabled shape=%s",
                chunk_hash[:16],
                tuple(decoded_tensor.shape),
            )
            return True

        layer_index_tensor = torch.tensor(
            sample.layer_indices, device=decoded_tensor.device, dtype=torch.long)
        token_index_tensor = torch.tensor(
            sample.token_indices, device=decoded_tensor.device, dtype=torch.long)
        dim_index_tensor = torch.tensor(
            sample.dim_indices, device=decoded_tensor.device, dtype=torch.long)
        actual = decoded_tensor.index_select(1, layer_index_tensor).index_select(
            2, token_index_tensor).index_select(3, dim_index_tensor)
        actual_cpu = actual.detach().cpu()
        expected_cpu = sample.values
        if torch.equal(actual_cpu, expected_cpu):
            logger.info(
                "[LMCACHE_BAM_WRITE_READ_VERIFY_OK] chunk_hash=%s "
                "shape=%s layers=%s tokens=%s dims=%s",
                chunk_hash[:16],
                tuple(sample.shape),
                sample.layer_indices,
                sample.token_indices,
                sample.dim_indices,
            )
            return True

        diff = (actual_cpu.float() - expected_cpu.float()).abs()
        mismatch_flat = torch.nonzero(actual_cpu != expected_cpu, as_tuple=False)
        first = mismatch_flat[0].tolist() if mismatch_flat.numel() > 0 else []
        detail = ""
        if len(first) == 4:
            kv_i, layer_i, token_i, dim_i = [int(x) for x in first]
            detail = (
                f" kv={kv_i}"
                f" layer={sample.layer_indices[layer_i]}"
                f" token={sample.token_indices[token_i]}"
                f" dim={sample.dim_indices[dim_i]}"
                f" expected={expected_cpu[kv_i, layer_i, token_i, dim_i].item()}"
                f" actual={actual_cpu[kv_i, layer_i, token_i, dim_i].item()}")
        logger.error(
            "[LMCACHE_BAM_WRITE_READ_VERIFY_FAIL] chunk_hash=%s "
            "reason=value_mismatch max_abs_diff=%s first_mismatch=%s%s",
            chunk_hash[:16],
            diff.max().item() if diff.numel() > 0 else "n/a",
            first,
            detail,
        )
        return False

    def register_existing_chunk(self, key: Any, *, slot_id: int,
                                page_offset: int, actual_tokens: int,
                                shape: torch.Size, dtype: torch.dtype) -> None:
        """注册已写入 SSD 的 chunk 元数据，用于新进程 cold-read。

        这个接口不触碰 BaM page cache，也不会写数据；它只恢复
        `chunk_hash -> slot_id/page_offset` 的映射，让新进程可以不依赖
        旧进程的 page cache，直接按同样的 BaM 布局去读盘。
        """
        chunk_hash = _extract_chunk_hash(key)
        slot_id = int(slot_id)
        page_offset = int(page_offset)
        if slot_id < 0:
            raise ValueError(f"slot_id must be non-negative, got {slot_id}")
        expected_page_offset = (
            self.base_row_offset + slot_id * self.layout.pages_per_chunk)
        if page_offset != expected_page_offset:
            raise ValueError(
                "page_offset must match base_row_offset + slot_id * pages_per_chunk: "
                f"slot_id={slot_id}, page_offset={page_offset}, "
                f"base_row_offset={self.base_row_offset}, "
                f"pages_per_chunk={self.layout.pages_per_chunk}")
        if slot_id >= self.chunk_capacity:
            raise ValueError(
                f"slot_id exceeds chunk_capacity: {slot_id} >= {self.chunk_capacity}")

        metadata = BaMChunkMetadata(
            slot_id=slot_id,
            page_offset=page_offset,
            actual_tokens=int(actual_tokens),
            shape=torch.Size(shape),
            dtype=dtype,
        )
        with self._slot_lock:
            self._chunk_slots[chunk_hash] = slot_id
            self._chunk_slots.move_to_end(chunk_hash)
            self._chunk_metadata[chunk_hash] = metadata
            self._write_read_verify_refs.pop(chunk_hash, None)

    def store_chunk(self, key: Any, tensor: torch.Tensor) -> None:
        # BaM shadow write 的主入口。
        # 形状变化大致是：
        #   LMCache tensor: [2, num_layers, actual_tokens, hidden_dim]
        #   pad 后 tensor: [2, num_layers, slot_tokens, hidden_dim]
        #   page bytes:     [pages_per_chunk, 128KB]
        #
        # 然后把 page bytes 作为 BaM row 写入 page cache / SSD。
        chunk_hash = _extract_chunk_hash(key)
        actual_tokens = int(tensor.shape[2])
        if _env_enabled("VLLM_BAM_WRITE_READ_VERIFY"):
            self._record_write_read_verify_reference(chunk_hash, tensor)
        pages = self.layout.encode_pages(tensor)
        slot_id = self._get_or_assign_slot(chunk_hash)
        page_offset = int(self.base_row_offset +
                          slot_id * self.layout.pages_per_chunk)

        start = time.perf_counter()
        if not pages.is_cuda:
            # LMCache 原始 buffer 常在 CPU / pinned CPU。
            # 这里显式搬到 BaM 控制 GPU，避免隐式设备迁移让调用链难读。
            pages = pages.to(device=f"cuda:{envs.VLLM_BAM_CTRL_IDX}",
                             non_blocking=False)
        self.row_store.store_rows(pages, page_offset)
        elapsed_s = time.perf_counter() - start

        metadata = BaMChunkMetadata(
            slot_id=slot_id,
            page_offset=page_offset,
            actual_tokens=actual_tokens,
            shape=torch.Size(tensor.shape),
            dtype=tensor.dtype,
        )
        with self._slot_lock:
            self._chunk_metadata[chunk_hash] = metadata

        total_bytes = int(pages.numel())
        gib_per_s = (total_bytes / elapsed_s /
                     (1024**3)) if elapsed_s > 0 else 0.0
        logger.info(
            "[LMCACHE_BAM_WRITE] chunk_hash=%s page_offset=%d actual_tokens=%d "
            "slot_tokens=%d page_count=%d page_bytes=%d total_bytes=%d "
            "elapsed_ms=%.3f bw_gib_s=%.3f",
            chunk_hash[:16],
            page_offset,
            actual_tokens,
            self.layout.slot_num_tokens,
            self.layout.pages_per_chunk,
            self.layout.page_bytes,
            total_bytes,
            elapsed_s * 1000.0,
            gib_per_s,
        )

    def load_chunk_tensor(self, key: Any) -> Optional[torch.Tensor]:
        # 这里是“同步读完整 chunk”的 baseline。
        # 路径是：
        #   chunk_hash -> metadata -> page ids -> BaM row_store.load_rows
        #   -> 128KB pages -> decode_pages -> LMCache tensor
        chunk_hash = _extract_chunk_hash(key)
        metadata = self._lookup_metadata(chunk_hash)
        if metadata is None:
            return None

        start = time.perf_counter()
        page_ids = torch.arange(
            metadata.page_offset,
            metadata.page_offset + self.layout.pages_per_chunk,
            device=f"cuda:{envs.VLLM_BAM_CTRL_IDX}",
            dtype=torch.int64,
        )
        pages = torch.empty(
            (self.layout.pages_per_chunk, self.layout.page_bytes),
            device=f"cuda:{envs.VLLM_BAM_CTRL_IDX}",
            dtype=torch.uint8,
        )
        self.row_store.load_rows(page_ids, pages)
        tensor = self.layout.decode_pages(pages, metadata)
        elapsed_s = time.perf_counter() - start

        total_bytes = int(pages.numel())
        gib_per_s = (total_bytes / elapsed_s /
                     (1024**3)) if elapsed_s > 0 else 0.0
        logger.info(
            "[LMCACHE_BAM_READ] chunk_hash=%s page_offset=%d actual_tokens=%d "
            "slot_tokens=%d page_count=%d page_bytes=%d total_bytes=%d "
            "elapsed_ms=%.3f bw_gib_s=%.3f",
            chunk_hash[:16],
            metadata.page_offset,
            metadata.actual_tokens,
            self.layout.slot_num_tokens,
            self.layout.pages_per_chunk,
            self.layout.page_bytes,
            total_bytes,
            elapsed_s * 1000.0,
            gib_per_s,
        )
        return tensor

    def load_chunk_tensor_prefetch(self, key: Any) -> Optional[torch.Tensor]:
        """用 page-level rowctx 三段式读取 chunk。

        这是 GPU-initiated 演进路线的中间层入口：

        1. CPU 根据 chunk metadata 生成 BaM page ids
        2. BaM rowctx submit/poll/get kernel 读取 128KB pages
        3. 将 pages decode 回 LMCache KV tensor

        默认主路径暂时不调用它，避免影响已有同步 baseline。
        """
        chunk_hash = _extract_chunk_hash(key)
        metadata = self._lookup_metadata(chunk_hash)
        if metadata is None:
            return None

        pipeline = self._ensure_prefetch_pipeline()
        return pipeline.load_chunk_tensor(
            chunk_hash=chunk_hash,
            metadata=metadata,
        )

    def load_chunk_tensor_kv_fast_path(self, key: Any) -> Optional[torch.Tensor]:
        """用 KVCache 专用 fast path 读取 chunk。

        这里是第 2 档路线在 vLLM 侧的入口。和 prefetch 中间层相比，它不再
        把上层接口暴露为通用 row/page 读取，而是明确表达为：

        ```text
        chunk metadata
          -> KV request descriptor
          -> BaM KV store
          -> [pages_per_chunk, 128KB]
          -> LMCache KV tensor
        ```

        当前内部仍复用 BaM rowctx；后续底层替换成 GPU worker 时，上层
        storage manager 不需要改变。
        """
        chunk_hash = _extract_chunk_hash(key)
        metadata = self._lookup_metadata(chunk_hash)
        if metadata is None:
            return None

        return self._ensure_kv_fast_path().load_chunk_tensor(
            chunk_hash=chunk_hash,
            metadata=metadata,
        )

    def _ensure_prefetch_pipeline(self) -> LMCacheBaMPagePipeline:
        """懒创建 page-level prefetch pipeline。

        真实 vLLM 路径里只有开启 `read_mode=prefetch` 时才需要它。
        这里集中创建，避免单 chunk、batch、early prefetch 各自重复初始化。
        """
        if self._prefetch_pipeline is None:
            self._prefetch_pipeline = LMCacheBaMPagePipeline(
                row_store=self.row_store,
                layout=self.layout,
                device=f"cuda:{envs.VLLM_BAM_CTRL_IDX}",
            )
        return self._prefetch_pipeline

    def _ensure_kv_fast_path(self) -> LMCacheBaMKVFastPath:
        """懒创建 KVCache 专用 fast path。

        它和 sync/prefetch 共享同一个 BaMRowStore，因此不会重复初始化 BaM
        controller/page cache。这里创建的只是一个 KV 语义适配层。
        """
        if self._kv_fast_path is None:
            self._kv_fast_path = LMCacheBaMKVFastPath(
                row_store=self.row_store,
                layout=self.layout,
                device=f"cuda:{envs.VLLM_BAM_CTRL_IDX}",
            )
        return self._kv_fast_path

    def prefetch_chunk(self, key: Any) -> bool:
        """提前提交一个 chunk 的 BaM page read 请求。

        这是“真实 vLLM 路径提前发起 IO”的关键接口，但它仍然保持保守：

        - CPU 根据 LMCache key 查 metadata，决定这个 chunk 是否值得预取
        - GPU 上生成/持有 page id 表 `[pages_per_chunk]`
        - BaM rowctx submit request，但此处不等待、不 refill
        - 后续 `load_prefetched_chunk_tensor()` 再 poll/complete/refill

        换句话说，这一步把 IO 发起时间从 `storage_manager.get(key)` 提前到
        `storage_manager.prefetch(key)`，但不会改变 LMCache retrieve 的语义。
        """
        chunk_hash = _extract_chunk_hash(key)
        metadata = self._lookup_metadata(chunk_hash)
        if metadata is None:
            return False

        with self._prefetch_lock:
            if chunk_hash in self._prefetch_requests:
                return True

        pipeline = self._ensure_prefetch_pipeline()
        request = pipeline.prepare_request(chunk_hash=chunk_hash,
                                           metadata=metadata)
        pipeline.submit_request(request)
        with self._prefetch_lock:
            self._prefetch_requests[chunk_hash] = request
        logger.info(
            "[LMCACHE_BAM_EARLY_PREFETCH] submitted chunk_hash=%s "
            "page_offset=%d page_count=%d",
            chunk_hash[:16],
            request.plan.page_offset,
            request.plan.page_count,
        )
        return True

    def load_prefetched_chunk_tensor(self, key: Any) -> Optional[torch.Tensor]:
        """消费 `prefetch_chunk()` 已提交的请求并还原成 LMCache tensor。

        如果没有 outstanding request，返回 None，让调用方走原来的 blocking
        `load_chunk_tensor_prefetch()`。这样 early prefetch 只是优化机会，
        不会成为正确性前提。
        """
        chunk_hash = _extract_chunk_hash(key)
        with self._prefetch_lock:
            request = self._prefetch_requests.pop(chunk_hash, None)
        if request is None:
            return None

        pipeline = self._ensure_prefetch_pipeline()
        pipeline.wait_request(request)

        # refill 仍在这里执行，因为 LMCache retrieve 需要拿到完整 MemoryObj。
        # 后续如果要进一步靠近 attention，可以把 refill 改成直接回填 paged KV。
        start = time.perf_counter()
        tensor = pipeline.refill_request(request)
        refill_ms = (time.perf_counter() - start) * 1000.0
        prefetched = request.prefetched
        if prefetched is None:
            raise RuntimeError(
                "BaM early-prefetch request completed without pages")

        total_ms = (time.perf_counter() - request.handle.total_start_s
                    ) * 1000.0 if request.handle is not None else refill_ms
        elapsed_s = total_ms / 1000.0
        gib_per_s = (request.plan.total_bytes / elapsed_s /
                     (1024**3)) if elapsed_s > 0 else 0.0
        logger.info(
            "[LMCACHE_BAM_EARLY_PREFETCH_CONSUME] chunk_hash=%s "
            "page_offset=%d page_count=%d submit_ms=%.3f poll_ms=%.3f "
            "poll_iters=%d get_ms=%.3f prefetch_ms=%.3f refill_ms=%.3f "
            "total_ms=%.3f bw_gib_s=%.3f",
            chunk_hash[:16],
            request.plan.page_offset,
            request.plan.page_count,
            request.handle.submit_ms if request.handle is not None else 0.0,
            prefetched.stats.poll_ms,
            prefetched.stats.poll_iters,
            prefetched.stats.get_ms,
            prefetched.stats.total_ms,
            refill_ms,
            total_ms,
            gib_per_s,
        )
        return tensor

    def load_chunk_tensors_prefetch_batch(
        self,
        keys: list[Any],
    ) -> dict[str, torch.Tensor]:
        """批量走 page-level rowctx 读取多个 chunk。

        这个接口给实验 replay 使用，不改变真实 vLLM/LMCache 的默认调用方式。

        数据流程：
          1. CPU 根据每个 key 查到 `BaMChunkMetadata`
          2. pipeline 为每个 chunk 生成 GPU page id 表 `[112]`
          3. 批量 submit 到 BaM rowctx，按 FIFO poll/complete
          4. 每个 chunk 的 `[112, 128KB]` pages 再 refill 回
             `[2, num_layers, actual_tokens, hidden_dim]`

        返回值用 `chunk_hash -> tensor`，调用方可以按原始 read 顺序取结果。
        """
        items: list[tuple[str, BaMChunkMetadata]] = []
        for key in keys:
            chunk_hash = _extract_chunk_hash(key)
            metadata = self._lookup_metadata(chunk_hash)
            if metadata is None:
                raise KeyError(f"BaM chunk not found: {chunk_hash}")
            items.append((chunk_hash, metadata))

        if not items:
            return {}

        return self._ensure_prefetch_pipeline().load_chunk_tensors_batch(items)

    def load_chunk_tensors_kv_fast_path_batch(
        self,
        keys: list[Any],
    ) -> dict[str, torch.Tensor]:
        """批量走 KVCache 专用 fast path。

        这是第 2 档路线的第一阶段 microbench 入口。它和已有
        `load_chunk_tensors_prefetch_batch()` 的输出一致，但内部上层接口已经
        换成 KV descriptor：

        ```text
        [key, ...]
          -> [(chunk_hash, metadata), ...]
          -> [BaMKVRequest, ...]
          -> [chunk, pages_per_chunk, 128KB]
          -> {chunk_hash: KV tensor}
        ```
        """
        items: list[tuple[str, BaMChunkMetadata]] = []
        for key in keys:
            chunk_hash = _extract_chunk_hash(key)
            metadata = self._lookup_metadata(chunk_hash)
            if metadata is None:
                raise KeyError(f"BaM chunk not found: {chunk_hash}")
            items.append((chunk_hash, metadata))

        if not items:
            return {}

        return self._ensure_kv_fast_path().load_chunk_tensors_batch(items)

    def load_chunk_tensors_kv_fast_path_direct_batch(
        self,
        keys: list[Any],
    ) -> dict[str, torch.Tensor]:
        """通过 direct-cache-load 旁路批量读取 chunk，并还原成 KV tensor。

        与 `load_chunk_tensors_kv_fast_path_batch()` 的区别：

        - 不创建新的 native KV batch request
        - 不进入 submit/poll/consume/runtime 生命周期
        - 只基于当前 metadata/page ids 直接同步读取 BaM cache 可见页

        因此这条接口最适合放在“主链已经有一条 persistent request 正在或刚刚
        完成”的调试校验场景里，用来安全地构造期望 tensor。
        """
        items: list[tuple[str, BaMChunkMetadata]] = []
        for key in keys:
            chunk_hash = _extract_chunk_hash(key)
            metadata = self._lookup_metadata(chunk_hash)
            if metadata is None:
                raise KeyError(f"BaM chunk not found: {chunk_hash}")
            items.append((chunk_hash, metadata))

        if not items:
            return {}

        return self._ensure_kv_fast_path().load_chunk_tensors_direct_batch(
            items)

    def load_chunk_tensors_kv_fast_path_from_live_request_pages(
        self,
        request_handle: Any,
        *,
        max_chunks: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """复用当前 live request 的 pages buffer，还原期望 chunk tensor。"""
        return self._ensure_kv_fast_path(
        ).load_chunk_tensors_from_live_request_pages(
            request_handle,
            max_chunks=max_chunks,
        )

    @staticmethod
    def _find_first_tensor_mismatch(
        actual_tensor: torch.Tensor,
        expected_tensor: torch.Tensor,
    ) -> tuple[list[int], float] | None:
        """返回第一处不一致位置及最大绝对误差。

        当前这个 helper 只服务 runtime direct placement verify 调试路径。

        因此这里刻意把 compare 放到 CPU 上做，而不是继续在 GPU 上做：
        - 避免 persistent service 常驻时，调试 compare 自己再次发起额外 CUDA
          kernel 干扰观察结果
        - 保持正式数据面完全不变，只让调试路径更稳定地给出 mismatch 结论
        """
        actual_cpu = actual_tensor.detach().to(device="cpu")
        expected_cpu = expected_tensor.detach().to(device="cpu")
        if torch.equal(actual_cpu, expected_cpu):
            return None
        mismatch_mask = actual_cpu.ne(expected_cpu)
        mismatch_pos = mismatch_mask.nonzero(as_tuple=False)
        first_pos = mismatch_pos[0].detach().cpu().tolist()
        max_abs = float(
            (actual_cpu.float() - expected_cpu.float()).abs().max().item())
        return first_pos, max_abs

    @staticmethod
    def _find_first_tensor_mismatch_in_sample_cpu_debug(
        actual_tensor: torch.Tensor,
        expected_tensor: torch.Tensor,
        *,
        sample_tokens: int,
        sample_heads: int,
        sample_dims: int,
    ) -> tuple[list[int], float, float, float] | None:
        """仅在 CPU 上比较一个很小的 packed 样本切片。

        这条 helper 只服务 runtime direct placement verify 调试路径，目标是：

        1. 不再像之前那样把整块 packed tensor 都搬回 CPU
        2. 先用很小的样本快速判断“最终 cache 里是不是已经明显写错”
        3. 若样本已经错，立即给出精确 token/head/dim 与实际值/期望值

        之所以默认只验一个很小的样本，而不是直接做 full compare，是因为
        当前 persistent service 常驻时，我们首先需要把“正式主链是否可跑”
        和“verify 调试本身是否过重”这两件事拆开。
        """
        capped_tokens = min(max(int(sample_tokens), 1), int(actual_tensor.shape[0]))
        capped_heads = min(max(int(sample_heads), 1), int(actual_tensor.shape[1]))
        capped_dims = min(max(int(sample_dims), 1), int(actual_tensor.shape[2]))
        actual_sample = actual_tensor[:capped_tokens, :capped_heads,
                                      :capped_dims].detach().to(device="cpu")
        expected_sample = expected_tensor[:capped_tokens, :capped_heads,
                                          :capped_dims].detach().to(device="cpu")
        if torch.equal(actual_sample, expected_sample):
            return None
        mismatch_mask = actual_sample.ne(expected_sample)
        mismatch_pos = mismatch_mask.nonzero(as_tuple=False)
        first_pos = mismatch_pos[0].detach().cpu().tolist()
        token_idx = int(first_pos[0])
        head_idx = int(first_pos[1])
        dim_idx = int(first_pos[2])
        actual_value = float(actual_sample[token_idx, head_idx, dim_idx].item())
        expected_value = float(
            expected_sample[token_idx, head_idx, dim_idx].item())
        max_abs = float((actual_sample.float() - expected_sample.float()).abs().
                        max().item())
        return first_pos, max_abs, actual_value, expected_value

    @staticmethod
    def _build_packed_verify_slot_indices_cpu_debug(
        *,
        slot_slice: torch.Tensor,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """仅供 verify 调试路径使用：在 CPU 上组织 packed gather 索引。

        这里返回的就是 CPU tensor，而不是再拷回 GPU。

        当前 verify 的目标已经进一步收窄成：
        - 不再发任何新的 GPU gather/helper kernel
        - 只在 CPU 上观察一个很小的 packed 样本

        因此这一步只负责把：
          slot -> (block_id, block_offset)
        在 CPU 上算好，后续 sample compare 会继续完全留在 CPU。

        原因是当前 persistent service 常驻时，即使只是很小的：
        - `torch.div(slot_slice, block_size)`
        - `torch.remainder(slot_slice, block_size)`

        这类辅助 CUDA kernel 也可能和后台 service 相互干扰，导致我们定位不了
        真正的数据面问题。

        因此这里把“slot -> (block_id, block_offset)”的索引组织显式下沉到 CPU：
        - CPU 负责算很小的一组索引
        - verify sample compare 后续也继续留在 CPU
        - 这样就不会再被这些辅助小 kernel 污染观察结果
        """
        slot_slice_host = slot_slice.detach().to(device="cpu", dtype=torch.long)
        slot_blocks_host = torch.div(
            slot_slice_host,
            block_size,
            rounding_mode="floor",
        ).to(torch.long)
        slot_offsets_host = torch.remainder(
            slot_slice_host,
            block_size,
        ).to(torch.long)
        return (
            slot_blocks_host,
            slot_offsets_host,
        )

    @staticmethod
    def _extract_packed_verify_sample_cpu_debug(
        *,
        layer_cache: torch.Tensor,
        expected_layer: torch.Tensor,
        slot_blocks_host: torch.Tensor,
        slot_offsets_host: torch.Tensor,
        num_kv_heads: int,
        head_size: int,
        pack_size: int,
        block_size: int,
        sample_tokens: int,
        sample_heads: int,
        sample_dims: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """仅供 verify 调试路径使用：在 CPU 上还原一个极小写入样本。

        当前 one-copy 写端已经统一到 LMCache 官方
        `multi_layer_kv_transfer(direction=false)` 的物理写入语义：

        ```text
        layer_cache[kv, slot_id, hidden]
        ```

        因此 verifier 也必须按同一套 flat paged-buffer ABI 读回。这里保留
        `slot_blocks_host/slot_offsets_host` 入参，是为了不大改调用链；实际读回
        时会先恢复 `slot_id = block * block_size + offset`，再在 CPU 上按
        `[2, page_buffer_size, num_kv_heads, head_size]` 取样比较。

        因此这条 helper 只增加极小的 D2H 连续切片拷贝，不会再向 GPU
        提交额外的 gather helper kernel。
        """
        capped_tokens = min(max(int(sample_tokens), 1), int(slot_blocks_host.numel()))
        capped_heads = min(max(int(sample_heads), 1), int(num_kv_heads))
        capped_dims = min(max(int(sample_dims), 1), int(head_size))

        expected_key_cpu = expected_layer[0].reshape(
            int(expected_layer.shape[1]),
            num_kv_heads,
            head_size,
        )[:capped_tokens, :capped_heads, :capped_dims].detach().to(device="cpu")
        expected_value_cpu = expected_layer[1].reshape(
            int(expected_layer.shape[1]),
            num_kv_heads,
            head_size,
        )[:capped_tokens, :capped_heads, :capped_dims].detach().to(device="cpu")

        actual_key_cpu = torch.empty_like(expected_key_cpu)
        actual_value_cpu = torch.empty_like(expected_value_cpu)

        unique_block_rows: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for token_idx in range(capped_tokens):
            block_id = int(slot_blocks_host[token_idx].item())
            if block_id in unique_block_rows:
                continue
            # 只把 sample 涉及到的 block 拷到 CPU，避免 verify 变成新的性能路径。
            # block row 的底层布局与官方 transfer 一致：
            #   [2, block_size, num_kv_heads, head_size]
            block_row_cpu = layer_cache[:, block_id].detach().to(
                device="cpu").view(2, block_size, num_kv_heads, head_size)
            unique_block_rows[block_id] = (
                block_row_cpu[0],
                block_row_cpu[1],
            )

        for token_idx in range(capped_tokens):
            block_id = int(slot_blocks_host[token_idx].item())
            block_offset = int(slot_offsets_host[token_idx].item())
            key_row_cpu, value_row_cpu = unique_block_rows[block_id]
            actual_key_cpu[token_idx].copy_(
                key_row_cpu[
                    block_offset,
                    :capped_heads,
                    :capped_dims,
                ])
            actual_value_cpu[token_idx].copy_(
                value_row_cpu[
                    block_offset,
                    :capped_heads,
                    :capped_dims,
                ])

        return (
            actual_key_cpu,
            actual_value_cpu,
            expected_key_cpu,
            expected_value_cpu,
            len(unique_block_rows),
        )

    def _run_host_reference_packed_verify_sample_cpu_debug(
        self,
        *,
        in_flight_request: _InFlightDirectPlacementRequest,
        chunk_index: int,
        actual_tokens: int,
        slot_slice: torch.Tensor,
        expected_layer: torch.Tensor,
        sample_tokens: int,
        sample_heads: int,
        sample_dims: int,
        num_kv_heads: int,
        head_size: int,
        pack_size: int,
        block_size: int,
        layer_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """用 host Triton 参考写端对同一份 live request pages 做一次小样本对照。

        这条 helper 的目的非常单一：

        - runtime persistent service 已经把 live request pages 直接写进了真实
          vLLM paged KV cache
        - 现在我们怀疑“最终 packed KV cache 写端”有 bug
        - 但还需要再定责：问题究竟出在
          1. runtime device scatter
          2. 还是更上游 `live request pages` / slot mapping / packed ABI 理解

        因此这里复用“同一份 live request pages + 同一份 slot_slice”，额外走一遍
        host Triton 参考写端，写到一份临时 scratch KV cache，然后只抽一个极小的
        packed sample 做 CPU compare。

        如果 host 参考写端正确、runtime 实际写入错误，那么根因就能进一步收敛到：
          `gids_nvme.cu` 里的 runtime device scatter / 时序逻辑
        """
        kv_read_handle = (getattr(in_flight_request, "runtime_cleanup_handle",
                                  None)
                          or getattr(in_flight_request, "kv_read_handle", None))
        # 当前 request 级句柄里保存的 `kv_read_handle` 实际上就是底层 native batch
        # handle，而不是外层 `LMCacheBaMKVBatchReadRequestHandle` 包装对象。
        #
        # 因此这里兼容两种传入形态：
        # 1. 直接就是 native handle
        # 2. 外层包装对象，再通过 `.native_handle` 取到底层句柄
        native_handle = None
        if kv_read_handle is not None:
            if hasattr(kv_read_handle, "request_table"):
                native_handle = kv_read_handle
            else:
                native_handle = getattr(kv_read_handle, "native_handle", None)
        if native_handle is None or not hasattr(native_handle, "request_table"):
            raise RuntimeError(
                "host reference verify requires live native request_table pages")

        request_table = native_handle.request_table
        page_count = int(request_table.page_count)
        start = int(chunk_index) * page_count
        end = start + page_count
        pages = request_table.pages[start:end]
        if int(pages.shape[0]) != page_count:
            raise RuntimeError(
                "host reference verify pages slice mismatch: "
                f"chunk_index={chunk_index} page_count={page_count} "
                f"got={tuple(pages.shape)}")

        # 这里显式新建 scratch cache，而不是复用真实 kv_caches。
        #
        # 原因：
        # 1. verify 的职责只是定责，不应该再污染真实运行时状态；
        # 2. 我们要观察的是“同一份 pages 经 host 参考写端后会写成什么”；
        # 3. scratch cache 完全隔离后，后续 sample compare 的结论才可信。
        scratch_kv_caches = [
            torch.zeros_like(layer_cache)
            for layer_cache in in_flight_request.kv_caches
        ]
        scratch_kv_cache_pointers_gpu = torch.tensor(
            tuple(int(layer_cache.data_ptr()) for layer_cache in scratch_kv_caches),
            dtype=torch.int64,
            device=scratch_kv_caches[0].device,
        )
        # 不能调用 `_fused_pages_to_vllm_cache()`：那个 helper 为正式热路径服务，
        # 内部固定使用 direct_placer 缓存的真实 kv_cache pointer table。
        #
        # 这里是调试参考写端，必须把数据写入 scratch cache，否则后续从 scratch
        # 读回时会看到全 0，并把 host reference 误判成 mismatch。
        in_flight_request.direct_placer._launch_fused_pages_to_vllm_cache(
            pages,
            slot_slice.contiguous(),
            scratch_kv_cache_pointers_gpu,
            actual_tokens=int(actual_tokens),
            page_buffer_size=int(
                getattr(in_flight_request.direct_placer, "_page_buffer_size")),
            num_kv_heads=int(num_kv_heads),
            head_size=int(head_size),
            block_size=int(block_size),
            pack_size=int(pack_size),
        )
        torch.cuda.synchronize(device=scratch_kv_caches[0].device)

        slot_blocks_host, slot_offsets_host = (
            self._build_packed_verify_slot_indices_cpu_debug(
                slot_slice=slot_slice,
                block_size=block_size,
            ))
        return self._extract_packed_verify_sample_cpu_debug(
            layer_cache=scratch_kv_caches[layer_id],
            expected_layer=expected_layer,
            slot_blocks_host=slot_blocks_host,
            slot_offsets_host=slot_offsets_host,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            pack_size=pack_size,
            block_size=block_size,
            sample_tokens=sample_tokens,
            sample_heads=sample_heads,
            sample_dims=sample_dims,
        )

    def _run_official_paged_cache_oracle_verify_sample_cpu_debug(
        self,
        *,
        in_flight_request: _InFlightDirectPlacementRequest,
        slot_slice: torch.Tensor,
        expected_layer: torch.Tensor,
        sample_tokens: int,
        sample_heads: int,
        sample_dims: int,
        num_kv_heads: int,
        head_size: int,
        pack_size: int,
        block_size: int,
        layer_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """用 vLLM 官方 `reshape_and_cache` 作为 packed 写端 oracle。

        这条 helper 的定位比 host fused 参考更“硬”：

        - `expected_layer` 已经是从 live request pages 还原出来的 dense K/V
        - 这里不再走我们自己的 fused/Triton 写端
        - 而是直接调用 vLLM 官方 `PagedAttention.write_to_paged_cache()`

        因而它回答的是：

        ```text
        如果 dense K/V 是对的，
        那么官方 paged cache 写端写出来的 packed 结果是什么？
        ```

        后续只要：
        - 官方 oracle 正确
        - host fused 错
        - runtime direct 错

        就可以把根因彻底收敛到“我们自己实现的 packed scatter 语义”，而不是
        再怀疑 `decode_pages()` 或 packed sample 抽取 helper。
        """
        actual_tokens = int(expected_layer.shape[1])
        scratch_layer_cache = torch.zeros_like(layer_cache)
        oracle_key_cache, oracle_value_cache = PagedAttention.split_kv_cache(
            scratch_layer_cache,
            int(num_kv_heads),
            int(head_size),
        )
        dense_key = expected_layer[0].reshape(
            actual_tokens,
            num_kv_heads,
            head_size,
        ).contiguous()
        dense_value = expected_layer[1].reshape(
            actual_tokens,
            num_kv_heads,
            head_size,
        ).contiguous()
        # 当前 verify 场景的 kv_cache_dtype 仍是 `auto/fp16` 主线，
        # 因此这里用 1.0 的 scale 即可复现官方非量化写端语义。
        oracle_scale = torch.tensor(
            1.0,
            dtype=torch.float32,
            device=scratch_layer_cache.device,
        )
        PagedAttention.write_to_paged_cache(
            dense_key,
            dense_value,
            oracle_key_cache,
            oracle_value_cache,
            slot_slice.contiguous(),
            str(getattr(in_flight_request, "kv_cache_dtype", "auto")),
            oracle_scale,
            oracle_scale,
        )
        torch.cuda.synchronize(device=scratch_layer_cache.device)

        slot_blocks_host, slot_offsets_host = (
            self._build_packed_verify_slot_indices_cpu_debug(
                slot_slice=slot_slice,
                block_size=block_size,
            ))
        return self._extract_packed_verify_sample_cpu_debug(
            layer_cache=scratch_layer_cache,
            expected_layer=expected_layer,
            slot_blocks_host=slot_blocks_host,
            slot_offsets_host=slot_offsets_host,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            pack_size=pack_size,
            block_size=block_size,
            sample_tokens=sample_tokens,
            sample_heads=sample_heads,
            sample_dims=sample_dims,
        )

    def _rewrite_runtime_direct_prefix_into_paged_kv_cache_with_official_write(
        self,
        *,
        in_flight_request: _InFlightDirectPlacementRequest,
    ) -> None:
        """用 vLLM 官方写端把当前 runtime prefix 重新落到真实 paged KV cache。

        当前已经通过 verify 明确了三件事：

        1. `live request pages -> dense expected tensor` 这一步是稳定可用的
        2. vLLM 官方 `reshape_and_cache` 语义是正确的
        3. 现阶段 runtime direct placement 自己的 packed scatter 仍然有错

        因此这条 helper 先走一条 correctness-first 的收敛路径：

        - 保留 GPU persistent poll / cleanup-only runtime 主线不变
        - 只在“最终真正要把 prefix 暴露给模型计算”之前
        - 用已经 materialize 出来的 dense prefix tensor
        - 调 vLLM 官方写端，覆盖真实 paged KV cache

        这样做的边界很清楚：

        - 不回退 read/poll/runtime 生命周期
        - 不重新引入额外的 BaM 读
        - 只把当前最后一跳的 packed cache 写入收敛到官方正确语义

        后续如果把 runtime device scatter 修对了，可以再把这层去掉；但在那之前，
        它能保证：

        `GPU 负责把页读回来`
          -> `CPU 只触发一次官方 GPU cache-write kernel`
          -> `模型消费到的 paged KV cache 内容正确`
        """
        dense_chunks = in_flight_request.materialized_prefix_chunk_tensors
        if not dense_chunks:
            return

        num_kv_heads = int(in_flight_request.num_kv_heads)
        head_size = int(in_flight_request.head_size)
        if num_kv_heads <= 0 or head_size <= 0:
            raise RuntimeError(
                "runtime direct official write repair requires "
                f"valid num_kv_heads/head_size, got "
                f"{num_kv_heads}/{head_size}")

        kv_caches = in_flight_request.kv_caches
        split_layer_caches = [
            PagedAttention.split_kv_cache(
                layer_cache,
                num_kv_heads,
                head_size,
            ) for layer_cache in kv_caches
        ]
        scale = torch.tensor(
            1.0,
            dtype=torch.float32,
            device=kv_caches[0].device,
        )

        rewrite_begin = time.perf_counter()
        total_tokens = 0
        for chunk_index, dense_chunk in enumerate(dense_chunks):
            actual_tokens = int(dense_chunk.shape[2])
            if actual_tokens <= 0:
                continue
            chunk_start = int(in_flight_request.chunk_starts[chunk_index])
            slot_slice = in_flight_request.slot_mapping[
                chunk_start:chunk_start + actual_tokens
            ].contiguous()
            total_tokens += actual_tokens
            for layer_id, (key_cache, value_cache) in enumerate(split_layer_caches):
                dense_key = dense_chunk[0, layer_id].reshape(
                    actual_tokens,
                    num_kv_heads,
                    head_size,
                ).contiguous()
                dense_value = dense_chunk[1, layer_id].reshape(
                    actual_tokens,
                    num_kv_heads,
                    head_size,
                ).contiguous()
                PagedAttention.write_to_paged_cache(
                    dense_key,
                    dense_value,
                    key_cache,
                    value_cache,
                    slot_slice,
                    str(in_flight_request.kv_cache_dtype),
                    scale,
                    scale,
                )

        # 这里显式只同步当前 stream，保证官方写端已经完成；
        # 不去做整卡同步，避免把同卡上不相关工作一起纳入等待。
        torch.cuda.current_stream(device=kv_caches[0].device).synchronize()
        logger.info(
            "[LMCACHE_BAM_DIRECT_PLACEMENT_RUNTIME_OFFICIAL_WRITE_REPAIR] "
            "chunks=%d tokens=%d layers=%d rewrite_ms=%.3f",
            len(dense_chunks),
            total_tokens,
            len(kv_caches),
            (time.perf_counter() - rewrite_begin) * 1000.0,
        )
        if envs.VLLM_BAM_DIRECT_PLACEMENT_VERIFY_OFFICIAL_WRITE:
            # official-write verifier 会把最终 paged KV cache 的小样本读回 CPU。
            #
            # 这类 D2H 调试读回会触发 CUDA 同步；如果 persistent service 此时
            # 仍在空转，调试路径可能被后台 service 拖住，表现为只打印
            # VERIFY_OFFICIAL_WRITE_BEGIN 后不再前进。
            #
            # 因此这里复用 runtime-write verify 已经验证过的保护：只在调试
            # 开关打开、且当前 runtime 已经 cleanup 后，尝试停止 idle service。
            # 正式路径不开这个开关时，GPU 后台 service 仍保持原来的生命周期。
            self._stop_kv_runtime_service_if_idle_for_verify_debug()
        self._verify_official_write_repair_against_paged_kv_cache(
            in_flight_request=in_flight_request,
            split_layer_caches=split_layer_caches,
        )

    def _verify_official_write_repair_against_paged_kv_cache(
        self,
        *,
        in_flight_request: _InFlightDirectPlacementRequest,
        split_layer_caches: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """校验 official write repair 是否真的写对最终 paged KV cache。

        这条调试支线只在
        `VLLM_BAM_DIRECT_PLACEMENT_VERIFY_OFFICIAL_WRITE=1` 时启用。

        当前主线为了先保证 correctness，会把 BaM live pages 解码成 dense
        chunk，再调用 vLLM 官方 `PagedAttention.write_to_paged_cache()` 写入
        真实 paged KV cache。模型后续的 xformers fallback 也是从这份真实
        cache 里按 block table 读取 prefix KV。

        因此这里不再引入 scratch cache / host reference / 旧 runtime scatter
        等额外分支，只做最小闭环：

        1. 从 dense chunk 取少量 K/V reference；
        2. 用同一段 `slot_mapping` 反算 `(block_id, block_offset)`；
        3. 按 vLLM 官方 packed ABI 从真实 key_cache/value_cache 读回；
        4. 精确比较 sampled values。

        如果这里失败，说明 official write repair 的输入解释或 slot 坐标已经错；
        如果这里通过，后续问题就应继续查 metadata rebuild 后的 block table
        是否和写入时的 slot_mapping 对齐。
        """
        if not envs.VLLM_BAM_DIRECT_PLACEMENT_VERIFY_OFFICIAL_WRITE:
            return

        dense_chunks = in_flight_request.materialized_prefix_chunk_tensors
        if not dense_chunks:
            logger.warning(
                "[LMCACHE_BAM_DIRECT_PLACEMENT_VERIFY_OFFICIAL_WRITE_SKIP] "
                "reason=no_materialized_prefix_chunks")
            return

        num_kv_heads = int(in_flight_request.num_kv_heads)
        head_size = int(in_flight_request.head_size)
        if num_kv_heads <= 0 or head_size <= 0:
            raise RuntimeError(
                "official write verify requires num_kv_heads/head_size, got "
                f"{num_kv_heads}/{head_size}")

        max_chunks = min(
            int(envs.VLLM_BAM_DIRECT_PLACEMENT_VERIFY_OFFICIAL_WRITE_MAX_CHUNKS),
            len(dense_chunks),
        )
        max_layers = min(
            int(envs.VLLM_BAM_DIRECT_PLACEMENT_VERIFY_OFFICIAL_WRITE_MAX_LAYERS),
            len(split_layer_caches),
        )
        sample_token_cap = int(
            envs.VLLM_BAM_DIRECT_PLACEMENT_VERIFY_OFFICIAL_WRITE_SAMPLE_TOKENS)
        sample_head_cap = int(
            envs.VLLM_BAM_DIRECT_PLACEMENT_VERIFY_OFFICIAL_WRITE_SAMPLE_HEADS)
        sample_dim_cap = int(
            envs.VLLM_BAM_DIRECT_PLACEMENT_VERIFY_OFFICIAL_WRITE_SAMPLE_DIMS)

        verify_begin = time.perf_counter()
        compared_values = 0
        logger.info(
            "[LMCACHE_BAM_DIRECT_PLACEMENT_VERIFY_OFFICIAL_WRITE_BEGIN] "
            "chunks=%d verify_chunks=%d verify_layers=%d sample_tokens=%d "
            "sample_heads=%d sample_dims=%d",
            len(dense_chunks),
            max_chunks,
            max_layers,
            sample_token_cap,
            sample_head_cap,
            sample_dim_cap,
        )

        for chunk_index, dense_chunk in enumerate(dense_chunks[:max_chunks]):
            actual_tokens = int(dense_chunk.shape[2])
            if actual_tokens <= 0:
                continue
            hidden_dim = int(dense_chunk.shape[3])
            if hidden_dim != num_kv_heads * head_size:
                raise RuntimeError(
                    "official write verify hidden size mismatch: "
                    f"chunk={chunk_index} hidden_dim={hidden_dim} "
                    f"num_kv_heads={num_kv_heads} head_size={head_size}")

            chunk_start = int(in_flight_request.chunk_starts[chunk_index])
            slot_slice = in_flight_request.slot_mapping[
                chunk_start:chunk_start + actual_tokens
            ].contiguous()
            if int(slot_slice.numel()) != actual_tokens:
                raise RuntimeError(
                    "official write verify slot slice length mismatch: "
                    f"chunk={chunk_index} actual_tokens={actual_tokens} "
                    f"slot_slice={int(slot_slice.numel())}")

            # 抽样覆盖 chunk 开头、page 边界和 chunk 尾部；这些位置最容易暴露
            # chunk 顺序、128KB page 边界、slot offset 解释错误。
            token_indices = self._verify_sample_indices(
                actual_tokens,
                (0, 1, 2, 15, 16, 127, 128, actual_tokens - 2,
                 actual_tokens - 1),
                fallback_count=min(sample_token_cap, actual_tokens),
            )[:sample_token_cap]
            head_indices = self._verify_sample_indices(
                num_kv_heads,
                (0, 1, num_kv_heads - 1),
                fallback_count=min(sample_head_cap, num_kv_heads),
            )[:sample_head_cap]
            dim_indices = self._verify_sample_indices(
                head_size,
                (0, 1, 2, 3, 7, 15, 63, 127, head_size - 1),
                fallback_count=min(sample_dim_cap, head_size),
            )[:sample_dim_cap]

            token_index_tensor = torch.tensor(
                token_indices, device=slot_slice.device, dtype=torch.long)
            logger.info(
                "[LMCACHE_BAM_DIRECT_PLACEMENT_VERIFY_OFFICIAL_WRITE_STAGE] "
                "stage=slot_sample_cpu_begin chunk=%d chunk_start=%d "
                "token_indices=%s",
                chunk_index,
                chunk_start,
                token_indices,
            )
            sampled_slots_cpu = slot_slice.index_select(
                0, token_index_tensor).detach().to(
                    device="cpu", dtype=torch.long)
            logger.info(
                "[LMCACHE_BAM_DIRECT_PLACEMENT_VERIFY_OFFICIAL_WRITE_STAGE] "
                "stage=slot_sample_cpu_done chunk=%d sampled_slots=%s",
                chunk_index,
                tuple(int(x.item()) for x in sampled_slots_cpu),
            )

            for layer_id in range(max_layers):
                key_cache, value_cache = split_layer_caches[layer_id]
                block_size = int(value_cache.shape[3])
                pack_size = int(key_cache.shape[4])
                if pack_size <= 0 or head_size % pack_size != 0:
                    raise RuntimeError(
                        "official write verify invalid key cache packed view: "
                        f"layer={layer_id} head_size={head_size} "
                        f"pack_size={pack_size}")

                expected_layer_cpu = dense_chunk[:, layer_id].reshape(
                    2,
                    actual_tokens,
                    num_kv_heads,
                    head_size,
                ).detach().to(device="cpu")
                logger.info(
                    "[LMCACHE_BAM_DIRECT_PLACEMENT_VERIFY_OFFICIAL_WRITE_STAGE] "
                    "stage=expected_layer_cpu_done chunk=%d layer=%d "
                    "shape=%s",
                    chunk_index,
                    layer_id,
                    tuple(expected_layer_cpu.shape),
                )

                unique_block_rows: dict[int, tuple[torch.Tensor,
                                                   torch.Tensor]] = {}
                for slot_id_tensor in sampled_slots_cpu:
                    slot_id = int(slot_id_tensor.item())
                    if slot_id < 0:
                        raise RuntimeError(
                            "official write verify sampled a negative slot: "
                            f"chunk={chunk_index} layer={layer_id} "
                            f"slot_id={slot_id}")
                    block_id = slot_id // block_size
                    if block_id in unique_block_rows:
                        continue
                    if block_id < 0 or block_id >= int(key_cache.shape[0]):
                        raise RuntimeError(
                            "official write verify block id out of range: "
                            f"chunk={chunk_index} layer={layer_id} "
                            f"slot_id={slot_id} block_id={block_id} "
                            f"num_blocks={int(key_cache.shape[0])}")
                    logger.info(
                        "[LMCACHE_BAM_DIRECT_PLACEMENT_VERIFY_OFFICIAL_WRITE_STAGE] "
                        "stage=cache_block_cpu_begin chunk=%d layer=%d "
                        "block=%d slot=%d",
                        chunk_index,
                        layer_id,
                        block_id,
                        slot_id,
                    )
                    # 这里只拷贝 sampled token 涉及到的少量 block row 到 CPU。
                    # 这是调试路径，不进入性能主线；好处是读回公式完全透明。
                    key_row_cpu = key_cache[block_id].detach().to(
                        device="cpu").view(
                            num_kv_heads,
                            head_size // pack_size,
                            block_size,
                            pack_size,
                        )
                    value_row_cpu = value_cache[block_id].detach().to(
                        device="cpu").view(
                            num_kv_heads,
                            head_size,
                            block_size,
                        )
                    unique_block_rows[block_id] = (key_row_cpu, value_row_cpu)
                    logger.info(
                        "[LMCACHE_BAM_DIRECT_PLACEMENT_VERIFY_OFFICIAL_WRITE_STAGE] "
                        "stage=cache_block_cpu_done chunk=%d layer=%d "
                        "block=%d",
                        chunk_index,
                        layer_id,
                        block_id,
                    )

                for sample_pos, token_idx in enumerate(token_indices):
                    slot_id = int(sampled_slots_cpu[sample_pos].item())
                    block_id = slot_id // block_size
                    block_offset = slot_id % block_size
                    key_row_cpu, value_row_cpu = unique_block_rows[block_id]
                    for head_idx in head_indices:
                        for dim_idx in dim_indices:
                            x_idx = int(dim_idx) // pack_size
                            x_offset = int(dim_idx) % pack_size
                            actual_key = key_row_cpu[
                                int(head_idx), x_idx, block_offset,
                                x_offset]
                            expected_key = expected_layer_cpu[
                                0, int(token_idx), int(head_idx), int(dim_idx)]
                            if actual_key.item() != expected_key.item():
                                raise RuntimeError(
                                    "official write repair key mismatch: "
                                    f"chunk={chunk_index} layer={layer_id} "
                                    f"token={int(token_idx)} slot={slot_id} "
                                    f"block={block_id} offset={block_offset} "
                                    f"head={int(head_idx)} dim={int(dim_idx)} "
                                    f"actual={float(actual_key.item())} "
                                    f"expected={float(expected_key.item())}")

                            actual_value = value_row_cpu[
                                int(head_idx), int(dim_idx), block_offset]
                            expected_value = expected_layer_cpu[
                                1, int(token_idx), int(head_idx), int(dim_idx)]
                            if actual_value.item() != expected_value.item():
                                raise RuntimeError(
                                    "official write repair value mismatch: "
                                    f"chunk={chunk_index} layer={layer_id} "
                                    f"token={int(token_idx)} slot={slot_id} "
                                    f"block={block_id} offset={block_offset} "
                                    f"head={int(head_idx)} dim={int(dim_idx)} "
                                    f"actual={float(actual_value.item())} "
                                    f"expected={float(expected_value.item())}")
                            compared_values += 2

                logger.info(
                    "[LMCACHE_BAM_DIRECT_PLACEMENT_VERIFY_OFFICIAL_WRITE_CHUNK] "
                    "chunk=%d layer=%d tokens=%s heads=%s dims=%s "
                    "sampled_slots=%s unique_blocks=%d",
                    chunk_index,
                    layer_id,
                    token_indices,
                    head_indices,
                    dim_indices,
                    tuple(int(x.item()) for x in sampled_slots_cpu),
                    len(unique_block_rows),
                )

        logger.info(
            "[LMCACHE_BAM_DIRECT_PLACEMENT_VERIFY_OFFICIAL_WRITE_OK] "
            "verify_chunks=%d verify_layers=%d compared_values=%d "
            "elapsed_ms=%.3f",
            max_chunks,
            max_layers,
            compared_values,
            (time.perf_counter() - verify_begin) * 1000.0,
        )

    def _verify_runtime_direct_placement_write_against_materialized_chunks(
        self,
        *,
        in_flight_request: _InFlightDirectPlacementRequest,
        consumable_chunks: int,
        expected_tensors: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """核对 runtime direct placement 的最终写入结果是否正确。

        这条校验路径刻意只在调试场景下打开，目的不是替代主逻辑，而是尽快回答：

        ```text
        GPU persistent service 已经把数据直接写进 vLLM paged KV cache
          -> 这些最终写进去的值
          -> 是否和“旧 materialize 语义下的期望 chunk tensor”一致？
        ```

        之所以不直接把整条旧 placement 路径再跑一遍，是因为这里真正需要验证
        的只有一件事：最终 cache 里的目标 slot 内容对不对。

        因此这里选择一条更轻、更聚焦，同时也更安全的数据面对照方式：

        1. 只复用当前 live request 已经镜像到 `request_table.pages` 的同批 pages，
           本地还原成“旧 materialize 语义”下的期望 chunk tensor
        2. 按当前 request 真正使用的 `slot_mapping`
           从最终 `kv_caches[layer]` 里抽取对应 token slot
        3. 逐层、逐 K/V、逐 token 向量做精确比对

        一旦这里发现不一致，就说明问题已经落在：

        - runtime direct placement 的 scatter 目标
        - 或其发布时序
        - 或其 layout 解释

        而不是上层 runtime metadata / rebuild 控制面。

        这里刻意不再允许校验路径自己退回额外的 BaM 读操作。否则就会把：

        - “验证当前 request 最终写入值是否正确”
        - “旁路再读一遍底层 pages”

        重新耦合起来，并再次把主链拖回此前已经确认会卡住的
        `read_feature async kernel..` 路径。
        """
        if not envs.VLLM_BAM_DIRECT_PLACEMENT_VERIFY_RUNTIME_WRITE:
            return
        if consumable_chunks <= 0:
            return
        if expected_tensors is None:
            logger.warning(
                "[LMCACHE_BAM_DIRECT_PLACEMENT_VERIFY_RUNTIME_WRITE_SKIP] "
                "reason=missing_live_request_pages consumable_chunks=%d",
                consumable_chunks,
            )
            return

        verify_start = time.perf_counter()

        def _debug_mark_stage(stage: str) -> None:
            """只打无同步阶段日志，避免再次被 persistent service 干扰。

            之前这里尝试过 `torch.cuda.synchronize(device=...)`，但在当前
            persistent service 常驻模型下，这类整卡/整设备同步本身就会被后台
            service 拖住，反而无法继续区分“真正的读写卡点”。

            因此当前调试策略收敛为：
            - 只记录阶段 begin/done
            - 不再主动等待任何 CUDA stream / device
            - 让下一轮日志自然告诉我们：最后到底停在哪个阶段
            """
            logger.info(
                "[LMCACHE_BAM_DIRECT_PLACEMENT_VERIFY_RUNTIME_WRITE_STAGE] "
                "stage=%s",
                stage,
            )

        # 这里默认故意只跑一个很小的“快速定责版”校验窗口：
        #
        # 1. 先只看最前面的少量 consumable chunk
        # 2. 先只看最前面的少量 layer
        #
        # 目标不是一次性做完整 correctness sweep，而是尽快回答：
        # - flat 写端是不是已经错了
        # - 如果 flat 没错，packed 读侧是不是解释错了
        #
        # 当前 persistent service 会常驻占着同一张卡。若这里继续按
        # “4 chunks * 28 layers * flat+packed 全量扫”去做，调试校验本身就会
        # 变成新的阻塞源，反而看不清真正的问题。
        max_verify_chunks = min(
            int(envs.VLLM_BAM_DIRECT_PLACEMENT_VERIFY_RUNTIME_WRITE_MAX_CHUNKS),
            int(consumable_chunks),
        )
        max_verify_layers = min(
            int(envs.VLLM_BAM_DIRECT_PLACEMENT_VERIFY_RUNTIME_WRITE_MAX_LAYERS),
            int(len(in_flight_request.kv_caches)),
        )
        sample_token_limit = int(
            envs.VLLM_BAM_DIRECT_PLACEMENT_VERIFY_RUNTIME_WRITE_SAMPLE_TOKENS)
        sample_dim_limit = int(
            envs.VLLM_BAM_DIRECT_PLACEMENT_VERIFY_RUNTIME_WRITE_SAMPLE_DIMS)
        enable_full_compare = bool(
            envs.VLLM_BAM_DIRECT_PLACEMENT_VERIFY_RUNTIME_WRITE_FULL_COMPARE)
        keys_to_verify = list(in_flight_request.keys[:max_verify_chunks])
        logger.info(
            "[LMCACHE_BAM_DIRECT_PLACEMENT_VERIFY_RUNTIME_WRITE_BEGIN] "
            "chunks=%d verify_chunks=%d verify_layers=%d sample_tokens=%d "
            "sample_dims=%d full_compare=%s source=live_request_pages",
            consumable_chunks,
            max_verify_chunks,
            max_verify_layers,
            sample_token_limit,
            sample_dim_limit,
            "true" if enable_full_compare else "false",
        )

        total_chunk_tokens = 0
        compared_packed_layer_slices = 0
        packed_verify_enabled = (
            int(getattr(in_flight_request, "num_kv_heads", 0)) > 0
            and int(getattr(in_flight_request, "head_size", 0)) > 0)
        if not packed_verify_enabled:
            raise RuntimeError(
                "runtime direct placement verify now requires packed-view "
                "metadata (num_kv_heads/head_size), but it is missing")
        for chunk_index, key in enumerate(keys_to_verify):
            chunk_hash = _extract_chunk_hash(key)
            expected_chunk = expected_tensors[chunk_hash]
            actual_tokens = int(expected_chunk.shape[2])
            chunk_start = int(in_flight_request.chunk_starts[chunk_index])
            logger.info(
                "[LMCACHE_BAM_DIRECT_PLACEMENT_VERIFY_RUNTIME_WRITE_STAGE] "
                "stage=slot_slice_begin chunk_index=%d chunk_hash=%s "
                "chunk_start=%d actual_tokens=%d",
                chunk_index,
                chunk_hash,
                chunk_start,
                actual_tokens,
            )
            slot_slice = in_flight_request.slot_mapping[
                chunk_start:chunk_start + actual_tokens].to(
                    device=in_flight_request.kv_caches[0].device,
                    dtype=torch.long,
                )
            _debug_mark_stage(f"slot_slice_ready_chunk_{chunk_index}")
            logger.info(
                "[LMCACHE_BAM_DIRECT_PLACEMENT_VERIFY_RUNTIME_WRITE_STAGE] "
                "stage=slot_slice_done chunk_index=%d chunk_hash=%s "
                "slot_count=%d",
                chunk_index,
                chunk_hash,
                int(slot_slice.numel()),
            )

            if int(slot_slice.numel()) != actual_tokens:
                raise RuntimeError(
                    "runtime direct placement verify slot slice length mismatch: "
                    f"chunk_hash={chunk_hash} expected_tokens={actual_tokens} "
                    f"slot_slice={int(slot_slice.numel())}")
            # 这里不再做 `(<0).any().item()` 这类 host 标量回读检查。
            #
            # 原因不是这些检查不重要，而是当前我们正在定位：
            #   persistent service 常驻时，verify 究竟卡在哪一步
            #
            # 这类 `.item()` 会强制把 CUDA 标量同步回 host，本身就会重新引入
            # 一次隐藏同步点，导致日志停在“检查语句”而不是实际的数据访问语句上，
            # 反而模糊真正的卡点。
            #
            # 当前调试阶段更重要的是先把流程推进到：
            #   flat_index_select_begin / done
            # 再判断真正的设备访问是否有问题。

            total_chunk_tokens += actual_tokens
            # 当前最终 KV cache 的真实语义已经确认是 paged/block packed 视图，
            # 不是早期假设的 flat token-slot 视图。
            #
            # 因此这里把 verify 主线收敛成 packed-only：
            # - 直接按 block / offset / head 的真实读侧口径抽取
            # - 再和 live request pages 还原出的期望 chunk tensor 对比
            #
            # 这样能直接回答当前真正关键的问题：
            #   GPU 后台写进最终 paged KV cache 的 packed 数据对不对
            #
            # 而不会再被错误的 flat 布局假设误导。
            for layer_id, layer_cache in enumerate(
                    in_flight_request.kv_caches[:max_verify_layers]):
                expected_layer = expected_chunk[:, layer_id, :, :]
                num_kv_heads = int(in_flight_request.num_kv_heads)
                head_size = int(in_flight_request.head_size)
                hidden_dim = int(expected_layer.shape[-1])
                if hidden_dim != num_kv_heads * head_size:
                    raise RuntimeError(
                        "runtime direct placement packed-view verify hidden size "
                        "mismatch: "
                        f"chunk_hash={chunk_hash} layer={layer_id} "
                        f"hidden_dim={hidden_dim} num_kv_heads={num_kv_heads} "
                        f"head_size={head_size}")
                if layer_cache.ndim != 3 or int(layer_cache.shape[0]) != 2:
                    raise RuntimeError(
                        "runtime direct placement packed-view verify expects "
                        f"layer cache shaped [2, num_blocks, width], got={tuple(layer_cache.shape)}")

                num_blocks = int(layer_cache.shape[1])
                flattened_width = int(layer_cache.shape[2])
                if hidden_dim <= 0 or flattened_width % hidden_dim != 0:
                    raise RuntimeError(
                        "runtime direct placement packed-view verify failed to "
                        "infer block size from layer cache: "
                        f"chunk_hash={chunk_hash} layer={layer_id} "
                        f"flattened_width={flattened_width} hidden_dim={hidden_dim}")
                block_size = flattened_width // hidden_dim
                if block_size <= 0:
                    raise RuntimeError(
                        "runtime direct placement packed-view verify inferred "
                        f"invalid block_size={block_size}")

                pack_size = 16 // int(layer_cache.element_size())
                if pack_size <= 0 or head_size % pack_size != 0:
                    raise RuntimeError(
                        "runtime direct placement packed-view verify cannot "
                        "build key packed view: "
                        f"head_size={head_size} pack_size={pack_size}")

                logger.info(
                    "[LMCACHE_BAM_DIRECT_PLACEMENT_VERIFY_RUNTIME_WRITE_STAGE] "
                    "stage=packed_extract_begin chunk_index=%d layer=%d "
                    "chunk_hash=%s block_size=%d num_blocks=%d num_kv_heads=%d "
                    "head_size=%d layer_shape=%s expected_layer_shape=%s",
                    chunk_index,
                    layer_id,
                    chunk_hash,
                    block_size,
                    num_blocks,
                    num_kv_heads,
                    head_size,
                    tuple(layer_cache.shape),
                    tuple(expected_layer.shape),
                )
                # 注意：这里的 block / offset 索引组织故意放在 CPU。
                #
                # 这是一条很薄的 verify 专用调试支线，不进入正式数据面。
                # 当前目标是先稳定观察“最终 packed cache 里到底写成了什么”，
                # 因此优先避免继续在 GPU 上发起辅助小 kernel。
                slot_blocks, slot_offsets = (
                    self._build_packed_verify_slot_indices_cpu_debug(
                        slot_slice=slot_slice,
                        block_size=block_size,
                    ))
                _debug_mark_stage(
                    f"packed_indices_ready_chunk_{chunk_index}_layer_{layer_id}")

                sample_tokens = min(actual_tokens, sample_token_limit)
                sample_heads = 1
                sample_dims = min(head_size, sample_dim_limit)
                logger.info(
                    "[LMCACHE_BAM_DIRECT_PLACEMENT_VERIFY_RUNTIME_WRITE_STAGE] "
                    "stage=packed_sample_extract_begin chunk_index=%d layer=%d "
                    "chunk_hash=%s sample_tokens=%d sample_heads=%d sample_dims=%d",
                    chunk_index,
                    layer_id,
                    chunk_hash,
                    sample_tokens,
                    sample_heads,
                    sample_dims,
                )
                (packed_key_sample, packed_value_sample, expected_key_sample,
                 expected_value_sample, unique_block_count) = (
                     self._extract_packed_verify_sample_cpu_debug(
                         layer_cache=layer_cache,
                         expected_layer=expected_layer,
                         slot_blocks_host=slot_blocks,
                         slot_offsets_host=slot_offsets,
                         num_kv_heads=num_kv_heads,
                         head_size=head_size,
                         pack_size=pack_size,
                         block_size=block_size,
                         sample_tokens=sample_tokens,
                         sample_heads=sample_heads,
                         sample_dims=sample_dims,
                     ))
                _debug_mark_stage(
                    f"packed_extract_done_chunk_{chunk_index}_layer_{layer_id}")
                logger.info(
                    "[LMCACHE_BAM_DIRECT_PLACEMENT_VERIFY_RUNTIME_WRITE_STAGE] "
                    "stage=packed_sample_cpu_copy_done chunk_index=%d layer=%d "
                    "chunk_hash=%s unique_blocks=%d packed_key_sample_shape=%s "
                    "packed_value_sample_shape=%s",
                    chunk_index,
                    layer_id,
                    chunk_hash,
                    unique_block_count,
                    tuple(packed_key_sample.shape),
                    tuple(packed_value_sample.shape),
                )
                logger.info(
                    "[LMCACHE_BAM_DIRECT_PLACEMENT_VERIFY_RUNTIME_WRITE_STAGE] "
                    "stage=packed_compare_begin chunk_index=%d layer=%d "
                    "chunk_hash=%s packed_key_shape=%s packed_value_shape=%s",
                    chunk_index,
                    layer_id,
                    chunk_hash,
                    tuple(packed_key_sample.shape),
                    tuple(packed_value_sample.shape),
                )
                compared_packed_layer_slices += 1

                packed_key_sample_mismatch = (
                    self._find_first_tensor_mismatch_in_sample_cpu_debug(
                        packed_key_sample,
                        expected_key_sample,
                        sample_tokens=sample_tokens,
                        sample_heads=sample_heads,
                        sample_dims=sample_dims,
                    ))
                logger.info(
                    "[LMCACHE_BAM_DIRECT_PLACEMENT_VERIFY_RUNTIME_WRITE_STAGE] "
                    "stage=packed_key_sample_compare_done chunk_index=%d "
                    "layer=%d chunk_hash=%s mismatch=%s",
                    chunk_index,
                    layer_id,
                    chunk_hash,
                    "true" if packed_key_sample_mismatch is not None else "false",
                )
                if packed_key_sample_mismatch is not None:
                    (first_pos, max_abs, actual_value,
                     expected_value_scalar) = packed_key_sample_mismatch
                    token_idx = int(first_pos[0])
                    head_idx = int(first_pos[1])
                    dim_idx = int(first_pos[2])
                    slot_id = int(slot_slice[token_idx].detach().to(device="cpu").item())
                    host_reference_summary = "host_reference=not_run"
                    official_oracle_summary = "official_oracle=not_run"
                    try:
                        (host_key_sample, _host_value_sample,
                         host_expected_key_sample, _host_expected_value_sample,
                         host_unique_block_count) = (
                             self._run_host_reference_packed_verify_sample_cpu_debug(
                                 in_flight_request=in_flight_request,
                                 chunk_index=chunk_index,
                                 actual_tokens=actual_tokens,
                                 slot_slice=slot_slice,
                                 expected_layer=expected_layer,
                                 sample_tokens=sample_tokens,
                                 sample_heads=sample_heads,
                                 sample_dims=sample_dims,
                                 num_kv_heads=num_kv_heads,
                                 head_size=head_size,
                                 pack_size=pack_size,
                                 block_size=block_size,
                                 layer_id=layer_id,
                             ))
                        host_key_mismatch = (
                            self._find_first_tensor_mismatch_in_sample_cpu_debug(
                                host_key_sample,
                                host_expected_key_sample,
                                sample_tokens=sample_tokens,
                                sample_heads=sample_heads,
                                sample_dims=sample_dims,
                            ))
                        if host_key_mismatch is None:
                            host_reference_summary = (
                                "host_reference=match "
                                f"unique_blocks={host_unique_block_count}")
                        else:
                            (host_first_pos, host_max_abs, host_actual_value,
                             host_expected_value_scalar) = host_key_mismatch
                            host_reference_summary = (
                                "host_reference=mismatch "
                                f"token_idx={int(host_first_pos[0])} "
                                f"head={int(host_first_pos[1])} "
                                f"dim={int(host_first_pos[2])} "
                                f"actual={host_actual_value} "
                                f"expected={host_expected_value_scalar} "
                                f"max_abs={host_max_abs} "
                                f"unique_blocks={host_unique_block_count}")
                    except Exception as host_reference_exc:
                        host_reference_summary = (
                            "host_reference=error "
                            f"type={type(host_reference_exc).__name__} "
                            f"msg={host_reference_exc}")
                    try:
                        (oracle_key_sample, _oracle_value_sample,
                         oracle_expected_key_sample,
                         _oracle_expected_value_sample,
                         oracle_unique_block_count) = (
                             self._run_official_paged_cache_oracle_verify_sample_cpu_debug(
                                 in_flight_request=in_flight_request,
                                 slot_slice=slot_slice,
                                 expected_layer=expected_layer,
                                 sample_tokens=sample_tokens,
                                 sample_heads=sample_heads,
                                 sample_dims=sample_dims,
                                 num_kv_heads=num_kv_heads,
                                 head_size=head_size,
                                 pack_size=pack_size,
                                 block_size=block_size,
                                 layer_cache=layer_cache,
                             ))
                        oracle_key_mismatch = (
                            self._find_first_tensor_mismatch_in_sample_cpu_debug(
                                oracle_key_sample,
                                oracle_expected_key_sample,
                                sample_tokens=sample_tokens,
                                sample_heads=sample_heads,
                                sample_dims=sample_dims,
                            ))
                        if oracle_key_mismatch is None:
                            official_oracle_summary = (
                                "official_oracle=match "
                                f"unique_blocks={oracle_unique_block_count}")
                        else:
                            (oracle_first_pos, oracle_max_abs,
                             oracle_actual_value,
                             oracle_expected_value_scalar) = oracle_key_mismatch
                            official_oracle_summary = (
                                "official_oracle=mismatch "
                                f"token_idx={int(oracle_first_pos[0])} "
                                f"head={int(oracle_first_pos[1])} "
                                f"dim={int(oracle_first_pos[2])} "
                                f"actual={oracle_actual_value} "
                                f"expected={oracle_expected_value_scalar} "
                                f"max_abs={oracle_max_abs} "
                                f"unique_blocks={oracle_unique_block_count}")
                    except Exception as official_oracle_exc:
                        official_oracle_summary = (
                            "official_oracle=error "
                            f"type={type(official_oracle_exc).__name__} "
                            f"msg={official_oracle_exc}")
                    raise RuntimeError(
                        "runtime direct placement packed-key sample mismatch: "
                        f"chunk_hash={chunk_hash} layer={layer_id} "
                        f"token_idx={token_idx} slot_id={slot_id} "
                        f"head={head_idx} dim={dim_idx} actual={actual_value} "
                        f"expected={expected_value_scalar} max_abs={max_abs} "
                        f"{host_reference_summary} {official_oracle_summary}")

                packed_value_sample_mismatch = (
                    self._find_first_tensor_mismatch_in_sample_cpu_debug(
                        packed_value_sample,
                        expected_value_sample,
                        sample_tokens=sample_tokens,
                        sample_heads=sample_heads,
                        sample_dims=sample_dims,
                    ))
                logger.info(
                    "[LMCACHE_BAM_DIRECT_PLACEMENT_VERIFY_RUNTIME_WRITE_STAGE] "
                    "stage=packed_value_sample_compare_done chunk_index=%d "
                    "layer=%d chunk_hash=%s mismatch=%s",
                    chunk_index,
                    layer_id,
                    chunk_hash,
                    "true" if packed_value_sample_mismatch is not None else "false",
                )
                if packed_value_sample_mismatch is not None:
                    (first_pos, max_abs, actual_value,
                     expected_value_scalar) = packed_value_sample_mismatch
                    token_idx = int(first_pos[0])
                    head_idx = int(first_pos[1])
                    dim_idx = int(first_pos[2])
                    slot_id = int(slot_slice[token_idx].detach().to(device="cpu").item())
                    raise RuntimeError(
                        "runtime direct placement packed-value sample mismatch: "
                        f"chunk_hash={chunk_hash} layer={layer_id} "
                        f"token_idx={token_idx} slot_id={slot_id} "
                        f"head={head_idx} dim={dim_idx} actual={actual_value} "
                        f"expected={expected_value_scalar} max_abs={max_abs}")

                logger.info(
                    "[LMCACHE_BAM_DIRECT_PLACEMENT_VERIFY_RUNTIME_WRITE_STAGE] "
                    "stage=packed_sample_compare_done chunk_index=%d layer=%d "
                    "chunk_hash=%s",
                    chunk_index,
                    layer_id,
                    chunk_hash,
                )

                if not enable_full_compare:
                    logger.info(
                        "[LMCACHE_BAM_DIRECT_PLACEMENT_VERIFY_RUNTIME_WRITE_STAGE] "
                        "stage=packed_full_compare_skipped chunk_index=%d "
                        "layer=%d chunk_hash=%s",
                        chunk_index,
                        layer_id,
                        chunk_hash,
                    )
                    continue

                slot_blocks_device = slot_blocks.to(
                    device=layer_cache.device,
                    dtype=torch.long,
                )
                slot_offsets_device = slot_offsets.to(
                    device=layer_cache.device,
                    dtype=torch.long,
                )
                flat_cache = layer_cache.view(
                    2,
                    num_blocks * block_size,
                    num_kv_heads,
                    head_size,
                )
                flat_slots_device = (
                    slot_blocks_device * int(block_size) + slot_offsets_device)
                packed_key = flat_cache[
                    0,
                    flat_slots_device,
                    :,
                    :,
                ]
                packed_value = flat_cache[
                    1,
                    flat_slots_device,
                    :,
                    :,
                ]
                expected_key = expected_layer[0].reshape(
                    actual_tokens,
                    num_kv_heads,
                    head_size,
                )
                expected_value = expected_layer[1].reshape(
                    actual_tokens,
                    num_kv_heads,
                    head_size,
                )

                packed_key_mismatch = self._find_first_tensor_mismatch(
                    packed_key,
                    expected_key,
                )
                logger.info(
                    "[LMCACHE_BAM_DIRECT_PLACEMENT_VERIFY_RUNTIME_WRITE_STAGE] "
                    "stage=packed_key_compare_done chunk_index=%d layer=%d "
                    "chunk_hash=%s mismatch=%s",
                    chunk_index,
                    layer_id,
                    chunk_hash,
                    "true" if packed_key_mismatch is not None else "false",
                )
                if packed_key_mismatch is not None:
                    first_pos, max_abs = packed_key_mismatch
                    token_idx = int(first_pos[0])
                    head_idx = int(first_pos[1])
                    dim_idx = int(first_pos[2])
                    slot_id = int(slot_slice[token_idx].item())
                    actual_value = packed_key[token_idx, head_idx, dim_idx].item()
                    expected_value = expected_key[token_idx, head_idx,
                                                  dim_idx].item()
                    raise RuntimeError(
                        "runtime direct placement packed-key mismatch: "
                        f"chunk_hash={chunk_hash} layer={layer_id} "
                        f"token_idx={token_idx} slot_id={slot_id} "
                        f"head={head_idx} dim={dim_idx} actual={actual_value} "
                        f"expected={expected_value} max_abs={max_abs}")

                packed_value_mismatch = self._find_first_tensor_mismatch(
                    packed_value,
                    expected_value,
                )
                logger.info(
                    "[LMCACHE_BAM_DIRECT_PLACEMENT_VERIFY_RUNTIME_WRITE_STAGE] "
                    "stage=packed_value_compare_done chunk_index=%d layer=%d "
                    "chunk_hash=%s mismatch=%s",
                    chunk_index,
                    layer_id,
                    chunk_hash,
                    "true" if packed_value_mismatch is not None else "false",
                )
                if packed_value_mismatch is not None:
                    first_pos, max_abs = packed_value_mismatch
                    token_idx = int(first_pos[0])
                    head_idx = int(first_pos[1])
                    dim_idx = int(first_pos[2])
                    slot_id = int(slot_slice[token_idx].item())
                    actual_value = packed_value[token_idx, head_idx,
                                                dim_idx].item()
                    expected_value_scalar = expected_value[token_idx, head_idx,
                                                           dim_idx].item()
                    raise RuntimeError(
                        "runtime direct placement packed-value mismatch: "
                        f"chunk_hash={chunk_hash} layer={layer_id} "
                        f"token_idx={token_idx} slot_id={slot_id} "
                        f"head={head_idx} dim={dim_idx} actual={actual_value} "
                        f"expected={expected_value_scalar} max_abs={max_abs}")

        verify_ms = (time.perf_counter() - verify_start) * 1000.0
        logger.info(
            "[LMCACHE_BAM_DIRECT_PLACEMENT_VERIFY_RUNTIME_WRITE_OK] "
            "chunks=%d verify_chunks=%d verify_layers=%d chunk_tokens=%d "
            "compared_packed_layer_slices=%d packed_verify_enabled=%s "
            "elapsed_ms=%.3f",
            consumable_chunks,
            max_verify_chunks,
            max_verify_layers,
            total_chunk_tokens,
            compared_packed_layer_slices,
            str(packed_verify_enabled).lower(),
            verify_ms,
        )

    def read_chunk_pages_kv_fast_path_batch(
        self,
        keys: list[Any],
    ) -> list[Any]:
        """批量读取 BaM pages，供 direct placement 使用。

        与 `load_chunk_tensors_kv_fast_path_batch()` 的区别：

        - 不调用 Triton refill
        - 不构造 `[2, layers, tokens, hidden]` LMCache tensor
        - 返回底层 BaM pages result，让上层直接写入 vLLM paged KV cache

        这正是 Tutti/TARDIS 路线里“减少中间 tensor/rebuild”的第一步。
        """
        items: list[tuple[str, BaMChunkMetadata]] = []
        for key in keys:
            chunk_hash = _extract_chunk_hash(key)
            metadata = self._lookup_metadata(chunk_hash)
            if metadata is None:
                raise KeyError(f"BaM chunk not found: {chunk_hash}")
            items.append((chunk_hash, metadata))

        if not items:
            return []

        return self._ensure_kv_fast_path().read_chunk_pages_batch(items)

    def submit_chunk_pages_kv_fast_path_batch_request(
        self,
        keys: list[Any],
    ) -> Any:
        """显式提交一批 KV page read，并返回后续可 poll/consume 的句柄。"""
        items: list[tuple[str, BaMChunkMetadata]] = []
        for key in keys:
            chunk_hash = _extract_chunk_hash(key)
            metadata = self._lookup_metadata(chunk_hash)
            if metadata is None:
                raise KeyError(f"BaM chunk not found: {chunk_hash}")
            items.append((chunk_hash, metadata))

        if not items:
            raise ValueError(
                "submit_chunk_pages_kv_fast_path_batch_request requires keys")

        return self._ensure_kv_fast_path().submit_chunk_pages_batch_request(
            items)

    def poll_chunk_pages_kv_fast_path_batch_request(
        self,
        request_handle: Any,
    ) -> Any:
        """推进一次 KV page read handle，并返回当前 frontier 快照。"""
        return self._ensure_kv_fast_path().poll_chunk_pages_batch_request(
            request_handle)

    def get_chunk_pages_kv_fast_path_batch_request_runtime_snapshot(
        self,
        request_handle: Any,
    ) -> Any:
        """读取一批 KV native read 当前对应的 runtime 观察快照。"""
        return self._ensure_kv_fast_path(
        ).get_chunk_pages_batch_request_runtime_snapshot(request_handle)

    def attach_chunk_pages_kv_fast_path_batch_request_runtime_direct_placement(
        self,
        request_handle: Any,
        *,
        slot_mapping: torch.Tensor,
        chunk_starts: torch.Tensor,
        kv_cache_pointers_gpu: torch.Tensor,
        page_buffer_size: int,
        block_size: int,
        page_token_capacity: int,
        pages_per_kv_layer: int,
        num_layers: int,
        num_kv_heads: int,
        head_size: int,
        pack_size: int,
    ) -> bool:
        """给当前 KV native batch request 绑定设备侧 direct placement 描述符。"""
        return self._ensure_kv_fast_path(
        ).attach_chunk_pages_batch_request_runtime_direct_placement(
            request_handle,
            slot_mapping=slot_mapping,
            chunk_starts=chunk_starts,
            kv_cache_pointers_gpu=kv_cache_pointers_gpu,
            page_buffer_size=page_buffer_size,
            block_size=block_size,
            page_token_capacity=page_token_capacity,
            pages_per_kv_layer=pages_per_kv_layer,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            pack_size=pack_size,
        )

    def attach_chunk_pages_kv_fast_path_batch_request_runtime_attention_metadata(
        self,
        request_handle: Any,
        *,
        attachment: BaMRuntimeAttentionMetadataAttachment,
    ) -> bool:
        """给当前 KV native batch request 绑定设备侧 attention metadata workspace。"""
        return self._ensure_kv_fast_path(
        ).attach_chunk_pages_batch_request_runtime_attention_metadata(
            request_handle,
            attachment=attachment,
        )

    def consume_chunk_pages_kv_fast_path_batch_request(
        self,
        request_handle: Any,
        *,
        timeout_s: float | None = None,
    ) -> list[Any]:
        """消费一批已经 submit 的 KV page read。"""
        return self._ensure_kv_fast_path().consume_chunk_pages_batch_request(
            request_handle,
            timeout_s=timeout_s,
        )

    def finalize_chunk_pages_kv_fast_path_batch_request_runtime_direct_placement(
        self,
        request_handle: Any,
        *,
        timeout_s: float | None = None,
    ) -> bool:
        """对 runtime-direct-placement 请求执行 cleanup-only finalize。"""
        return self._ensure_kv_fast_path(
        ).finalize_chunk_pages_batch_request_runtime_direct_placement(
            request_handle,
            timeout_s=timeout_s,
        )

    def read_chunk_pages_kv_fast_path_direct_batch(
        self,
        keys: list[Any],
        *,
        max_chunks: int | None = None,
    ) -> list[Any]:
        """直接从当前 BaM store 读取指定前缀 chunk 子集的 pages。"""
        if max_chunks is not None:
            keys = keys[:max(int(max_chunks), 0)]
        items: list[tuple[str, BaMChunkMetadata]] = []
        for key in keys:
            chunk_hash = _extract_chunk_hash(key)
            metadata = self._lookup_metadata(chunk_hash)
            if metadata is None:
                raise KeyError(f"BaM chunk not found: {chunk_hash}")
            items.append((chunk_hash, metadata))

        if not items:
            return []
        return self._ensure_kv_fast_path().read_chunk_pages_direct_batch(items)

    def get_last_direct_placement_state_snapshot(
        self,
    ) -> Optional[BaMDirectPlacementBatchStateSnapshot]:
        """返回最近一次 direct placement 的状态快照。

        当前主线还没有把这份状态真正接到 attention consume 侧，因此这里先提供
        只读快照接口，方便：

        - 在测试里直接断言 ready 语义
        - 在联调中核对 direct placement 究竟推进到了哪个阶段
        - 后续把它接给更细粒度的 prefix consume 逻辑
        """
        if self._last_direct_placement_state_tracker is None:
            return None
        return self._last_direct_placement_state_tracker.snapshot()

    def _collect_direct_placement_entries(
        self,
        *,
        token_database: Any,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> list[tuple[int, int, Any]]:
        """收集本轮 direct placement 可由 BaM 服务的前缀 chunks。

        这里刻意把“命中判断”和“真正发起 I/O”拆开，原因有两个：

        1. 让 `LMCacheBaMStorageManager` 不需要知道 token_database 的迭代细节，
           它只负责把 direct placement 请求转交给 BaM store。
        2. 保持 LMCache prefix 语义不变。`process_tokens()` 产出的 chunk 是按前缀
           顺序排列的，一旦中间有一个 chunk 在 BaM 中缺失，就必须停止；否则
           后面的 chunk 即使存在，也不能越过缺口直接注回 vLLM KV cache。

        返回值中的三元组含义：

        ```text
        (start, end, key)
        ```

        其中 `start/end` 使用的是当前 `tokens` / `slot_mapping` 这一段局部坐标，
        后续 placement kernel 会用它把 chunk 对应的 token 范围映射回正确 slot。
        """
        entries: list[tuple[int, int, Any]] = []
        for start, end, key in token_database.process_tokens(tokens, mask):
            metadata = self.get_chunk_metadata(key)
            if metadata is None:
                # direct placement 只能消费“连续前缀命中”。
                # 因此一旦这里遇到第一个 miss，就必须立刻停止，并把断点
                # 记录下来，帮助排查“为什么前几个 chunk 能复用、最后一个
                # 不能复用”的具体位置。
                logger.info(
                    "[LMCACHE_BAM_DIRECT_PLACEMENT_PREFIX_BREAK] "
                    "prefix_chunks=%d miss_range=[%d,%d) chunk_hash=%s",
                    len(entries),
                    int(start),
                    int(end),
                    _extract_chunk_hash(key)[:16],
                )
                break
            entries.append((int(start), int(end), key))
        return entries

    def _build_direct_placement_descriptor(
        self,
        *,
        entries: list[tuple[int, int, Any]],
        results: list[Any],
    ) -> BaMDirectPlacementBatchDescriptor:
        """把本轮命中的 chunk 与 BaM 读结果收成稳定 descriptor。

        这一步和 direct placement 内部 `_build_plan()` 的职责不同：

        - `_build_plan()` 面向“当前这次真正怎么执行 placement”
        - 这里面向“这次 batch 在控制面上由哪些 chunk 组成”

        因此这里保留 chunk hash / token range / total_bytes 这些更适合状态跟踪与
        后续按需 consume 的信息，而不掺入临时的 CUDA tensor / launch 细节。
        """
        if len(entries) != len(results):
            raise ValueError(
                "direct placement descriptor requires matching entries/results: "
                f"{len(entries)} vs {len(results)}")

        chunk_descriptors: list[BaMDirectPlacementChunkDescriptor] = []
        total_tokens = 0
        total_bytes = 0
        for (start, end, key), result in zip(entries, results):
            actual_tokens = int(result.descriptor.actual_tokens)
            total_chunk_bytes = int(result.descriptor.total_bytes)
            chunk_descriptors.append(
                BaMDirectPlacementChunkDescriptor(
                    chunk_hash=_extract_chunk_hash(key),
                    chunk_start=int(start),
                    chunk_end=int(end),
                    actual_tokens=actual_tokens,
                    total_bytes=total_chunk_bytes,
                ))
            total_tokens += actual_tokens
            total_bytes += total_chunk_bytes

        return BaMDirectPlacementBatchDescriptor(
            chunks=tuple(chunk_descriptors),
            total_tokens=total_tokens,
            total_bytes=total_bytes,
        )

    def _build_direct_placement_descriptor_from_metadata(
        self,
        *,
        entries: list[tuple[int, int, Any]],
    ) -> BaMDirectPlacementBatchDescriptor:
        """仅基于 metadata 构造 direct placement descriptor。

        request-handle 主线里，`start()` 只做 BaM read submit，不会同步等待
        `results`。因此 batch 的稳定控制面 descriptor 不能再依赖“已经 consume 完的
        read result”，而应该直接由 chunk metadata 构造。
        """
        chunk_descriptors: list[BaMDirectPlacementChunkDescriptor] = []
        total_tokens = 0
        total_bytes = 0
        chunk_total_bytes = int(self.layout.pages_per_chunk *
                                self.layout.page_bytes)

        for start, end, key in entries:
            chunk_hash = _extract_chunk_hash(key)
            metadata = self.get_chunk_metadata(key)
            if metadata is None:
                raise KeyError(
                    "BaM chunk metadata disappeared while building direct "
                    f"placement descriptor: {chunk_hash}")
            actual_tokens = int(metadata.actual_tokens)
            chunk_descriptors.append(
                BaMDirectPlacementChunkDescriptor(
                    chunk_hash=chunk_hash,
                    chunk_start=int(start),
                    chunk_end=int(end),
                    actual_tokens=actual_tokens,
                    total_bytes=chunk_total_bytes,
                ))
            total_tokens += actual_tokens
            total_bytes += chunk_total_bytes

        return BaMDirectPlacementBatchDescriptor(
            chunks=tuple(chunk_descriptors),
            total_tokens=total_tokens,
            total_bytes=total_bytes,
        )

    def _log_direct_placement_state(
        self,
        *,
        stage: str,
        tracker: BaMDirectPlacementStateTracker,
    ) -> None:
        """打印当前 direct placement batch 的 ready 状态摘要。"""
        snapshot = tracker.snapshot()
        logger.info(
            "[LMCACHE_BAM_DIRECT_PLACEMENT_STATE] stage=%s chunks=%d "
            "read_ready=%d/%d staged_ready=%d/%d cache_ready=%d/%d "
            "read_tokens=%d/%d staged_tokens=%d/%d cache_tokens=%d/%d "
            "consumable_chunks=%d/%d consumable_tokens=%d/%d",
            stage,
            len(snapshot.chunk_states),
            snapshot.read_ready_chunks,
            len(snapshot.chunk_states),
            snapshot.staged_ready_chunks,
            len(snapshot.chunk_states),
            snapshot.cache_ready_chunks,
            len(snapshot.chunk_states),
            snapshot.read_ready_tokens,
            snapshot.descriptor.total_tokens,
            snapshot.staged_ready_tokens,
            snapshot.descriptor.total_tokens,
            snapshot.cache_ready_tokens,
            snapshot.descriptor.total_tokens,
            snapshot.consumable_chunks,
            len(snapshot.chunk_states),
            snapshot.consumable_tokens,
            snapshot.descriptor.total_tokens,
        )

    @staticmethod
    def _advance_direct_read_ready_frontier(
        tracker: BaMDirectPlacementStateTracker,
        *,
        target_ready_chunks: int,
    ) -> BaMDirectPlacementBatchStateSnapshot:
        """把 read-ready frontier 递增推进到目标 chunk 数。

        这里刻意只按“从 batch 开头起的连续前缀”推进，是因为当前 KV read 主线和
        上层 prefix 语义都只关心 contiguous frontier：

        - 底层 frontier_table 暴露的是 `read_ready_frontier_chunks`
        - 上层 request 也只能先消费连续前缀

        因此这层 helper 不去处理离散 chunk ready，而是把 request-handle 先稳稳
        收敛在“连续前缀”这条主线上。
        """
        tracker.mark_chunks_read_ready_upto(max(int(target_ready_chunks), 0))
        return tracker.snapshot()

    @staticmethod
    def _sync_tracker_from_native_read_frontier(
        tracker: BaMDirectPlacementStateTracker,
        *,
        poll_snapshot: Any,
    ) -> BaMDirectPlacementBatchStateSnapshot:
        """把 native KV read frontier 同步到 direct placement tracker。

        当前这层桥接刻意只同步：

        - `read_ready_frontier_chunks`

        而不会直接消费 native frontier 里的：

        - `cache_ready_frontier_chunks`
        - `consumable_frontier_chunks`

        原因是这两列目前仍然属于“BaM native read request 生命周期”的保留
        观测值，还不等价于“最终 vLLM paged KV cache 已经可被 attention 消费”。
        如果现在就把它们直接映射到 direct placement tracker，会把：

        - pages 已经读回
        - 最终 KV cache 已经就绪

        这两个语义错误混在一起。

        因此当前先维持一个清晰分层：

        - native frontier 只负责提供统一的 read-ready 连续前缀
        - placement tracker 继续独立维护 staged/cache-ready 连续前缀

        后续真的把 consume/frontier/cache-ready 全部下沉到 GPU 时，只需要扩展
        这里的映射规则，不需要再重拆 request-handle 主流程。
        """
        return LMCacheBaMStore._advance_direct_read_ready_frontier(
            tracker,
            target_ready_chunks=int(poll_snapshot.read_ready_frontier_chunks),
        )

    @staticmethod
    def _sync_tracker_from_placement_frontier(
        tracker: BaMDirectPlacementStateTracker,
        *,
        frontier_snapshot: BaMDirectPlacementFrontierSnapshot,
    ) -> BaMDirectPlacementBatchStateSnapshot:
        """把 placement execution 导出的 frontier 快照同步回 tracker。

        当前 direct placement 已经开始向 request-level frontier ABI 收口，但：

        - `ret_mask`
        - 状态日志
        - 现有单测

        仍然主要围绕 `BaMDirectPlacementStateTracker.snapshot()` 组织。因此这里保留
        一个很薄的镜像同步层：

        - 上层控制面优先消费 placement frontier 接口
        - tracker 继续作为当前阶段的本地状态镜像

        这样下一步如果 tracker 真被 frontier table/ABI 取代，只需要删掉这层
        镜像，而不必再次改散所有调用点。
        """
        tracker.mark_chunks_read_ready_upto(
            int(frontier_snapshot.read_ready_frontier_chunks))
        tracker.mark_chunks_staged_ready_upto(
            int(frontier_snapshot.staged_ready_frontier_chunks))
        tracker.mark_chunks_cache_ready_upto(
            int(frontier_snapshot.cache_ready_frontier_chunks))
        return tracker.snapshot()

    @staticmethod
    def _build_request_frontier_snapshot_from_tracker(
        *,
        tracker: BaMDirectPlacementStateTracker,
        launch_frontier_chunks: int,
    ) -> BaMDirectPlacementFrontierSnapshot:
        """基于当前 request tracker 构造统一的 request-level frontier 快照。

        这层 helper 的定位，是给 request handle 提供一份稳定的统一 ABI，而不是
        让更高层 runtime 再去理解：

        - 现在是否还在 native read-ready 阶段
        - frontier wave 是否已经 launch
        - tracker 当前有哪些 flag 已推进

        当前先保留最小语义：

        - `launch_frontier_chunks`
          表示当前已经真正 launch 到 placement 的连续 chunk 前缀长度
        - `read_ready/cache_ready/consumable`
          都来自 request tracker 的当前镜像状态

        这样后续即使 placement frontier 真正改成 GPU-resident 更新，上层看到的
        仍然是同一套 `BaMDirectPlacementFrontierSnapshot`。
        """
        snapshot = tracker.snapshot()
        status = 1
        if int(snapshot.read_ready_chunks) > 0:
            status = 2
        if int(snapshot.cache_ready_chunks) > 0:
            status = 3
        if int(snapshot.consumable_chunks) > 0:
            status = 4
        return BaMDirectPlacementFrontierSnapshot(
            frontier_row=(
                int(status),
                int(launch_frontier_chunks),
                int(snapshot.read_ready_chunks),
                int(snapshot.cache_ready_chunks),
                int(snapshot.consumable_chunks),
                int(len(snapshot.chunk_states)),
                0,
            ),
            launch_frontier_chunks=int(launch_frontier_chunks),
            read_ready_frontier_chunks=int(snapshot.read_ready_chunks),
            staged_ready_frontier_chunks=int(snapshot.staged_ready_chunks),
            cache_ready_frontier_chunks=int(snapshot.cache_ready_chunks),
            consumable_frontier_chunks=int(snapshot.consumable_chunks),
            total_chunks=int(len(snapshot.chunk_states)),
            read_ready_frontier_tokens=int(snapshot.read_ready_tokens),
            staged_ready_frontier_tokens=int(snapshot.staged_ready_tokens),
            cache_ready_frontier_tokens=int(snapshot.cache_ready_tokens),
            consumable_frontier_tokens=int(snapshot.consumable_tokens),
            error_code=0,
        )

    @staticmethod
    def _allocate_direct_placement_request_frontier_table(
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """为一个 direct placement request 分配统一 frontier table。

        这里故意直接对齐到底层 native KV frontier ABI：

        ```text
        frontier_table: [7] int64
        ```

        这样当前 request handle、placement execution、以及后续可能接手的 GPU
        runtime，都能围绕同一张表工作。
        """
        table_device = torch.device("cpu")
        if device.type == "cuda" and torch.cuda.is_available():
            table_device = device
        return torch.zeros((7,), dtype=torch.int64, device=table_device)

    @staticmethod
    def _extract_native_kv_frontier_table(
        request_handle: Any | None,
    ) -> Optional[torch.Tensor]:
        """从 KV fast path request handle 中提取底层 native frontier table。

        当前 request-handle 三段式下，如果异步 submit 成功，最底层 native batch
        handle 已经自带一张由 BaM KV runtime 维护的 `gpu_frontier_table`。

        这张表的价值在于：

        - 它是真正由底层 KV native runtime 持有和推进的 GPU-visible frontier
        - 它和后续 placement execution 需要的 frontier ABI 已经对齐

        因此在 direct placement request 的 read-ready 阶段，优先直接复用这张表，
        比再额外分配一张平行表更贴近后续 GPU runtime 接手的最终方向。
        """
        if request_handle is None:
            return None
        native_handle = getattr(request_handle, "native_handle", None)
        request_table = getattr(native_handle, "request_table", None)
        frontier_table = getattr(request_table, "gpu_frontier_table", None)
        if isinstance(frontier_table, torch.Tensor):
            return frontier_table
        return None

    @classmethod
    def _request_frontier_table_owned_by_native_read(
        cls,
        in_flight_request: _InFlightDirectPlacementRequest,
    ) -> bool:
        """判断当前 request frontier table 是否仍由 native read runtime 持有。

        只有满足下面三个条件时，才认为“当前 table 所有权仍在 native read 阶段”：

        1. 当前 request 还保留着 live `kv_read_handle`
        2. request.frontier_table 与 native handle 自带的 gpu_frontier_table 是同一张

        这样可以避免 host 侧在 read-ready 阶段反复把 tracker 行写回去，覆盖掉
        native runtime 本来已经写好的 `launch/read_ready/...` frontier 信息。
        """
        native_frontier_table = cls._extract_native_kv_frontier_table(
            in_flight_request.kv_read_handle)
        if native_frontier_table is None:
            return False
        return native_frontier_table is in_flight_request.frontier_table

    @staticmethod
    def _publish_direct_placement_request_frontier_table(
        *,
        frontier_table: torch.Tensor | None,
        frontier_snapshot: BaMDirectPlacementFrontierSnapshot,
    ) -> None:
        """把一份 request frontier 快照同步写回共享 frontier table。

        这里刻意沿用 execution 侧已经收敛下来的“单调不回退”语义，而不是直接
        `copy_` 整行，原因是 request-level frontier 现在也正在逐步变成：

        - host 侧：只负责薄轮询 / 读表 / 收口
        - 更底层 runtime：可能更早、更频繁地推进 frontier

        一旦未来 frontier table 的更新权进一步往 GPU runtime 下放，如果这里仍然
        用 host 侧旧 snapshot 整行覆盖，就会把已经更靠前的 frontier 误回退。

        因此当前策略是：

        - 只按列原位更新
        - 只允许状态单调前进
        - staged 仍不进入统一 7 列 ABI
        """
        if frontier_table is None:
            return
        frontier_row = frontier_snapshot.frontier_row
        frontier_table[_DIRECT_FRONTIER_COL_STATUS] = max(
            int(frontier_table[_DIRECT_FRONTIER_COL_STATUS]),
            int(frontier_row[_DIRECT_FRONTIER_COL_STATUS]),
        )
        frontier_table[_DIRECT_FRONTIER_COL_LAUNCH] = max(
            int(frontier_table[_DIRECT_FRONTIER_COL_LAUNCH]),
            int(frontier_row[_DIRECT_FRONTIER_COL_LAUNCH]),
        )
        frontier_table[_DIRECT_FRONTIER_COL_READ_READY] = max(
            int(frontier_table[_DIRECT_FRONTIER_COL_READ_READY]),
            int(frontier_row[_DIRECT_FRONTIER_COL_READ_READY]),
        )
        frontier_table[_DIRECT_FRONTIER_COL_CACHE_READY] = max(
            int(frontier_table[_DIRECT_FRONTIER_COL_CACHE_READY]),
            int(frontier_row[_DIRECT_FRONTIER_COL_CACHE_READY]),
        )
        frontier_table[_DIRECT_FRONTIER_COL_CONSUMABLE] = max(
            int(frontier_table[_DIRECT_FRONTIER_COL_CONSUMABLE]),
            int(frontier_row[_DIRECT_FRONTIER_COL_CONSUMABLE]),
        )
        frontier_table[_DIRECT_FRONTIER_COL_TOTAL] = max(
            int(frontier_table[_DIRECT_FRONTIER_COL_TOTAL]),
            int(frontier_row[_DIRECT_FRONTIER_COL_TOTAL]),
        )
        frontier_table[_DIRECT_FRONTIER_COL_ERROR] = max(
            int(frontier_table[_DIRECT_FRONTIER_COL_ERROR]),
            int(frontier_row[_DIRECT_FRONTIER_COL_ERROR]),
        )

    @staticmethod
    def _tracker_not_ahead_of_frontier(
        *,
        tracker_snapshot: BaMDirectPlacementBatchStateSnapshot,
        frontier_snapshot: BaMDirectPlacementFrontierSnapshot,
    ) -> bool:
        """判断 tracker 是否没有跑到 frontier 快照前面。

        request-level getter 现在需要同时面对两类来源：

        1. execution/native runtime 已导出的统一 frontier ABI
        2. host tracker 的本地镜像

        当 tracker 明显没有比 frontier 更靠前时，我们应直接信 frontier；
        只有 tracker 某些 ready 计数已经更大时，才需要做一次“launch 边界来自
        frontier、ready 状态来自 tracker”的轻量合并。
        """
        return (
            int(tracker_snapshot.read_ready_chunks) <=
            int(frontier_snapshot.read_ready_frontier_chunks)
            and int(tracker_snapshot.staged_ready_chunks) <=
            int(frontier_snapshot.staged_ready_frontier_chunks)
            and int(tracker_snapshot.cache_ready_chunks) <=
            int(frontier_snapshot.cache_ready_frontier_chunks)
            and int(tracker_snapshot.consumable_chunks) <=
            int(frontier_snapshot.consumable_frontier_chunks)
        )

    @staticmethod
    def _read_direct_placement_request_frontier_row(
        frontier_table: torch.Tensor | None,
    ) -> tuple[int, ...]:
        """读取共享 frontier table 的当前 host 快照。"""
        if frontier_table is None:
            return ()
        return tuple(int(value) for value in frontier_table.detach().cpu().tolist())

    @staticmethod
    def _count_descriptor_tokens_upto(
        descriptor: BaMDirectPlacementBatchDescriptor,
        ready_chunks: int,
    ) -> int:
        """按连续前缀 chunk 数统计 token 数。"""
        final_ready_chunks = min(max(int(ready_chunks), 0), len(descriptor.chunks))
        return sum(
            int(chunk.actual_tokens)
            for chunk in descriptor.chunks[:final_ready_chunks]
        )

    @classmethod
    def _build_request_frontier_snapshot_from_table(
        cls,
        *,
        tracker_snapshot: BaMDirectPlacementBatchStateSnapshot,
        frontier_row: tuple[int, ...],
    ) -> BaMDirectPlacementFrontierSnapshot:
        """基于共享 frontier table 重建统一 request-level frontier 快照。

        这层 helper 的核心原则是：

        - frontier row 是当前 request-level frontier 的主事实来源
        - tracker 只负责补充当前 ABI 里还没有显式编码的 staged 信息
        - token 数通过 tracker descriptor 的稳定 chunk 元数据按前缀重建

        这样 request-level getter 就能真正围绕共享 table 工作，而不是继续把
        tracker/mirror 当成主来源。
        """
        descriptor = tracker_snapshot.descriptor
        status = int(frontier_row[0]) if len(frontier_row) > 0 else 0
        launch_frontier_chunks = max(
            int(frontier_row[1]) if len(frontier_row) > 1 else 0, 0)
        read_ready_frontier_chunks = max(
            int(frontier_row[2]) if len(frontier_row) > 2 else 0, 0)
        cache_ready_frontier_chunks = max(
            int(frontier_row[3]) if len(frontier_row) > 3 else 0, 0)
        consumable_frontier_chunks = max(
            int(frontier_row[4]) if len(frontier_row) > 4 else 0, 0)
        total_chunks = max(
            int(frontier_row[5]) if len(frontier_row) > 5 else 0,
            len(tracker_snapshot.chunk_states),
        )
        error_code = int(frontier_row[6]) if len(frontier_row) > 6 else 0
        return BaMDirectPlacementFrontierSnapshot(
            frontier_row=(
                int(status),
                int(launch_frontier_chunks),
                int(read_ready_frontier_chunks),
                int(cache_ready_frontier_chunks),
                int(consumable_frontier_chunks),
                int(total_chunks),
                int(error_code),
            ),
            launch_frontier_chunks=int(launch_frontier_chunks),
            read_ready_frontier_chunks=int(read_ready_frontier_chunks),
            staged_ready_frontier_chunks=int(tracker_snapshot.staged_ready_chunks),
            cache_ready_frontier_chunks=int(cache_ready_frontier_chunks),
            consumable_frontier_chunks=int(consumable_frontier_chunks),
            total_chunks=int(total_chunks),
            read_ready_frontier_tokens=cls._count_descriptor_tokens_upto(
                descriptor, read_ready_frontier_chunks),
            staged_ready_frontier_tokens=int(
                tracker_snapshot.staged_ready_tokens),
            cache_ready_frontier_tokens=cls._count_descriptor_tokens_upto(
                descriptor, cache_ready_frontier_chunks),
            consumable_frontier_tokens=cls._count_descriptor_tokens_upto(
                descriptor, consumable_frontier_chunks),
            error_code=int(error_code),
        )

    def _get_direct_placement_request_frontier(
        self,
        *,
        in_flight_request: _InFlightDirectPlacementRequest,
    ) -> BaMDirectPlacementFrontierSnapshot:
        """读取当前 request handle 的统一 frontier 快照。

        当前主线已经收口到：

        1. `start/poll` 阶段统一暴露 native read / tracker 可见的 request frontier
        2. native read 若自带共享 frontier table，则优先直接读取那张表
        3. 只有当前表不再由 native read 持有时，host 才把 tracker 快照同步回去

        这样可以保证 `get_frontier()` 是纯观察接口，不会在读取过程中暗中推进
        placement 或回退更靠前的共享 frontier。
        """
        tracker_snapshot = in_flight_request.state_tracker.snapshot()
        frontier_table = in_flight_request.frontier_table
        native_table_owns_frontier = \
            self._request_frontier_table_owned_by_native_read(
                in_flight_request)
        request_frontier_snapshot = self._build_request_frontier_snapshot_from_tracker(
            tracker=in_flight_request.state_tracker,
            launch_frontier_chunks=int(in_flight_request.first_wave_launch_chunks),
        )
        if not native_table_owns_frontier:
            self._publish_direct_placement_request_frontier_table(
                frontier_table=frontier_table,
                frontier_snapshot=request_frontier_snapshot,
            )
        tracker_snapshot = in_flight_request.state_tracker.snapshot()
        frontier_row = self._read_direct_placement_request_frontier_row(
            frontier_table)
        if frontier_row:
            return self._build_request_frontier_snapshot_from_table(
                tracker_snapshot=tracker_snapshot,
                frontier_row=frontier_row,
            )
        return request_frontier_snapshot

    @staticmethod
    def _count_ready_chunks_in_range(
        snapshot: BaMDirectPlacementBatchStateSnapshot,
        *,
        chunk_start: int,
        chunk_count: int,
        ready_attr: str,
    ) -> int:
        """统计某一段 chunk 范围内已经 ready 的数量。"""
        chunk_end = min(chunk_start + chunk_count, len(snapshot.chunk_states))
        ready_chunks = 0
        for chunk_state in snapshot.chunk_states[chunk_start:chunk_end]:
            if bool(getattr(chunk_state, ready_attr)):
                ready_chunks += 1
        return ready_chunks

    @staticmethod
    def _mark_ready_range_if_empty(
        tracker: BaMDirectPlacementStateTracker,
        *,
        chunk_start: int,
        chunk_count: int,
        ready_attr: str,
    ) -> None:
        """仅在某一波完全没有推进 ready 状态时，按范围补一个保守兜底。

        兜底范围必须收缩到“本轮真实 launch 的 chunk 区间”，不能再把整批都
        粗暴标成 ready，否则会破坏 request frontier 的连续语义。
        """
        snapshot = tracker.snapshot()
        ready_chunks = LMCacheBaMStore._count_ready_chunks_in_range(
            snapshot,
            chunk_start=chunk_start,
            chunk_count=chunk_count,
            ready_attr=ready_attr,
        )
        if ready_chunks > 0:
            return

        chunk_end = min(chunk_start + chunk_count, len(snapshot.chunk_states))
        for chunk_index in range(chunk_start, chunk_end):
            if ready_attr == "staged_ready":
                tracker.mark_chunk_staged_ready(chunk_index)
            elif ready_attr == "cache_ready":
                tracker.mark_chunk_cache_ready(chunk_index)
            else:
                raise ValueError(f"unsupported ready_attr: {ready_attr!r}")

    @staticmethod
    def _resolve_wave_return_target_chunks(
        *,
        launch_start_chunk: int,
        launch_chunks: int,
        return_target_chunks: int | None,
    ) -> int:
        """规范化当前 wave 的连续前缀返回目标。"""
        if return_target_chunks is None:
            return_target_chunks = int(launch_start_chunk) + int(launch_chunks)
        normalized_target = max(int(return_target_chunks), 0)
        if normalized_target < int(launch_start_chunk):
            raise ValueError(
                "return_target_chunks must not be smaller than launch_start_chunk: "
                f"{normalized_target} vs {launch_start_chunk}")
        return normalized_target

    def _launch_direct_placement_wave(
        self,
        *,
        direct_placer: BaMDirectKVPlacer,
        results: list[Any],
        kv_caches: list[torch.Tensor],
        slot_mapping: torch.Tensor,
        chunk_starts: list[int],
        state_tracker: BaMDirectPlacementStateTracker,
        frontier_table: torch.Tensor | None = None,
        launch_start_chunk: int,
        launch_chunk_count: int | None,
        return_target_chunks: int | None,
        wave_name: str,
        do_prepare: bool = True,
    ) -> _InFlightDirectPlacementWave:
        """launch 一波 direct placement，并返回可继续收口的 in-flight wave。

        这层 helper 的职责很单一：

        - 只组织一波 placement 的 prepare / launch
        - 只产出一个显式 in-flight wave 对象
        - 不在这里做等待/收口，避免再把 launch 与 wait 混回去

        注意：
        当前主线虽然最终仍会在同一次 direct retrieve 内等待 wave 满足返回
        语义，但这里已经先把 launch 边界独立了出来。后续如果要把 execution
        句柄再往上层 runtime 提一层，就不需要再重新拆主流程。

        参数语义：

        - `launch_chunk_count`
          这一波真实提交给 placement 的 chunk 数。
        - `return_target_chunks`
          这一波结束后，希望暴露给上层推理引擎的“连续可消费前缀 chunk 数”。

        """
        launch_chunks = (int(launch_chunk_count)
                         if launch_chunk_count is not None else
                         max(len(results) - int(launch_start_chunk), 0))
        if launch_chunks <= 0:
            raise ValueError("launch_chunks must be positive for in-flight wave")
        normalized_return_target_chunks = self._resolve_wave_return_target_chunks(
            launch_start_chunk=launch_start_chunk,
            launch_chunks=launch_chunks,
            return_target_chunks=return_target_chunks,
        )

        if do_prepare:
            direct_placer.prepare_for_batch(
                results=results,
                kv_caches=kv_caches,
                slot_mapping=slot_mapping,
                chunk_starts=chunk_starts,
                launch_start_chunk=launch_start_chunk,
                max_chunks_to_launch=launch_chunk_count,
            )
        launched_batch = direct_placer.start_batch(
            results=results,
            kv_caches=kv_caches,
            slot_mapping=slot_mapping,
            chunk_starts=chunk_starts,
            launch_start_chunk=launch_start_chunk,
            max_chunks_to_launch=launch_chunk_count,
        )
        try:
            direct_execution = direct_placer.execution_from_launched_batch(
                launched_batch=launched_batch,
                state_tracker=state_tracker,
                frontier_table=frontier_table,
            )
        except TypeError as exc:
            # 兼容旧测试桩 / 过渡实现：
            # 当前主线已经支持 execution 复用外部 request frontier table，但部分
            # 测试桩仍然只有旧签名。这里仅在明确是“不认识 frontier_table 这个
            # 关键字”时回退旧调用方式，避免把其它真实 TypeError 吞掉。
            if "frontier_table" not in str(exc):
                raise
            direct_execution = direct_placer.execution_from_launched_batch(
                launched_batch=launched_batch,
                state_tracker=state_tracker,
            )
        # 当前主线已经收敛为单波 placement：
        # launch 完成后立刻做一次非阻塞 ready 推进，把已经自然完成的状态同步进
        # tracker，后续 poll/finalize 只消费同一条主线，而不再保留额外的
        # launch-control 实验分支。
        direct_execution.advance_ready()
        return _InFlightDirectPlacementWave(
            direct_execution=direct_execution,
            direct_placer=direct_placer,
            launched_batch=launched_batch,
            wave_name=wave_name,
            launch_start_chunk=int(launch_start_chunk),
            launch_chunks=int(launch_chunks),
            return_target_chunks=int(normalized_return_target_chunks),
        )

    def _wait_direct_placement_wave(
        self,
        *,
        in_flight_wave: _InFlightDirectPlacementWave,
        state_tracker: BaMDirectPlacementStateTracker,
    ) -> tuple[Any, BaMDirectPlacementBatchStateSnapshot]:
        """等待一个已经 launch 的 direct placement wave 满足返回语义。"""
        direct_execution = in_flight_wave.direct_execution
        target_frontier_wait_v2 = getattr(
            direct_execution,
            "wait_until_contiguous_cache_ready_frontier",
            None,
        )
        target_frontier_wait = getattr(
            direct_execution,
            "wait_until_contiguous_cache_ready",
            None,
        )
        wave_local_wait = getattr(
            direct_execution,
            "wait_until_launched_range_cache_ready",
            None,
        )
        final_batch_snapshot = None
        if callable(target_frontier_wait_v2):
            placement_frontier_snapshot = target_frontier_wait_v2(
                in_flight_wave.return_target_chunks)
            if placement_frontier_snapshot is not None:
                final_batch_snapshot = self._sync_tracker_from_placement_frontier(
                    state_tracker,
                    frontier_snapshot=placement_frontier_snapshot,
                )
            # 这里优先走新的 `get_stats()` 显式接口。
            #
            # 之所以仍然兼容历史上的 `_build_stats()`，是因为当前测试桩和部分过渡
            # 执行器还没有完全切到新接口。如果在 storage 边界直接强制所有调用方
            # 升级，会让这次“拆 launch / wait 边界”的控制面改动无谓扩散。
            #
            # 因此这里做一层很薄的兼容适配：
            # - 真实主线实现：走 `get_stats()`；
            # - 旧测试桩 / 过渡执行器：回退到 `_build_stats()`。
            #
            # 这样既能保持上层接口收敛，也不会把兼容逻辑污染到 direct placement
            # 数据面里。
            if hasattr(direct_execution, "get_stats"):
                place_stats = direct_execution.get_stats()
            else:
                place_stats = direct_execution._build_stats()
        elif callable(target_frontier_wait):
            final_batch_snapshot = target_frontier_wait(
                in_flight_wave.return_target_chunks)
            # 这里优先走新的 `get_stats()` 显式接口。
            #
            # 之所以仍然兼容历史上的 `_build_stats()`，是因为当前测试桩和部分过渡
            # 执行器还没有完全切到新接口。如果在 storage 边界直接强制所有调用方
            # 升级，会让这次“拆 launch / wait 边界”的控制面改动无谓扩散。
            #
            # 因此这里做一层很薄的兼容适配：
            # - 真实主线实现：走 `get_stats()`；
            # - 旧测试桩 / 过渡执行器：回退到 `_build_stats()`。
            #
            # 这样既能保持上层接口收敛，也不会把兼容逻辑污染到 direct placement
            # 数据面里。
            if hasattr(direct_execution, "get_stats"):
                place_stats = direct_execution.get_stats()
            else:
                place_stats = direct_execution._build_stats()
        elif callable(wave_local_wait):
            place_stats, final_batch_snapshot = wave_local_wait()
        else:
            place_stats, final_batch_snapshot = direct_execution.wait()
        in_flight_wave.direct_placer.log_launched_batch_step_timings(
            in_flight_wave.launched_batch)

        self._mark_ready_range_if_empty(
            state_tracker,
            chunk_start=in_flight_wave.launch_start_chunk,
            chunk_count=in_flight_wave.launch_chunks,
            ready_attr="staged_ready",
        )
        self._mark_ready_range_if_empty(
            state_tracker,
            chunk_start=in_flight_wave.launch_start_chunk,
            chunk_count=in_flight_wave.launch_chunks,
            ready_attr="cache_ready",
        )
        snapshot = final_batch_snapshot or state_tracker.snapshot()
        logger.info(
            "[LMCACHE_BAM_DIRECT_PLACEMENT_WAVE_DONE] wave=%s "
            "launch_start_chunk=%d launch_chunks=%d staged_ready_in_wave=%d "
            "cache_ready_in_wave=%d return_target_chunks=%d "
            "consumable_chunks=%d consumable_tokens=%d "
            "place_ms=%.3f",
            in_flight_wave.wave_name,
            in_flight_wave.launch_start_chunk,
            in_flight_wave.launch_chunks,
            self._count_ready_chunks_in_range(
                snapshot,
                chunk_start=in_flight_wave.launch_start_chunk,
                chunk_count=in_flight_wave.launch_chunks,
                ready_attr="staged_ready",
            ),
            self._count_ready_chunks_in_range(
                snapshot,
                chunk_start=in_flight_wave.launch_start_chunk,
                chunk_count=in_flight_wave.launch_chunks,
                ready_attr="cache_ready",
            ),
            in_flight_wave.return_target_chunks,
            snapshot.consumable_chunks,
            snapshot.consumable_tokens,
            float(place_stats.place_ms),
        )
        return place_stats, snapshot

    def _run_direct_placement_wave(
        self,
        *,
        direct_placer: BaMDirectKVPlacer,
        results: list[Any],
        kv_caches: list[torch.Tensor],
        slot_mapping: torch.Tensor,
        chunk_starts: list[int],
        state_tracker: BaMDirectPlacementStateTracker,
        launch_start_chunk: int,
        launch_chunk_count: int | None,
        return_target_chunks: int | None,
        wave_name: str,
        do_prepare: bool = True,
    ) -> tuple[Any, BaMDirectPlacementBatchStateSnapshot]:
        """兼容当前调用方的一站式 wave helper。"""
        launch_chunks = (int(launch_chunk_count)
                         if launch_chunk_count is not None else
                         max(len(results) - int(launch_start_chunk), 0))
        if launch_chunks <= 0:
            return type(
                "PlaceStats", (),
                {
                    "impl": envs.VLLM_BAM_DIRECT_PLACEMENT_IMPL.strip().lower(),
                    "refill_ms": 0.0,
                    "transfer_ms": 0.0,
                    "fused_ms": 0.0,
                    "place_ms": 0.0,
                })(), state_tracker.snapshot()
        in_flight_wave = self._launch_direct_placement_wave(
            direct_placer=direct_placer,
            results=results,
            kv_caches=kv_caches,
            slot_mapping=slot_mapping,
            chunk_starts=chunk_starts,
            state_tracker=state_tracker,
            launch_start_chunk=launch_start_chunk,
            launch_chunk_count=launch_chunk_count,
            return_target_chunks=return_target_chunks,
            wave_name=wave_name,
            do_prepare=do_prepare,
        )
        return self._wait_direct_placement_wave(
            in_flight_wave=in_flight_wave,
            state_tracker=state_tracker,
        )

    def _start_direct_placement_request(
        self,
        *,
        token_database: Any,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor],
        kv_caches: list[torch.Tensor],
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        num_kv_heads: int,
        head_size: int,
    ) -> Optional[_InFlightDirectPlacementRequest]:
        """启动一次 direct placement request，并返回后续可 poll/finalize 的句柄。

        这是把“整次 request 的控制面”从旧的一口气大函数中拆出来的第一步。
        当前它只负责：

        1. 收集 prefix hit entries
        2. 读取 BaM pages
        3. 初始化 request 级 state tracker
        4. 完成 prepare，并 launch 第一波 frontier wave

        它刻意不做：

        - 等待 wave 完成
        - 构造 ret_mask
        - 做 rebuilt 语义收口

        这样后续无论是：
        - 继续保留当前同步 finalize
        - 还是把 handle 上提给 runtime 做周期性 poll

        都能复用同一套 request 启动逻辑。
        """
        if not envs.VLLM_BAM_KV_FAST_PATH:
            self._last_direct_placement_state_tracker = None
            return None

        masked_tokens = int(mask.sum().item()) if mask is not None else len(tokens)
        direct_total_start = time.perf_counter()
        logger.info(
            "[LMCACHE_BAM_DIRECT_PLACEMENT_ENTER] tokens=%d masked_tokens=%d "
            "slot_mapping=%d",
            len(tokens),
            masked_tokens,
            int(slot_mapping.numel()),
        )
        entries = self._collect_direct_placement_entries(
            token_database=token_database,
            tokens=tokens,
            mask=mask,
        )
        collect_entries_ms = (time.perf_counter() - direct_total_start) * 1000.0
        if not entries:
            self._last_direct_placement_state_tracker = None
            logger.info(
                "[LMCACHE_BAM_DIRECT_PLACEMENT_NO_PREFIX_HIT] "
                "tokens=%d masked_tokens=%d",
                len(tokens),
                masked_tokens,
            )
            return None

        keys = [key for _, _, key in entries]
        chunk_ranges = ",".join(f"[{start},{end})" for start, end, _ in entries)
        chunk_hashes = ",".join(_extract_chunk_hash(key)[:16] for key in keys)
        logger.info(
            "[LMCACHE_BAM_DIRECT_PLACEMENT_PREFIX_HIT] chunks=%d ranges=%s "
            "chunk_hashes=%s",
            len(entries),
            chunk_ranges,
            chunk_hashes,
        )
        direct_placer = self._ensure_direct_kv_placer(
            kv_cache_dtype=kv_cache_dtype)
        logger.info(
            "[LMCACHE_BAM_DIRECT_PLACEMENT_PRIME_BEGIN] chunks=%d",
            len(entries),
        )
        ensure_pointer_state = getattr(
            direct_placer,
            "ensure_kv_cache_pointer_state",
            None,
        )
        if callable(ensure_pointer_state):
            ensure_pointer_state(kv_caches=kv_caches)
        logger.info(
            "[LMCACHE_BAM_DIRECT_PLACEMENT_PRIME_DONE] chunks=%d",
            len(entries),
        )

        read_submit_start = time.perf_counter()
        kv_read_handle: Any | None = None
        blocking_results: list[Any] | None = None
        logger.info(
            "[LMCACHE_BAM_DIRECT_PLACEMENT_READ_SUBMIT_BEGIN] batch_size=%d "
            "chunk_hashes=%s",
            len(keys),
            chunk_hashes,
        )
        try:
            kv_read_handle = self.submit_chunk_pages_kv_fast_path_batch_request(
                keys)
        except Exception:
            # runtime / persistent 主线不允许在 request 内部偷偷回退成 blocking
            # read。否则同一次 direct placement request 会同时混入：
            #
            # - GPU persistent service 语义
            # - host materialize 语义
            #
            # 这样日志、正确性和性能口径都会失真。
            #
            # 只有在显式 legacy 路径里，才允许继续用 blocking read 兜底。
            if self._persistent_runtime_mainline_enabled():
                raise RuntimeError(
                    "runtime direct placement requires native batch submit; "
                    "blocking fallback is disabled in persistent mode"
                ) from None
            logger.exception(
                "[LMCACHE_BAM_DIRECT_PLACEMENT_READ_SUBMIT] "
                "async submit failed; fall back to blocking batch read")
            blocking_results = self.read_chunk_pages_kv_fast_path_batch(keys)
        read_submit_ms = (time.perf_counter() - read_submit_start) * 1000.0

        # token_database 和 slot_mapping 使用同一段 tokens 的局部坐标系。
        # 例如 mask 前缀有 False 时，第一条可 retrieve 的 chunk 起点可能不是 0。
        # 这里必须保留这个局部偏移，不能人为重编号，否则 placement 会把数据写
        # 到错误的 vLLM physical slot。
        chunk_starts = [start for start, _, _ in entries]
        prefix_hit_chunks = len(entries)
        prefix_hit_tokens = sum(
            int(self.get_chunk_metadata(key).actual_tokens)  # type: ignore[union-attr]
            for key in keys)
        descriptor_start = time.perf_counter()
        batch_descriptor = self._build_direct_placement_descriptor_from_metadata(
            entries=entries)
        descriptor_ms = (time.perf_counter() - descriptor_start) * 1000.0
        tracker_start = time.perf_counter()
        state_tracker = BaMDirectPlacementStateTracker(batch_descriptor)
        if blocking_results is not None:
            state_tracker.mark_all_read_ready()
        self._last_direct_placement_state_tracker = state_tracker
        frontier_table = self._extract_native_kv_frontier_table(kv_read_handle)
        if frontier_table is None:
            frontier_table = self._allocate_direct_placement_request_frontier_table(
                device=slot_mapping.device,
            )
        self._log_direct_placement_state(
            stage=("read_ready" if blocking_results is not None else "submitted"),
            tracker=state_tracker,
        )
        if frontier_table is not self._extract_native_kv_frontier_table(
                kv_read_handle):
            self._publish_direct_placement_request_frontier_table(
                frontier_table=frontier_table,
                frontier_snapshot=self._build_request_frontier_snapshot_from_tracker(
                    tracker=state_tracker,
                    launch_frontier_chunks=0,
                ),
            )
        tracker_init_ms = (time.perf_counter() - tracker_start) * 1000.0
        (runtime_direct_placement_attached,
         runtime_direct_placement_attachment) = (
             self._try_attach_runtime_direct_placement(
                 kv_read_handle=kv_read_handle,
                 direct_placer=direct_placer,
                 kv_caches=kv_caches,
                 slot_mapping=slot_mapping,
                 chunk_starts=chunk_starts,
                 num_kv_heads=int(num_kv_heads),
                 head_size=int(head_size),
             ))
        if (self._runtime_direct_path_required()
                and not runtime_direct_placement_attached):
            raise RuntimeError(
                "persistent direct placement requires runtime one-copy attach; "
                "host-side materialized finalize is disabled")
        first_wave_launch_chunks = prefix_hit_chunks
        # 对正常推理引擎主线来说，“当前请求应该返回的 prefix 长度”必须先于
        # placement 执行语义被确定下来。
        #
        # 当前默认就是：
        #   命中了多少连续 prefix chunk
        #     -> 就准备返回多少连续 prefix chunk
        #
        first_wave_return_target_chunks = first_wave_launch_chunks
        first_wave_launch_tokens = sum(
            int(chunk_descriptor.actual_tokens)
            for chunk_descriptor in batch_descriptor.chunks[
                :first_wave_launch_chunks])
        pipeline_name = self._direct_placement_pipeline_name(
            runtime_direct_placement_attached=runtime_direct_placement_attached,
        )
        logger.info(
            "[LMCACHE_BAM_DIRECT_PLACEMENT_PIPELINE] pipeline=%s "
            "kv_executor=%s runtime_enabled=%s persistent_enabled=%s "
            "runtime_one_copy=%s runtime_attach=%s",
            pipeline_name,
            os.getenv("VLLM_BAM_KV_EXECUTOR", "rowctx"),
            str(self._gpu_worker_runtime_enabled()).lower(),
            str(self._gpu_worker_persistent_enabled()).lower(),
            str(bool(envs.VLLM_BAM_DIRECT_PLACEMENT_RUNTIME_ONE_COPY)).lower(),
            str(bool(runtime_direct_placement_attached)).lower(),
        )
        logger.info(
            "[LMCACHE_BAM_DIRECT_PLACEMENT_LAUNCH_POLICY] total_chunks=%d "
            "prefix_hit_chunks=%d prefix_hit_tokens=%d "
            "frontier_launch_chunks=%d frontier_launch_tokens=%d "
            "return_target_chunks=%d",
            prefix_hit_chunks,
            prefix_hit_chunks,
            prefix_hit_tokens,
            first_wave_launch_chunks,
            first_wave_launch_tokens,
            first_wave_return_target_chunks,
        )
        return _InFlightDirectPlacementRequest(
            tokens=tokens,
            kv_caches=kv_caches,
            slot_mapping=slot_mapping,
            kv_read_handle=kv_read_handle,
            results=blocking_results,
            state_tracker=state_tracker,
            frontier_table=frontier_table,
            direct_placer=direct_placer,
            kv_cache_dtype=kv_cache_dtype,
            chunk_starts=chunk_starts,
            keys=keys,
            prefix_hit_chunks=prefix_hit_chunks,
            prefix_hit_tokens=prefix_hit_tokens,
            first_wave_launch_chunks=first_wave_launch_chunks,
            first_wave_return_target_chunks=first_wave_return_target_chunks,
            read_ready_frontier_chunks=(
                len(blocking_results) if blocking_results is not None else 0),
            bootstrap_profile=_DirectPlacementRequestBootstrapProfile(
                collect_entries_ms=collect_entries_ms,
                read_submit_ms=read_submit_ms,
                descriptor_ms=descriptor_ms,
                tracker_init_ms=tracker_init_ms,
                direct_total_start_time=direct_total_start,
            ),
            num_kv_heads=int(num_kv_heads),
            head_size=int(head_size),
            runtime_direct_placement_attached=(
                runtime_direct_placement_attached),
            runtime_direct_placement_attachment=(
                runtime_direct_placement_attachment),
        )

    def _poll_direct_placement_request(
        self,
        *,
        in_flight_request: _InFlightDirectPlacementRequest,
    ) -> BaMDirectPlacementBatchStateSnapshot:
        """非阻塞推进一次 request 当前已 launch frontier 的 ready 状态。

        当前版本只推进 request 当前已知的连续前缀 frontier，不主动做 finalize。

        这里刻意把“轮询 ready”和“真正收口 finalize”分开，原因是：

        - 对普通 materialized 路径，poll 只能先可靠暴露 `read_ready`
        - 对 runtime direct placement 路径，poll 可以直接暴露 GPU 后台发布的
          `cache_ready/consumable`

        返回值始终落到 request 级 snapshot，而不是 wave 局部结构，这样以后上层
        runtime 如果直接持有 request handle，就不需要理解 wave 细节。
        """
        if in_flight_request.results is None and \
                in_flight_request.kv_read_handle is not None:
            previous_snapshot = in_flight_request.state_tracker.snapshot()
            self._maybe_log_direct_placement_native_runtime_context(
                in_flight_request=in_flight_request,
            )
            kv_poll_snapshot = self.poll_chunk_pages_kv_fast_path_batch_request(
                in_flight_request.kv_read_handle)
            snapshot = self._sync_tracker_from_native_read_frontier(
                in_flight_request.state_tracker,
                poll_snapshot=kv_poll_snapshot,
            )
            if in_flight_request.runtime_direct_placement_attached:
                in_flight_request.state_tracker.mark_chunks_staged_ready_upto(
                    int(kv_poll_snapshot.cache_ready_frontier_chunks))
                in_flight_request.state_tracker.mark_chunks_cache_ready_upto(
                    int(kv_poll_snapshot.consumable_frontier_chunks))
                snapshot = in_flight_request.state_tracker.snapshot()
            if not self._request_frontier_table_owned_by_native_read(
                    in_flight_request):
                self._publish_direct_placement_request_frontier_table(
                    frontier_table=in_flight_request.frontier_table,
                    frontier_snapshot=(
                        self._build_request_frontier_snapshot_from_tracker(
                            tracker=in_flight_request.state_tracker,
                            launch_frontier_chunks=0,
                        )),
                )
            if snapshot.read_ready_chunks > \
                    in_flight_request.read_ready_frontier_chunks:
                in_flight_request.read_ready_frontier_chunks = int(
                    snapshot.read_ready_chunks)
            frontier_advanced = (
                int(snapshot.read_ready_chunks) >
                int(previous_snapshot.read_ready_chunks)
                or int(snapshot.cache_ready_chunks) >
                int(previous_snapshot.cache_ready_chunks)
                or int(snapshot.consumable_chunks) >
                int(previous_snapshot.consumable_chunks)
            )
            if frontier_advanced:
                logger.info(
                    "[LMCACHE_BAM_DIRECT_PLACEMENT_READ_FRONTIER] "
                    "launch_frontier_chunks=%d/%d "
                    "read_ready_chunks=%d/%d read_ready_tokens=%d/%d "
                    "cache_ready_chunks=%d/%d cache_ready_tokens=%d/%d "
                    "consumable_chunks=%d/%d consumable_tokens=%d/%d "
                    "native_cache_ready_chunks=%d/%d "
                    "native_consumable_chunks=%d/%d "
                    "batch_ready=%s poll_iters=%d host_status=%d",
                    kv_poll_snapshot.launch_frontier_chunks,
                    kv_poll_snapshot.total_chunks,
                    snapshot.read_ready_chunks,
                    len(snapshot.chunk_states),
                    snapshot.read_ready_tokens,
                    snapshot.descriptor.total_tokens,
                    snapshot.cache_ready_chunks,
                    len(snapshot.chunk_states),
                    snapshot.cache_ready_tokens,
                    snapshot.descriptor.total_tokens,
                    snapshot.consumable_chunks,
                    len(snapshot.chunk_states),
                    snapshot.consumable_tokens,
                    snapshot.descriptor.total_tokens,
                    kv_poll_snapshot.cache_ready_frontier_chunks,
                    kv_poll_snapshot.total_chunks,
                    kv_poll_snapshot.consumable_frontier_chunks,
                    kv_poll_snapshot.total_chunks,
                    kv_poll_snapshot.ready,
                    kv_poll_snapshot.poll_iters,
                    kv_poll_snapshot.host_status,
                )
                if kv_poll_snapshot.ready:
                    try:
                        ready_runtime_snapshot = (
                            self.
                            get_chunk_pages_kv_fast_path_batch_request_runtime_snapshot(
                                in_flight_request.kv_read_handle))
                        ready_runtime_row = getattr(
                            ready_runtime_snapshot, "matched_runtime_row", None)
                        if ready_runtime_row is not None:
                            logger.info(
                                "[LMCACHE_BAM_DIRECT_PLACEMENT_RUNTIME_READY] "
                                "request_id=%d runtime_row=%s",
                                int(getattr(ready_runtime_snapshot,
                                            "request_id", 0)),
                                tuple(int(value)
                                      for value in ready_runtime_row),
                            )
                    except Exception:
                        logger.debug(
                            "[LMCACHE_BAM_DIRECT_PLACEMENT_RUNTIME_READY] "
                            "runtime snapshot unavailable",
                            exc_info=True,
                        )
            return snapshot

        return in_flight_request.state_tracker.snapshot()

    def _maybe_log_direct_placement_native_runtime_context(
        self,
        *,
        in_flight_request: _InFlightDirectPlacementRequest,
    ) -> None:
        """在 native read 阶段打印一次 runtime attach 上下文。

        这条日志不参与控制逻辑，只服务当前 GPU-resident runtime 主线的排查：

        - 请求 submit 后有没有真的进入 runtime slot
        - 当前 frontier_table / request_table / completion_table 指针是否稳定
        - persistent service 是否已经处于运行状态
        """
        if in_flight_request.native_runtime_context_logged:
            return
        if in_flight_request.kv_read_handle is None:
            return
        try:
            runtime_snapshot = (
                self.get_chunk_pages_kv_fast_path_batch_request_runtime_snapshot(
                    in_flight_request.kv_read_handle))
        except Exception:
            logger.debug(
                "[LMCACHE_BAM_DIRECT_PLACEMENT_RUNTIME_ATTACH] "
                "runtime snapshot unavailable",
                exc_info=True,
            )
            return
        matched_row = getattr(runtime_snapshot, "matched_runtime_row", None)
        if matched_row is None:
            return
        logger.info(
            "[LMCACHE_BAM_DIRECT_PLACEMENT_RUNTIME_ATTACH] "
            "request_id=%d backend=%s service_running=%s active_count=%d "
            "request_table_ptr=0x%x frontier_table_ptr=0x%x "
            "completion_table_ptr=0x%x runtime_row=%s",
            int(getattr(runtime_snapshot, "request_id", 0)),
            str(getattr(runtime_snapshot, "worker_backend", "unknown")),
            bool(getattr(runtime_snapshot, "service_running", False)),
            int(getattr(runtime_snapshot, "active_count", 0)),
            int(getattr(runtime_snapshot, "request_table_ptr", 0)),
            int(getattr(runtime_snapshot, "frontier_table_ptr", 0)),
            int(getattr(runtime_snapshot, "completion_table_ptr", 0)),
            tuple(int(value) for value in matched_row),
        )
        in_flight_request.native_runtime_context_logged = True

    def _consume_direct_placement_read_request(
        self,
        *,
        in_flight_request: _InFlightDirectPlacementRequest,
    ) -> _DirectPlacementFinalizeReadOutcome:
        """按当前 pipeline 收口 direct placement 对应的 native read。

        这层现在只做调度，不再把三条链路的细节揉在一起：

        - rowctx_baseline / gpu_worker_persistent_materialized:
          consume 出 materialized pages，后续走已验证正确的 materialized placement；
        - gpu_worker_persistent_one_copy:
          只做 cleanup-only read finalize，后续由 one-copy pipeline 收口。
        """
        if in_flight_request.results is not None:
            return _DirectPlacementFinalizeReadOutcome(
                read_finalize_mode=_DIRECT_FINALIZE_MODE_RESULTS_MATERIALIZED,
                pipeline_name=self._direct_placement_pipeline_name(
                    read_finalize_mode=(
                        _DIRECT_FINALIZE_MODE_RESULTS_MATERIALIZED),
                    runtime_direct_placement_attached=False,
                ),
                runtime_cleanup_handle=in_flight_request.kv_read_handle,
                runtime_verify_expected_tensors=None,
            )

        if in_flight_request.kv_read_handle is None:
            raise RuntimeError(
                "direct placement finalize requires kv_read_handle "
                "before results are available")
        if (self._runtime_direct_path_required()
                and not in_flight_request.runtime_direct_placement_attached):
            raise RuntimeError(
                "persistent direct placement lost runtime one-copy attachment "
                "before finalize")

        if in_flight_request.runtime_direct_placement_attached:
            return self._finalize_one_copy_read_request(
                in_flight_request=in_flight_request)
        return self._consume_materialized_read_request(
            in_flight_request=in_flight_request)

    def _log_direct_read_consume_begin(
        self,
        *,
        in_flight_request: _InFlightDirectPlacementRequest,
    ) -> float:
        """统一打印 read consume/finalize 起点，返回计时起点。

        读收口虽然分成 materialized 和 one-copy 两条 helper，但日志前缀保持一致，
        方便继续用旧 grep 观察 direct placement 生命周期。
        """
        read_consume_start = time.perf_counter()
        logger.info(
            "[LMCACHE_BAM_DIRECT_PLACEMENT_READ_CONSUME_BEGIN] "
            "batch_size=%d",
            len(in_flight_request.keys),
        )
        return read_consume_start

    def _mark_direct_read_ready_and_log(
        self,
        *,
        in_flight_request: _InFlightDirectPlacementRequest,
        ready_chunks: int,
        read_consume_start: float,
        read_finalize_mode: str,
    ) -> None:
        """发布 read-ready frontier，并打印统一的 read consume 完成日志。"""
        in_flight_request.state_tracker.mark_chunks_read_ready_upto(ready_chunks)
        in_flight_request.read_ready_frontier_chunks = int(ready_chunks)
        self._log_direct_placement_state(
            stage="read_ready",
            tracker=in_flight_request.state_tracker,
        )
        read_consume_wall_ms = (
            time.perf_counter() - read_consume_start) * 1000.0
        logger.info(
            "[LMCACHE_BAM_DIRECT_PLACEMENT_READ_CONSUME_DONE] "
            "batch_size=%d consume_wall_ms=%.3f mode=%s",
            int(ready_chunks),
            read_consume_wall_ms,
            read_finalize_mode,
        )

    def _prepare_one_copy_runtime_verify_expected_tensors(
        self,
        *,
        in_flight_request: _InFlightDirectPlacementRequest,
    ) -> dict[str, torch.Tensor] | None:
        """为 one-copy 写后校验准备 expected dense tensors。

        这是强调试支线，不参与默认正确 fast path。把它单独拆出来后，one-copy
        主线的 cleanup 语义不会再和 verify live refill 逻辑缠在一起。
        """
        if not envs.VLLM_BAM_DIRECT_PLACEMENT_VERIFY_RUNTIME_WRITE:
            return None

        # one-copy correctness repair 现在会先用“正确 materialized 线路”同源的
        # refill kernel 准备 dense prefix。raw verify 的 reference 应该优先复用
        # 这份已经准备好的 tensor：
        #
        # - 避免为了 verify 再从 live request pages 重复 refill 一遍；
        # - 保证 raw verify、official repair、后续 xformers 诊断看到的是同一份
        #   correctness anchor；
        # - 避免重新引入“旁路 decode 自洽但模型输出错误”的循环校验。
        existing_dense = in_flight_request.materialized_prefix_chunk_tensors
        if existing_dense:
            prefix_keys = in_flight_request.keys[:len(existing_dense)]
            logger.info(
                "[LMCACHE_BAM_DIRECT_PLACEMENT_VERIFY_RUNTIME_WRITE_PREPARE_REUSE] "
                "chunks=%d source=materialized_prefix_refill",
                len(existing_dense),
            )
            return {
                _extract_chunk_hash(key): dense_chunk
                for key, dense_chunk in zip(prefix_keys, existing_dense)
            }

        if (envs.VLLM_BAM_DIRECT_PLACEMENT_RUNTIME_ONE_COPY and not envs.
                VLLM_BAM_DIRECT_PLACEMENT_VERIFY_RUNTIME_WRITE_ALLOW_LIVE_REFILL):
            logger.warning(
                "[LMCACHE_BAM_DIRECT_PLACEMENT_VERIFY_RUNTIME_WRITE_SKIP] "
                "reason=live_refill_disabled_for_runtime_one_copy "
                "batch_size=%d prefix_hit_chunks=%d",
                len(in_flight_request.keys),
                int(in_flight_request.prefix_hit_chunks),
            )
            return None
        logger.info(
            "[LMCACHE_BAM_DIRECT_PLACEMENT_VERIFY_RUNTIME_WRITE_PREPARE_BEGIN] "
            "batch_size=%d prefix_hit_chunks=%d source=live_request_pages",
            len(in_flight_request.keys),
            int(in_flight_request.prefix_hit_chunks),
        )
        try:
            tensors = (
                self.load_chunk_tensors_kv_fast_path_from_live_request_pages(
                    in_flight_request.kv_read_handle,
                    max_chunks=in_flight_request.prefix_hit_chunks,
                ))
            logger.info(
                "[LMCACHE_BAM_DIRECT_PLACEMENT_VERIFY_RUNTIME_WRITE_PREPARE_DONE] "
                "prepared_chunks=%d",
                len(tensors),
            )
            return tensors
        except Exception:
            logger.exception(
                "[LMCACHE_BAM_DIRECT_PLACEMENT_VERIFY_RUNTIME_WRITE_PREPARE_FAIL] "
                "failed to build expected tensors from live request pages")
            return None

    def _prepare_one_copy_dense_prefix_workspace(
        self,
        *,
        in_flight_request: _InFlightDirectPlacementRequest,
    ) -> None:
        """为 one-copy correctness repair / dense consume 准备 dense prefix tensor。

        这一步只属于 `gpu_worker_persistent_one_copy` 实验线。当前 one-copy 写端
        还没有完全收敛到官方 paged KV 语义，因此这里保留一份从 live pages
        materialize 出来的 dense prefix，供后续 official-write repair 和调试校验
        使用。materialized fast path 不走这里。
        """
        if in_flight_request.prefix_hit_chunks <= 0:
            return
        try:
            logger.info(
                "[LMCACHE_BAM_DIRECT_PLACEMENT_DENSE_PREFIX_PREPARE_BEGIN] "
                "chunks=%d source=live_request_pages",
                int(in_flight_request.prefix_hit_chunks),
            )
            dense_prefix_tensors = (
                self.load_chunk_tensors_kv_fast_path_from_live_request_pages(
                    in_flight_request.kv_read_handle,
                    max_chunks=in_flight_request.prefix_hit_chunks,
                ))
            logger.info(
                "[LMCACHE_BAM_DIRECT_PLACEMENT_DENSE_PREFIX_PREPARE_LOAD_DONE] "
                "loaded_chunks=%d",
                len(dense_prefix_tensors),
            )
            in_flight_request.materialized_prefix_chunk_tensors = tuple(
                dense_prefix_tensors[_extract_chunk_hash(key)]
                for key in in_flight_request.keys[
                    :in_flight_request.prefix_hit_chunks]
            )
            self._verify_one_copy_dense_prefix_against_write_reference(
                in_flight_request=in_flight_request)
            logger.info(
                "[LMCACHE_BAM_DIRECT_PLACEMENT_DENSE_PREFIX_PREPARE_DONE] "
                "chunks=%d",
                len(in_flight_request.materialized_prefix_chunk_tensors),
            )
        except Exception:
            logger.exception(
                "[LMCACHE_BAM_DIRECT_PLACEMENT_DENSE_PREFIX_PREPARE_FAIL] "
                "failed to materialize dense prefix chunk tensors "
                "from live request pages")
            in_flight_request.materialized_prefix_chunk_tensors = None

    def _resolve_one_copy_runtime_verify_expected_tensors(
        self,
        *,
        in_flight_request: _InFlightDirectPlacementRequest,
        runtime_verify_expected_tensors: dict[str, torch.Tensor] | None,
    ) -> dict[str, torch.Tensor] | None:
        """取得 one-copy raw 写入校验所需的 expected dense tensors。

        one-copy 当前有两种可能已经拿到“正确参考值”：

        1. `VERIFY_RUNTIME_WRITE_ALLOW_LIVE_REFILL=1` 时，读收口阶段会直接从
           live request pages 构造 `dict[chunk_hash, dense_tensor]`；
        2. correctness repair 为了调用 vLLM 官方写端，会准备
           `materialized_prefix_chunk_tensors`，其顺序与当前 prefix keys 一一对应。

        raw one-copy 校验必须发生在 official-write repair 之前，否则 repair 会把
        GPU persistent service 原始 scatter 的错误覆盖掉。这里优先复用已有 dict，
        不存在时再把 dense prefix tuple 轻量组回 dict，避免为了调试再次触发 BaM
        读路径。
        """
        if runtime_verify_expected_tensors is not None:
            return runtime_verify_expected_tensors

        dense_chunks = in_flight_request.materialized_prefix_chunk_tensors
        if not dense_chunks:
            return None

        prefix_keys = in_flight_request.keys[:len(dense_chunks)]
        return {
            _extract_chunk_hash(key): dense_chunk
            for key, dense_chunk in zip(prefix_keys, dense_chunks)
        }

    def _verify_one_copy_raw_runtime_write_before_repair(
        self,
        *,
        in_flight_request: _InFlightDirectPlacementRequest,
        runtime_verify_expected_tensors: dict[str, torch.Tensor] | None,
    ) -> bool:
        """在 official repair 前校验 GPU service 的原始 one-copy 写入。

        这条 helper 是 one-copy 下一阶段最关键的“定责钩子”：

        - 如果这里通过，说明 persistent service 直接写 paged KV cache 已经正确，
          后续可以安全移除 official-write repair；
        - 如果这里失败，错误会暴露在 repair 之前，mismatch 日志就能直接指向
          CUDA scatter 的源布局、目标 packed offset 或 slot 映射问题。

        返回值表示本轮是否已经做过 runtime-write verify。调用方用它避免在
        repair 之后再次调用同一个 verifier，从而把 raw 结果和 repair 结果混在一起。
        """
        if not envs.VLLM_BAM_DIRECT_PLACEMENT_VERIFY_RUNTIME_WRITE:
            return False

        expected_tensors = self._resolve_one_copy_runtime_verify_expected_tensors(
            in_flight_request=in_flight_request,
            runtime_verify_expected_tensors=runtime_verify_expected_tensors,
        )
        logger.info(
            "[LMCACHE_BAM_DIRECT_PLACEMENT_RAW_RUNTIME_WRITE_VERIFY_BEGIN] "
            "chunks=%d expected_source=%s",
            int(in_flight_request.prefix_hit_chunks),
            "available" if expected_tensors is not None else "missing",
        )
        if expected_tensors is None:
            logger.warning(
                "[LMCACHE_BAM_DIRECT_PLACEMENT_RAW_RUNTIME_WRITE_VERIFY_SKIP] "
                "reason=missing_expected_tensors chunks=%d",
                int(in_flight_request.prefix_hit_chunks),
            )
            return False
        # raw verify 会从最终 paged KV cache 做少量 D2H 抽样读回。当前 V100
        # persistent service 是常驻 CTA，如果此时 service 已经没有活跃 request
        # 但仍在空转，D2H 调试读回可能被后台 kernel 拖住，表现为日志停在
        # `packed_sample_extract_begin`。
        #
        # 因此这里和 official-write verify 使用同一个安全边界：只在 runtime
        # 已经 idle 时退役 service；如果还有活跃请求，底层不会停止它。这样 raw
        # verify 看到的仍然是 repair 前的原始 one-copy 写入结果，但不会被空转
        # persistent CTA 卡住。
        self._stop_kv_runtime_service_if_idle_for_verify_debug()
        self._verify_runtime_direct_placement_write_against_materialized_chunks(
            in_flight_request=in_flight_request,
            consumable_chunks=int(in_flight_request.prefix_hit_chunks),
            expected_tensors=expected_tensors,
        )
        logger.info(
            "[LMCACHE_BAM_DIRECT_PLACEMENT_RAW_RUNTIME_WRITE_VERIFY_DONE] "
            "chunks=%d",
            int(in_flight_request.prefix_hit_chunks),
        )
        return True

    def _verify_one_copy_dense_prefix_against_write_reference(
        self,
        *,
        in_flight_request: _InFlightDirectPlacementRequest,
    ) -> None:
        """可选校验 one-copy dense prefix 是否等于 shadow write 源数据。"""
        if not _env_enabled("VLLM_BAM_WRITE_READ_VERIFY"):
            return
        if _env_enabled("VLLM_BAM_WRITE_READ_VERIFY_SYNC_COMPARE"):
            # 写读源数据校验会把少量 decoded tensor 样本从 GPU 拷回 CPU。
            # 这个同步点只属于强调试支线；为了避免它和已经 idle 的 persistent
            # service 互相等待，这里在真正做 D2H 对比前先尝试停止后台 service。
            self._stop_kv_runtime_service_if_idle_for_verify_debug()
        verified = True
        prefix_keys = in_flight_request.keys[:in_flight_request.prefix_hit_chunks]
        for key, decoded_tensor in zip(
                prefix_keys,
                in_flight_request.materialized_prefix_chunk_tensors or ()):
            verified = (
                self._verify_decoded_chunk_against_write_reference(
                    chunk_hash=_extract_chunk_hash(key),
                    decoded_tensor=decoded_tensor,
                ) and verified)
        if not verified:
            raise RuntimeError(
                "BaM write/read verify failed; decoded prefix chunk does not "
                "match shadow write source")

    def _finalize_one_copy_read_request(
        self,
        *,
        in_flight_request: _InFlightDirectPlacementRequest,
    ) -> _DirectPlacementFinalizeReadOutcome:
        """收口 `gpu_worker_persistent_one_copy` 的 read 阶段。

        当前主线已经重新切回真正 one-copy：

        ```text
        GPU persistent service:
          BaM cache page -> vLLM paged KV cache

        host finalize:
          cleanup-only / detach request lifecycle
        ```

        这里不再消费出 `BaMKVReadResult[]`，也不再把请求绕回
        `_finalize_materialized_pipeline()`。如果显式打开 runtime-write verify，
        会在 cleanup 前用 live request pages 构造一份小规模 correctness anchor；
        默认性能路径不做这一步。
        """
        read_consume_start = self._log_direct_read_consume_begin(
            in_flight_request=in_flight_request)
        runtime_cleanup_handle = in_flight_request.kv_read_handle
        runtime_verify_expected_tensors = (
            self._prepare_one_copy_runtime_verify_expected_tensors(
                in_flight_request=in_flight_request))

        runtime_cleanup_done = (
            self.finalize_chunk_pages_kv_fast_path_batch_request_runtime_direct_placement(
                in_flight_request.kv_read_handle,
                timeout_s=float(envs.VLLM_ENGINE_ITERATION_TIMEOUT_S),
            ))
        if not runtime_cleanup_done:
            raise RuntimeError(
                "runtime direct placement did not reach cleanup-only "
                "completion")
        self._mark_direct_read_ready_and_log(
            in_flight_request=in_flight_request,
            ready_chunks=in_flight_request.prefix_hit_chunks,
            read_consume_start=read_consume_start,
            read_finalize_mode=_DIRECT_FINALIZE_MODE_RUNTIME_DIRECT,
        )
        in_flight_request.kv_read_handle = None
        in_flight_request.runtime_cleanup_handle = runtime_cleanup_handle
        return _DirectPlacementFinalizeReadOutcome(
            read_finalize_mode=_DIRECT_FINALIZE_MODE_RUNTIME_DIRECT,
            pipeline_name=_DIRECT_PIPELINE_GPU_WORKER_PERSISTENT_ONE_COPY,
            runtime_cleanup_handle=runtime_cleanup_handle,
            runtime_verify_expected_tensors=runtime_verify_expected_tensors,
        )

    def _consume_materialized_read_request(
        self,
        *,
        in_flight_request: _InFlightDirectPlacementRequest,
    ) -> _DirectPlacementFinalizeReadOutcome:
        """收口 materialized 链路的 read 阶段。

        这条 helper 同时服务：

        - `rowctx_baseline`
        - `gpu_worker_persistent_materialized`

        两者之后都会进入 `_finalize_materialized_pipeline()`，因此这里的唯一职责是
        消费出按 chunk 切分好的 pages，并发布 read-ready frontier。
        """
        read_consume_start = self._log_direct_read_consume_begin(
            in_flight_request=in_flight_request)
        runtime_cleanup_handle = in_flight_request.kv_read_handle
        in_flight_request.results = (
            self.consume_chunk_pages_kv_fast_path_batch_request(
                in_flight_request.kv_read_handle,
                timeout_s=float(envs.VLLM_ENGINE_ITERATION_TIMEOUT_S),
            ))
        self._mark_direct_read_ready_and_log(
            in_flight_request=in_flight_request,
            ready_chunks=len(in_flight_request.results),
            read_consume_start=read_consume_start,
            read_finalize_mode=_DIRECT_FINALIZE_MODE_RESULTS_MATERIALIZED,
        )
        in_flight_request.kv_read_handle = None
        in_flight_request.runtime_cleanup_handle = runtime_cleanup_handle
        return _DirectPlacementFinalizeReadOutcome(
            read_finalize_mode=_DIRECT_FINALIZE_MODE_RESULTS_MATERIALIZED,
            pipeline_name=self._direct_placement_pipeline_name(
                read_finalize_mode=_DIRECT_FINALIZE_MODE_RESULTS_MATERIALIZED,
                runtime_direct_placement_attached=False,
            ),
            runtime_cleanup_handle=runtime_cleanup_handle,
            runtime_verify_expected_tensors=None,
        )

    def _finalize_persistent_one_copy_pipeline(
        self,
        *,
        in_flight_request: _InFlightDirectPlacementRequest,
        runtime_cleanup_handle: Any | None,
        runtime_verify_expected_tensors: dict[str, torch.Tensor] | None,
    ) -> _DirectPlacementFinalizeBackendOutcome:
        """收口 one-copy 实验主线。

        这条路径只做两件事：

        1. 把 tracker 推到连续 consumable frontier
        2. 不再做任何 host 侧 placement / wave launch
        """
        state_tracker = in_flight_request.state_tracker
        bootstrap_profile = in_flight_request.bootstrap_profile

        state_tracker.mark_chunks_staged_ready_upto(
            in_flight_request.prefix_hit_chunks)
        state_tracker.mark_chunks_cache_ready_upto(
            in_flight_request.prefix_hit_chunks)
        raw_runtime_write_verified = (
            self._verify_one_copy_raw_runtime_write_before_repair(
                in_flight_request=in_flight_request,
                runtime_verify_expected_tensors=runtime_verify_expected_tensors,
            ))
        # 这里不再执行 official-write repair。真正的 one-copy 语义是：
        # GPU persistent service 已经把 BaM cache page 直接写进最终 vLLM paged
        # KV cache；host 只发布 frontier/ret_mask。若输出仍错，必须继续修
        # runtime scatter 或读侧解释，而不是再用 repair 覆盖错误。
        place_stats = type(
            "PlaceStats", (),
            {
                "impl": "gpu_runtime_direct",
                "refill_ms": 0.0,
                "transfer_ms": 0.0,
                "fused_ms": 0.0,
                "place_ms": 0.0,
            })()
        first_wave_snapshot = state_tracker.snapshot()
        frontier_wave_ms = 0.0
        bootstrap_profile.prepare_ms = 0.0
        cache_ready_log_start = time.perf_counter()
        self._log_direct_placement_state(
            stage="cache_ready",
            tracker=state_tracker,
        )
        cache_ready_log_ms = (
            time.perf_counter() - cache_ready_log_start) * 1000.0
        return _DirectPlacementFinalizeBackendOutcome(
            backend_name="runtime_direct_cleanup",
            snapshot=first_wave_snapshot,
            place_stats=place_stats,
            frontier_wave_ms=frontier_wave_ms,
            cache_ready_log_ms=cache_ready_log_ms,
            raw_runtime_write_verified=raw_runtime_write_verified,
        )

    def _finalize_materialized_pipeline(
        self,
        *,
        in_flight_request: _InFlightDirectPlacementRequest,
    ) -> _DirectPlacementFinalizeBackendOutcome:
        """收口 materialized placement 主线。

        这条路径同时覆盖两种保留链路：

        - rowctx_baseline
        - gpu_worker_persistent_materialized

        它们的共同点是：最终写入 vLLM paged KV cache 仍走已经验证正确的
        materialized placement，而不是让 persistent service 直接写最终 cache。
        """
        bootstrap_profile = in_flight_request.bootstrap_profile
        state_tracker = in_flight_request.state_tracker
        logger.info(
            "[LMCACHE_BAM_DIRECT_PLACEMENT_FINALIZE_BEGIN] "
            "mode=%s chunks=%d launch_chunks=%d "
            "return_target_chunks=%d",
            _DIRECT_FINALIZE_MODE_RESULTS_MATERIALIZED,
            int(len(in_flight_request.results or [])),
            int(in_flight_request.first_wave_launch_chunks),
            int(in_flight_request.first_wave_return_target_chunks),
        )
        # materialized finalize 代表当前请求已经完成 BaM page read，并把 pages
        # 物化到 `request_table.pages`。接下来的 direct placement/refill 是另一组
        # GPU kernel，它不应该和空转的 persistent service CTA 抢同一张卡的执行资源。
        #
        # 因此这里在 host placement 前建立一个明确边界：
        # - 如果 runtime 仍有活跃请求，底层不会停止 service；
        # - 如果本请求已经 cleanup 完成且 runtime 空闲，则退役 service；
        # - 然后再启动已验证正确的 rowctx/materialized placement 路径。
        self._stop_kv_runtime_service_if_idle(
            source="materialized_finalize",
            reason="before_host_placement",
        )
        prepare_start = time.perf_counter()
        logger.info(
            "[LMCACHE_BAM_DIRECT_PLACEMENT_FINALIZE_PREPARE_BEGIN] "
            "chunks=%d",
            int(len(in_flight_request.results or [])),
        )
        prepare_bam_results_for_vllm_kvcache(
            results=in_flight_request.results,
            layout=self.layout,
            kv_caches=in_flight_request.kv_caches,
            slot_mapping=in_flight_request.slot_mapping,
            chunk_starts=in_flight_request.chunk_starts,
            kv_cache_dtype=in_flight_request.kv_cache_dtype,
            placer=in_flight_request.direct_placer,
            launch_start_chunk=0,
            max_chunks_to_launch=(
                in_flight_request.first_wave_launch_chunks),
        )
        bootstrap_profile.prepare_ms = (
            time.perf_counter() - prepare_start) * 1000.0
        logger.info(
            "[LMCACHE_BAM_DIRECT_PLACEMENT_FINALIZE_PREPARE_DONE] "
            "prepare_ms=%.3f",
            bootstrap_profile.prepare_ms,
        )

        bootstrap_profile.frontier_wave_start_time = time.perf_counter()
        logger.info(
            "[LMCACHE_BAM_DIRECT_PLACEMENT_FINALIZE_WAVE_BEGIN] "
            "launch_chunks=%d return_target_chunks=%d",
            int(in_flight_request.first_wave_launch_chunks),
            int(in_flight_request.first_wave_return_target_chunks),
        )
        place_stats, first_wave_snapshot = self._run_direct_placement_wave(
            direct_placer=in_flight_request.direct_placer,
            results=in_flight_request.results,
            kv_caches=in_flight_request.kv_caches,
            slot_mapping=in_flight_request.slot_mapping,
            chunk_starts=in_flight_request.chunk_starts,
            state_tracker=state_tracker,
            launch_start_chunk=0,
            launch_chunk_count=in_flight_request.first_wave_launch_chunks,
            return_target_chunks=(
                in_flight_request.first_wave_return_target_chunks),
            wave_name="frontier",
            do_prepare=False,
        )
        frontier_wave_ms = 0.0
        if bootstrap_profile.frontier_wave_start_time is not None:
            frontier_wave_ms = (
                time.perf_counter() - bootstrap_profile.frontier_wave_start_time
            ) * 1000.0
        logger.info(
            "[LMCACHE_BAM_DIRECT_PLACEMENT_FINALIZE_WAVE_DONE] "
            "frontier_wave_ms=%.3f cache_ready_chunks=%d consumable_chunks=%d "
            "impl=%s place_ms=%.3f",
            frontier_wave_ms,
            int(first_wave_snapshot.cache_ready_chunks),
            int(first_wave_snapshot.consumable_chunks),
            str(getattr(place_stats, "impl", "unknown")),
            float(getattr(place_stats, "place_ms", 0.0)),
        )
        cache_ready_log_start = time.perf_counter()
        self._log_direct_placement_state(
            stage="cache_ready",
            tracker=state_tracker,
        )
        cache_ready_log_ms = (
            time.perf_counter() - cache_ready_log_start) * 1000.0
        return _DirectPlacementFinalizeBackendOutcome(
            backend_name="materialized_host_finalize",
            snapshot=first_wave_snapshot,
            place_stats=place_stats,
            frontier_wave_ms=frontier_wave_ms,
            cache_ready_log_ms=cache_ready_log_ms,
        )

    def _direct_placement_pipeline_name(
        self,
        *,
        read_finalize_mode: str | None = None,
        runtime_direct_placement_attached: bool = False,
    ) -> str:
        """把内部 finalize mode 映射成当前保留的三条 KV 链路名称。

        这里故意只做只读判定，不参与任何控制流：

        - 内部函数继续用 `runtime_direct/results_materialized` 做稳定分叉；
        - 日志、文档和脚本统一用 pipeline 名称表达“到底跑的是哪条链路”。

        这样可以避免后续再把当前输出正确的
        `gpu_worker_persistent_materialized` 误认为是 legacy fallback。
        """
        if (read_finalize_mode == _DIRECT_FINALIZE_MODE_RUNTIME_DIRECT
                or runtime_direct_placement_attached):
            return _DIRECT_PIPELINE_GPU_WORKER_PERSISTENT_ONE_COPY
        if self._persistent_runtime_mainline_enabled():
            return _DIRECT_PIPELINE_GPU_WORKER_PERSISTENT_MATERIALIZED
        return _DIRECT_PIPELINE_ROWCTX_BASELINE

    def _finalize_direct_placement_with_backend(
        self,
        *,
        read_finalize_mode: str,
        in_flight_request: _InFlightDirectPlacementRequest,
        runtime_cleanup_handle: Any | None,
        runtime_verify_expected_tensors: dict[str, torch.Tensor] | None,
    ) -> _DirectPlacementFinalizeBackendOutcome:
        """按显式 finalize consume backend 收口 direct placement request。"""
        if read_finalize_mode == _DIRECT_FINALIZE_MODE_RUNTIME_DIRECT:
            return self._finalize_persistent_one_copy_pipeline(
                in_flight_request=in_flight_request,
                runtime_cleanup_handle=runtime_cleanup_handle,
                runtime_verify_expected_tensors=runtime_verify_expected_tensors,
            )
        if read_finalize_mode == _DIRECT_FINALIZE_MODE_RESULTS_MATERIALIZED:
            return self._finalize_materialized_pipeline(
                in_flight_request=in_flight_request,
            )
        raise ValueError(
            f"Unknown direct placement finalize mode: {read_finalize_mode}")

    def _finalize_direct_placement_request(
        self,
        *,
        in_flight_request: _InFlightDirectPlacementRequest,
    ) -> torch.Tensor:
        """同步收口一次已经 start 的 direct placement request。

        这里保留当前主线需要的同步语义：

        - 先把 native read consume 成当前请求可见的结果
        - 再执行单波 placement 并等待返回目标满足
        - 再构造 ret_mask 并做返回语义校验

        但和旧版本不同的是，所有“已启动 request 的稳定上下文”都来自
        `in_flight_request`，不再依赖一个大函数里的局部变量链式传递。
        """
        bootstrap_profile = in_flight_request.bootstrap_profile
        state_tracker = in_flight_request.state_tracker
        read_outcome = self._consume_direct_placement_read_request(
            in_flight_request=in_flight_request)
        # 每次 finalize 都先把“本次 request 对上层是否可直接信任 runtime
        # metadata”重置掉，避免旧 request 某轮留下的状态混到新一轮判断中。
        in_flight_request.runtime_metadata_fast_path_authoritative = False
        in_flight_request.runtime_metadata_consumable_tokens = 0
        read_finalize_mode = str(read_outcome.read_finalize_mode)
        pipeline_name = str(read_outcome.pipeline_name)
        runtime_cleanup_handle = read_outcome.runtime_cleanup_handle
        runtime_verify_expected_tensors = read_outcome.runtime_verify_expected_tensors
        finalize_backend_outcome = self._finalize_direct_placement_with_backend(
            read_finalize_mode=read_finalize_mode,
            in_flight_request=in_flight_request,
            runtime_cleanup_handle=runtime_cleanup_handle,
            runtime_verify_expected_tensors=runtime_verify_expected_tensors,
        )
        first_wave_snapshot = finalize_backend_outcome.snapshot
        place_stats = finalize_backend_outcome.place_stats
        frontier_wave_ms = finalize_backend_outcome.frontier_wave_ms
        cache_ready_log_ms = finalize_backend_outcome.cache_ready_log_ms
        raw_runtime_write_verified = (
            finalize_backend_outcome.raw_runtime_write_verified)

        # ret_mask 必须绑定到当前这次单波 placement 真正形成的 contiguous
        # consumable frontier，而不能简单等同于“这次命中了多少 prefix chunk”。
        #
        # 只有已经连续 cache-ready 的前缀，才能安全暴露给 LMCache / vLLM。
        ret_mask_start = time.perf_counter()
        ret_mask = self._build_consumable_ret_mask(
            tokens=in_flight_request.tokens,
            snapshot=first_wave_snapshot,
        )
        ret_mask_ms = (time.perf_counter() - ret_mask_start) * 1000.0
        return_snapshot = first_wave_snapshot
        if read_finalize_mode == _DIRECT_FINALIZE_MODE_RUNTIME_DIRECT:
            # 这条校验只服务当前最关键的定位问题：
            #
            #   persistent service + cleanup-only direct placement
            #     已经跑通
            #   但最终模型输出仍然错误
            #
            # 这时我们首先要确认的，不是 metadata fast path，而是：
            #
            #   最终写进 vLLM paged KV cache 的值
            #   是否已经与旧 materialize 语义完全一致
            #
            # 因此这里把对照点放在 cleanup-only 收口之后、ret_mask 已可见之前。
            #
            # 注意：这里还会额外做一件只属于 verify 调试支线的事：
            # - 如果当前 persistent service 已经空闲，就先把它停掉
            # - 再去直接回读最终 paged KV cache
            #
            # 原因是当前已经验证到：
            # - 主线 `consumable` 已经形成
            # - 但 verify 若在 service 常驻时直接读最终 kv_cache，仍可能被拖住
            #
            # 因此这里显式把“主线已经完成”和“调试读取最终 cache”拆开。
            if (envs.VLLM_BAM_DIRECT_PLACEMENT_VERIFY_RUNTIME_WRITE
                    and not raw_runtime_write_verified):
                self._stop_kv_runtime_service_if_idle_for_verify_debug()
                self._verify_runtime_direct_placement_write_against_materialized_chunks(
                    in_flight_request=in_flight_request,
                    consumable_chunks=int(return_snapshot.consumable_chunks),
                    expected_tensors=runtime_verify_expected_tensors,
                )
            elif raw_runtime_write_verified:
                logger.info(
                    "[LMCACHE_BAM_DIRECT_PLACEMENT_VERIFY_RUNTIME_WRITE_SKIP] "
                    "reason=raw_runtime_write_already_verified_before_repair "
                    "consumable_chunks=%d",
                    int(return_snapshot.consumable_chunks),
                )
        # 只有 runtime direct cleanup-only 这条主线，才能说明：
        #
        # 1. GPU persistent service 已经完成
        #      BaM cache -> vLLM KV cache
        # 2. 并且与这条 request 绑定的 attention metadata workspace 也应该已经
        #    处于“可直接消费”的完成态
        #
        # 因此这里把“可直接走 runtime metadata fast path”的 authoritative
        # request-level 语义显式发布给上层，而不再让 adapter 去额外猜测
        # 某个中间 ready_flag 是否恰好被读成 1。
        if (in_flight_request.runtime_direct_placement_attached and
                read_finalize_mode == _DIRECT_FINALIZE_MODE_RUNTIME_DIRECT):
            # `runtime_metadata_consumable_tokens` 的发布不应再绑死在 metadata
            # attachment 是否启用上。
            #
            # 原因是当前默认主线里，这条 metadata attachment 实验支线本来就是关的；
            # 但即便如此，我们仍然需要把“这次 request 实际已经恢复了多少连续
            # prefix token”稳定发布给上层日志/调试口径。
            in_flight_request.runtime_metadata_consumable_tokens = int(
                return_snapshot.consumable_tokens)
        if (in_flight_request.runtime_direct_placement_attached and
                in_flight_request.runtime_attention_metadata_attached and
                read_finalize_mode == _DIRECT_FINALIZE_MODE_RUNTIME_DIRECT):
            in_flight_request.runtime_metadata_fast_path_authoritative = True
            in_flight_request.runtime_metadata_consumable_tokens = int(
                return_snapshot.consumable_tokens)
        self._validate_direct_placement_return_semantics(
            tokens=in_flight_request.tokens,
            ret_mask=ret_mask,
            return_snapshot=return_snapshot,
            prefix_hit_chunks=in_flight_request.prefix_hit_chunks,
            return_target_chunks=in_flight_request.first_wave_return_target_chunks,
        )

        final_log_start = time.perf_counter()
        if in_flight_request.results is not None:
            total_bytes = sum(
                int(result.descriptor.total_bytes)
                for result in in_flight_request.results
            )
            batch_stats = in_flight_request.results[0].stats
            read_submit_ms = float(batch_stats.submit_ms)
            read_poll_ms = float(batch_stats.poll_ms)
            read_poll_iters = int(batch_stats.poll_iters)
            read_get_ms = float(batch_stats.get_ms)
            read_total_ms = float(
                getattr(
                    batch_stats,
                    "total_ms",
                    getattr(batch_stats, "submit_ms", 0.0) +
                    getattr(batch_stats, "poll_ms", 0.0) +
                    getattr(batch_stats, "get_ms", 0.0),
                ))
            read_executor_name = str(
                getattr(batch_stats, "executor_name", "rowctx"))
            read_worker_backend = str(
                getattr(batch_stats, "worker_backend", "rowctx"))
        else:
            if runtime_cleanup_handle is None:
                raise RuntimeError(
                    "runtime cleanup-only direct placement lost kv_read_handle "
                    "before logging")
            native_handle = getattr(runtime_cleanup_handle, "native_handle", None)
            total_bytes = int(state_tracker.snapshot().descriptor.total_bytes)
            # 单测桩可能只关心“是否走到了 cleanup-only 主线”，不会完整构造
            # native_handle。这里允许在缺少底层句柄时退化成 descriptor 级统计，
            # 避免让日志/profile 假设反过来卡住主逻辑验证。
            read_submit_ms = float(
                getattr(native_handle, "submit_ms", 0.0)
            ) if native_handle is not None else 0.0
            read_poll_ms = float(
                getattr(native_handle, "poll_ms", 0.0) or 0.0
            ) if native_handle is not None else 0.0
            read_poll_iters = int(
                getattr(native_handle, "poll_iters", 0)
            ) if native_handle is not None else 0
            read_get_ms = float(
                getattr(native_handle, "get_ms", 0.0) or 0.0
            ) if native_handle is not None else 0.0
            read_total_ms = float(
                getattr(native_handle, "total_ms", 0.0)
                or (read_submit_ms + read_poll_ms + read_get_ms)
            ) if native_handle is not None else 0.0
            read_executor_name = str(finalize_backend_outcome.backend_name)
            read_worker_backend = str(
                getattr(native_handle, "worker_backend", "rowctx")
            ) if native_handle is not None else "gpu_runtime_direct_cleanup"
        total_tokens = int(return_snapshot.consumable_tokens)
        logger.info(
            "[LMCACHE_BAM_DIRECT_PLACEMENT_READ] batch_size=%d total_bytes=%d "
            "submit_ms=%.3f poll_ms=%.3f poll_iters=%d get_ms=%.3f "
            "read_ms=%.3f executor=%s worker_backend=%s pipeline=%s "
            "finalize_mode=%s",
            in_flight_request.prefix_hit_chunks,
            total_bytes,
            read_submit_ms,
            read_poll_ms,
            read_poll_iters,
            read_get_ms,
            read_total_ms,
            read_executor_name,
            read_worker_backend,
            pipeline_name,
            read_finalize_mode,
        )
        logger.info(
            "[LMCACHE_BAM_DIRECT_PLACEMENT] chunks=%d tokens=%d "
            "impl=%s read_ms=%.3f refill_ms=%.3f transfer_ms=%.3f "
            "fused_ms=%.3f place_ms=%.3f total_ms=%.3f pipeline=%s",
            in_flight_request.prefix_hit_chunks,
            total_tokens,
            place_stats.impl,
            read_total_ms,
            place_stats.refill_ms,
            place_stats.transfer_ms,
            place_stats.fused_ms,
            place_stats.place_ms,
            read_total_ms + place_stats.place_ms,
            pipeline_name,
        )
        final_log_ms = (time.perf_counter() - final_log_start) * 1000.0
        direct_total_ms = (
            time.perf_counter() - bootstrap_profile.direct_total_start_time
        ) * 1000.0
        logger.info(
            "[LMCACHE_BAM_DIRECT_PLACEMENT_PROFILE] collect_entries_ms=%.3f "
            "read_submit_ms=%.3f read_ms=%.3f descriptor_ms=%.3f "
            "tracker_init_ms=%.3f "
            "prepare_ms=%.3f frontier_wave_wall_ms=%.3f "
            "cache_ready_log_ms=%.3f ret_mask_ms=%.3f "
            "final_log_ms=%.3f direct_total_ms=%.3f",
            bootstrap_profile.collect_entries_ms,
            bootstrap_profile.read_submit_ms,
            read_total_ms,
            bootstrap_profile.descriptor_ms,
            bootstrap_profile.tracker_init_ms,
            bootstrap_profile.prepare_ms,
            frontier_wave_ms,
            cache_ready_log_ms,
            ret_mask_ms,
            final_log_ms,
            direct_total_ms,
        )
        return ret_mask

    def start_direct_placement_request(
        self,
        *,
        token_database: Any,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor],
        kv_caches: list[torch.Tensor],
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str = "auto",
        num_kv_heads: int = 0,
        head_size: int = 0,
    ) -> Optional[Any]:
        """公开 direct placement request 的 start 边界。

        当前返回值仍然故意声明为 `Any`，原因是这只是第一版对外暴露的 request
        handle，调用方只需要把它当成一个“后续可 poll/finalize 的不透明句柄”
        来传递，而不应该依赖内部 dataclass 字段。

        这样后续如果我们继续调整 request handle 的内部组织方式，就不需要同步改
        上层 runtime / adapter 的类型依赖。
        """
        return self._start_direct_placement_request(
            token_database=token_database,
            tokens=tokens,
            mask=mask,
            kv_caches=kv_caches,
            slot_mapping=slot_mapping,
            kv_cache_dtype=kv_cache_dtype,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
        )

    def poll_direct_placement_request(
        self,
        *,
        in_flight_request: Any,
    ) -> BaMDirectPlacementBatchStateSnapshot:
        """公开 direct placement request 的非阻塞 poll 边界。"""
        return self._poll_direct_placement_request(
            in_flight_request=in_flight_request,
        )

    def get_direct_placement_request_frontier(
        self,
        *,
        in_flight_request: Any,
    ) -> BaMDirectPlacementFrontierSnapshot:
        """公开 direct placement request 的统一 frontier 观察接口。

        这层接口不主动推进状态，只读取当前 request handle 已可见的 frontier。

        注意这条接口现在主要服务调试 / 观察，不建议作为高频调度热路径使用。
        对运行时调度来说，应优先复用 `poll_direct_placement_request()` 的返回值。

        后续如果更高层 runtime 想把：

        - `poll()` 负责推进
        - `get_frontier()` 负责观察

        显式拆开，就可以直接复用这里，而不用再理解 request/wave/tracker 的内部
        组织方式。
        """
        return self._get_direct_placement_request_frontier(
            in_flight_request=in_flight_request,
        )

    def get_direct_placement_request_frontier_table(
        self,
        *,
        in_flight_request: Any,
    ) -> Optional[torch.Tensor]:
        """公开 direct placement request 的统一 frontier table。

        这条接口当前的意义主要是：

        - 让更高层 runtime 可以直接持有同一张 GPU-visible frontier table
        - 避免调用方去依赖 request handle 的内部 dataclass 字段

        当前它只是一个很薄的 accessor；但这正好符合第三步现阶段的目标：
        先把“统一 frontier ABI 的所有权”稳定下来，再继续下压更新逻辑。
        """
        return getattr(in_flight_request, "frontier_table", None)

    def get_direct_placement_request_runtime_snapshot(
        self,
        *,
        in_flight_request: Any,
    ) -> Optional[Any]:
        """公开 direct placement request 当前对应的底层 runtime 观察快照。

        当前这层只桥接 native KV read 阶段，因为现阶段真正接入 GPU worker
        runtime / persistent service 的还是底层 BaM page read。

        后续如果 placement 自身也进入统一 GPU runtime，这个接口可以继续沿用，
        只需要把来源从 `kv_read_handle` 扩展到更完整的 request runtime 结构。
        """
        kv_read_handle = getattr(in_flight_request, "kv_read_handle", None)
        if kv_read_handle is None:
            return None
        return self.get_chunk_pages_kv_fast_path_batch_request_runtime_snapshot(
            kv_read_handle)

    def finalize_direct_placement_request(
        self,
        *,
        in_flight_request: Any,
    ) -> torch.Tensor:
        """公开 direct placement request 的同步 finalize 边界。"""
        return self._finalize_direct_placement_request(
            in_flight_request=in_flight_request,
        )

    def _build_consumable_ret_mask(
        self,
        *,
        tokens: torch.Tensor,
        snapshot: BaMDirectPlacementBatchStateSnapshot,
    ) -> torch.Tensor:
        """基于当前 contiguous cache-ready frontier 构造返回给 LMCache 的 ret_mask。

        这里刻意不用“所有命中的 chunk”直接生成 mask，而是只暴露当前真正
        `consumable` 的连续前缀。原因是：

        - prefix 语义要求必须是“从开头连续命中”
        - 后续如果某个 chunk 已经命中但还没 placement 完成，它不能被上层当成
          已恢复 prefix 使用

        因此 ret_mask 的口径应当绑定到：

        ```text
        contiguous cache-ready frontier
        ```

        而不是“这轮一共命中了多少个 chunk”。
        """
        ret_mask = torch.zeros_like(tokens, dtype=torch.bool, device="cpu")
        for chunk_state in snapshot.chunk_states:
            if not chunk_state.cache_ready:
                break
            ret_mask[chunk_state.descriptor.chunk_start:chunk_state.descriptor.
                     chunk_end] = True
        return ret_mask

    @staticmethod
    def _count_ret_mask_tokens(ret_mask: torch.Tensor) -> int:
        """统计返回给 LMCache / vLLM 的 prefix token 数。"""
        return int(ret_mask.sum().item())

    def _validate_direct_placement_return_semantics(
        self,
        *,
        tokens: torch.Tensor,
        ret_mask: torch.Tensor,
        return_snapshot: BaMDirectPlacementBatchStateSnapshot,
        prefix_hit_chunks: int,
        return_target_chunks: int,
    ) -> None:
        """校验 direct placement 主线的“返回语义”是否自洽。

        这里故意把检查集中在 store 内部，而不是分散到多个调用点，原因是：

        - 这里同时能看到底层状态快照和最终返回给上层的 `ret_mask`
        - 这正是“正常推理引擎语义”最容易被未来优化破坏的边界

        当前要求至少满足：

        1. `ret_mask` 只能覆盖当前 contiguous consumable frontier
        2. `ret_mask` token 数必须等于 snapshot 的 `consumable_tokens`
        3. 当前返回目标不能超过 prefix hit，也不能超过 snapshot 已暴露的范围
        """
        ret_mask_tokens = self._count_ret_mask_tokens(ret_mask)
        consumable_tokens = int(return_snapshot.consumable_tokens)
        consumable_chunks = int(return_snapshot.consumable_chunks)

        if ret_mask_tokens != consumable_tokens:
            raise RuntimeError(
                "direct placement return semantics mismatch: "
                f"ret_mask_tokens={ret_mask_tokens} "
                f"consumable_tokens={consumable_tokens} "
                f"consumable_chunks={consumable_chunks}")

        if consumable_chunks > int(prefix_hit_chunks):
            raise RuntimeError(
                "direct placement consumable frontier exceeds prefix hit range: "
                f"consumable_chunks={consumable_chunks} "
                f"prefix_hit_chunks={prefix_hit_chunks}")

        if consumable_chunks < int(return_target_chunks):
            raise RuntimeError(
                "direct placement returned before target prefix became consumable: "
                f"consumable_chunks={consumable_chunks} "
                f"return_target_chunks={return_target_chunks}")

        if ret_mask.shape != tokens.shape:
            raise RuntimeError(
                "direct placement ret_mask shape mismatch: "
                f"ret_mask_shape={tuple(ret_mask.shape)} "
                f"tokens_shape={tuple(tokens.shape)}")

        logger.info(
            "[LMCACHE_BAM_DIRECT_PLACEMENT_RETURN_SEMANTICS] "
            "prefix_hit_chunks=%d return_target_chunks=%d "
            "consumable_chunks=%d ret_mask_tokens=%d",
            prefix_hit_chunks,
            return_target_chunks,
            consumable_chunks,
            ret_mask_tokens,
        )

    def direct_place_chunks_to_vllm_kvcache(
        self,
        *,
        token_database: Any,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor],
        kv_caches: list[torch.Tensor],
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str = "auto",
        num_kv_heads: int,
        head_size: int,
    ) -> Optional[torch.Tensor]:
        """把 BaM 命中的前缀 chunks 直接写入 vLLM paged KV cache。

        这是 Direct Placement v0 在 `LMCacheBaMStore` 里的数据面总入口。
        它把原来分散在 storage manager 里的几步操作收口到一起：

        ```text
        token_database.process_tokens()
          -> 找到 BaM 可服务的前缀 chunks
          -> BaM KV fast path batch read pages
          -> direct placement 写入 vLLM paged KV cache
          -> 返回 ret_mask 给 LMCache adapter
        ```

        返回值语义：

        - `torch.Tensor`:
            BaM 已经成功服务至少一个 chunk，返回当前这部分命中的 ret_mask。
            LMCache adapter 会按这份 mask 重建 model input。
        - `None`:
            当前请求不应继续走 direct placement，调用方应回退到 LMCache 原始
            retrieve 路径。典型场景包括：
            - BaM 还没初始化 / 当前没写入过 metadata
            - 这个前缀在 BaM 中 0 命中
            - KV fast path 未启用

        这里把“0 命中”定义为 `None` 而不是全 False mask，是为了避免误吞掉
        原始 LMCache SSD retrieve。否则 direct placement 开关一开，BaM 没命中
        时就会直接返回 miss，后面的 LMCache fallback 根本没有机会执行。
        """
        in_flight_request = self.start_direct_placement_request(
            token_database=token_database,
            tokens=tokens,
            mask=mask,
            kv_caches=kv_caches,
            slot_mapping=slot_mapping,
            kv_cache_dtype=kv_cache_dtype,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
        )
        if in_flight_request is None:
            return None

        # 当前同步主线仍然会在同一次 direct retrieve 里把 request 收口，但这里
        # 先显式保留一次 request 级 poll，作为后续 runtime 周期性推进 ready
        # 状态的统一入口。
        self.poll_direct_placement_request(
            in_flight_request=in_flight_request,
        )
        return self.finalize_direct_placement_request(
            in_flight_request=in_flight_request,
        )

    def _ensure_direct_kv_placer(
        self,
        *,
        kv_cache_dtype: str,
    ) -> BaMDirectKVPlacer:
        """按 BaM store 生命周期复用一个 direct KV placer。"""
        if self._direct_kv_placer is None:
            self._direct_kv_placer = BaMDirectKVPlacer(
                layout=self.layout,
                kv_cache_dtype=kv_cache_dtype,
            )
        return self._direct_kv_placer

    def _gpu_worker_runtime_enabled(self) -> bool:
        """判断当前是否启用了 GPU-visible runtime slot。"""
        return _env_enabled("GIDS_KV_GPU_WORKER_RUNTIME_ENABLE")

    def _gpu_worker_persistent_enabled(self) -> bool:
        """判断当前是否启用了 persistent service CTA。"""
        return _env_enabled("GIDS_KV_GPU_WORKER_PERSISTENT_ENABLE")

    def _persistent_runtime_mainline_enabled(self) -> bool:
        """判断当前是否处在 GPU worker runtime/persistent 主线上。

        这里直接读取环境变量，而不是 `getattr(vllm.envs, "GIDS_*", ...)`。
        原因是 `GIDS_*` 属于 BaM/GIDS 底层运行时开关，并不由 vLLM 统一导出；
        如果继续从 `envs` 兜默认值，包装脚本虽然传了参数，storage 仍可能误判为
        “没开 runtime/persistent”，从而错误地放行旧回退路径。
        """
        return (self._gpu_worker_runtime_enabled()
                or self._gpu_worker_persistent_enabled())

    def _runtime_direct_path_required(self) -> bool:
        """判断当前请求是否必须严格走 runtime one-copy 主线。"""
        return (self._persistent_runtime_mainline_enabled()
                and bool(
                    envs.
                    VLLM_BAM_DIRECT_PLACEMENT_REQUIRE_RUNTIME_ONE_COPY))

    def _try_attach_runtime_direct_placement(
        self,
        *,
        kv_read_handle: Any | None,
        direct_placer: BaMDirectKVPlacer,
        kv_caches: list[torch.Tensor],
        slot_mapping: torch.Tensor,
        chunk_starts: list[int],
        num_kv_heads: int = 0,
        head_size: int = 0,
    ) -> tuple[bool, BaMRuntimeDirectPlacementAttachment | None]:
        """尽量把最终 direct placement 描述符直接挂到 live native request 上。

        成功后，GPU persistent service 就能在 `request_table.pages` ready 后，
        继续把数据直接 scatter 到最终 paged KV cache，而不需要 host 再发一波
        placement kernel。
        """
        if kv_read_handle is None:
            if self._runtime_direct_path_required():
                raise RuntimeError(
                    "runtime one-copy direct placement requires a live "
                    "native batch handle")
            return False, None
        # runtime one-copy 仍然保留显式总开关；
        # 但如果当前主线声明“必须走 one-copy”，这里就不能再静默跳回
        # materialized finalize。
        if not envs.VLLM_BAM_DIRECT_PLACEMENT_RUNTIME_ONE_COPY:
            if self._runtime_direct_path_required():
                raise RuntimeError(
                    "persistent direct placement requires "
                    "VLLM_BAM_DIRECT_PLACEMENT_RUNTIME_ONE_COPY=1")
            logger.info(
                "[LMCACHE_BAM_DIRECT_PLACEMENT_RUNTIME_ATTACH_SKIP] "
                "reason=runtime_one_copy_disabled",
            )
            return False, None
        try:
            attachment = direct_placer.build_runtime_direct_placement_attachment(
                kv_caches=kv_caches,
                slot_mapping=slot_mapping,
                chunk_starts=chunk_starts,
                num_kv_heads=int(num_kv_heads),
                head_size=int(head_size),
            )
            attached = (
                self.attach_chunk_pages_kv_fast_path_batch_request_runtime_direct_placement(
                    kv_read_handle,
                    slot_mapping=attachment.slot_mapping,
                    chunk_starts=attachment.chunk_starts,
                    kv_cache_pointers_gpu=attachment.kv_cache_pointers_gpu,
                    page_buffer_size=attachment.page_buffer_size,
                    block_size=attachment.block_size,
                    page_token_capacity=attachment.page_token_capacity,
                    pages_per_kv_layer=attachment.pages_per_kv_layer,
                    num_layers=attachment.num_layers,
                    num_kv_heads=attachment.num_kv_heads,
                    head_size=attachment.head_size,
                    pack_size=attachment.pack_size,
                ))
            if attached:
                # 注意：这里不要再对 `attachment.slot_mapping` 做
                # `min()/max().item()` 这类会强制同步 CUDA tensor 的调试读取。
                #
                # 原因有两点：
                #
                # 1. 这一层正处在 runtime direct placement attach 的热路径上，
                #    attach 完成后马上就要进入 launch/poll 主线；
                # 2. `slot_mapping` 常驻在 GPU，上述聚合 + `.item()` 会把当前线程
                #    强制卡在一次设备同步上。最近一轮日志正是停在
                #    `KV_WORKER_RUNTIME_WRITE_SLOT` 之后、这条 attach 日志之前，
                #    说明问题不是 slot 没写进去，而是这类“为了打日志而同步设备”
                #    把控制面卡住了。
                #
                # 因此这里仅保留不会触发额外 CUDA 同步的静态 shape / 容量信息。
                logger.info(
                    "[LMCACHE_BAM_DIRECT_PLACEMENT_RUNTIME_ATTACH] "
                    "chunks=%d page_buffer_size=%d block_size=%d "
                    "page_token_capacity=%d pages_per_kv_layer=%d "
                    "num_layers=%d num_kv_heads=%d head_size=%d pack_size=%d "
                    "slot_mapping_len=%d",
                    len(chunk_starts),
                    attachment.page_buffer_size,
                    attachment.block_size,
                    attachment.page_token_capacity,
                    attachment.pages_per_kv_layer,
                    attachment.num_layers,
                    attachment.num_kv_heads,
                    attachment.head_size,
                    attachment.pack_size,
                    int(attachment.slot_mapping.numel()),
                )
                return True, attachment
        except Exception:
            if self._runtime_direct_path_required():
                raise
            logger.exception(
                "[LMCACHE_BAM_DIRECT_PLACEMENT_RUNTIME_ATTACH] failed; "
                "fall back to host-side placement")
        if self._runtime_direct_path_required():
            raise RuntimeError(
                "persistent direct placement requires successful runtime "
                "one-copy attach")
        return False, None

    def attach_direct_placement_request_runtime_attention_metadata(
        self,
        *,
        in_flight_request: _InFlightDirectPlacementRequest,
        attachment: BaMRuntimeAttentionMetadataAttachment,
    ) -> bool:
        """把单条 sequence 的 attention metadata workspace 挂到 live request 上。

        当前主线刻意把“数据搬运目标”和“attention metadata workspace”拆成两份
        attachment，原因是两者虽然都由同一个 persistent service CTA 消费，但
        职责完全不同：

        - direct placement attachment:
          描述 `BaM cache -> vLLM paged KV cache` 的最终写入目标
        - attention metadata attachment:
          描述 finalize 后当前 sequence 要给 attention / sampling 使用的元数据

        这样后续继续清理旧的前台 rebuild 逻辑时，就不需要把两种语义重新揉回
        一个大结构里。
        """
        kv_read_handle = in_flight_request.kv_read_handle
        if kv_read_handle is None:
            return False
        attached = (
            self.attach_chunk_pages_kv_fast_path_batch_request_runtime_attention_metadata(
                kv_read_handle,
                attachment=attachment,
            ))
        if attached:
            in_flight_request.runtime_attention_metadata_attached = True
            in_flight_request.runtime_attention_metadata_attachment = attachment
            logger.info(
                "[LMCACHE_BAM_DIRECT_PLACEMENT_RUNTIME_METADATA_ATTACH] "
                "tokens=%d total_seq_len=%d vllm_num_computed_tokens=%d "
                "block_size=%d do_sample=%s is_chunk_prefill=%s",
                int(attachment.full_query_slot_mapping_src.numel()),
                int(attachment.total_seq_len),
                int(attachment.vllm_num_computed_tokens),
                int(attachment.block_size),
                str(bool(attachment.do_sample)).lower(),
                str(bool(attachment.is_chunk_prefill)).lower(),
            )
        return attached

    def enqueue_kv_fast_path_prefetch_key(self, key: Any) -> bool:
        """登记本轮 LMCache retrieve 可能会读取的 key。

        第一版真实 vLLM batch fast path 仍然由 CPU 做粗粒度决策：

        ```text
        LMCache engine.prefetch(tokens, mask)
          -> storage_manager.prefetch(key)
          -> enqueue key
        ```

        这里不立即 submit BaM IO，而是把同一轮 retrieve 的 key 收集起来。
        第一次 `get(key)` 进来时，再把这批 key 一次性走
        `load_chunk_tensors_kv_fast_path_batch()`，避免逐 key 串行 read/refill。
        """
        chunk_hash = _extract_chunk_hash(key)
        metadata = self._lookup_metadata(chunk_hash)
        if metadata is None:
            return False

        with self._prefetch_lock:
            self._kv_batch_pending_keys[chunk_hash] = key
            logger.info(
                "[LMCACHE_BAM_KV_FAST_PATH_PREFETCH_ENQUEUE] "
                "chunk_hash=%s pending=%d",
                chunk_hash[:16],
                len(self._kv_batch_pending_keys),
            )
        return True

    def consume_kv_fast_path_tensor(self, key: Any) -> Optional[torch.Tensor]:
        """消费 KV fast path batch 结果。

        返回值语义：

        - 有 batch 结果：返回已经读好的 tensor
        - 有 pending keys 但还没加载：触发一次 batch read，然后返回当前 key
        - 当前 key 不在 batch 中：返回 None，让调用方退回单 chunk fast path

        这保持了正确性：即使某个 key 没被 prefetch 阶段收集，也仍能在 get
        阶段用单 chunk fast path 读取。
        """
        chunk_hash = _extract_chunk_hash(key)
        with self._prefetch_lock:
            tensor = self._kv_batch_loaded_tensors.pop(chunk_hash, None)
            if tensor is not None:
                return tensor

            if chunk_hash not in self._kv_batch_pending_keys:
                return None

            pending_items = list(self._kv_batch_pending_keys.items())
            self._kv_batch_pending_keys.clear()

        # BaM IO 和 Triton refill 不应在锁内执行；否则其它 LMCache 调用会被
        # Python lock 额外阻塞。
        keys = [item_key for _, item_key in pending_items]
        tensors = self.load_chunk_tensors_kv_fast_path_batch(keys)

        with self._prefetch_lock:
            for loaded_hash, loaded_tensor in tensors.items():
                self._kv_batch_loaded_tensors[loaded_hash] = loaded_tensor
            tensor = self._kv_batch_loaded_tensors.pop(chunk_hash, None)

        logger.info(
            "[LMCACHE_BAM_KV_FAST_PATH_BATCH_CONSUME] requested=%s "
            "batch_size=%d hit=%s",
            chunk_hash[:16],
            len(pending_items),
            tensor is not None,
        )
        return tensor


class LMCacheMemoryObjAdapter:
    """把 BaM 读回 tensor 回填到 LMCache 的 memory_obj。"""

    @staticmethod
    def populate_kv_blob(memory_obj: Any, tensor: torch.Tensor) -> None:
        target_tensor = getattr(memory_obj, "tensor", None)
        if target_tensor is None:
            raise ValueError("memory_obj.tensor is required for BaM prefer-load")

        if tuple(target_tensor.shape) != tuple(tensor.shape):
            raise ValueError(
                "shape mismatch between BaM tensor and allocated memory_obj: "
                f"bam={tuple(tensor.shape)} allocated={tuple(target_tensor.shape)}")
        if target_tensor.dtype != tensor.dtype:
            raise ValueError(
                "dtype mismatch between BaM tensor and allocated memory_obj: "
                f"bam={tensor.dtype} allocated={target_tensor.dtype}")

        # 优先回填 raw_data，语义上更接近 LMCache 原始磁盘路径的
        # `readinto(byte_buffer)`。
        # 这样做的好处是：
        # - 不依赖某个具体 tensor layout 的隐式 copy
        # - 更接近 LMCache 原生路径的 byte-level 语义
        target_raw = getattr(memory_obj, "raw_data", None)
        if target_raw is not None:
            if target_raw.dtype != torch.uint8:
                raise ValueError(
                    "memory_obj.raw_data must be uint8 for BaM prefer-load, "
                    f"got {target_raw.dtype}")

            expected_bytes = int(target_tensor.numel() * target_tensor.element_size())
            target_bytes = target_raw[:expected_bytes]
            source_bytes = tensor.contiguous().view(torch.uint8).reshape(-1).to(
                device=target_bytes.device, non_blocking=False)
            if target_bytes.numel() != source_bytes.numel():
                raise ValueError(
                    "byte size mismatch between BaM payload and LMCache buffer: "
                    f"payload={source_bytes.numel()} buffer={target_bytes.numel()}")
            target_bytes.copy_(source_bytes, non_blocking=False)
        else:
            # 少数实现可能没有 raw_data，这里回退到 tensor 视图 copy。
            target_tensor.copy_(tensor.to(device=target_tensor.device,
                                          non_blocking=False))

        from lmcache.experimental.memory_management import MemoryFormat
        memory_obj.metadata.fmt = MemoryFormat.KV_BLOB


class LMCacheBaMStorageManager:
    """包装 LMCache storage manager，增加 BaM shadow / prefer-load 能力。"""

    def __init__(self, storage_manager: Any) -> None:
        self._storage_manager = storage_manager
        self._bam_store: Optional[LMCacheBaMStore] = None
        self._bam_init_attempted = False
        self._bam_init_failed = False
        self._bam_prefer_load_disabled = False
        # 只在前几次 prefer-load 时做一次内容对照，快速判断问题是在
        # BaM KV_BLOB 本身，还是后续 vLLM attention 路径。
        self._prefer_load_verify_budget = 2
        self._closed = False
        atexit.register(self.close)

    def close(self) -> None:
        """在 storage manager 生命周期结束时收口 BaM runtime。"""
        if self._closed:
            return
        self._closed = True
        if self._bam_store is None:
            return
        try:
            self._bam_store.close()
        except Exception:
            logger.exception(
                "[LMCACHE_BAM_RUNTIME_IDLE_STOP] failed during manager.close")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _log_bam_runtime_context(self, tensor: torch.Tensor) -> None:
        dev_path = "/dev/libnvm0"
        dev_exists = os.path.exists(dev_path)
        dev_mode = "n/a"
        if dev_exists:
            dev_mode = oct(os.stat(dev_path).st_mode & 0o777)

        logger.info(
            "[LMCACHE_BAM] init context pid=%d euid=%d exe=%s cwd=%s "
            "tensor_shape=%s tensor_dtype=%s tensor_device=%s "
            "bam_import_path=%s ld_library_path=%s device_exists=%s "
            "device_mode=%s cache_size_mb=%d num_ssd=%d ssd_list=%s ctrl_idx=%d",
            os.getpid(),
            os.geteuid(),
            sys.executable,
            os.getcwd(),
            tuple(tensor.shape),
            tensor.dtype,
            tensor.device,
            envs.VLLM_BAM_IMPORT_PATH,
            os.environ.get("LD_LIBRARY_PATH", ""),
            dev_exists,
            dev_mode,
            envs.VLLM_BAM_CACHE_SIZE_MB,
            envs.VLLM_BAM_NUM_SSD,
            parse_optional_int_list(envs.VLLM_BAM_SSD_LIST),
            envs.VLLM_BAM_CTRL_IDX,
        )
        logger.info(
            "[LMCACHE_BAM] env gids_kv_worker_poll_impl=%s",
            os.environ.get("GIDS_KV_WORKER_POLL_IMPL", "<unset>"),
        )

    def _ensure_bam_store(self, memory_obj: Any) -> Optional[LMCacheBaMStore]:
        # BaM store 只在第一次真正看到 tensor 时初始化。
        # 这样可以拿到准确的 shape/dtype，避免提前猜测布局。
        tensor = getattr(memory_obj, "tensor", None)
        if tensor is None:
            return None

        if self._bam_store is not None:
            return self._bam_store
        if self._bam_init_failed:
            return None

        if not self._bam_init_attempted:
            self._bam_init_attempted = True
            self._log_bam_runtime_context(tensor)
            try:
                self._bam_store = LMCacheBaMStore.from_kv_shape(
                    tensor.shape, tensor.dtype)
            except Exception:
                self._bam_init_failed = True
                logger.exception(
                    "[LMCACHE_BAM] failed to initialize BaM store; "
                    "falling back to original LMCache path")
                return None

        return self._bam_store

    def _try_prefer_bam_load(self, key: Any) -> Optional[Any]:
        # prefer-load 的顺序是：
        #   1. 根据 chunk_hash 找到元数据
        #   2. 在 LMCache storage 侧先申请一个同形状的 memory_obj
        #   3. 从 BaM 读回 tensor。默认保持原 sync/prefetch 行为；
        #      开启 VLLM_BAM_KV_FAST_PATH 后走 KVCache 专用 descriptor 路径。
        #   4. 把 tensor 回填进 LMCache 的 memory_obj
        if self._bam_store is None:
            return None

        metadata = self._bam_store.get_chunk_metadata(key)
        if metadata is None:
            return None

        memory_obj = self._storage_manager.allocate(metadata.shape, metadata.dtype)
        if memory_obj is None:
            return None

        if envs.VLLM_BAM_KV_FAST_PATH:
            logger.info(
                "[LMCACHE_BAM_KV_FAST_PATH_GET] chunk_hash=%s "
                "mode=batch_or_single",
                _extract_chunk_hash(key)[:16],
            )
            tensor = self._bam_store.consume_kv_fast_path_tensor(key)
            if tensor is None:
                logger.info(
                    "[LMCACHE_BAM_KV_FAST_PATH_GET_FALLBACK_SINGLE] "
                    "chunk_hash=%s batch_miss=True",
                    _extract_chunk_hash(key)[:16],
                )
                tensor = self._bam_store.load_chunk_tensor_kv_fast_path(key)
        elif envs.VLLM_BAM_LMCACHE_READ_MODE == "prefetch":
            # 如果上游已经调用过 `storage_manager.prefetch(key)`，
            # 这里优先消费那次已提交的 BaM request；否则回到原来的 blocking
            # prefetch 读。这样 early prefetch 是纯优化，不是正确性前提。
            tensor = self._bam_store.load_prefetched_chunk_tensor(key)
            if tensor is None:
                tensor = self._bam_store.load_chunk_tensor_prefetch(key)
        else:
            tensor = self._bam_store.load_chunk_tensor(key)
        if tensor is None:
            return None

        self._maybe_verify_prefer_load_tensor(key, tensor)
        LMCacheMemoryObjAdapter.populate_kv_blob(memory_obj, tensor)
        logger.info(
            "[LMCACHE_BAM] prefer-load hit chunk_hash=%s",
            _extract_chunk_hash(key)[:16],
        )
        return memory_obj

    def prefetch(self, key: Any) -> None:
        """复用 LMCache V0 的 storage_manager.prefetch 语义发起 BaM 预读。

        LMCache 原生 `CacheEngine.prefetch(tokens, mask)` 会按 chunk key 调用
        `storage_manager.prefetch(key)`。这里把同名接口接到 BaM：

        - 开启 BaM prefer-load + read_mode=prefetch 时：提交 BaM page read
        - 开启 KV fast path 时：不再透传原生 LMCache disk prefetch，避免
          BaM 读取和 LMCache SSD 预取同时发生，污染性能口径
        - 其他模式：透传给原始 LMCache storage manager

        这让真实 vLLM 路径可以在 `retrieve()` 前先发起 BaM IO，而 `get()`
        时只需要等待/消费 outstanding request。
        """
        if (envs.VLLM_BAM_LMCACHE_PREFER_LOAD_ENABLE
                and envs.VLLM_BAM_KV_FAST_PATH
                and not self._bam_prefer_load_disabled
                and self._bam_store is not None):
            enqueued = self._bam_store.enqueue_kv_fast_path_prefetch_key(key)
            if enqueued:
                # 只有当前 chunk 已经成功登记到 BaM 本轮 batch 队列时，
                # 才能安全跳过 LMCache 原生 disk prefetch。
                #
                # 否则会出现这样的问题：
                # 1. prefetch 阶段没有把 chunk 纳入 BaM batch
                # 2. 这里又提前 return，LMCache 原生预取也被跳过
                # 3. 后续 retrieve 只能在更晚的 get 阶段临时兜底，甚至可能
                #    因为上层等待预取结果而表现为“卡住”
                #
                # 因此这里必须把“BaM 已接管”和“BaM 未接管”两种情况分开。
                logger.info(
                    "[LMCACHE_BAM_KV_FAST_PATH_PREFETCH_SKIP] chunk_hash=%s "
                    "enqueued=%s skip original LMCache disk prefetch",
                    _extract_chunk_hash(key)[:16],
                    enqueued,
                )
                return

            # 走到这里说明：
            # - KV fast path 已启用
            # - 但当前 chunk 还没有在 BaM 侧形成可用 metadata，通常是因为
            #   这轮请求的 shadow store 还在推进，或 chunk 还没被写入 BaM
            #
            # 这里我们**不要**回退到 LMCache 原生 disk prefetch：
            # - 回退会把 BaM/LMCache 两条路径重新搅在一起，容易把排错问题
            #   放大成“看起来卡住”
            # - 只要后续 `get()` 时 metadata 已经写入，prefer-load 仍然可以走
            #   BaM 单 chunk read 补上这个 key
            #
            # 因此这里更合适的语义是“暂时不预取，保留给后续 BaM read”。
            logger.info(
                "[LMCACHE_BAM_KV_FAST_PATH_PREFETCH_DEFER] chunk_hash=%s "
                "enqueued=%s defer to later BaM read instead of LMCache "
                "fallback",
                _extract_chunk_hash(key)[:16],
                enqueued,
            )
            return

        if (envs.VLLM_BAM_LMCACHE_PREFER_LOAD_ENABLE
                and envs.VLLM_BAM_LMCACHE_READ_MODE == "prefetch"
                and not envs.VLLM_BAM_KV_FAST_PATH
                and not self._bam_prefer_load_disabled
                and self._bam_store is not None):
            try:
                if self._bam_store.prefetch_chunk(key):
                    return
            except Exception:
                # 预取失败不应影响 LMCache 原生路径；真正 get 时仍可 fallback。
                logger.exception(
                    "[LMCACHE_BAM] early prefetch failed; "
                    "fall back to original LMCache prefetch")

        prefetch = getattr(self._storage_manager, "prefetch", None)
        if prefetch is not None:
            prefetch(key)

    def _maybe_verify_prefer_load_tensor(self, key: Any,
                                         bam_tensor: torch.Tensor) -> None:
        """把 BaM 结果和原始 LMCache 结果做一次直接对照。

        这里只在前几次 prefer-load 做校验，目的是快速判断问题到底在
        BaM KV_BLOB、页布局，还是后续 LMCache/vLLM 消费路径。
        """
        if self._prefer_load_verify_budget <= 0:
            return

        self._prefer_load_verify_budget -= 1
        try:
            original_memory_obj = self._storage_manager.get(key)
            if original_memory_obj is None or original_memory_obj.tensor is None:
                logger.warning(
                    "[LMCACHE_BAM_VERIFY] original LMCache tensor missing; "
                    "skip verification for chunk_hash=%s",
                    _extract_chunk_hash(key)[:16],
                )
                return

            original_tensor = original_memory_obj.tensor
            if tuple(original_tensor.shape) != tuple(bam_tensor.shape):
                logger.error(
                    "[LMCACHE_BAM_VERIFY] shape mismatch chunk_hash=%s "
                    "bam=%s original=%s",
                    _extract_chunk_hash(key)[:16],
                    tuple(bam_tensor.shape),
                    tuple(original_tensor.shape),
                )
                return
            if original_tensor.dtype != bam_tensor.dtype:
                logger.error(
                    "[LMCACHE_BAM_VERIFY] dtype mismatch chunk_hash=%s "
                    "bam=%s original=%s",
                    _extract_chunk_hash(key)[:16],
                    bam_tensor.dtype,
                    original_tensor.dtype,
                )
                return

            bam_cpu = bam_tensor.detach().to("cpu", non_blocking=False)
            original_cpu = original_tensor.detach().to("cpu", non_blocking=False)
            exact_equal = bool(torch.equal(bam_cpu, original_cpu))
            max_abs_diff = float(
                (bam_cpu - original_cpu).abs().max().item())
            mean_abs_diff = float(
                (bam_cpu - original_cpu).abs().float().mean().item())

            logger.info(
                "[LMCACHE_BAM_VERIFY] chunk_hash=%s exact_equal=%s "
                "max_abs_diff=%.6f mean_abs_diff=%.6f shape=%s dtype=%s",
                _extract_chunk_hash(key)[:16],
                exact_equal,
                max_abs_diff,
                mean_abs_diff,
                tuple(bam_cpu.shape),
                bam_cpu.dtype,
            )
        except Exception:
            logger.exception(
                "[LMCACHE_BAM_VERIFY] failed to compare BaM tensor with "
                "original LMCache tensor")

    def _maybe_verify_direct_retrieve_against_original_storage(
        self,
        *,
        in_flight_request: Any,
    ) -> None:
        """校验 direct retrieve 读回的 dense chunk 是否等于 LMCache 原始源。

        这一步只服务当前 correctness 排查，不参与正式热路径。它和 store 内部
        `WRITE_READ_VERIFY` 的区别是：

        - store 内部校验：BaM 读回结果 vs shadow write 当时记录的轻量样本；
        - 这里的校验：BaM 读回结果 vs 当前原始 LMCache storage manager.get(key)。

        后者能回答一个更独立的问题：BaM 读回的 prefix chunk 是否和 LMCache
        自己会返回给 vLLM 的 chunk 语义一致。如果这里通过，说明问题更可能在
        后续 paged KV 写入或 attention consume；如果这里失败，则说明 BaM
        shadow/read/page decode 的源数据语义就已经和 LMCache 原路径分叉。

        默认仍然保留轻量抽样，避免把普通 verify 路径拖得过重。当前如果需要
        定位 request_2 乱码，可以打开：

            VLLM_BAM_DIRECT_RETRIEVE_CROSS_SOURCE_FULL_COMPARE=1

        这会对所有命中的 prefix chunk 做逐元素精确比较。该开关只用于诊断，
        不进入正式性能口径。
        """
        if not _env_enabled("VLLM_BAM_WRITE_READ_VERIFY"):
            return
        if not _env_enabled("VLLM_BAM_WRITE_READ_VERIFY_SYNC_COMPARE"):
            logger.warning(
                "[LMCACHE_BAM_DIRECT_RETRIEVE_CROSS_SOURCE_SKIP] "
                "reason=sync_compare_disabled")
            return
        if self._bam_store is None:
            return

        materialized_tensors = getattr(
            in_flight_request, "materialized_prefix_chunk_tensors", None)
        if materialized_tensors is None:
            logger.warning(
                "[LMCACHE_BAM_DIRECT_RETRIEVE_CROSS_SOURCE_SKIP] "
                "reason=no_materialized_prefix_tensors")
            return

        prefix_hit_chunks = int(getattr(in_flight_request, "prefix_hit_chunks", 0))
        keys = list(getattr(in_flight_request, "keys", []))
        compare_count = min(prefix_hit_chunks, len(keys), len(materialized_tensors))
        if compare_count <= 0:
            logger.warning(
                "[LMCACHE_BAM_DIRECT_RETRIEVE_CROSS_SOURCE_SKIP] "
                "reason=no_prefix_chunks prefix_hit_chunks=%d keys=%d tensors=%d",
                prefix_hit_chunks,
                len(keys),
                len(materialized_tensors),
            )
            return

        verified = True
        full_compare = _env_enabled(
            "VLLM_BAM_DIRECT_RETRIEVE_CROSS_SOURCE_FULL_COMPARE")
        logger.info(
            "[LMCACHE_BAM_DIRECT_RETRIEVE_CROSS_SOURCE_BEGIN] chunks=%d "
            "full_compare=%s",
            compare_count,
            str(bool(full_compare)).lower(),
        )
        for idx in range(compare_count):
            key = keys[idx]
            chunk_hash = _extract_chunk_hash(key)
            bam_tensor = materialized_tensors[idx]
            try:
                original_memory_obj = self._storage_manager.get(key)
                original_tensor = getattr(original_memory_obj, "tensor", None)
                if original_tensor is None:
                    logger.error(
                        "[LMCACHE_BAM_DIRECT_RETRIEVE_CROSS_SOURCE_FAIL] "
                        "chunk_hash=%s reason=original_missing",
                        chunk_hash[:16],
                    )
                    verified = False
                    continue

                if tuple(original_tensor.shape) != tuple(bam_tensor.shape):
                    logger.error(
                        "[LMCACHE_BAM_DIRECT_RETRIEVE_CROSS_SOURCE_FAIL] "
                        "chunk_hash=%s reason=shape_mismatch original=%s bam=%s",
                        chunk_hash[:16],
                        tuple(original_tensor.shape),
                        tuple(bam_tensor.shape),
                    )
                    verified = False
                    continue

                if original_tensor.dtype != bam_tensor.dtype:
                    logger.error(
                        "[LMCACHE_BAM_DIRECT_RETRIEVE_CROSS_SOURCE_FAIL] "
                        "chunk_hash=%s reason=dtype_mismatch original=%s bam=%s",
                        chunk_hash[:16],
                        original_tensor.dtype,
                        bam_tensor.dtype,
                    )
                    verified = False
                    continue

                if full_compare:
                    # 全量 compare 只在 correctness 排查时打开：
                    #
                    # - 原始 LMCache tensor 是“LMCache 原生 retrieve 会交给
                    #   vLLM 的真实 chunk”；
                    # - bam_tensor 是当前 BaM direct retrieve 从 live pages 解码
                    #   出来的 dense chunk；
                    # - 两者逐元素相等，才能说明 BaM shadow/read/page decode 这层
                    #   与 LMCache 原生路径完全等价。
                    #
                    # 这里把两边都搬到 CPU 做比较，避免为了诊断再发额外 GPU
                    # kernel，也避免和 persistent service 的设备侧工作交织。
                    original_cpu = original_tensor.detach().to(
                        device="cpu",
                        copy=True,
                    )
                    bam_cpu = bam_tensor.detach().to(
                        device="cpu",
                        copy=True,
                    )
                    if torch.equal(original_cpu, bam_cpu):
                        logger.info(
                            "[LMCACHE_BAM_DIRECT_RETRIEVE_CROSS_SOURCE_FULL_OK] "
                            "chunk_hash=%s shape=%s dtype=%s",
                            chunk_hash[:16],
                            tuple(bam_cpu.shape),
                            bam_cpu.dtype,
                        )
                        continue

                    diff = (bam_cpu.float() - original_cpu.float()).abs()
                    mismatch_flat = torch.nonzero(
                        bam_cpu != original_cpu, as_tuple=False)
                    first = (mismatch_flat[0].tolist()
                             if mismatch_flat.numel() > 0 else [])
                    detail = ""
                    if len(first) == 4:
                        kv_i, layer_i, token_i, dim_i = [int(x) for x in first]
                        detail = (
                            f" kv={kv_i}"
                            f" layer={layer_i}"
                            f" token={token_i}"
                            f" dim={dim_i}"
                            f" original={original_cpu[kv_i, layer_i, token_i, dim_i].item()}"
                            f" bam={bam_cpu[kv_i, layer_i, token_i, dim_i].item()}")
                    logger.error(
                        "[LMCACHE_BAM_DIRECT_RETRIEVE_CROSS_SOURCE_FULL_FAIL] "
                        "chunk_hash=%s max_abs_diff=%s first_mismatch=%s%s",
                        chunk_hash[:16],
                        diff.max().item() if diff.numel() > 0 else "n/a",
                        first,
                        detail,
                    )
                    verified = False
                    continue

                # 复用 store 里的稳定采样坐标，覆盖开头、128-token page 边界
                # 和 chunk 尾部。这里只抽样，不做整块 full compare，避免调试
                # 逻辑把性能口径和主线行为搅在一起。
                sample = self._bam_store._build_write_read_verify_sample(
                    chunk_hash=chunk_hash,
                    tensor=original_tensor,
                )
                layer_index_tensor = torch.tensor(
                    sample.layer_indices,
                    device=bam_tensor.device,
                    dtype=torch.long,
                )
                token_index_tensor = torch.tensor(
                    sample.token_indices,
                    device=bam_tensor.device,
                    dtype=torch.long,
                )
                dim_index_tensor = torch.tensor(
                    sample.dim_indices,
                    device=bam_tensor.device,
                    dtype=torch.long,
                )
                actual = bam_tensor.index_select(
                    1, layer_index_tensor).index_select(
                        2, token_index_tensor).index_select(3, dim_index_tensor)
                actual_cpu = actual.detach().cpu()
                expected_cpu = sample.values
                if torch.equal(actual_cpu, expected_cpu):
                    logger.info(
                        "[LMCACHE_BAM_DIRECT_RETRIEVE_CROSS_SOURCE_OK] "
                        "chunk_hash=%s shape=%s layers=%s tokens=%s dims=%s",
                        chunk_hash[:16],
                        tuple(sample.shape),
                        sample.layer_indices,
                        sample.token_indices,
                        sample.dim_indices,
                    )
                    continue

                diff = (actual_cpu.float() - expected_cpu.float()).abs()
                mismatch_flat = torch.nonzero(
                    actual_cpu != expected_cpu, as_tuple=False)
                first = (mismatch_flat[0].tolist()
                         if mismatch_flat.numel() > 0 else [])
                detail = ""
                if len(first) == 4:
                    kv_i, layer_i, token_i, dim_i = [int(x) for x in first]
                    detail = (
                        f" kv={kv_i}"
                        f" layer={sample.layer_indices[layer_i]}"
                        f" token={sample.token_indices[token_i]}"
                        f" dim={sample.dim_indices[dim_i]}"
                        f" original={expected_cpu[kv_i, layer_i, token_i, dim_i].item()}"
                        f" bam={actual_cpu[kv_i, layer_i, token_i, dim_i].item()}")
                logger.error(
                    "[LMCACHE_BAM_DIRECT_RETRIEVE_CROSS_SOURCE_FAIL] "
                    "chunk_hash=%s reason=value_mismatch max_abs_diff=%s "
                    "first_mismatch=%s%s",
                    chunk_hash[:16],
                    diff.max().item() if diff.numel() > 0 else "n/a",
                    first,
                    detail,
                )
                verified = False
            except Exception:
                logger.exception(
                    "[LMCACHE_BAM_DIRECT_RETRIEVE_CROSS_SOURCE_FAIL] "
                    "chunk_hash=%s reason=exception",
                    chunk_hash[:16],
                )
                verified = False

        if verified:
            logger.info(
                "[LMCACHE_BAM_DIRECT_RETRIEVE_CROSS_SOURCE_OK] chunks=%d",
                compare_count,
            )
        else:
            logger.error(
                "[LMCACHE_BAM_DIRECT_RETRIEVE_CROSS_SOURCE_FAIL] chunks=%d",
                compare_count,
            )

    def put(self, key: Any, memory_obj: Any) -> None:
        # 写路径顺序：
        #   1. 尝试初始化 BaM store
        #   2. 先 shadow 一份到 BaM
        #   3. 再把原始 memory_obj 交回 LMCache 原路径
        # 这样不会破坏原生 SSD baseline。
        bam_store = self._ensure_bam_store(memory_obj)
        if bam_store is not None and getattr(memory_obj, "tensor", None) is not None:
            try:
                bam_store.store_chunk(key, memory_obj.tensor)
            except Exception:
                logger.exception(
                    "[LMCACHE_BAM] shadow store failed; keep LMCache original path")

        self._storage_manager.put(key, memory_obj)

    def get(self, key: Any) -> Optional[Any]:
        # 读路径顺序：
        #   1. 如果开启 prefer-load，先尝试 BaM
        #   2. 失败则回退 LMCache 原路径
        #   3. 这样可以单独评估 BaM 路径的命中收益
        if (envs.VLLM_BAM_LMCACHE_PREFER_LOAD_ENABLE
                and not self._bam_prefer_load_disabled):
            try:
                memory_obj = self._try_prefer_bam_load(key)
                if memory_obj is not None:
                    return memory_obj
            except Exception:
                # BaM rowctx/prefetch 一旦在真实 vLLM 路径里失败，继续尝试通常
                # 会反复踩同一个 outstanding request / FIFO 状态。这里直接禁用
                # 后续 prefer-load，让本轮请求安全回退 LMCache 原生 SSD 路径。
                self._bam_prefer_load_disabled = True
                logger.exception(
                    "[LMCACHE_BAM] prefer-load failed; disable BaM prefer-load "
                    "for this storage manager and fall back to LMCache")

        memory_obj = self._storage_manager.get(key)
        if (memory_obj is not None
                and envs.VLLM_BAM_LMCACHE_PREFER_LOAD_ENABLE):
            try:
                logger.info(
                    "[LMCACHE_BAM] prefer-load miss/fallback chunk_hash=%s",
                    _extract_chunk_hash(key)[:16],
                )
            except Exception:
                pass
        return memory_obj

    def direct_retrieve_to_vllm_kvcache(
        self,
        *,
        token_database: Any,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor],
        kv_caches: list[torch.Tensor],
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str = "auto",
        num_kv_heads: int,
        head_size: int,
    ) -> Optional[torch.Tensor]:
        """把 BaM 中命中的 chunks 直接写入 vLLM paged KV cache。

        这个方法是 Direct Placement v0 的 storage-manager 入口。它复用
        LMCache 已有的 `token_database.process_tokens()`，因此不会重新实现
        prefix hash / mask / chunk key 逻辑。

        数据流：

        ```text
        tokens + mask
          -> token_database.process_tokens()
          -> [CacheEngineKey, ...]
          -> BaM KV batch read pages
          -> direct placement prepare/start/wait
          -> ret_mask
        ```

        返回值语义与 `LMCacheBaMStore.direct_place_chunks_to_vllm_kvcache()`
        保持一致：

        - `torch.Tensor`: BaM 已经完成至少一个 chunk 的 direct placement
        - `None`: 当前应回退到原始 LMCache retrieve

        注意：这里仅在显式 direct-placement 开关开启时由 vLLM adapter 调用；
        普通 `get()` / LMCache SSD / 现有 BaM prefer-load 路径完全不受影响。
        """
        if self._bam_store is None:
            return None
        if not envs.VLLM_BAM_KV_FAST_PATH:
            return None

        return self._bam_store.direct_place_chunks_to_vllm_kvcache(
            token_database=token_database,
            tokens=tokens,
            mask=mask,
            kv_caches=kv_caches,
            slot_mapping=slot_mapping,
            kv_cache_dtype=kv_cache_dtype,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
        )

    def start_direct_retrieve_to_vllm_kvcache(
        self,
        *,
        token_database: Any,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor],
        kv_caches: list[torch.Tensor],
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str = "auto",
        num_kv_heads: int,
        head_size: int,
    ) -> Optional[Any]:
        """公开 direct retrieve request 的 start 边界，供更高层 runtime 编排。

        当前 `num_kv_heads/head_size` 还保留在接口上，主要是为了和现有
        `direct_retrieve_to_vllm_kvcache()` 的调用签名保持一致，避免后续 adapter
        切换时再做一轮分叉。现在这两个参数还会继续下传到 request handle，
        只服务运行时诊断：
        - flat slot-major 写入视角
        - packed paged-KV 消费视角
        """
        if self._bam_store is None:
            return None
        if not envs.VLLM_BAM_KV_FAST_PATH:
            return None

        return self._bam_store.start_direct_placement_request(
            token_database=token_database,
            tokens=tokens,
            mask=mask,
            kv_caches=kv_caches,
            slot_mapping=slot_mapping,
            kv_cache_dtype=kv_cache_dtype,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
        )

    def poll_direct_retrieve_to_vllm_kvcache(
        self,
        *,
        in_flight_request: Any,
    ) -> Optional[BaMDirectPlacementBatchStateSnapshot]:
        """公开 direct retrieve request 的非阻塞 poll 边界。"""
        if self._bam_store is None:
            return None
        return self._bam_store.poll_direct_placement_request(
            in_flight_request=in_flight_request,
        )

    def get_direct_retrieve_to_vllm_kvcache_runtime_snapshot(
        self,
        *,
        in_flight_request: Any,
    ) -> Optional[Any]:
        """公开 direct retrieve request 当前对应的底层 runtime 观察快照。"""
        if self._bam_store is None:
            return None
        return self._bam_store.get_direct_placement_request_runtime_snapshot(
            in_flight_request=in_flight_request,
        )

    def attach_direct_retrieve_to_vllm_kvcache_runtime_attention_metadata(
        self,
        *,
        in_flight_request: Any,
        attachment: BaMRuntimeAttentionMetadataAttachment,
    ) -> bool:
        """把单条 sequence 的 runtime attention metadata workspace 挂到 live request。"""
        if self._bam_store is None:
            return False
        return self._bam_store.attach_direct_placement_request_runtime_attention_metadata(
            in_flight_request=in_flight_request,
            attachment=attachment,
        )

    def stop_direct_retrieve_to_vllm_kvcache_runtime_service_if_idle(self) -> bool:
        """在 direct retrieve request 已全部 cleanup 后尝试停掉空转 runtime service。

        这条接口只服务当前 LMCache adapter 的 validation rebuild 主线：

        - direct retrieve 的数据面与 cleanup 已结束
        - 前台还要继续做本地 metadata rebuild / xformers fallback
        - 如果 persistent service 仍在空转，前台后续 CUDA kernel 可能被拖住

        因此这里不引入新的状态分支，只把底层已有的
        `native_runtime_stop_service_if_idle()` 薄薄转发出来。
        """
        if self._bam_store is None:
            return False
        kv_fast_path = self._bam_store._kv_fast_path
        if kv_fast_path is None:
            return False
        return bool(
            kv_fast_path.kv_store.native_runtime_stop_service_if_idle())

    def finalize_direct_retrieve_to_vllm_kvcache(
        self,
        *,
        in_flight_request: Any,
    ) -> Optional[torch.Tensor]:
        """公开 direct retrieve request 的 finalize 边界。"""
        if self._bam_store is None:
            return None
        ret_mask = self._bam_store.finalize_direct_placement_request(
            in_flight_request=in_flight_request,
        )
        self._maybe_verify_direct_retrieve_against_original_storage(
            in_flight_request=in_flight_request,
        )
        return ret_mask

    def __getattr__(self, name: str) -> Any:
        return getattr(self._storage_manager, name)
