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
#     最激进 one-copy 实验线。GPU persistent service 直接写最终 vLLM paged
#     KV cache；前台只观察 consumable frontier 并做 cleanup，不再夹带
#     materialized repair/verify 支线。
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
    这样 `_finalize_direct_placement_request()` 后续只需要按 finalize mode 分派，
    不再把 one-copy 调试期的 expected tensor / repair 状态一路传递。
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


@dataclass(frozen=True)
class _DirectPlacementFinalizeBackendOutcome:
    """描述一次 finalize consume backend 的统一返回值。"""

    backend_name: str
    snapshot: BaMDirectPlacementBatchStateSnapshot
    place_stats: Any
    frontier_wave_ms: float
    cache_ready_log_ms: float


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

    def get_last_direct_placement_state_snapshot(
        self,
    ) -> Optional[BaMDirectPlacementBatchStateSnapshot]:
        """返回最近一次 direct placement 的状态快照。

        这是三条 KV 链路共用的只读观察接口：
        - 单测用它断言 frontier/ret_mask 语义；
        - 联调用它确认 read/stage/cache/consumable 推进到哪里；
        - 正式数据面不依赖它做额外搬运。
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
        """收集本轮 direct placement 可由 BaM 服务的连续前缀 chunks。

        这一步是三条 KV 链路共同的控制面入口，只负责判断“哪些 chunk 可以由
        BaM 接管”，不发起 I/O，也不做 placement。

        LMCache 的 prefix 语义要求命中必须连续：一旦中间某个 chunk 在 BaM 中
        缺失，后面的 chunk 即使存在，也不能越过缺口直接写回 vLLM KV cache。
        因此这里遇到第一个 miss 就停止。
        """
        entries: list[tuple[int, int, Any]] = []
        for start, end, key in token_database.process_tokens(tokens, mask):
            metadata = self.get_chunk_metadata(key)
            if metadata is None:
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

    def _finalize_one_copy_read_request(
        self,
        *,
        in_flight_request: _InFlightDirectPlacementRequest,
    ) -> _DirectPlacementFinalizeReadOutcome:
        """收口 `gpu_worker_persistent_one_copy` 的 read 阶段。

        当前主线恢复为之前可跑通的 one-copy 语义：

        ```text
        GPU persistent service:
          CQ service / ctx refresh
          BaM cache page -> vLLM paged KV cache
          发布 CONSUMED / consumable frontier

        host finalize:
          cleanup-only / detach request lifecycle
        ```

        这里不再消费出 `BaMKVReadResult[]`，也不再把请求绕回
        `_finalize_materialized_pipeline()`。one-copy 的正确性以底层 runtime
        scatter 为准，前台不再夹带 live-pages refill / official-write repair。
        """
        read_consume_start = self._log_direct_read_consume_begin(
            in_flight_request=in_flight_request)
        runtime_cleanup_handle = in_flight_request.kv_read_handle

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
        )

    def _finalize_persistent_one_copy_pipeline(
        self,
        *,
        in_flight_request: _InFlightDirectPlacementRequest,
        runtime_cleanup_handle: Any | None,
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
        # 这里不再执行 official-write repair 或 runtime-write verify。真正的
        # one-copy 语义是：
        # GPU persistent service 已经把 BaM cache page 直接写进最终 vLLM paged
        # KV cache；host 只发布 frontier/ret_mask。若输出仍错，必须继续修
        # runtime scatter 或读侧解释，而不是再用 repair/verify 支线覆盖错误。
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
    ) -> _DirectPlacementFinalizeBackendOutcome:
        """按显式 finalize consume backend 收口 direct placement request。"""
        if read_finalize_mode == _DIRECT_FINALIZE_MODE_RUNTIME_DIRECT:
            return self._finalize_persistent_one_copy_pipeline(
                in_flight_request=in_flight_request,
                runtime_cleanup_handle=runtime_cleanup_handle,
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
        finalize_backend_outcome = self._finalize_direct_placement_with_backend(
            read_finalize_mode=read_finalize_mode,
            in_flight_request=in_flight_request,
            runtime_cleanup_handle=runtime_cleanup_handle,
        )
        first_wave_snapshot = finalize_backend_outcome.snapshot
        place_stats = finalize_backend_outcome.place_stats
        frontier_wave_ms = finalize_backend_outcome.frontier_wave_ms
        cache_ready_log_ms = finalize_backend_outcome.cache_ready_log_ms

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
            "[LMCACHE_BAM] env kv_executor=%s runtime=%s persistent=%s",
            os.environ.get("VLLM_BAM_KV_EXECUTOR", "<unset>"),
            os.environ.get("GIDS_KV_GPU_WORKER_RUNTIME_ENABLE", "<unset>"),
            os.environ.get("GIDS_KV_GPU_WORKER_PERSISTENT_ENABLE", "<unset>"),
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
        return ret_mask

    def __getattr__(self, name: str) -> Any:
        return getattr(self._storage_manager, name)
