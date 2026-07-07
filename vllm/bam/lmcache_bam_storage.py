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
    BaMDirectKVPlacer, BaMDirectPlacementBatchDescriptor,
    BaMDirectPlacementBatchStateSnapshot, BaMDirectPlacementChunkDescriptor,
    BaMDirectPlacementExecution, BaMDirectPlacementStateTracker,
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


@dataclass(frozen=True)
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
    read_ms: float
    descriptor_ms: float
    tracker_init_ms: float
    prepare_ms: float
    direct_total_start_time: float
    frontier_wave_start_time: float


@dataclass(frozen=True)
class _InFlightDirectPlacementRequest:
    """描述一次已经 start、后续可被 poll/finalize 的 direct placement request。

    这层对象是“request 级控制面”的第一版句柄。它和 `_InFlightDirectPlacementWave`
    的区别是：

    - `Wave` 只描述单次 placement launch 的 runtime 边界
    - `Request` 描述一次完整 direct retrieve 的稳定上下文

    因此 request handle 会额外持有：

    - prefix 命中信息
    - 当前 request 对上层准备返回的 frontier 目标
    - followup wave 的实验配置
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
    results: list[Any]
    state_tracker: BaMDirectPlacementStateTracker
    direct_placer: BaMDirectKVPlacer
    chunk_starts: list[int]
    frontier_wave: _InFlightDirectPlacementWave
    prefix_hit_chunks: int
    prefix_hit_tokens: int
    first_wave_launch_chunks: int
    first_wave_return_target_chunks: int
    followup_chunk_limit: int
    bootstrap_profile: _DirectPlacementRequestBootstrapProfile


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

    def _resolve_direct_placement_frontier_chunk_limit(
        self,
        *,
        total_chunks: int,
    ) -> int | None:
        """解析本轮 direct placement 需要真正 launch 的连续前缀 chunk 数。

        当前语义保持“默认不裁剪”：

        - `VLLM_BAM_DIRECT_PLACEMENT_FRONTIER_CHUNKS <= 0`
          表示本轮命中的连续 prefix chunks 全量 launch
        - `1 <= N < total_chunks`
          表示本轮只 launch 前 N 个 chunk
        - `N >= total_chunks`
          等价于不裁剪，仍走全量 launch

        返回 `None` 表示“按默认主线全量 launch”，这样下游执行器仍然可以保留
        “默认全量”和“显式前缀裁剪”两种控制面分支，后续继续往真正的 frontier
        consume 推进时更容易扩展。
        """
        configured_limit = int(
            envs.VLLM_BAM_DIRECT_PLACEMENT_FRONTIER_CHUNKS)
        if configured_limit <= 0 or configured_limit >= int(total_chunks):
            return None
        return int(configured_limit)

    def _resolve_direct_placement_followup_chunk_limit(
        self,
        *,
        total_chunks: int,
        first_wave_chunks: int,
    ) -> int:
        """解析真实 store 路径里第二波 followup placement 的 chunk 数。

        这里故意把 followup wave 设计成一个独立开关，而不是复用第一波的
        `FRONTIER_CHUNKS`，原因是两者关注点不同：

        - 第一波决定“当前要尽快暴露多少可消费前缀”
        - 第二波决定“在第一波之后，额外再补多少已命中的 chunk”

        当前返回值语义：

        - `0`: 不执行 followup wave
        - `N>0`: 从 `first_wave_chunks` 开始，再额外执行至多 `N` 个 chunk

        注意：
        这个 followup 目前仍是“真实 store 控制面验证版”，默认关闭，避免影响
        已经稳定的全前缀主路径。
        """
        configured_limit = int(
            envs.VLLM_BAM_DIRECT_PLACEMENT_FOLLOWUP_CHUNKS)
        remaining_chunks = max(int(total_chunks) - int(first_wave_chunks), 0)
        if configured_limit <= 0 or remaining_chunks <= 0:
            return 0
        return min(int(configured_limit), remaining_chunks)

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

        旧版 store 的兜底逻辑是“如果外部实现没有推进状态，就整批标成 ready”。
        在单波全量 launch 的时代这没问题，但现在已经开始支持：

        - 第一波只 launch 前若干个 chunk
        - 第二波从中间某个 chunk 偏移继续 launch

        因此兜底必须收缩到“本轮真实 launch 的 chunk 范围”，不能再把整批都
        粗暴标成 ready，否则会破坏我们刚建立好的 frontier 语义。
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
        direct_execution = direct_placer.execution_from_launched_batch(
            launched_batch=launched_batch,
            state_tracker=state_tracker,
        )
        # 先做一次非阻塞推进，把已经天然 ready 的状态同步进 tracker。
        # 当前同步版本虽然最终还是会 wait，但把这一步单独保留出来后，未来要把
        # wave 改成更细粒度轮询时，就不需要再拆 execution 接口。
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
        if callable(target_frontier_wait):
            _final_state_snapshot = target_frontier_wait(
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
            place_stats, _final_state_snapshot = wave_local_wait()
        else:
            place_stats, _final_state_snapshot = direct_execution.wait()
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
        snapshot = state_tracker.snapshot()
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

        read_start = time.perf_counter()
        logger.info(
            "[LMCACHE_BAM_DIRECT_PLACEMENT_READ_BEGIN] batch_size=%d "
            "chunk_hashes=%s",
            len(keys),
            chunk_hashes,
        )
        results = self.read_chunk_pages_kv_fast_path_batch(keys)
        read_ms = (time.perf_counter() - read_start) * 1000.0
        if not results:
            self._last_direct_placement_state_tracker = None
            logger.info(
                "[LMCACHE_BAM_DIRECT_PLACEMENT_READ_EMPTY] batch_size=%d "
                "read_ms=%.3f",
                len(keys),
                read_ms,
            )
            return None

        # token_database 和 slot_mapping 使用同一段 tokens 的局部坐标系。
        # 例如 mask 前缀有 False 时，第一条可 retrieve 的 chunk 起点可能不是 0。
        # 这里必须保留这个局部偏移，不能人为重编号，否则 placement 会把数据写
        # 到错误的 vLLM physical slot。
        chunk_starts = [start for start, _, _ in entries]
        prefix_hit_chunks = len(entries)
        prefix_hit_tokens = sum(
            int(result.descriptor.actual_tokens) for result in results)
        launch_chunk_limit = (
            self._resolve_direct_placement_frontier_chunk_limit(
                total_chunks=prefix_hit_chunks))
        descriptor_start = time.perf_counter()
        batch_descriptor = self._build_direct_placement_descriptor(
            entries=entries,
            results=results,
        )
        descriptor_ms = (time.perf_counter() - descriptor_start) * 1000.0
        tracker_start = time.perf_counter()
        state_tracker = BaMDirectPlacementStateTracker(batch_descriptor)
        state_tracker.mark_all_read_ready()
        self._last_direct_placement_state_tracker = state_tracker
        self._log_direct_placement_state(stage="read_ready", tracker=state_tracker)
        tracker_init_ms = (time.perf_counter() - tracker_start) * 1000.0
        direct_placer = self._ensure_direct_kv_placer(
            kv_cache_dtype=kv_cache_dtype)
        first_wave_launch_chunks = (launch_chunk_limit
                                    if launch_chunk_limit is not None else
                                    prefix_hit_chunks)
        # 对正常推理引擎主线来说，“当前请求应该返回的 prefix 长度”必须先于
        # placement 执行语义被确定下来。
        #
        # 当前默认就是：
        #   命中了多少连续 prefix chunk
        #     -> 就准备返回多少连续 prefix chunk
        #
        # 只有显式打开 frontier 限制实验时，返回目标才会收缩到首波 launch 范围。
        first_wave_return_target_chunks = first_wave_launch_chunks
        first_wave_launch_tokens = sum(
            int(result.descriptor.actual_tokens)
            for result in results[:first_wave_launch_chunks])
        followup_chunk_limit = (
            self._resolve_direct_placement_followup_chunk_limit(
                total_chunks=prefix_hit_chunks,
                first_wave_chunks=first_wave_launch_chunks,
            ))
        logger.info(
            "[LMCACHE_BAM_DIRECT_PLACEMENT_LAUNCH_POLICY] total_chunks=%d "
            "prefix_hit_chunks=%d prefix_hit_tokens=%d "
            "frontier_launch_chunks=%d frontier_launch_tokens=%d "
            "return_target_chunks=%d launch_chunk_limit=%s "
            "followup_chunk_limit=%d",
            prefix_hit_chunks,
            prefix_hit_chunks,
            prefix_hit_tokens,
            first_wave_launch_chunks,
            first_wave_launch_tokens,
            first_wave_return_target_chunks,
            ("all" if launch_chunk_limit is None else
             str(launch_chunk_limit)),
            followup_chunk_limit,
        )
        # 第一波是真正决定“当前请求要返回多少可消费前缀”的关键 wave。
        # 在真正启动第一波 placement 之前，先把一次性准备成本前移掉：
        # - Triton/JIT warmup
        # - pointer 初始化
        #
        # 这样 `DIRECT_RETRIEVE elapsed_ms` 会更接近 steady-state，而不会被
        # 首发编译/初始化一次性成本干扰。
        prepare_start = time.perf_counter()
        prepare_bam_results_for_vllm_kvcache(
            results=results,
            layout=self.layout,
            kv_caches=kv_caches,
            slot_mapping=slot_mapping,
            chunk_starts=chunk_starts,
            kv_cache_dtype=kv_cache_dtype,
            placer=direct_placer,
            launch_start_chunk=0,
            max_chunks_to_launch=launch_chunk_limit,
        )
        prepare_ms = (time.perf_counter() - prepare_start) * 1000.0
        frontier_wave_start = time.perf_counter()
        frontier_wave = self._launch_direct_placement_wave(
            direct_placer=direct_placer,
            results=results,
            kv_caches=kv_caches,
            slot_mapping=slot_mapping,
            chunk_starts=chunk_starts,
            state_tracker=state_tracker,
            launch_start_chunk=0,
            launch_chunk_count=launch_chunk_limit,
            return_target_chunks=first_wave_return_target_chunks,
            wave_name="frontier",
            do_prepare=False,
        )
        return _InFlightDirectPlacementRequest(
            tokens=tokens,
            kv_caches=kv_caches,
            slot_mapping=slot_mapping,
            results=results,
            state_tracker=state_tracker,
            direct_placer=direct_placer,
            chunk_starts=chunk_starts,
            frontier_wave=frontier_wave,
            prefix_hit_chunks=prefix_hit_chunks,
            prefix_hit_tokens=prefix_hit_tokens,
            first_wave_launch_chunks=first_wave_launch_chunks,
            first_wave_return_target_chunks=first_wave_return_target_chunks,
            followup_chunk_limit=followup_chunk_limit,
            bootstrap_profile=_DirectPlacementRequestBootstrapProfile(
                collect_entries_ms=collect_entries_ms,
                read_ms=read_ms,
                descriptor_ms=descriptor_ms,
                tracker_init_ms=tracker_init_ms,
                prepare_ms=prepare_ms,
                direct_total_start_time=direct_total_start,
                frontier_wave_start_time=frontier_wave_start,
            ),
        )

    def _poll_direct_placement_request(
        self,
        *,
        in_flight_request: _InFlightDirectPlacementRequest,
    ) -> BaMDirectPlacementBatchStateSnapshot:
        """非阻塞推进一次 request 当前已 launch frontier 的 ready 状态。

        当前版本先只轮询第一波 frontier wave，因为真正决定“本次请求现在能不能
        返回”的仍然是连续 consumable prefix frontier。

        返回值始终落到 request 级 snapshot，而不是 wave 局部结构，这样以后上层
        runtime 如果直接持有 request handle，就不需要理解 wave 细节。
        """
        frontier_snapshot = (
            in_flight_request.frontier_wave.direct_execution.advance_ready())
        if frontier_snapshot is not None:
            return frontier_snapshot
        return in_flight_request.state_tracker.snapshot()

    def _finalize_direct_placement_request(
        self,
        *,
        in_flight_request: _InFlightDirectPlacementRequest,
    ) -> torch.Tensor:
        """同步收口一次已经 start 的 direct placement request。

        这里保留当前主线需要的同步语义：

        - 先等待 frontier wave 满足当前请求的返回目标
        - 再构造 ret_mask 并做返回语义校验
        - 最后如果 followup 实验开关开启，再继续补齐 resident cache

        但和旧版本不同的是，所有“已启动 request 的稳定上下文”都来自
        `in_flight_request`，不再依赖一个大函数里的局部变量链式传递。
        """
        bootstrap_profile = in_flight_request.bootstrap_profile
        state_tracker = in_flight_request.state_tracker
        place_stats, first_wave_snapshot = self._wait_direct_placement_wave(
            in_flight_wave=in_flight_request.frontier_wave,
            state_tracker=state_tracker,
        )
        frontier_wave_ms = (
            time.perf_counter() - bootstrap_profile.frontier_wave_start_time
        ) * 1000.0
        cache_ready_log_start = time.perf_counter()
        self._log_direct_placement_state(stage="cache_ready", tracker=state_tracker)
        cache_ready_log_ms = (time.perf_counter() - cache_ready_log_start) * 1000.0

        # ret_mask 必须绑定到第一波完成时的 contiguous consumable frontier，
        # 而不能绑定到后续 followup wave 的最终 resident 状态。
        #
        # 否则第一波只想先返回 1 个 chunk、第二波只是顺手把后续 chunk 补进
        # cache 的情况下，上层会错误地把后续 chunk 也当成当前请求已经可消费的
        # prefix。
        ret_mask_start = time.perf_counter()
        ret_mask = self._build_consumable_ret_mask(
            tokens=in_flight_request.tokens,
            snapshot=first_wave_snapshot,
        )
        ret_mask_ms = (time.perf_counter() - ret_mask_start) * 1000.0
        return_snapshot = first_wave_snapshot
        self._validate_direct_placement_return_semantics(
            tokens=in_flight_request.tokens,
            ret_mask=ret_mask,
            return_snapshot=return_snapshot,
            prefix_hit_chunks=in_flight_request.prefix_hit_chunks,
            return_target_chunks=in_flight_request.first_wave_return_target_chunks,
        )

        # 第二波 followup 是一个默认关闭的实验控制面能力。
        # 它当前的目标不是立刻改变返回给上层的 hit_tokens，而是把：
        # - 从 chunk 偏移处继续 launch
        # - 继续复用同一个 tracker 推进 ready 状态
        # - 真实 store 日志里可观察到第二波行为
        # 这条链路先在真实路径中接通。
        followup_total_ms = 0.0
        followup_log_ms = 0.0
        if in_flight_request.followup_chunk_limit > 0:
            logger.info(
                "[LMCACHE_BAM_DIRECT_PLACEMENT_FOLLOWUP_BEGIN] "
                "start_chunk=%d launch_chunks=%d",
                in_flight_request.first_wave_launch_chunks,
                in_flight_request.followup_chunk_limit,
            )
            followup_start = time.perf_counter()
            followup_wave = self._launch_direct_placement_wave(
                direct_placer=in_flight_request.direct_placer,
                results=in_flight_request.results,
                kv_caches=in_flight_request.kv_caches,
                slot_mapping=in_flight_request.slot_mapping,
                chunk_starts=in_flight_request.chunk_starts,
                state_tracker=state_tracker,
                launch_start_chunk=in_flight_request.first_wave_launch_chunks,
                launch_chunk_count=in_flight_request.followup_chunk_limit,
                return_target_chunks=(
                    in_flight_request.first_wave_launch_chunks +
                    in_flight_request.followup_chunk_limit
                ),
                wave_name="followup",
            )
            _followup_stats, followup_snapshot = self._wait_direct_placement_wave(
                in_flight_wave=followup_wave,
                state_tracker=state_tracker,
            )
            followup_total_ms = (time.perf_counter() - followup_start) * 1000.0
            followup_log_start = time.perf_counter()
            self._log_direct_placement_state(
                stage="followup_cache_ready",
                tracker=state_tracker,
            )
            logger.info(
                "[LMCACHE_BAM_DIRECT_PLACEMENT_FOLLOWUP_DONE] "
                "resident_cache_ready_chunks=%d resident_consumable_chunks=%d "
                "resident_consumable_tokens=%d return_consumable_chunks=%d "
                "return_consumable_tokens=%d",
                followup_snapshot.cache_ready_chunks,
                followup_snapshot.consumable_chunks,
                followup_snapshot.consumable_tokens,
                return_snapshot.consumable_chunks,
                return_snapshot.consumable_tokens,
            )
            followup_log_ms = (time.perf_counter() - followup_log_start) * 1000.0

        final_log_start = time.perf_counter()
        total_bytes = sum(
            int(result.descriptor.total_bytes)
            for result in in_flight_request.results
        )
        total_tokens = int(return_snapshot.consumable_tokens)
        batch_stats = in_flight_request.results[0].stats
        logger.info(
            "[LMCACHE_BAM_DIRECT_PLACEMENT_READ] batch_size=%d total_bytes=%d "
            "submit_ms=%.3f poll_ms=%.3f poll_iters=%d get_ms=%.3f "
            "read_ms=%.3f executor=%s worker_backend=%s",
            len(in_flight_request.results),
            total_bytes,
            batch_stats.submit_ms,
            batch_stats.poll_ms,
            batch_stats.poll_iters,
            batch_stats.get_ms,
            bootstrap_profile.read_ms,
            getattr(batch_stats, "executor_name", "rowctx"),
            getattr(batch_stats, "worker_backend", "rowctx"),
        )
        logger.info(
            "[LMCACHE_BAM_DIRECT_PLACEMENT] chunks=%d tokens=%d "
            "impl=%s read_ms=%.3f refill_ms=%.3f transfer_ms=%.3f "
            "fused_ms=%.3f place_ms=%.3f total_ms=%.3f",
            in_flight_request.prefix_hit_chunks,
            total_tokens,
            place_stats.impl,
            bootstrap_profile.read_ms,
            place_stats.refill_ms,
            place_stats.transfer_ms,
            place_stats.fused_ms,
            place_stats.place_ms,
            bootstrap_profile.read_ms + place_stats.place_ms,
        )
        final_log_ms = (time.perf_counter() - final_log_start) * 1000.0
        direct_total_ms = (
            time.perf_counter() - bootstrap_profile.direct_total_start_time
        ) * 1000.0
        logger.info(
            "[LMCACHE_BAM_DIRECT_PLACEMENT_PROFILE] collect_entries_ms=%.3f "
            "read_ms=%.3f descriptor_ms=%.3f tracker_init_ms=%.3f "
            "prepare_ms=%.3f frontier_wave_wall_ms=%.3f "
            "cache_ready_log_ms=%.3f ret_mask_ms=%.3f "
            "followup_wave_wall_ms=%.3f followup_log_ms=%.3f "
            "final_log_ms=%.3f direct_total_ms=%.3f",
            bootstrap_profile.collect_entries_ms,
            bootstrap_profile.read_ms,
            bootstrap_profile.descriptor_ms,
            bootstrap_profile.tracker_init_ms,
            bootstrap_profile.prepare_ms,
            frontier_wave_ms,
            cache_ready_log_ms,
            ret_mask_ms,
            followup_total_ms,
            followup_log_ms,
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
        切换时再做一轮分叉。当前 store 侧 request handle 本身还不直接依赖这两个
        参数，但保留这层签名有助于未来继续向 runtime 抬升控制面。
        """
        del num_kv_heads, head_size
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

    def finalize_direct_retrieve_to_vllm_kvcache(
        self,
        *,
        in_flight_request: Any,
    ) -> Optional[torch.Tensor]:
        """公开 direct retrieve request 的 finalize 边界。"""
        if self._bam_store is None:
            return None
        return self._bam_store.finalize_direct_placement_request(
            in_flight_request=in_flight_request,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._storage_manager, name)
