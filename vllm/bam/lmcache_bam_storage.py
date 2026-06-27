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
        self._prefetch_lock = threading.Lock()
        self._prefetch_requests: Dict[str, LMCacheBaMChunkReadRequest] = {}

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
            "base_row_offset=%d",
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
        #   3. 从 BaM 读回 tensor。sync 模式走完整 chunk 同步读；
        #      prefetch 模式走 page-level submit/poll/get/refill 中间层。
        #   4. 把 tensor 回填进 LMCache 的 memory_obj
        if self._bam_store is None:
            return None

        metadata = self._bam_store.get_chunk_metadata(key)
        if metadata is None:
            return None

        memory_obj = self._storage_manager.allocate(metadata.shape, metadata.dtype)
        if memory_obj is None:
            return None

        if envs.VLLM_BAM_LMCACHE_READ_MODE == "prefetch":
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
        - 其他模式：透传给原始 LMCache storage manager

        这让真实 vLLM 路径可以在 `retrieve()` 前先发起 BaM IO，而 `get()`
        时只需要等待/消费 outstanding request。
        """
        if (envs.VLLM_BAM_LMCACHE_PREFER_LOAD_ENABLE
                and envs.VLLM_BAM_LMCACHE_READ_MODE == "prefetch"
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

    def __getattr__(self, name: str) -> Any:
        return getattr(self._storage_manager, name)
