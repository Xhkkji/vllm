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
    """为 BaM 槽位选择稳定的 token 容量。"""
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
    """记录单个 LMCache chunk 在 BaM 里的落点和还原信息。"""

    slot_id: int
    page_offset: int
    actual_tokens: int
    shape: torch.Size
    dtype: torch.dtype


@dataclass(frozen=True)
class LMCacheBaMPageLayout:
    """描述 LMCache KV chunk 在 BaM 中的固定页布局。"""

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
        """把 KV chunk 编成固定数量的 128KB page。"""
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
        """把固定页还原回真实长度的 KV chunk。"""
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
                 chunk_capacity: int) -> None:
        self.row_store = row_store
        self.layout = layout
        self.chunk_capacity = int(chunk_capacity)

        self._chunk_slots: "OrderedDict[str, int]" = OrderedDict()
        self._chunk_metadata: Dict[str, BaMChunkMetadata] = {}
        self._slot_lock = threading.Lock()

    @classmethod
    def from_kv_shape(cls, kv_shape: torch.Size,
                      dtype: torch.dtype) -> "LMCacheBaMStore":
        layout = LMCacheBaMPageLayout.from_kv_shape(kv_shape, dtype)
        chunk_capacity = int(envs.VLLM_BAM_LMCACHE_SHADOW_CHUNKS)
        num_rows = int(layout.pages_per_chunk * chunk_capacity)

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
            "ctrl_idx=%d",
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
        )
        return cls(row_store=row_store,
                   layout=layout,
                   chunk_capacity=chunk_capacity)

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

    def store_chunk(self, key: Any, tensor: torch.Tensor) -> None:
        chunk_hash = _extract_chunk_hash(key)
        actual_tokens = int(tensor.shape[2])
        pages = self.layout.encode_pages(tensor)
        slot_id = self._get_or_assign_slot(chunk_hash)
        page_offset = int(slot_id * self.layout.pages_per_chunk)

        start = time.perf_counter()
        if not pages.is_cuda:
            # LMCache 原始 buffer 常在 CPU / pinned CPU。
            # 这里统一显式搬到 BaM 控制 GPU，路径更直白。
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

        # 优先回填 raw_data，语义上更接近 LMCache 原始磁盘路径的 readinto(byte_buffer)。
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
        if self._bam_store is None:
            return None

        metadata = self._bam_store.get_chunk_metadata(key)
        if metadata is None:
            return None

        memory_obj = self._storage_manager.allocate(metadata.shape, metadata.dtype)
        if memory_obj is None:
            return None

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

    def _maybe_verify_prefer_load_tensor(self, key: Any,
                                         bam_tensor: torch.Tensor) -> None:
        """把 BaM 结果和原始 LMCache 结果做一次直接对照。"""
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
        bam_store = self._ensure_bam_store(memory_obj)
        if bam_store is not None and getattr(memory_obj, "tensor", None) is not None:
            try:
                bam_store.store_chunk(key, memory_obj.tensor)
            except Exception:
                logger.exception(
                    "[LMCACHE_BAM] shadow store failed; keep LMCache original path")

        self._storage_manager.put(key, memory_obj)

    def get(self, key: Any) -> Optional[Any]:
        if envs.VLLM_BAM_LMCACHE_PREFER_LOAD_ENABLE:
            try:
                memory_obj = self._try_prefer_bam_load(key)
                if memory_obj is not None:
                    return memory_obj
            except Exception:
                logger.exception(
                    "[LMCACHE_BAM] prefer-load failed; falling back to LMCache")

        memory_obj = self._storage_manager.get(key)
        if memory_obj is not None and envs.VLLM_BAM_LMCACHE_PREFER_LOAD_ENABLE:
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
