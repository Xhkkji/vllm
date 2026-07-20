# SPDX-License-Identifier: Apache-2.0
"""LMCache V0 的可选 LMCache-style GDS wrapper。

默认不开启；只有设置 `VLLM_GDS_LMCACHE_SHADOW_ENABLE=1` 或
`VLLM_GDS_LMCACHE_PREFER_LOAD_ENABLE=1` 时才会包住 LMCache storage manager。
这样不会影响原生 SSD 路径，也不会影响已经实现好的 BaM 路径。
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional

import torch

import vllm.envs as envs
from vllm.bam.gds_baseline.lmcache_style_gds_store import (
    LMCacheStyleGDSChunkStore, LMCacheStyleGDSConfig)
from vllm.bam.lmcache_bam_storage import (LMCacheMemoryObjAdapter,
                                          _extract_chunk_hash)
from vllm.logger import init_logger

logger = init_logger(__name__)


def _resolve_gds_device() -> str:
    if envs.VLLM_GDS_LMCACHE_DEVICE:
        return envs.VLLM_GDS_LMCACHE_DEVICE
    return f"cuda:{envs.VLLM_BAM_CTRL_IDX}"


class LMCacheGDSStorageManager:
    """给 LMCache V0 storage manager 增加可选 GDS shadow/prefer-load。"""

    def __init__(self, storage_manager: Any) -> None:
        self._storage_manager = storage_manager
        self._gds_store: Optional[LMCacheStyleGDSChunkStore] = None
        self._gds_init_attempted = False
        self._gds_init_failed = False
        # GDS prefer-load 的正确性对照会额外调用一次原生 LMCache get()。
        # 这对排查很有用，但会把“SSD->GPU(GDS)”性能口径污染成
        # “GDS 读 + 原生 SSD 读”。因此默认保留历史的前 2 个 chunk 校验，
        # 性能脚本可以显式设为 0，只观察 GDS prefer-load 本身。
        self._prefer_load_verify_budget = max(
            int(os.getenv("VLLM_GDS_LMCACHE_VERIFY_BUDGET", "2")), 0)

    def _log_gds_runtime_context(self, tensor: torch.Tensor) -> None:
        logger.info(
            "[LMCACHE_GDS] init context pid=%d euid=%d exe=%s cwd=%s "
            "tensor_shape=%s tensor_dtype=%s tensor_device=%s gds_path=%s "
            "use_gds=%s use_direct_io=%s device=%s fmt=%s "
            "use_registered_buffer=%s registered_buffer_mb=%d",
            os.getpid(),
            os.geteuid(),
            sys.executable,
            os.getcwd(),
            tuple(tensor.shape),
            tensor.dtype,
            tensor.device,
            envs.VLLM_GDS_LMCACHE_PATH,
            envs.VLLM_GDS_LMCACHE_USE_GDS,
            envs.VLLM_GDS_LMCACHE_USE_DIRECT_IO,
            _resolve_gds_device(),
            envs.VLLM_GDS_LMCACHE_FMT,
            envs.VLLM_GDS_LMCACHE_USE_REGISTERED_BUFFER,
            envs.VLLM_GDS_LMCACHE_REGISTERED_BUFFER_MB,
        )

    def _ensure_gds_store(self, memory_obj: Any) -> Optional[LMCacheStyleGDSChunkStore]:
        tensor = getattr(memory_obj, "tensor", None)
        if tensor is None:
            return None
        if self._gds_store is not None:
            return self._gds_store
        if self._gds_init_failed:
            return None

        if not self._gds_init_attempted:
            self._gds_init_attempted = True
            self._log_gds_runtime_context(tensor)
            try:
                config = LMCacheStyleGDSConfig(
                    gds_path=envs.VLLM_GDS_LMCACHE_PATH,
                    device=_resolve_gds_device(),
                    use_gds=envs.VLLM_GDS_LMCACHE_USE_GDS,
                    use_direct_io=envs.VLLM_GDS_LMCACHE_USE_DIRECT_IO,
                    fmt=envs.VLLM_GDS_LMCACHE_FMT,
                    use_registered_buffer=(
                        envs.VLLM_GDS_LMCACHE_USE_REGISTERED_BUFFER),
                    registered_buffer_size=(
                        envs.VLLM_GDS_LMCACHE_REGISTERED_BUFFER_MB * 1024 * 1024),
                )
                self._gds_store = LMCacheStyleGDSChunkStore(config=config)
            except Exception:
                self._gds_init_failed = True
                logger.exception(
                    "[LMCACHE_GDS] failed to initialize GDS store; "
                    "falling back to original LMCache path")
                return None

        return self._gds_store

    def _try_prefer_gds_load(self, key: Any) -> Optional[Any]:
        if self._gds_store is None:
            return None

        metadata = self._gds_store.get_chunk_metadata(key)
        if metadata is None:
            return None

        memory_obj = self._storage_manager.allocate(metadata.shape, metadata.dtype)
        if memory_obj is None:
            return None

        tensor = self._gds_store.load_chunk_tensor(key)
        if tensor is None:
            return None

        self._maybe_verify_prefer_load_tensor(key, tensor)
        LMCacheMemoryObjAdapter.populate_kv_blob(memory_obj, tensor)
        logger.info(
            "[LMCACHE_GDS] prefer-load hit chunk_hash=%s",
            _extract_chunk_hash(key)[:16],
        )
        return memory_obj

    def _maybe_verify_prefer_load_tensor(self, key: Any,
                                         gds_tensor: torch.Tensor) -> None:
        if self._prefer_load_verify_budget <= 0:
            return
        self._prefer_load_verify_budget -= 1
        try:
            original_memory_obj = self._storage_manager.get(key)
            if original_memory_obj is None or original_memory_obj.tensor is None:
                logger.warning(
                    "[LMCACHE_GDS_VERIFY] original LMCache tensor missing; "
                    "skip verification for chunk_hash=%s",
                    _extract_chunk_hash(key)[:16],
                )
                return
            original_tensor = original_memory_obj.tensor
            gds_cpu = gds_tensor.detach().to("cpu", non_blocking=False)
            original_cpu = original_tensor.detach().to("cpu", non_blocking=False)
            exact_equal = bool(torch.equal(gds_cpu, original_cpu))
            max_abs_diff = float((gds_cpu - original_cpu).abs().max().item())
            mean_abs_diff = float(
                (gds_cpu - original_cpu).abs().float().mean().item())
            logger.info(
                "[LMCACHE_GDS_VERIFY] chunk_hash=%s exact_equal=%s "
                "max_abs_diff=%.6f mean_abs_diff=%.6f shape=%s dtype=%s",
                _extract_chunk_hash(key)[:16],
                exact_equal,
                max_abs_diff,
                mean_abs_diff,
                tuple(gds_cpu.shape),
                gds_cpu.dtype,
            )
        except Exception:
            logger.exception(
                "[LMCACHE_GDS_VERIFY] failed to compare GDS tensor with "
                "original LMCache tensor")

    def put(self, key: Any, memory_obj: Any) -> None:
        gds_store = self._ensure_gds_store(memory_obj)
        if (envs.VLLM_GDS_LMCACHE_SHADOW_ENABLE and gds_store is not None
                and getattr(memory_obj, "tensor", None) is not None):
            try:
                gds_store.put_chunk(key, memory_obj.tensor)
            except Exception:
                logger.exception(
                    "[LMCACHE_GDS] shadow store failed; keep LMCache original path")

        self._storage_manager.put(key, memory_obj)

    def get(self, key: Any) -> Optional[Any]:
        if envs.VLLM_GDS_LMCACHE_PREFER_LOAD_ENABLE:
            try:
                memory_obj = self._try_prefer_gds_load(key)
                if memory_obj is not None:
                    return memory_obj
            except Exception:
                logger.exception(
                    "[LMCACHE_GDS] prefer-load failed; falling back to LMCache")

        memory_obj = self._storage_manager.get(key)
        if memory_obj is not None and envs.VLLM_GDS_LMCACHE_PREFER_LOAD_ENABLE:
            try:
                logger.info(
                    "[LMCACHE_GDS] prefer-load miss/fallback chunk_hash=%s",
                    _extract_chunk_hash(key)[:16],
                )
            except Exception:
                pass
        return memory_obj

    def __getattr__(self, name: str) -> Any:
        return getattr(self._storage_manager, name)
