# SPDX-License-Identifier: Apache-2.0
"""LMCache BaM 的 KVCache 专用 fast path。

这层对应 GPU-initiated 路线的“第 2 档起步”：

  LMCache/BaM metadata
    -> KV chunk descriptor
    -> BaM KV store 读回 [page_count, 128KB] pages
    -> GPU refill 成 LMCache KV tensor

当前第一阶段的 BaM KV store 仍复用 BaM_IOStack 里的 rowctx 接口，所以它
不是最终 persistent GPU worker。但它已经把上层接口从通用 feature path
中拆出来，后续底层可以替换为真正的 `gids_kv_cache`，而不需要重写
vLLM/LMCache 接入逻辑。
"""

from __future__ import annotations

import time
from typing import Any, Sequence

import torch

from vllm.bam.lmcache_bam_refill import refill_pages_to_lmcache_tensor
from vllm.bam.row_store_loader import import_bam_kv_store
from vllm.logger import init_logger

logger = init_logger(__name__)

_KV_FAST_PATH_TIMEOUT_S = 10.0


def _kv_batch_status_name(status: int) -> str:
    """把 BaM KV request/status table 的状态值转成日志友好的名字。"""
    return {
        0: "INIT",
        1: "SUBMITTED",
        2: "IO_DONE",
        3: "CONSUMED",
        4: "ERROR",
    }.get(int(status), f"UNKNOWN({status})")


def _kv_chunk_status_summary(statuses: object) -> str:
    """把 per-chunk status table 压缩成日志友好的摘要。

    例如 8 个 chunk 全部完成时输出 `8xCONSUMED`。如果未来底层支持乱序
    completion，这里会输出不同状态的计数，方便直接看出分布。
    """
    if not statuses:
        return "none"
    counts: dict[int, int] = {}
    for status in statuses:  # type: ignore[union-attr]
        status_int = int(status)
        counts[status_int] = counts.get(status_int, 0) + 1
    return ",".join(
        f"{count}x{_kv_batch_status_name(status)}"
        for status, count in sorted(counts.items()))


def _kv_completion_status_summary(statuses: object) -> str:
    """压缩 completion table 中的 per-chunk status。"""
    return _kv_chunk_status_summary(statuses)


def _kv_completion_bytes_summary(values: object) -> str:
    """压缩 completion table 中的 bytes_done。

    稳定路径下同一批 chunk 的 bytes_done 应该相同，例如
    `8x14680064`。如果未来支持不同大小 chunk，这里会显示多个桶。
    """
    if not values:
        return "none"
    counts: dict[int, int] = {}
    for value in values:  # type: ignore[union-attr]
        value_int = int(value)
        counts[value_int] = counts.get(value_int, 0) + 1
    return ",".join(f"{count}x{value}" for value, count in sorted(counts.items()))


def _kv_completion_error_summary(values: object) -> str:
    """压缩 completion table 中的 error_code。正常情况下应为 `Nx0`。"""
    return _kv_completion_bytes_summary(values)


class LMCacheBaMKVFastPath:
    """KVCache 专用读取路径。

    这个类刻意只依赖三个东西：

    - `row_store`: 已初始化好的 BaMRowStore，负责底层设备和 rowctx 生命周期
    - `layout`: LMCache KV chunk 到 128KB pages 的固定布局
    - `metadata`: 每个 chunk 的 page_offset / actual_tokens / dtype / shape

    它不关心 LMCache storage manager，也不直接操作 memory_obj。这样可以
    保持接口解耦：storage 层负责 key/memory_obj，fast path 只负责 KV 数据面。
    """

    def __init__(self, *, row_store: Any, layout: Any,
                 device: str | torch.device) -> None:
        self.layout = layout
        self.device = torch.device(device)
        kv_store_cls = import_bam_kv_store()
        self.kv_store = kv_store_cls(
            row_store=row_store,
            page_bytes=layout.page_bytes,
            device=self.device,
        )

    def _make_request(self, *, chunk_id: int, metadata: Any) -> Any:
        """把 BaM metadata 转成 KV 专用 descriptor。

        形状关系：

        ```text
        metadata.page_offset
          -> chunk 起始 128KB page id

        layout.pages_per_chunk
          -> 当前 chunk 固定读取页数，例如 Qwen2.5-7B fp16 为 112

        metadata.actual_tokens
          -> 读回后 refill 时裁掉 padding
        ```
        """
        return self.kv_store.make_request(
            chunk_id=chunk_id,
            page_offset=metadata.page_offset,
            page_count=self.layout.pages_per_chunk,
            actual_tokens=metadata.actual_tokens,
        )

    def _refill_pages(self, pages: torch.Tensor, metadata: Any) -> torch.Tensor:
        """把 `[page_count, 128KB]` pages 还原成 LMCache KV tensor。"""
        try:
            # CPU 只负责 launch；真正的 page -> KV 元素映射在 Triton/GPU 中执行。
            return refill_pages_to_lmcache_tensor(pages, metadata, self.layout)
        except Exception:
            logger.exception(
                "[LMCACHE_BAM_KV_FAST_PATH_REFILL] GPU refill failed; "
                "fallback to PyTorch decode_pages")
            return self.layout.decode_pages(pages, metadata)

    def load_chunk_tensor(self, *, chunk_hash: str,
                          metadata: Any) -> torch.Tensor:
        """读取单个 chunk 并还原成 LMCache tensor。

        第一阶段流程：

        ```text
        CPU:
          metadata -> BaMKVRequest
          submit/wait request
          launch refill

        BaM/GPU:
          rowctx read 112 个 128KB pages
          pages buffer 写在 GPU 上
          refill kernel 还原 KV tensor
        ```
        """
        total_start = time.perf_counter()
        request = self._make_request(chunk_id=metadata.slot_id,
                                     metadata=metadata)
        result = self.kv_store.read_pages(request,
                                          timeout_s=_KV_FAST_PATH_TIMEOUT_S)

        refill_start = time.perf_counter()
        tensor = self._refill_pages(result.pages, metadata)
        refill_ms = (time.perf_counter() - refill_start) * 1000.0
        total_ms = (time.perf_counter() - total_start) * 1000.0

        elapsed_s = total_ms / 1000.0
        gib_per_s = (result.descriptor.total_bytes / elapsed_s /
                     (1024**3)) if elapsed_s > 0 else 0.0
        logger.info(
            "[LMCACHE_BAM_KV_FAST_PATH_READ] chunk_hash=%s page_offset=%d "
            "page_count=%d page_bytes=%d actual_tokens=%d total_bytes=%d "
            "submit_ms=%.3f poll_ms=%.3f poll_iters=%d get_ms=%.3f "
            "read_ms=%.3f refill_ms=%.3f total_ms=%.3f bw_gib_s=%.3f",
            chunk_hash[:16],
            result.descriptor.page_offset,
            result.descriptor.page_count,
            result.descriptor.page_bytes,
            result.descriptor.actual_tokens,
            result.descriptor.total_bytes,
            result.stats.submit_ms,
            result.stats.poll_ms,
            result.stats.poll_iters,
            result.stats.get_ms,
            result.stats.total_ms,
            refill_ms,
            total_ms,
            gib_per_s,
        )
        return tensor

    def load_chunk_tensors_batch(
        self,
        items: Sequence[tuple[str, Any]],
    ) -> dict[str, torch.Tensor]:
        """批量读取多个 chunk 并还原成 LMCache tensor。

        这是 KV fast path 第一阶段最重要的验证入口：

        ```text
        [(chunk_hash, metadata), ...]
          -> [BaMKVRequest, ...]
          -> BaM KV batch read
          -> 每个 chunk 得到 [112, 128KB] pages
          -> 逐个 GPU refill
        ```

        当前 refill 仍逐 chunk 调用，先保证路径简洁和正确。后续如果 batch read
        稳定，可以再做 batch refill 或直接回填 vLLM paged KV cache。
        """
        if not items:
            return {}

        total_start = time.perf_counter()
        requests = [
            self._make_request(chunk_id=metadata.slot_id, metadata=metadata)
            for _, metadata in items
        ]
        results = self.kv_store.read_pages_batch(
            requests,
            timeout_s=_KV_FAST_PATH_TIMEOUT_S,
        )

        refill_start = time.perf_counter()
        tensors: dict[str, torch.Tensor] = {}
        for (chunk_hash, metadata), result in zip(items, results):
            tensors[chunk_hash] = self._refill_pages(result.pages, metadata)
        refill_ms = (time.perf_counter() - refill_start) * 1000.0
        total_ms = (time.perf_counter() - total_start) * 1000.0

        total_bytes = sum(result.descriptor.total_bytes for result in results)
        elapsed_s = total_ms / 1000.0
        gib_per_s = (total_bytes / elapsed_s /
                     (1024**3)) if elapsed_s > 0 else 0.0
        # 同一批 native KV read 当前共享一次 submit/poll/get；取第一个 result
        # 的 stats 就能代表整批 BaM IO 的分段耗时。这里把它打到 vLLM 日志，
        # 方便区分“SSD/BaM 读慢”和“GPU refill/JIT 慢”。
        batch_stats = results[0].stats
        logger.info(
            "[LMCACHE_BAM_KV_FAST_PATH_BATCH_READ] batch_size=%d total_bytes=%d "
            "submit_ms=%.3f poll_ms=%.3f poll_iters=%d get_ms=%.3f "
            "read_ms=%.3f refill_ms=%.3f total_ms=%.3f bw_gib_s=%.3f "
            "status=%s->%s->%s gpu_status=%s->%s->%s "
            "chunk_gpu_status=%s->%s->%s request_table=%s "
            "completion_status=%s->%s->%s completion_bytes=%s->%s "
            "completion_error=%s->%s executor=%s worker_backend=%s",
            len(items),
            total_bytes,
            batch_stats.submit_ms,
            batch_stats.poll_ms,
            batch_stats.poll_iters,
            batch_stats.get_ms,
            batch_stats.total_ms,
            refill_ms,
            total_ms,
            gib_per_s,
            _kv_batch_status_name(getattr(batch_stats, "submit_status", 1)),
            _kv_batch_status_name(getattr(batch_stats, "ready_status", 2)),
            _kv_batch_status_name(getattr(batch_stats, "consumed_status", 3)),
            _kv_batch_status_name(
                getattr(batch_stats, "submit_gpu_status", 1)),
            _kv_batch_status_name(getattr(batch_stats, "ready_gpu_status", 2)),
            _kv_batch_status_name(
                getattr(batch_stats, "consumed_gpu_status", 3)),
            _kv_chunk_status_summary(
                getattr(batch_stats, "submit_chunk_statuses", ())),
            _kv_chunk_status_summary(
                getattr(batch_stats, "ready_chunk_statuses", ())),
            _kv_chunk_status_summary(
                getattr(batch_stats, "consumed_chunk_statuses", ())),
            getattr(batch_stats, "request_table_mode", "offsets"),
            _kv_completion_status_summary(
                getattr(batch_stats, "submit_completion_statuses", ())),
            _kv_completion_status_summary(
                getattr(batch_stats, "ready_completion_statuses", ())),
            _kv_completion_status_summary(
                getattr(batch_stats, "consumed_completion_statuses", ())),
            _kv_completion_bytes_summary(
                getattr(batch_stats, "ready_completion_bytes", ())),
            _kv_completion_bytes_summary(
                getattr(batch_stats, "consumed_completion_bytes", ())),
            _kv_completion_error_summary(
                getattr(batch_stats, "ready_completion_errors", ())),
            _kv_completion_error_summary(
                getattr(batch_stats, "consumed_completion_errors", ())),
            getattr(batch_stats, "executor_name", "rowctx"),
            getattr(batch_stats, "worker_backend", "rowctx"),
        )
        return tensors

    def read_chunk_pages_batch(
        self,
        items: Sequence[tuple[str, Any]],
    ) -> list[Any]:
        """批量读取 chunk pages，但不还原成 LMCache tensor。

        Direct Placement 路径会直接把 BaM pages 写入 vLLM paged KV cache。
        因此这里刻意只返回底层 `BaMKVReadResult`：

        ```text
        [(chunk_hash, metadata), ...]
          -> BaM KV batch read
          -> [BaMKVReadResult(page_count x 128KB), ...]
          -> direct placement kernel / reshape_and_cache_flash
        ```

        这样可以复用当前已经验证过的 `gpu_worker + kv_cq_service_v1` I/O
        路径，同时绕开 `refill_pages_to_lmcache_tensor()`。
        """
        if not items:
            return []

        requests = [
            self._make_request(chunk_id=metadata.slot_id, metadata=metadata)
            for _, metadata in items
        ]
        return self.kv_store.read_pages_batch(
            requests,
            timeout_s=_KV_FAST_PATH_TIMEOUT_S,
        )
