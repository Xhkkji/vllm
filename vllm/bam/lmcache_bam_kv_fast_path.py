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

from dataclasses import dataclass
import time
from typing import Any, Sequence

import torch

from vllm.bam.lmcache_bam_direct_placement import (
    BaMRuntimeAttentionMetadataAttachment)
from vllm.bam.lmcache_bam_refill import refill_pages_to_lmcache_tensor
from vllm.bam.row_store_loader import import_bam_kv_store
from vllm.logger import init_logger

logger = init_logger(__name__)

_KV_FAST_PATH_TIMEOUT_S = 10.0
_KV_FRONTIER_COL_STATUS = 0
_KV_FRONTIER_COL_LAUNCH_FRONTIER_CHUNKS = 1
_KV_FRONTIER_COL_READ_READY_FRONTIER_CHUNKS = 2
_KV_FRONTIER_COL_CACHE_READY_FRONTIER_CHUNKS = 3
_KV_FRONTIER_COL_CONSUMABLE_FRONTIER_CHUNKS = 4
_KV_FRONTIER_COL_TOTAL_CHUNKS = 5
_KV_FRONTIER_COL_ERROR_CODE = 6


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


def _kv_frontier_value(frontier_row: Sequence[int], column: int) -> int:
    """安全读取 frontier row 中的某一列。

    这里统一做“缺列则返回 0”的薄封装，避免上层 request-handle 代码里到处散落
    越界判断。当前 host frontier mirror 在老扩展/调试场景下可能暂时缺失部分列，
    但这不应该让 KV fast path 的控制面直接崩掉。
    """
    if len(frontier_row) <= int(column):
        return 0
    return int(frontier_row[int(column)])


@dataclass(frozen=True)
class LMCacheBaMKVBatchReadPollSnapshot:
    """一次 batch read handle 轮询后的只读快照。

    这层对象只保存 direct placement request 当前真正需要观察的控制面信息：

    - `ready`
      整个 native batch 是否已经到达底层声明的可消费边界。对 runtime
      one-copy 来说，这个边界是 GPU persistent service 已经完成
      direct placement 并发布 `CONSUMED`。
    - `frontier_row`
      来自 BaM KV request 的 request-level frontier ABI
    - `read_ready_frontier_chunks`
      当前从 batch 开头起，连续多少个 chunk 已经 page-read ready

    注意：
    `cache_ready/consumable` 这两列当前分两种语义来源：

    1. 普通 native read / materialized finalize 路径
       这两列还只是底层 KV request 生命周期的保留列，不能直接视为
       “最终 vLLM KV cache 已经可计算”。
    2. runtime direct placement 已 attach 的路径
       persistent GPU service 会在同一张 frontier table 上直接发布：
       `BaM cache -> 最终 KV cache` 的完成前缀。
       这时 `consumable_frontier_chunks` 就已经是上层可直接调度的权威语义。

    这样上层就不必直接依赖 `BaMKVStatusSnapshot` 的内部字段细节，可以在后续
    继续把 request-handle 往 runtime 抬升时保持接口收敛。
    """

    ready: bool
    poll_iters: int
    host_status: int
    frontier_row: tuple[int, ...]
    launch_frontier_chunks: int
    read_ready_frontier_chunks: int
    cache_ready_frontier_chunks: int
    consumable_frontier_chunks: int
    total_chunks: int
    error_code: int


@dataclass
class LMCacheBaMKVBatchReadRequestHandle:
    """一批已经 submit、后续可 poll/consume 的 KV page read request。"""

    items: tuple[tuple[str, Any], ...]
    native_handle: Any
    last_poll_snapshot: LMCacheBaMKVBatchReadPollSnapshot | None = None
    runtime_direct_placement_attached: bool = False
    runtime_attention_metadata_attached: bool = False


@dataclass(frozen=True)
class LMCacheBaMKVPreparedBatchRead:
    """一批尚未 submit 的 KV read descriptor。

    这层对象是接下来 GPU-initiated demand-load 的干净边界：

    - CPU/上层调度只负责提前把“可能要读哪些 chunk”整理成紧凑 descriptor；
    - 这里不调用 `submit_native_batch()`，因此不会提前发起 CPU submit；
    - 当前 Python 版本在真正 direct placement start 时消费这份 descriptor；
    - 后续如果底层增加 GPU-side descriptor ring，可以把同样字段写入 ring，
      由 GPU persistent service 自己取 descriptor 并发起 SSD read。

    换句话说，它是“计划/描述符”，不是“已经在飞的 I/O handle”。
    """

    items: tuple[tuple[str, Any], ...]
    requests: tuple[Any, ...]
    source: str
    created_at_s: float

    @property
    def batch_key(self) -> tuple[str, ...]:
        """按 chunk 顺序生成稳定 key，避免 descriptor 绑定错 batch。"""
        return tuple(chunk_hash for chunk_hash, _ in self.items)

    @property
    def total_bytes(self) -> int:
        """本 batch 预计读取的总字节数，只用于日志/调试。"""
        return sum(
            int(getattr(request, "total_bytes", 0))
            for request in self.requests
        )


@dataclass(frozen=True)
class LMCacheBaMKVBatchReadRuntimeSnapshot:
    """KV batch read request 当前对应的 runtime 观察快照。

    这层对象的目的，是把 BaM_IOStack 里的 runtime slot 表细节收成一份更薄、
    更稳定的上层观察接口。当前 direct placement / deferred runtime 可以用它来
    回答三个问题：

    - persistent service 当前是否真的在运行
    - 这批 native read 是否已经登记进 runtime slot
    - 当前 request/frontier/completion 三张 GPU-visible 表的所有权是否稳定

    它不参与 ready 判定，也不改变现有 submit/poll/consume 语义，只作为后续
    GPU-resident frontier/service kernel 阶段的观测桥。
    """

    service_running: bool
    active_count: int
    request_id: int
    worker_backend: str
    request_table_ptr: int
    frontier_table_ptr: int
    completion_table_ptr: int
    matched_runtime_row: tuple[int, ...] | None
    runtime_rows: tuple[tuple[int, ...], ...]


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

    def _make_requests_for_items(
        self,
        items: Sequence[tuple[str, Any]],
    ) -> list[Any]:
        """把 `(chunk_hash, metadata)` 列表统一转换成底层 KV request。"""
        return [
            self._make_request(chunk_id=metadata.slot_id, metadata=metadata)
            for _, metadata in items
        ]

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
        results = self.consume_chunk_pages_batch_request(
            self.submit_chunk_pages_batch_request(items),
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

        这样可以复用当前已经验证过的 `gpu_worker + persistent/runtime`
        I/O 路径，同时绕开 `refill_pages_to_lmcache_tensor()`。
        """
        if not items:
            return []

        return self.consume_chunk_pages_batch_request(
            self.submit_chunk_pages_batch_request(items),
            timeout_s=_KV_FAST_PATH_TIMEOUT_S,
        )

    def submit_chunk_pages_batch_request(
        self,
        items: Sequence[tuple[str, Any]],
    ) -> LMCacheBaMKVBatchReadRequestHandle:
        """显式提交一批 chunk pages read，不等待完成。

        这是把 KV fast path 从“阻塞读黑盒”推进成 request-handle 三段式的关键
        边界：

        ```text
        items
          -> BaMKVRequest[]
          -> submit_native_batch()
          -> 后续由 poll/consume 分别推进
        ```

        这样 direct placement / runtime 层就可以把 BaM I/O 的生命周期跨越多个
        engine iteration 保留下来，而不是在 `start()` 阶段同步等完整批次读完。
        """
        if not items:
            raise ValueError(
                "submit_chunk_pages_batch_request requires non-empty items")

        requests = self._make_requests_for_items(items)
        native_handle = self.kv_store.submit_native_batch(requests)
        return LMCacheBaMKVBatchReadRequestHandle(
            items=tuple(items),
            native_handle=native_handle,
        )

    def prepare_chunk_pages_batch_request(
        self,
        items: Sequence[tuple[str, Any]],
        *,
        source: str,
    ) -> LMCacheBaMKVPreparedBatchRead:
        """只构造 KV read descriptor，不发起底层 submit。

        这是把“上层预取策略”和“底层 I/O 发起”解耦的核心接口。之前的
        early-submit 原型会在 connector/storage 阶段直接调用 native submit，
        本质仍是 CPU submit。现在这里明确只做 descriptor 准备：

        ```text
        chunk metadata -> BaMKVRequest[] -> prepared batch
        ```

        真正 read 什么时候启动，由后续 direct placement request 或未来的
        GPU-side descriptor consumer 决定。
        """
        if not items:
            raise ValueError(
                "prepare_chunk_pages_batch_request requires non-empty items")
        requests = self._make_requests_for_items(items)
        return LMCacheBaMKVPreparedBatchRead(
            items=tuple(items),
            requests=tuple(requests),
            source=str(source),
            created_at_s=time.perf_counter(),
        )

    def submit_prepared_chunk_pages_batch_request(
        self,
        prepared: LMCacheBaMKVPreparedBatchRead,
    ) -> LMCacheBaMKVBatchReadRequestHandle:
        """消费已准备好的 descriptor，并在当前 CPU 入口提交 native read。

        当前底层还没有暴露“GPU 消费 descriptor ring 并自行 submit”的 Python
        入口，所以这里仍然是兼容执行点。但它和旧逻辑的关键区别是：

        - submit 没有再提前发生在 connector/storage 预取阶段；
        - descriptor 生命周期已经独立出来，后续替换成 GPU-side submit 时，
          上层只需要把 `prepared.requests/items` 写入设备 ring。
        """
        if not prepared.items or not prepared.requests:
            raise ValueError(
                "submit_prepared_chunk_pages_batch_request requires "
                "non-empty prepared descriptor")
        native_handle = self.kv_store.submit_native_batch(prepared.requests)
        return LMCacheBaMKVBatchReadRequestHandle(
            items=prepared.items,
            native_handle=native_handle,
        )

    def poll_chunk_pages_batch_request(
        self,
        handle: LMCacheBaMKVBatchReadRequestHandle,
    ) -> LMCacheBaMKVBatchReadPollSnapshot:
        """推进一次 batch read request，并返回当前 frontier 快照。

        注意这里的语义是：

        - persistent runtime 已活跃时，底层 `poll_native_batch()` 会尽量退化成
          只读 request/runtime 状态；
        - 兼容路径下，仍可能顺手推进一次 batch/service poll façade；
        - 但不会做 consume，因此还不会把 pages buffer 暴露给上层。

        这正好对应当前 direct placement request 的第一阶段目标：
        先让“read-ready frontier”在 runtime 里变成可观察状态；后续再把
        consume/placement 进一步往更细粒度推进。
        """
        ready = bool(self.kv_store.poll_native_batch(handle.native_handle))
        status_snapshot = self.kv_store.snapshot_native_batch(
            handle.native_handle)
        frontier_row = tuple(status_snapshot.frontier_row or ())
        poll_snapshot = LMCacheBaMKVBatchReadPollSnapshot(
            ready=ready,
            poll_iters=int(handle.native_handle.poll_iters),
            host_status=int(status_snapshot.host_status),
            frontier_row=frontier_row,
            launch_frontier_chunks=_kv_frontier_value(
                frontier_row,
                _KV_FRONTIER_COL_LAUNCH_FRONTIER_CHUNKS,
            ),
            read_ready_frontier_chunks=_kv_frontier_value(
                frontier_row,
                _KV_FRONTIER_COL_READ_READY_FRONTIER_CHUNKS,
            ),
            cache_ready_frontier_chunks=_kv_frontier_value(
                frontier_row,
                _KV_FRONTIER_COL_CACHE_READY_FRONTIER_CHUNKS,
            ),
            consumable_frontier_chunks=_kv_frontier_value(
                frontier_row,
                _KV_FRONTIER_COL_CONSUMABLE_FRONTIER_CHUNKS,
            ),
            total_chunks=max(
                int(len(handle.items)),
                _kv_frontier_value(frontier_row, _KV_FRONTIER_COL_TOTAL_CHUNKS),
            ),
            error_code=_kv_frontier_value(frontier_row,
                                          _KV_FRONTIER_COL_ERROR_CODE),
        )
        handle.last_poll_snapshot = poll_snapshot
        return poll_snapshot

    def consume_chunk_pages_batch_request(
        self,
        handle: LMCacheBaMKVBatchReadRequestHandle,
        *,
        timeout_s: float | None = None,
    ) -> list[Any]:
        """消费一批已经 submit 的 KV pages read。

        如果 batch 还没 ready，这里会继续阻塞等待到底层 `IO_DONE`；这保证了：

        - 同步路径仍然可以直接复用这组 API；
        - 而 runtime/deferred 路径也可以在真正 finalize 前，先跨轮次保留同一批
          live handle。
        """
        native_result = self.kv_store.wait_native_batch(
            handle.native_handle,
            timeout_s=timeout_s,
        )
        return native_result.results

    def get_chunk_pages_batch_request_runtime_snapshot(
        self,
        handle: LMCacheBaMKVBatchReadRequestHandle,
    ) -> LMCacheBaMKVBatchReadRuntimeSnapshot:
        """读取一批 KV native read 当前对应的 runtime 观察快照。

        这层接口不主动 poll，也不推进 consume，只把底层 runtime slot/service
        状态收成上层可直接消费的稳定结构。
        """
        runtime_snapshot = self.kv_store.runtime_state_for_native_batch(
            handle.native_handle)
        return LMCacheBaMKVBatchReadRuntimeSnapshot(
            service_running=bool(runtime_snapshot.service_running),
            active_count=int(runtime_snapshot.active_count),
            request_id=int(runtime_snapshot.request_id),
            worker_backend=str(runtime_snapshot.worker_backend),
            request_table_ptr=int(runtime_snapshot.request_table_ptr),
            frontier_table_ptr=int(runtime_snapshot.frontier_table_ptr),
            completion_table_ptr=int(runtime_snapshot.completion_table_ptr),
            matched_runtime_row=runtime_snapshot.matched_runtime_row,
            runtime_rows=tuple(runtime_snapshot.runtime_rows),
        )

    def attach_chunk_pages_batch_request_runtime_direct_placement(
        self,
        handle: LMCacheBaMKVBatchReadRequestHandle,
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
        """给当前 native batch request 绑定设备侧 direct placement 描述符。"""
        attached = bool(self.kv_store.attach_native_batch_runtime_placement(
            handle.native_handle,
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
            slot_mapping_len=int(slot_mapping.numel()),
            kv_cache_ptrs_len=int(kv_cache_pointers_gpu.numel()),
        ))
        handle.runtime_direct_placement_attached = attached
        return attached

    def attach_chunk_pages_batch_request_runtime_attention_metadata(
        self,
        handle: LMCacheBaMKVBatchReadRequestHandle,
        *,
        attachment: BaMRuntimeAttentionMetadataAttachment,
    ) -> bool:
        """给当前 native batch request 绑定 attention metadata workspace。"""
        attached = bool(
            self.kv_store.attach_native_batch_runtime_attention_metadata(
                handle.native_handle,
                attachment=attachment,
            ))
        handle.runtime_attention_metadata_attached = attached
        return attached

    def finalize_chunk_pages_batch_request_runtime_direct_placement(
        self,
        handle: LMCacheBaMKVBatchReadRequestHandle,
        *,
        timeout_s: float | None = None,
    ) -> bool:
        """对 runtime-direct-placement 请求执行 cleanup-only finalize。

        这条路径不会再构造 Python `results`，只在底层已经进入 `CONSUMED` 时，
        完成 request 生命周期收尾与资源回收。
        """
        finalized = bool(self.kv_store.finalize_runtime_attached_native_batch(
            handle.native_handle,
            timeout_s=timeout_s,
        ))
        return finalized
