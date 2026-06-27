# SPDX-License-Identifier: Apache-2.0
"""LMCache BaM 的 page-level prefetch/refill 中间层。

这层的目标不是马上改 attention kernel，而是先把当前
`chunk_hash -> 同步读完整 chunk` 的路径拆成更接近 GPU-initiated 的三段：

1. planner：把 chunk metadata 转成 BaM page ids
2. prefetcher：复用 BaM rowctx submit/poll/get 读取 128KB pages
3. refiller：把 pages 还原成 LMCache KV tensor

当前第一版仍由 CPU/Python 调用这些接口，但 IO 的 submit/poll/get 已经走
BaM 现有 GPU kernel。后续可以逐步把 page request 生成、poll 和 refill
继续下沉到 GPU。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch

from vllm.bam.lmcache_bam_refill import refill_pages_to_lmcache_tensor
from vllm.logger import init_logger

logger = init_logger(__name__)

# BaM rowctx 的 SSD miss 路径目前还不够稳定。
# 如果 poll 长时间不返回，就让上层回退 LMCache 原生路径，避免真实 vLLM
# 请求无限卡住 GPU。热命中路径通常 poll_iters=0，不会受这个保护影响。
_PREFETCH_POLL_TIMEOUT_S = 10.0


class BaMPrefetchTimeoutError(TimeoutError):
    """BaM page-level prefetch 等待完成超时。"""


@dataclass(frozen=True)
class BaMPageReadPlan:
    """一次 chunk 读取对应的 BaM page 请求计划。

    `page_ids` 是 CUDA int64 tensor，内容是 BaM row/page id。
    当前 vLLM-BaM 把一个 128KB BaM page 当成一行，所以 page id 可以直接
    传给 row_store。

    这层不关心 LMCache 的 Python key，也不关心具体 KV 张量，只关心：
    - 这次要读哪些 page
    - 一共多少字节
    - page 在 GPU 侧如何组织成连续 buffer
    """

    chunk_hash: str
    page_offset: int
    page_count: int
    page_bytes: int
    total_bytes: int
    page_ids: torch.Tensor


@dataclass(frozen=True)
class BaMPrefetchStats:
    """记录 prefetch 三段式耗时，便于之后和同步 load_rows 对比。"""

    submit_ms: float
    poll_ms: float
    get_ms: float
    total_ms: float
    poll_iters: int

    @property
    def bw_gib_s(self) -> float:
        # 调用方会在日志中按实际 total_bytes 计算带宽。
        return 0.0


@dataclass(frozen=True)
class BaMPrefetchedPages:
    """一次 prefetch 完成后的结果。

    `pages` 的形状固定是 `[page_count, 128KB]`，dtype 是 `uint8`。
    也就是说，这里已经从“page id 请求”变成了“真实字节页缓冲”。
    """

    plan: BaMPageReadPlan
    pages: torch.Tensor
    stats: BaMPrefetchStats


@dataclass
class BaMPageReadHandle:
    """一次 page-level BaM 读取的提交句柄。

    这个对象是后续往更纯 GPU-initiated 演进时最重要的边界：

    - `plan.page_ids` 是 GPU 上的请求表，描述要读哪些 128KB page
    - `request` 是 BaM rowctx 返回的底层请求句柄
    - `pages` 是 GPU 上的完成缓冲，形状固定为 `[page_count, 128KB]`

    当前版本仍由 Python 调用 `submit/poll/complete`，但调用者不再直接接触
    BaM rowctx 细节。之后如果把 poll 或 request table 生成下沉到 GPU，
    主要替换这里的实现即可。
    """

    plan: BaMPageReadPlan
    request: Any
    pages: torch.Tensor
    submit_ms: float
    total_start_s: float


@dataclass
class LMCacheBaMChunkReadRequest:
    """一个 LMCache chunk 级别的 BaM 读取请求。

    这层是 `chunk metadata` 和 `BaM page IO` 之间的中间层。
    它把一次读请求拆成几个生命周期阶段：

    1. prepare：根据 metadata 生成 GPU page id 表
    2. submit：把 page id 表交给 BaM rowctx
    3. poll：检查 IO 是否完成
    4. complete：取回 `[page_count, 128KB]` 字节页
    5. refill：把字节页还原成 LMCache KV tensor

    当前请求对象仍由 CPU/Python 创建和推进；但关键数据结构已经在 GPU 上：
    - `plan.page_ids` 是 GPU tensor
    - `handle.pages` 是 GPU 完成缓冲

    后续要继续做 GPU-initiated 时，可以保持这个对象的上层语义不变，
    只替换 submit/poll/complete 的内部实现。
    """

    chunk_hash: str
    metadata: Any
    plan: BaMPageReadPlan
    handle: Optional[BaMPageReadHandle] = None
    prefetched: Optional[BaMPrefetchedPages] = None
    poll_start_s: Optional[float] = None
    poll_iters: int = 0

    @property
    def submitted(self) -> bool:
        return self.handle is not None

    @property
    def completed(self) -> bool:
        return self.prefetched is not None


@dataclass
class LMCacheBaMBatchReadRequest:
    """一批 LMCache chunk 级别的 BaM 读取请求。

    这个对象对应“批量请求表”的第一版 Python 表达：

    - `requests` 里每个元素仍是一个 chunk 的请求
    - 每个 chunk 内部有自己的 GPU `page_ids`
    - batch 层只负责按 BaM rowctx 的 FIFO 语义推进完成顺序

    当前实现有意保持保守：CPU 负责把多个 chunk request 编成 batch，
    然后一次性提交到底层 BaM rowctx；完成阶段按提交顺序 poll/complete。
    这样可以先验证“多 chunk outstanding”这件事本身是否稳定，后续再考虑
    把 batch request table 压成一个更紧凑的 GPU 结构。
    """

    requests: list[LMCacheBaMChunkReadRequest]
    total_start_s: float = 0.0
    next_complete_index: int = 0

    @property
    def submitted(self) -> bool:
        return all(request.submitted for request in self.requests)

    @property
    def completed(self) -> bool:
        return self.next_complete_index >= len(self.requests)

    @property
    def size(self) -> int:
        return len(self.requests)


class LMCacheBaMPagePlanner:
    """把 LMCache chunk metadata 转成 BaM page ids。

    这里不理解 Python key，也不关心 LMCache memory object。它只做整数映射：

    chunk metadata:
      page_offset = chunk 起始 BaM page
      pages_per_chunk = 固定页数，例如 Qwen2.5-7B fp16 下是 112

    输出:
      page_ids = [page_offset, page_offset + 1, ..., page_offset + page_count - 1]

    这意味着一个完整 chunk 读取的 page 请求，本质上就是一段连续 page id。
    """

    def __init__(self, device: str) -> None:
        self.device = torch.device(device)

    def plan_full_chunk(self, *, chunk_hash: str, page_offset: int,
                        page_count: int, page_bytes: int) -> BaMPageReadPlan:
        if page_count <= 0:
            raise ValueError(f"page_count must be positive, got {page_count}")
        if page_bytes <= 0:
            raise ValueError(f"page_bytes must be positive, got {page_bytes}")

        # page_ids 是后续 BaM rowctx 接口的输入。
        # 这里先在 CPU 逻辑里确定范围，但 tensor 本身放在 GPU 上，
        # 便于后续 BaM kernel 直接读取。
        page_ids = torch.arange(
            int(page_offset),
            int(page_offset) + int(page_count),
            device=self.device,
            dtype=torch.int64,
        )
        return BaMPageReadPlan(
            chunk_hash=chunk_hash,
            page_offset=int(page_offset),
            page_count=int(page_count),
            page_bytes=int(page_bytes),
            total_bytes=int(page_count) * int(page_bytes),
            page_ids=page_ids,
        )


class LMCacheBaMPagePrefetcher:
    """复用 BaM rowctx 三段式接口读取 128KB pages。

    当前 BaMRowStore 已经提供：

    - prefetch_rows(row_ids): submit
    - poll_prefetch(request): try poll
    - get_prefetched_rows(request, out_rows): get/copy out

    这里把它们收敛成一个清晰的 page-level API。
    默认阻塞等待读完成，但内部保留 submit/poll/get 三段耗时，
    后续可以拆成异步流水。
    """

    def __init__(self, row_store: Any, device: str) -> None:
        self.row_store = row_store
        self.device = torch.device(device)

    def submit(self, plan: BaMPageReadPlan) -> BaMPageReadHandle:
        """提交一次 page read，不等待完成。

        这是当前实现中最接近 GPU-initiated 的接口：

        1. 上层已经把 chunk metadata 转成了 GPU `page_ids`
        2. 这里把 `page_ids` 交给 BaM rowctx submit kernel
        3. 返回 handle，后续可以由调用者选择何时 poll / complete

        目前 CPU 仍负责 launch submit；真正的数据通路和请求上下文复用 BaM
        已有的 rowctx 逻辑，不重新实现底层 IO。
        """
        # 结果缓冲是 `[page_count, 128KB]` 的 uint8 CUDA tensor。
        # 这一步还没有恢复成 LMCache tensor，只是先把页字节取回来。
        pages = torch.empty(
            (plan.page_count, plan.page_bytes),
            device=self.device,
            dtype=torch.uint8,
        )

        total_start = time.perf_counter()
        submit_start = time.perf_counter()
        # submit：把 page ids 提交给 BaM rowctx。
        request = self.row_store.prefetch_rows(plan.page_ids)
        submit_ms = (time.perf_counter() - submit_start) * 1000.0
        return BaMPageReadHandle(
            plan=plan,
            request=request,
            pages=pages,
            submit_ms=submit_ms,
            total_start_s=total_start,
        )

    def poll(self, handle: BaMPageReadHandle) -> bool:
        """推进并检查一次 BaM page read 是否完成。

        当前 BaM rowctx 是 FIFO 完成队列，所以调用者应按 submit 顺序 poll/get。
        未来如果底层支持 GPU 侧队列，这个方法可以替换成读取 GPU completion
        table，而不影响 planner/refiller。
        """
        return bool(self.row_store.poll_prefetch(handle.request))

    def complete(self, handle: BaMPageReadHandle, *, poll_ms: float,
                 poll_iters: int) -> BaMPrefetchedPages:
        """取回已经完成的 pages。

        `get_prefetched_rows` 会把 BaM rowctx 中完成的 128KB rows 写入
        `handle.pages`。这一步之后，数据仍然是 page bytes，还没有恢复成
        LMCache KV tensor。
        """
        get_start = time.perf_counter()
        self.row_store.get_prefetched_rows(handle.request, handle.pages)
        get_ms = (time.perf_counter() - get_start) * 1000.0

        total_ms = (time.perf_counter() - handle.total_start_s) * 1000.0
        stats = BaMPrefetchStats(
            submit_ms=handle.submit_ms,
            poll_ms=poll_ms,
            get_ms=get_ms,
            total_ms=total_ms,
            poll_iters=int(poll_iters),
        )
        return BaMPrefetchedPages(
            plan=handle.plan,
            pages=handle.pages,
            stats=stats,
        )

    def read_pages(self, plan: BaMPageReadPlan) -> BaMPrefetchedPages:
        """阻塞读取 pages 的便捷封装。

        真实 vLLM/LMCache 接入先用这个 blocking 版本保证行为可控。
        但内部已经拆成 `submit -> poll -> complete`，所以后续可以把
        submit 提前到 attention 前、把 poll 与计算重叠，而不用改布局层。
        """
        handle = self.submit(plan)

        poll_start = time.perf_counter()
        # poll：等 page 请求完成。当前版本仍由 CPU 轮询，
        # 但实际 I/O 已由 BaM kernel / rowctx 执行。
        poll_iters = 0
        while not self.poll(handle):
            poll_iters += 1
            pass
        poll_ms = (time.perf_counter() - poll_start) * 1000.0
        return self.complete(handle, poll_ms=poll_ms, poll_iters=poll_iters)


class LMCacheBaMPageRefiller:
    """把 BaM pages 还原成 LMCache KV tensor。

    当前优先使用 Triton GPU kernel 回到 LMCache 使用的
    `[2, num_layers, actual_tokens, hidden_dim]`。

    如果 Triton 不可用或 kernel 失败，则回退到原来的 PyTorch decode，
    避免影响已有路径。

    后续如果要直接回填 vLLM paged KV cache，可以在这里新增另一个 refill
    方法，而不影响 planner/prefetcher。
    """

    def __init__(self, layout: Any) -> None:
        self.layout = layout

    def decode_to_lmcache_tensor(self, prefetched: BaMPrefetchedPages,
                                 metadata: Any) -> torch.Tensor:
        try:
            # GPU refill：CPU 只负责 launch。
            # 具体 pages -> KV tensor 的元素映射由 Triton kernel 在 GPU 上完成。
            return refill_pages_to_lmcache_tensor(prefetched.pages, metadata,
                                                  self.layout)
        except Exception:
            logger.exception(
                "[LMCACHE_BAM_PREFETCH_REFILL] GPU refill failed; "
                "fallback to PyTorch decode_pages")
            return self.layout.decode_pages(prefetched.pages, metadata)


class LMCacheBaMPagePipeline:
    """组合 planner/prefetcher/refiller 的最小中间层。

    这层给现有 BaM store 一个清晰的实验入口：

    metadata -> page_ids -> rowctx prefetch -> pages -> LMCache tensor

    其中数据形状变化可以粗略理解成：

    - 输入 metadata 对应一个 chunk
    - planner 输出 `[page_count]` 的 page id 列表
    - prefetcher 输出 `[page_count, 128KB]` 的字节页
    - refiller 输出 `[2, num_layers, actual_tokens, hidden_dim]` 的 KV tensor

    默认主路径暂时不强制使用它，避免影响已经能跑的同步 baseline。
    """

    def __init__(self, *, row_store: Any, layout: Any, device: str) -> None:
        self.planner = LMCacheBaMPagePlanner(device=device)
        self.prefetcher = LMCacheBaMPagePrefetcher(row_store=row_store,
                                                   device=device)
        self.refiller = LMCacheBaMPageRefiller(layout=layout)

    def prepare_request(self, *, chunk_hash: str,
                        metadata: Any) -> LMCacheBaMChunkReadRequest:
        """准备一次 chunk 读取请求，但不发起 IO。

        这里做的事情很少，但边界很重要：

        - 输入仍是 LMCache/BaM metadata
        - 输出变成 GPU 上的 page id 请求表

        例如 Qwen2.5-7B fp16 的一个完整 chunk：

        ```text
        metadata.page_offset = 112
        layout.pages_per_chunk = 112
        plan.page_ids = [112, 113, ..., 223]  # CUDA int64 tensor
        ```

        未来如果 vLLM 在 attention 前已经知道要用哪些 chunk，可以提前调用
        这个方法生成请求表，再稍后统一 submit。
        """
        plan = self.planner.plan_full_chunk(
            chunk_hash=chunk_hash,
            page_offset=metadata.page_offset,
            page_count=self.refiller.layout.pages_per_chunk,
            page_bytes=self.refiller.layout.page_bytes,
        )
        return LMCacheBaMChunkReadRequest(
            chunk_hash=chunk_hash,
            metadata=metadata,
            plan=plan,
        )

    def prepare_batch(
        self,
        items: Sequence[tuple[str, Any]],
    ) -> LMCacheBaMBatchReadRequest:
        """准备一批 chunk 读取请求，但不发起 IO。

        输入 `items` 是 `(chunk_hash, metadata)` 列表。输出 batch 中每个
        chunk 都已经生成了自己的 GPU page id 表：

        ```text
        batch.requests[i].plan.page_ids: [page_count] int64 CUDA tensor
        ```

        这一步仍由 CPU 根据 metadata 做粗粒度调度决策；但真正交给 BaM 的
        page 请求表已经在 GPU 上，后续 submit/poll/complete 可以逐步下沉。
        """
        requests = [
            self.prepare_request(chunk_hash=chunk_hash, metadata=metadata)
            for chunk_hash, metadata in items
        ]
        return LMCacheBaMBatchReadRequest(requests=requests)

    def submit_request(self,
                       request: LMCacheBaMChunkReadRequest
                       ) -> LMCacheBaMChunkReadRequest:
        """提交请求到 BaM rowctx，不等待完成。

        这是当前保守 GPU-initiated 路线的发起点。CPU 只负责调用 submit，
        BaM rowctx 使用 GPU 上的 `request.plan.page_ids` 作为请求表。
        """
        if request.handle is not None:
            return request
        logger.info(
            "[LMCACHE_BAM_PREFETCH_SUBMIT] chunk_hash=%s page_offset=%d "
            "page_count=%d page_bytes=%d",
            request.chunk_hash[:16],
            request.plan.page_offset,
            request.plan.page_count,
            request.plan.page_bytes,
        )
        request.handle = self.prefetcher.submit(request.plan)
        request.poll_start_s = time.perf_counter()
        logger.info(
            "[LMCACHE_BAM_PREFETCH_SUBMITTED] chunk_hash=%s submit_ms=%.3f",
            request.chunk_hash[:16],
            request.handle.submit_ms,
        )
        return request

    def submit_batch(self,
                     batch: LMCacheBaMBatchReadRequest
                     ) -> LMCacheBaMBatchReadRequest:
        """把一批 chunk 请求提交到 BaM rowctx。

        当前版本复用单 chunk 的 `submit_request()`，但调用边界已经变成 batch：

        - CPU 一次准备多个 chunk 的请求
        - BaM rowctx 中可以同时存在多个 outstanding request
        - 完成阶段必须按提交顺序推进，匹配 rowctx FIFO 语义

        这正是后续往 GPU-initiated 继续推进时需要的中间层：上层不再关心
        单个 chunk 何时 submit，只关心“一批即将被 attention 消费的 chunk”。
        """
        if not batch.requests:
            return batch
        if batch.total_start_s == 0.0:
            batch.total_start_s = time.perf_counter()
        logger.info("[LMCACHE_BAM_PREFETCH_BATCH_SUBMIT] batch_size=%d",
                    batch.size)
        for request in batch.requests:
            self.submit_request(request)
        return batch

    def poll_request(self, request: LMCacheBaMChunkReadRequest) -> bool:
        """推进并检查请求是否完成。

        当前实现是 CPU 轮询 BaM rowctx；后续可以把这里替换成 GPU completion
        table 或调度器事件，而上层仍然只看到 `poll_request()`。
        """
        if request.prefetched is not None:
            return True
        if request.handle is None:
            raise RuntimeError("BaM chunk read request must be submitted first")
        ready = self.prefetcher.poll(request.handle)
        if not ready:
            request.poll_iters += 1
        return ready

    def complete_request(self,
                         request: LMCacheBaMChunkReadRequest
                         ) -> BaMPrefetchedPages:
        """取回完成的 128KB pages。

        输入请求必须已经 submit；如果尚未 ready，底层 `complete()` 会通过
        BaM row store 的保护逻辑等待。正常路径建议先 poll 到 ready 再调用。
        """
        if request.prefetched is not None:
            return request.prefetched
        if request.handle is None:
            raise RuntimeError("BaM chunk read request must be submitted first")

        poll_start_s = request.poll_start_s
        if poll_start_s is None:
            poll_ms = 0.0
        else:
            poll_ms = (time.perf_counter() - poll_start_s) * 1000.0
        request.prefetched = self.prefetcher.complete(
            request.handle,
            poll_ms=poll_ms,
            poll_iters=request.poll_iters,
        )
        return request.prefetched

    def poll_batch(self, batch: LMCacheBaMBatchReadRequest) -> bool:
        """按 FIFO 顺序推进 batch completion。

        BaM rowctx 当前更接近 FIFO 完成队列：先提交的 request 应该先 poll/get。
        因此这里不会乱序轮询所有 chunk，而是只看 `next_complete_index` 指向的
        第一个未完成请求；它 ready 后立刻 complete，再继续尝试后续请求。

        返回值表示整个 batch 是否已经全部完成。
        """
        if batch.completed:
            return True
        if not batch.submitted:
            raise RuntimeError("BaM batch read request must be submitted first")

        while not batch.completed:
            request = batch.requests[batch.next_complete_index]
            if not self.poll_request(request):
                return False
            self.complete_request(request)
            batch.next_complete_index += 1
        return True

    def wait_batch(self,
                   batch: LMCacheBaMBatchReadRequest
                   ) -> LMCacheBaMBatchReadRequest:
        """阻塞等待整个 batch 完成。

        这仍然是 CPU orchestrated：CPU 负责等待 completion。
        但与单请求路径相比，多个 chunk 已经可以先一起 submit，给底层 BaM
        rowctx 暴露更多 outstanding work，用于验证后续 GPU-initiated 批量
        读取是否有价值。
        """
        if not batch.submitted:
            self.submit_batch(batch)
        while not self.poll_batch(batch):
            elapsed_s = 0.0
            if batch.total_start_s != 0.0:
                elapsed_s = time.perf_counter() - batch.total_start_s
            if elapsed_s > _PREFETCH_POLL_TIMEOUT_S:
                current = batch.requests[batch.next_complete_index]
                raise BaMPrefetchTimeoutError(
                    "BaM batch prefetch poll timeout: "
                    f"batch_size={batch.size} "
                    f"next_index={batch.next_complete_index} "
                    f"chunk_hash={current.chunk_hash[:16]} "
                    f"page_offset={current.plan.page_offset} "
                    f"page_count={current.plan.page_count} "
                    f"poll_iters={current.poll_iters} "
                    f"elapsed_s={elapsed_s:.3f}")
            pass
        return batch

    def wait_request(self,
                     request: LMCacheBaMChunkReadRequest
                     ) -> LMCacheBaMChunkReadRequest:
        """阻塞等待请求完成并取回 pages。

        这是为了保持当前 LMCache/replay 路径简单可控。后续真实 vLLM 路径可以
        不用这个 blocking helper，而是在计算间隙多次调用 `poll_request()`。
        """
        if request.handle is None:
            self.submit_request(request)
        while not self.poll_request(request):
            elapsed_s = 0.0
            if request.poll_start_s is not None:
                elapsed_s = time.perf_counter() - request.poll_start_s
            if elapsed_s > _PREFETCH_POLL_TIMEOUT_S:
                raise BaMPrefetchTimeoutError(
                    "BaM prefetch poll timeout: "
                    f"chunk_hash={request.chunk_hash[:16]} "
                    f"page_offset={request.plan.page_offset} "
                    f"page_count={request.plan.page_count} "
                    f"poll_iters={request.poll_iters} "
                    f"elapsed_s={elapsed_s:.3f}")
            pass
        self.complete_request(request)
        return request

    def refill_request(self,
                       request: LMCacheBaMChunkReadRequest) -> torch.Tensor:
        """把请求读回的 BaM pages 还原成 LMCache KV tensor。

        形状变化：

        ```text
        request.prefetched.pages: [page_count, 128KB] uint8 CUDA tensor
          -> Triton refill
          -> [2, num_layers, actual_tokens, hidden_dim]
        ```
        """
        if request.prefetched is None:
            raise RuntimeError("BaM chunk read request must be completed first")
        return self.refiller.decode_to_lmcache_tensor(request.prefetched,
                                                      request.metadata)

    def refill_batch(
        self,
        batch: LMCacheBaMBatchReadRequest,
    ) -> dict[str, torch.Tensor]:
        """把 batch 中已完成的 pages 逐个还原成 LMCache KV tensor。

        当前 refill 仍按 chunk 粒度调用 Triton kernel：

        ```text
        chunk pages: [112, 128KB]
          -> refill_request()
          -> KV tensor: [2, num_layers, actual_tokens, hidden_dim]
        ```

        这样代码最简单，也不会影响已验证过的单 chunk refill。后续如果要继续
        优化，可以把多个 chunk 的 pages 拼成更大的 batch refill kernel。
        """
        if not batch.completed:
            raise RuntimeError("BaM batch read request must be completed first")
        return {
            request.chunk_hash: self.refill_request(request)
            for request in batch.requests
        }

    def load_chunk_tensor(self, *, chunk_hash: str,
                          metadata: Any) -> Optional[torch.Tensor]:
        total_start = time.perf_counter()
        request = self.prepare_request(chunk_hash=chunk_hash, metadata=metadata)
        self.submit_request(request)
        self.wait_request(request)

        # refill 是独立阶段：输入 `[page_count, 128KB]` 字节页，
        # 输出 `[2, num_layers, actual_tokens, hidden_dim]` KV tensor。
        # 单独统计它，方便区分 SSD/page read 和格式回填的开销。
        refill_start = time.perf_counter()
        tensor = self.refill_request(request)
        refill_ms = (time.perf_counter() - refill_start) * 1000.0
        total_ms = (time.perf_counter() - total_start) * 1000.0

        prefetched = request.prefetched
        if prefetched is None:
            raise RuntimeError("BaM chunk read request completed without pages")

        elapsed_s = total_ms / 1000.0
        gib_per_s = (request.plan.total_bytes / elapsed_s /
                     (1024**3)) if elapsed_s > 0 else 0.0
        logger.info(
            "[LMCACHE_BAM_PREFETCH_READ] chunk_hash=%s page_offset=%d "
            "page_count=%d page_bytes=%d total_bytes=%d "
            "submit_ms=%.3f poll_ms=%.3f poll_iters=%d get_ms=%.3f "
            "prefetch_ms=%.3f refill_ms=%.3f total_ms=%.3f "
            "bw_gib_s=%.3f",
            chunk_hash[:16],
            request.plan.page_offset,
            request.plan.page_count,
            request.plan.page_bytes,
            request.plan.total_bytes,
            prefetched.stats.submit_ms,
            prefetched.stats.poll_ms,
            prefetched.stats.poll_iters,
            prefetched.stats.get_ms,
            prefetched.stats.total_ms,
            refill_ms,
            total_ms,
            gib_per_s,
        )
        return tensor

    def load_chunk_tensors_batch(
        self,
        items: Sequence[tuple[str, Any]],
    ) -> dict[str, torch.Tensor]:
        """批量读取多个 chunk 并还原为 LMCache KV tensor。

        这是 replay 阶段验证 batch prefetch 的入口。完整流程是：

        1. prepare_batch：为每个 chunk 生成 GPU page id 表
        2. submit_batch：把所有请求提交给 BaM rowctx
        3. wait_batch：按 FIFO poll/complete，得到 `[page_count, 128KB]`
        4. refill_batch：逐个 Triton refill 回 LMCache tensor

        返回值按 `chunk_hash -> tensor` 组织，方便上层 replay 做校验。
        """
        if not items:
            return {}

        total_start = time.perf_counter()
        batch = self.prepare_batch(items)
        self.submit_batch(batch)
        self.wait_batch(batch)

        refill_start = time.perf_counter()
        tensors = self.refill_batch(batch)
        refill_ms = (time.perf_counter() - refill_start) * 1000.0
        total_ms = (time.perf_counter() - total_start) * 1000.0

        total_bytes = sum(request.plan.total_bytes for request in batch.requests)
        elapsed_s = total_ms / 1000.0
        gib_per_s = (total_bytes / elapsed_s /
                     (1024**3)) if elapsed_s > 0 else 0.0
        logger.info(
            "[LMCACHE_BAM_PREFETCH_BATCH_READ] batch_size=%d total_bytes=%d "
            "completed=%d refill_ms=%.3f total_ms=%.3f bw_gib_s=%.3f",
            batch.size,
            total_bytes,
            batch.next_complete_index,
            refill_ms,
            total_ms,
            gib_per_s,
        )
        return tensors
