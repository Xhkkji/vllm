# SPDX-License-Identifier: Apache-2.0
"""BaM pages 直接放置到 vLLM paged KV cache 的第一版实现。

这层是 Direct Placement v0：

```text
BaM 128KB pages
  -> GPU refill 成 LMCache 标准 KV tensor
  -> 调 LMCache 官方 multi_layer_kv_transfer 写入 paged KV cache
```

它和现有 `lmcache_bam_refill.py` 的区别是：

- 旧路径先把完整 chunk 还原成 LMCache tensor：
  `[page_count, 128KB] -> [2, num_layers, tokens, hidden]`
- 本路径不再走 LMCache storage/retrieve 的旧控制面，但第一版仍复用
  LMCache 已验证的 GPU connector kernel 完成最后一跳。

为什么不直接调用 vLLM `reshape_and_cache`？

当前真实路径使用 XFormers/PagedAttention V0。LMCache V0 的官方 connector
把每层 `kv_cache[layer]` 当成扁平 paged buffer：

```text
[2, num_blocks * block_size, hidden_dim]
```

而 vLLM `reshape_and_cache` 会按 PagedAttention 的 key packed layout 写入。
前一版虽然能写进去，但模型输出变成乱码，说明写入格式与 LMCache/vLLM
这条真实 connector 路径不一致。因此 v0 先复用 LMCache 的
`multi_layer_kv_transfer` 保正确；后续 v1 再把
`BaM pages -> vLLM paged KV cache` 融成一个专用 CUDA/Triton kernel。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

import torch

import vllm.envs as envs
from vllm.bam.lmcache_bam_refill import refill_pages_to_lmcache_tensor_into
from vllm.logger import init_logger

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - 运行环境可能禁用 Triton
    triton = None
    tl = None

logger = init_logger(__name__)


if triton is not None:

    @triton.jit
    def _bam_pages_to_flat_paged_cache_kernel(
        pages_ptr,
        kv_cache_pointers_ptr,
        slot_mapping_ptr,
        total_elements: tl.constexpr,
        actual_tokens: tl.constexpr,
        hidden_dim: tl.constexpr,
        page_token_capacity: tl.constexpr,
        pages_per_kv_layer: tl.constexpr,
        num_layers: tl.constexpr,
        page_buffer_size: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """把一个 chunk 的某层 K 或 V 从 BaM pages 直接写到 flat paged cache。

        这里匹配 LMCache V0 `multi_layer_kv_transfer` 的 flat paged buffer 口径：

        ```text
        kv_cache[layer] 逻辑视图: [2, page_buffer_size, hidden_dim]
        slot_mapping[token]     : vLLM physical token slot
        ```

        每个 program 处理一段 `(token, hidden)` 元素。CPU 只负责 launch，
        数据寻址和 scatter 写入都在 GPU 上完成。
        """
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_elements

        hidden = offsets % hidden_dim
        token = (offsets // hidden_dim) % actual_tokens
        layer = (offsets // (hidden_dim * actual_tokens)) % num_layers
        kv_id = offsets // (hidden_dim * actual_tokens * num_layers)
        slot = tl.load(slot_mapping_ptr + token, mask=mask, other=-1)
        valid = mask & (slot >= 0)

        token_page = token // page_token_capacity
        token_in_page = token - token_page * page_token_capacity
        page_id = (kv_id * num_layers * pages_per_kv_layer +
                   layer * pages_per_kv_layer + token_page)
        page_width_elems = page_token_capacity * hidden_dim
        src_offsets = (page_id * page_width_elems +
                       token_in_page * hidden_dim + hidden)

        layer_cache_ptr = tl.load(kv_cache_pointers_ptr + layer)
        layer_cache_ptr = layer_cache_ptr.to(
            tl.pointer_type(pages_ptr.dtype.element_ty))
        dst_offsets = (layer_cache_ptr +
                       kv_id * page_buffer_size * hidden_dim +
                       slot * hidden_dim + hidden)
        values = tl.load(pages_ptr + src_offsets, mask=valid)
        tl.store(dst_offsets, values, mask=valid)


@dataclass(frozen=True)
class BaMDirectPlacementStats:
    """一次 direct placement 的统计信息。"""

    chunks: int
    tokens: int
    read_ms: float
    refill_ms: float
    transfer_ms: float
    fused_ms: float
    place_ms: float
    total_ms: float
    impl: str


@dataclass(frozen=True)
class BaMDirectPlacementChunkDescriptor:
    """描述一个可被 direct placement 消费的 prefix chunk。

    这个 descriptor 只携带“当前 chunk 对 direct placement 有意义的稳定信息”：

    - `chunk_hash`: 上层 LMCache / token database 中的 chunk 标识
    - `chunk_start/chunk_end`: 它在本轮局部 token 坐标里的覆盖范围
    - `actual_tokens`: 这个 chunk 真正有效的 token 数
    - `total_bytes`: 本次从 BaM 侧读回来的 page 总字节数

    后续如果继续推进到更细粒度的 ready / consume，这个对象可以直接作为
    “控制面 descriptor”的最小原型继续往下传，而不需要再从日志或临时变量
    里反推 chunk 边界。
    """

    chunk_hash: str
    chunk_start: int
    chunk_end: int
    actual_tokens: int
    total_bytes: int


@dataclass(frozen=True)
class BaMDirectPlacementBatchDescriptor:
    """描述一次 direct placement batch 的稳定控制面。"""

    chunks: tuple[BaMDirectPlacementChunkDescriptor, ...]
    total_tokens: int
    total_bytes: int


@dataclass(frozen=True)
class BaMDirectPlacementChunkStateSnapshot:
    """一次 direct placement 中单个 chunk 的状态快照。"""

    descriptor: BaMDirectPlacementChunkDescriptor
    read_ready: bool
    staged_ready: bool
    cache_ready: bool


@dataclass(frozen=True)
class BaMDirectPlacementBatchStateSnapshot:
    """一次 direct placement batch 的状态快照。

    这里把状态拆成三个阶段：

    - `read_ready`:
        BaM 已经把这个 chunk 对应的 pages 返回给上层。
    - `staged_ready`:
        这个 chunk 已经完成 placement 数据面处理，至少已经进入可被后续
        placement/consume 使用的 staging 形态。
        对当前实现来说：
        - `lmcache` 路径：代表 merged refill 已经完成
        - `fused` 路径：和最终 cache write 同步完成
    - `cache_ready`:
        这个 chunk 已经真正写入最终 vLLM paged KV cache，可被后续 attention
        按“已恢复 prefix”语义消费。

    当前同步版本最终仍会在函数返回前把所有 chunk 推进到 `cache_ready`。
    但先把这套状态显式化，后面做“按 chunk / 按 layer ready 后再消费”时，
    就不用再回头改 direct placement 的控制面接口。
    """

    descriptor: BaMDirectPlacementBatchDescriptor
    chunk_states: tuple[BaMDirectPlacementChunkStateSnapshot, ...]
    read_ready_chunks: int
    staged_ready_chunks: int
    cache_ready_chunks: int
    read_ready_tokens: int
    staged_ready_tokens: int
    cache_ready_tokens: int
    consumable_chunks: int
    consumable_tokens: int


class BaMDirectPlacementStateTracker:
    """跟踪一次 direct placement batch 的 ready 状态。

    这个 tracker 当前先服务两个目标：

    1. 把“哪些 chunk 已经读回 / 已经写入 cache”从隐式流程中抽出来，
       方便日志和测试直接观察；
    2. 为后续把当前同步主线推进成“部分 ready 后即可消费”的版本保留统一状态
       接口，避免到时候再从头拆控制面。

    注意：
    当前真正的数据路径仍然是同步收口的，因此这个 tracker 现在更多是“把语义
    显式化”，而不是已经实现了真正的异步消费。
    """

    def __init__(self, descriptor: BaMDirectPlacementBatchDescriptor) -> None:
        self._descriptor = descriptor
        self._read_ready = [False] * len(descriptor.chunks)
        self._staged_ready = [False] * len(descriptor.chunks)
        self._cache_ready = [False] * len(descriptor.chunks)

    @property
    def descriptor(self) -> BaMDirectPlacementBatchDescriptor:
        """返回这次 batch 的稳定 descriptor。"""
        return self._descriptor

    def mark_chunk_read_ready(self, chunk_index: int) -> None:
        """把一个 chunk 标记为“pages 已经读回”。"""
        self._read_ready[chunk_index] = True

    def mark_all_read_ready(self) -> None:
        """把整个 batch 标记为“pages 已经全部读回”。"""
        for chunk_index in range(len(self._read_ready)):
            self._read_ready[chunk_index] = True

    def mark_chunk_staged_ready(self, chunk_index: int) -> None:
        """把一个 chunk 标记为“placement staging 已完成”。"""
        self._staged_ready[chunk_index] = True

    def mark_all_staged_ready(self) -> None:
        """把整个 batch 标记为“placement staging 已完成”。"""
        for chunk_index in range(len(self._staged_ready)):
            self._staged_ready[chunk_index] = True

    def mark_chunk_cache_ready(self, chunk_index: int) -> None:
        """把一个 chunk 标记为“最终 vLLM cache 已可见”。"""
        self._cache_ready[chunk_index] = True

    def mark_all_cache_ready(self) -> None:
        """把整个 batch 标记为“最终 vLLM cache 已可见”。"""
        for chunk_index in range(len(self._cache_ready)):
            self._cache_ready[chunk_index] = True

    def snapshot(self) -> BaMDirectPlacementBatchStateSnapshot:
        """导出当前 batch 的不可变状态快照。"""
        chunk_states = tuple(
            BaMDirectPlacementChunkStateSnapshot(
                descriptor=chunk_descriptor,
                read_ready=self._read_ready[chunk_index],
                staged_ready=self._staged_ready[chunk_index],
                cache_ready=self._cache_ready[chunk_index],
            ) for chunk_index, chunk_descriptor in enumerate(
                self._descriptor.chunks))
        consumable_chunks = self.get_contiguous_cache_ready_chunk_count()
        consumable_tokens = self.get_contiguous_cache_ready_token_count()
        return BaMDirectPlacementBatchStateSnapshot(
            descriptor=self._descriptor,
            chunk_states=chunk_states,
            read_ready_chunks=sum(1 for ready in self._read_ready if ready),
            staged_ready_chunks=sum(
                1 for ready in self._staged_ready if ready),
            cache_ready_chunks=sum(1 for ready in self._cache_ready if ready),
            read_ready_tokens=self._count_ready_tokens(self._read_ready),
            staged_ready_tokens=self._count_ready_tokens(self._staged_ready),
            cache_ready_tokens=self._count_ready_tokens(self._cache_ready),
            consumable_chunks=consumable_chunks,
            consumable_tokens=consumable_tokens,
        )

    def _count_ready_tokens(self, ready_flags: Sequence[bool]) -> int:
        """按 ready 标志统计对应 token 数。"""
        return sum(
            chunk.actual_tokens for chunk, ready in zip(self._descriptor.chunks,
                                                        ready_flags)
            if ready)

    def get_contiguous_cache_ready_chunk_count(self) -> int:
        """统计从 batch 开头起，连续已经 cache-ready 的 chunk 数。

        direct placement 在 prefix 语义上只能消费“连续前缀”。因此即使后面的
        chunk 已经 ready，只要前面有一个 chunk 还没 ready，后面的 chunk 也
        不能提前暴露给上层作为可消费前缀。
        """
        ready_chunks = 0
        for ready in self._cache_ready:
            if not ready:
                break
            ready_chunks += 1
        return ready_chunks

    def get_contiguous_cache_ready_token_count(self) -> int:
        """统计当前真正可消费的连续前缀 token 数。"""
        consumable_tokens = 0
        for chunk, ready in zip(self._descriptor.chunks, self._cache_ready):
            if not ready:
                break
            consumable_tokens += chunk.actual_tokens
        return consumable_tokens


@dataclass(frozen=True)
class _BaMDirectPlacementLaunchedBatch:
    """一次已经完成 kernel launch 的 direct placement batch。

    这个对象只保存 launch 完成后的执行期信息：

    - placement plan
    - 各阶段 timing events
    - 每个 chunk 对应的完成 event
    - 本次执行的实现类型与计时起点

    之所以把它和 `StateTracker` 分开，是为了让“状态推进”和“launch 细节”
    解耦：

    - `StateTracker` 只表达 ready / consumable 语义
    - `LaunchedBatch` 只表达 GPU 上有哪些事件可被轮询

    这样后续如果底层 placement 执行形态再变化，只要还能提供类似的
    completion event 集合，就不需要改动上层 consumable frontier 逻辑。
    """

    impl: str
    plan: "_BaMDirectPlacementPlan"
    chunk_index_offset: int
    device: torch.device
    total_start_time: float
    refill_events: tuple[tuple[torch.cuda.Event, torch.cuda.Event], ...]
    transfer_events: tuple[tuple[torch.cuda.Event, torch.cuda.Event], ...]
    fused_events: tuple[tuple[torch.cuda.Event, torch.cuda.Event], ...]
    refill_step_events: tuple[
        tuple[int, int, int, torch.cuda.Event, torch.cuda.Event], ...]
    fused_step_events: tuple[
        tuple[int, int, torch.cuda.Event, torch.cuda.Event], ...]
    transfer_completion_event: torch.cuda.Event | None


class BaMDirectPlacementExecution:
    """direct placement 的执行句柄。

    这是把当前“已 launch 的 placement 工作”显式对象化后的第一步。

    当前版本的作用：

    1. 允许调用方在不改 placement 数据面的前提下，查询 ready 状态的推进；
    2. 把“等待 GPU 全部完成”和“根据 completion event 更新 frontier”拆开；
    3. 为后续真正的 chunk-ready -> chunk-consumable 消费接口保留统一入口。

    需要特别说明：

    - 当前上层 store 仍然会在同一次 direct retrieve 里调用 `wait()`，
      因此整条链路最终仍是同步收口的。
    - 但状态推进逻辑已经不再硬编码在 `place_batch()` 末尾，而是收敛到了
      这个 execution 句柄里。后续如果改成后台推进，只需要让调用方改成
      周期性 `advance_ready()`，而不需要再次拆 placement 主逻辑。
    """

    def __init__(
        self,
        *,
        launched_batch: _BaMDirectPlacementLaunchedBatch,
        state_tracker: BaMDirectPlacementStateTracker | None,
    ) -> None:
        self._launched_batch = launched_batch
        self._state_tracker = state_tracker
        self._committed_staged_chunks = 0
        self._committed_cache_chunks = 0
        self._finished = False

    @property
    def state_tracker(self) -> BaMDirectPlacementStateTracker | None:
        """返回与本次执行绑定的状态跟踪器。"""
        return self._state_tracker

    def snapshot(self) -> Optional[BaMDirectPlacementBatchStateSnapshot]:
        """返回当前 ready 状态快照。"""
        if self._state_tracker is None:
            return None
        return self._state_tracker.snapshot()

    def get_stats(self) -> BaMDirectPlacementStats:
        """返回当前执行句柄可见的 placement 统计信息。

        当前主线下，这个接口主要服务两类调用方：

        1. 已经完整等待到 wave 返回条件成立的同步收口路径
        2. 后续可能出现的“launch / wait 分离”控制面封装

        把统计信息暴露成显式接口后，上层就不需要再直接碰 `_build_stats()`
        这种内部实现细节，执行句柄的边界也会更清晰。
        """
        return self._build_stats()

    def advance_ready(self) -> Optional[BaMDirectPlacementBatchStateSnapshot]:
        """非阻塞推进当前已完成 chunk 的 ready 状态。

        这一步只做两件事：

        - 查询已经完成的 event
        - 按顺序推进 `staged_ready / cache_ready`

        它不会主动等待 GPU，因此适合后续被更高层的异步循环反复调用。
        """
        if self._launched_batch.impl == "fused":
            self._advance_fused_ready_nonblocking()
        else:
            self._advance_lmcache_ready_nonblocking()
        return self.snapshot()

    def wait(self) -> tuple[BaMDirectPlacementStats, Optional[
            BaMDirectPlacementBatchStateSnapshot]]:
        """等待当前 wave 对应的 launched range 全部 cache-ready。

        旧实现这里直接做 `torch.cuda.synchronize(device)`，会把当前设备上与本波
        无关的 CUDA 工作也一并阻塞住。现在改为：

        - 只轮询这次 launched batch 自己的 completion events
        - 只在这些 event 都 ready 后再收口统计与状态

        这样等待边界会更贴近真正的 GPU-initiated 语义，也为后续继续做
        “只等到部分 frontier ready”保留了统一接口。
        """
        return self.wait_until_launched_range_cache_ready()

    def wait_until_launched_range_cache_ready(
        self,
        *,
        timeout_s: float | None = None,
    ) -> tuple[BaMDirectPlacementStats, Optional[
            BaMDirectPlacementBatchStateSnapshot]]:
        """等待当前 wave 自己负责的 chunk 范围全部进入 cache-ready。

        这里的“当前 wave”严格指 `launched_batch.plan.entries` 里的那一段 chunk，
        不会额外等待当前设备上的其它 CUDA 工作。
        """
        if not self._finished:
            self._poll_until_ready(
                ready_predicate=self._is_launched_range_cache_ready,
                timeout_s=timeout_s,
                reason=(
                    "launched range cache-ready "
                    f"({len(self._launched_batch.plan.entries)} chunks)"
                ),
            )
            self._finish_after_ready_poll()
        return self.get_stats(), self.snapshot()

    def wait_until_contiguous_cache_ready(
        self,
        target_chunks: int,
        *,
        timeout_s: float | None = None,
    ) -> Optional[BaMDirectPlacementBatchStateSnapshot]:
        """等待连续可消费前缀推进到目标 chunk 数，供后续更细粒度收口复用。

        这个接口和 `wait_until_launched_range_cache_ready()` 的区别在于：

        - launched range：关注“本 wave 自己提交的工作是否全部做完”
        - contiguous cache-ready：关注“从 batch 开头起，当前可暴露给上层消费的
          连续前缀是否已经达到目标”

        如果本 wave 已经全部做完，但由于前面还存在空洞，导致 `target_chunks`
        仍不可达，则函数会返回当前 snapshot，而不是继续无意义死等。
        """
        target_chunks = max(int(target_chunks), 0)
        snapshot = self.advance_ready()
        if snapshot is None:
            return None
        if snapshot.consumable_chunks >= target_chunks:
            return snapshot

        self._poll_until_ready(
            ready_predicate=lambda: self._snapshot_reaches_contiguous_target(
                target_chunks),
            timeout_s=timeout_s,
            reason=f"contiguous cache-ready frontier>={target_chunks}",
            stop_when_wave_finishes=True,
        )
        return self.snapshot()

    def _advance_fused_ready_nonblocking(self) -> None:
        """推进 fused 路径下已经完成的 chunk ready 状态。

        对 fused 路径来说，一个 chunk 的一次 kernel 完成就等价于：

        - 这个 chunk 已经完成 staging
        - 这个 chunk 已经直接写入最终 vLLM paged KV cache

        因此这里一旦发现某个 step event 已完成，就可以直接把这个 chunk 推进到
        `staged_ready + cache_ready`。
        """
        while self._committed_cache_chunks < len(
                self._launched_batch.fused_step_events):
            _chunk_start, _actual_tokens, _step_start, step_end = \
                self._launched_batch.fused_step_events[
                    self._committed_cache_chunks]
            if not self._event_is_ready(step_end):
                break
            global_chunk_index = (
                self._launched_batch.chunk_index_offset +
                self._committed_cache_chunks)
            if self._state_tracker is not None:
                self._state_tracker.mark_chunk_staged_ready(global_chunk_index)
                self._state_tracker.mark_chunk_cache_ready(global_chunk_index)
            self._committed_staged_chunks += 1
            self._committed_cache_chunks += 1

    def _advance_lmcache_ready_nonblocking(self) -> None:
        """推进 lmcache 路径下已经完成的 chunk ready 状态。

        `lmcache` 路径分成两段：

        1. 每个 chunk 先进入 merged refill staging
        2. 最后通过一次统一 transfer 写入最终 cache

        因此这里的 ready 推进也分两段：

        - refill step 完成后，逐 chunk 标记 `staged_ready`
        - transfer 完成后，再把所有 chunk 统一推进为 `cache_ready`
        """
        while self._committed_staged_chunks < len(
                self._launched_batch.refill_step_events):
            _chunk_start, _token_offset, _actual_tokens, _step_start, step_end = \
                self._launched_batch.refill_step_events[
                    self._committed_staged_chunks]
            if not self._event_is_ready(step_end):
                break
            global_chunk_index = (
                self._launched_batch.chunk_index_offset +
                self._committed_staged_chunks)
            if self._state_tracker is not None:
                self._state_tracker.mark_chunk_staged_ready(global_chunk_index)
            self._committed_staged_chunks += 1

        transfer_completion_event = self._launched_batch.transfer_completion_event
        if (transfer_completion_event is not None and
                self._event_is_ready(transfer_completion_event)):
            while self._committed_cache_chunks < len(
                    self._launched_batch.plan.entries):
                global_chunk_index = (
                    self._launched_batch.chunk_index_offset +
                    self._committed_cache_chunks)
                if self._state_tracker is not None:
                    self._state_tracker.mark_chunk_cache_ready(
                        global_chunk_index)
                self._committed_cache_chunks += 1

    def _mark_all_remaining_ready(self) -> None:
        """在同步收口路径下补齐剩余 ready 状态。"""
        if self._state_tracker is None:
            return
        total_chunks = len(self._launched_batch.plan.entries)
        while self._committed_staged_chunks < total_chunks:
            global_chunk_index = (self._launched_batch.chunk_index_offset +
                                  self._committed_staged_chunks)
            self._state_tracker.mark_chunk_staged_ready(
                global_chunk_index)
            self._committed_staged_chunks += 1
        while self._committed_cache_chunks < total_chunks:
            global_chunk_index = (self._launched_batch.chunk_index_offset +
                                  self._committed_cache_chunks)
            self._state_tracker.mark_chunk_cache_ready(
                global_chunk_index)
            self._committed_cache_chunks += 1

    @staticmethod
    def _event_is_ready(event: torch.cuda.Event) -> bool:
        """检查一个 CUDA event 是否已经完成。"""
        query = getattr(event, "query", None)
        if query is None:
            # 测试桩可能没有实现 `query()`；这类场景可以保守地把它视为已完成，
            # 由单测直接验证状态推进逻辑。
            return True
        return bool(query())

    def _is_launched_range_cache_ready(self) -> bool:
        """判断本次 launched batch 负责的 chunk 是否已全部 cache-ready。"""
        return (self._committed_cache_chunks >=
                len(self._launched_batch.plan.entries))

    def _snapshot_reaches_contiguous_target(self, target_chunks: int) -> bool:
        """判断当前 snapshot 是否已经达到目标连续前缀长度。"""
        snapshot = self.snapshot()
        return snapshot is not None and snapshot.consumable_chunks >= target_chunks

    def _resolve_wait_timeout_s(self, timeout_s: float | None) -> float:
        """统一解析 event 轮询超时，避免 query 路径异常时陷入无界死循环。"""
        if timeout_s is not None:
            return max(float(timeout_s), 0.0)
        return max(float(envs.VLLM_ENGINE_ITERATION_TIMEOUT_S), 1.0)

    def _poll_until_ready(
        self,
        *,
        ready_predicate,
        timeout_s: float | None,
        reason: str,
        stop_when_wave_finishes: bool = False,
    ) -> None:
        """基于当前 batch 的 completion events 做轮询等待。

        这层 helper 的设计目标是：

        - 只观察本次 launched batch 自己的事件
        - 不做整卡 synchronize
        - 当目标已经达到，或者本 wave 已经完全结束且不可能再推进时，立刻返回
        - 如果超过超时仍无进展，则抛出明确异常，而不是无限空转
        """
        timeout_s = self._resolve_wait_timeout_s(timeout_s)
        deadline = time.perf_counter() + timeout_s
        while True:
            self.advance_ready()
            if ready_predicate():
                return
            if stop_when_wave_finishes and self._is_launched_range_cache_ready():
                return
            if time.perf_counter() >= deadline:
                raise RuntimeError(
                    "BaMDirectPlacementExecution timed out while waiting for "
                    f"{reason}; impl={self._launched_batch.impl} "
                    f"chunk_offset={self._launched_batch.chunk_index_offset} "
                    f"launched_chunks={len(self._launched_batch.plan.entries)} "
                    f"committed_staged={self._committed_staged_chunks} "
                    f"committed_cache={self._committed_cache_chunks} "
                    f"timeout_s={timeout_s:.3f}")

    def _finish_after_ready_poll(self) -> None:
        """在基于 event 的等待完成后收口状态。"""
        if self._finished:
            return
        # 真实运行时经过上面的 event 轮询，这里理论上已经自然推进到了终态；
        # 仍保留一次最小兜底，兼容测试桩或未来某条特殊实现遗漏状态推进。
        self._mark_all_remaining_ready()
        self._finished = True

    def _build_stats(self) -> BaMDirectPlacementStats:
        """在同步收口后汇总本次 placement 的耗时统计。"""
        total_ms = (time.perf_counter() -
                    self._launched_batch.total_start_time) * 1000.0
        refill_ms = BaMDirectKVPlacer._sum_cuda_event_ms(
            self._launched_batch.refill_events)
        transfer_ms = BaMDirectKVPlacer._sum_cuda_event_ms(
            self._launched_batch.transfer_events)
        fused_ms = BaMDirectKVPlacer._sum_cuda_event_ms(
            self._launched_batch.fused_events)
        return BaMDirectPlacementStats(
            chunks=len(self._launched_batch.plan.entries),
            tokens=self._launched_batch.plan.total_tokens,
            read_ms=0.0,
            refill_ms=refill_ms,
            transfer_ms=transfer_ms,
            fused_ms=fused_ms,
            place_ms=refill_ms + transfer_ms + fused_ms,
            total_ms=total_ms,
            impl=self._launched_batch.impl,
        )


@dataclass(frozen=True)
class _BaMDirectPlacementEntry:
    """一次 direct placement 中单个 chunk 的执行计划。

    这个对象只描述“当前这个 chunk 应该怎么被放进去”，不关心：

    - prefix lookup 是怎么命中的
    - BaM pages 是怎么读出来的
    - 上层 LMCache/vLLM 如何调度

    它只保留 direct placement 数据面真正需要的信息：

    - `result`: 底层 BaM batch read 返回的 pages 结果
    - `chunk_start`: 当前 chunk 在本轮 `slot_mapping` 里的局部起点
    - `actual_tokens`: 这个 chunk 真正有效的 token 数
    - `slot_mapping`: 当前 chunk 对应的 slot 映射切片

    这样后续如果继续推进到真正 GPU-visible `KVPlacementPlan`，可以直接以
    这个对象为原型，逐步把 Python 侧字段收缩成更底层的 descriptor。
    """

    result: Any
    chunk_start: int
    actual_tokens: int
    slot_mapping: torch.Tensor


@dataclass(frozen=True)
class _BaMDirectPlacementPlan:
    """一次 direct placement batch 的执行计划。"""

    entries: tuple[_BaMDirectPlacementEntry, ...]
    total_tokens: int


class BaMDirectKVPlacer:
    """把 BaM page batch 直接写入 vLLM paged KV cache。

    输入的 `results` 来自 `BaMKVStore.read_pages_batch()`，其中每个 chunk 的
    pages 形状固定为：

    ```text
    [pages_per_chunk, 128KB] uint8 CUDA
    ```

    目标 `kv_caches` 是 vLLM V0 每层一个 tensor：

    ```text
    kv_caches[layer_id]: [2, num_blocks, block_size * hidden_dim]
    ```

    LMCache 的 `multi_layer_kv_transfer` 会根据 `slot_mapping` 把连续 token
    的 K/V 写入正确 physical slot。这里 slot_mapping 的坐标系必须和
    LMCache adapter 传给原生 `engine.retrieve()` 的一致。
    """

    def __init__(self, *, layout: Any, kv_cache_dtype: str = "auto") -> None:
        self.layout = layout
        self.kv_cache_dtype = kv_cache_dtype
        self._kv_cache_pointers_cpu: torch.Tensor | None = None
        self._kv_cache_pointers_gpu: torch.Tensor | None = None
        self._kv_cache_pointer_values: tuple[int, ...] = ()
        self._page_buffer_size = 0
        self._block_size = 0
        # merged refill 的第一个真实 step 目前会承担明显的一次性 Triton/JIT
        # 初始化成本。这里记录“当前这组 shape/layout 是否已经预热过”，把这部分
        # 开销尽量前移到首次 placement 入口，而不是落到 request_2 的热路径上。
        self._merged_refill_warmup_done = False
        self._merged_refill_warmup_signature: tuple[int, ...] | None = None
        # fused 直写路径也存在同样的问题：首个真实 chunk 的第一次 Triton launch
        # 会承担编译/初始化成本。这里单独记录 fused 路径的 warmup 状态，避免把
        # 这部分一次性开销算进真实 request_2 的 placement 时间里。
        self._fused_warmup_done = False
        self._fused_warmup_signature: tuple[int, ...] | None = None

    def place_batch(
        self,
        *,
        results: Sequence[Any],
        kv_caches: Sequence[torch.Tensor],
        slot_mapping: torch.Tensor,
        chunk_starts: Sequence[int],
        state_tracker: BaMDirectPlacementStateTracker | None = None,
        launch_start_chunk: int = 0,
        max_chunks_to_launch: int | None = None,
    ) -> BaMDirectPlacementStats:
        """把多个 BaM chunk 的 pages 写入 vLLM paged KV cache。

        参数含义：

        - `results`: BaM batch read 结果，顺序必须和 `chunk_starts` 一致。
        - `kv_caches`: vLLM 每层 KV cache。
        - `slot_mapping`: 本次 retrieve token 对应的 vLLM slot 映射。
        - `chunk_starts`: 每个 chunk 在 `slot_mapping` 里的起始 token offset。

        这里不做 prefix/chunk lookup；那些属于 CPU 控制面，调用方已经通过
        LMCache token_database 完成。这里仅做数据面放置。
        """
        if len(results) != len(chunk_starts):
            raise ValueError(
                "results and chunk_starts length mismatch: "
                f"{len(results)} vs {len(chunk_starts)}")
        if len(kv_caches) != int(self.layout.num_layers):
            raise ValueError(
                "kv_caches layer count mismatch: "
                f"expected={self.layout.num_layers}, got={len(kv_caches)}")
        if not slot_mapping.is_cuda:
            raise ValueError("slot_mapping must be CUDA tensor")

        launched_batch = self.start_batch(
            results=results,
            kv_caches=kv_caches,
            slot_mapping=slot_mapping,
            chunk_starts=chunk_starts,
            launch_start_chunk=launch_start_chunk,
            max_chunks_to_launch=max_chunks_to_launch,
        )
        execution = BaMDirectPlacementExecution(
            launched_batch=launched_batch,
            state_tracker=state_tracker,
        )
        stats, _snapshot = execution.wait()
        self.log_launched_batch_step_timings(launched_batch)
        return stats

    def start_batch(
        self,
        *,
        results: Sequence[Any],
        kv_caches: Sequence[torch.Tensor],
        slot_mapping: torch.Tensor,
        chunk_starts: Sequence[int],
        launch_start_chunk: int = 0,
        max_chunks_to_launch: int | None = None,
    ) -> _BaMDirectPlacementLaunchedBatch:
        """launch 一次 direct placement，并返回可继续推进 ready 的执行期对象。

        和 `place_batch()` 的区别是：

        - `place_batch()` 会一直等到本次 placement 完成再返回统计信息
        - `start_batch()` 只负责 launch kernel，并把后续可被轮询的 event /
          plan / timing 组织成一个可继续推进的执行对象

        这正是后续把当前同步主线推进成“placement 在后台推进、上层只消费当前
        frontier”时需要的最小接口。
        """
        total_start = time.perf_counter()
        self._ensure_lmcache_connector_state(kv_caches)
        impl = envs.VLLM_BAM_DIRECT_PLACEMENT_IMPL.strip().lower()
        if impl not in ("lmcache", "fused"):
            raise ValueError(
                "VLLM_BAM_DIRECT_PLACEMENT_IMPL must be 'lmcache' or 'fused', "
                f"got {impl!r}")
        if impl == "fused" and triton is None:
            raise RuntimeError("Triton is required for fused direct placement")

        plan = self._build_plan(
            results=results,
            slot_mapping=slot_mapping,
            chunk_starts=chunk_starts,
        )
        launch_plan = self._build_launch_plan(
            plan=plan,
            launch_start_chunk=launch_start_chunk,
            max_chunks_to_launch=max_chunks_to_launch,
        )
        if not launch_plan.entries:
            return _BaMDirectPlacementLaunchedBatch(
                impl=impl,
                plan=launch_plan,
                chunk_index_offset=int(launch_start_chunk),
                device=slot_mapping.device,
                total_start_time=total_start,
                refill_events=(),
                transfer_events=(),
                fused_events=(),
                refill_step_events=(),
                fused_step_events=(),
                transfer_completion_event=None,
            )
        if impl == "fused":
            self._maybe_warmup_fused(launch_plan, kv_caches)
        else:
            self._maybe_warmup_merged_refill(launch_plan)

        # PyTorch/Triton/CUDA extension launch 都是异步的。不能用 Python
        # `time.perf_counter()` 包住函数调用来判断真实 GPU 耗时，否则看到的
        # 只是 CPU launch 开销。这里用 CUDA event 记录同一条 stream 上的阶段
        # 边界，后续由 execution 决定何时等待、何时推进 ready 状态。
        refill_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        transfer_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        fused_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        refill_step_events: list[tuple[int, int, int, torch.cuda.Event,
                                       torch.cuda.Event]] = []
        fused_step_events: list[tuple[int, int, torch.cuda.Event,
                                      torch.cuda.Event]] = []
        transfer_completion_event: torch.cuda.Event | None = None
        if impl == "fused":
            fused_start, fused_end = self._new_cuda_event_pair()
            fused_start.record()
            fused_step_events = self._fused_plan_entries_to_vllm_cache(
                launch_plan, kv_caches)
            fused_end.record()
            fused_events.append((fused_start, fused_end))
        else:
            refill_start, refill_end = self._new_cuda_event_pair()
            refill_start.record()
            kv_tensors, refill_step_events = self._refill_plan_entries(
                launch_plan)
            refill_end.record()
            refill_events.append((refill_start, refill_end))

            transfer_start, transfer_end = self._new_cuda_event_pair()
            transfer_start.record()
            self._lmcache_transfer_plan_entries(kv_tensors, launch_plan,
                                                kv_caches)
            transfer_end.record()
            transfer_events.append((transfer_start, transfer_end))
            transfer_completion_event = transfer_end

        return _BaMDirectPlacementLaunchedBatch(
            impl=impl,
            plan=launch_plan,
            chunk_index_offset=int(launch_start_chunk),
            device=slot_mapping.device,
            total_start_time=total_start,
            refill_events=tuple(refill_events),
            transfer_events=tuple(transfer_events),
            fused_events=tuple(fused_events),
            refill_step_events=tuple(refill_step_events),
            fused_step_events=tuple(fused_step_events),
            transfer_completion_event=transfer_completion_event,
        )

    def execution_from_launched_batch(
        self,
        *,
        launched_batch: _BaMDirectPlacementLaunchedBatch,
        state_tracker: BaMDirectPlacementStateTracker | None = None,
    ) -> BaMDirectPlacementExecution:
        """把一次已 launch 的 batch 封装成 execution 句柄。"""
        return BaMDirectPlacementExecution(
            launched_batch=launched_batch,
            state_tracker=state_tracker,
        )

    def log_launched_batch_step_timings(
        self,
        launched_batch: _BaMDirectPlacementLaunchedBatch,
    ) -> None:
        """在同步收口后打印每个 chunk 的 step timing。"""
        if launched_batch.impl == "fused":
            for chunk_start, actual_tokens, step_start, step_end in \
                    launched_batch.fused_step_events:
                logger.info(
                    "[LMCACHE_BAM_DIRECT_PLACEMENT_FUSED_STEP_DONE] "
                    "chunk_start=%d actual_tokens=%d step_ms=%.3f",
                    chunk_start,
                    actual_tokens,
                    float(step_start.elapsed_time(step_end)),
                )
            return

        for chunk_start, token_offset, actual_tokens, step_start, step_end in \
                launched_batch.refill_step_events:
            logger.info(
                "[LMCACHE_BAM_DIRECT_PLACEMENT_MERGED_REFILL_STEP_DONE] "
                "chunk_start=%d token_offset=%d actual_tokens=%d step_ms=%.3f",
                chunk_start,
                token_offset,
                actual_tokens,
                float(step_start.elapsed_time(step_end)),
            )

    def prepare_for_batch(
        self,
        *,
        results: Sequence[Any],
        kv_caches: Sequence[torch.Tensor],
        slot_mapping: torch.Tensor,
        chunk_starts: Sequence[int],
        launch_start_chunk: int = 0,
        max_chunks_to_launch: int | None = None,
    ) -> None:
        """在真实 direct placement 计时前预先完成一次性准备工作。

        这一步只做“首次成本前移”，不做真正的数据放置，目的有两个：

        1. 让 Triton/JIT warmup 不再记入 request_2 的 direct retrieve 热路径；
        2. 复用后续 `place_batch()` 会使用的同一份 plan / pointer 初始化前提。

        约束：

        - 这里只允许做幂等的准备动作；
        - 不允许改写真实 `kv_caches` 的语义内容；
        - 对 steady-state 再次调用应接近 no-op。
        """
        prepare_total_start = time.perf_counter()
        if len(results) != len(chunk_starts):
            raise ValueError(
                "results and chunk_starts length mismatch: "
                f"{len(results)} vs {len(chunk_starts)}")
        if not slot_mapping.is_cuda:
            raise ValueError("slot_mapping must be CUDA tensor")

        ensure_state_start = time.perf_counter()
        self._ensure_lmcache_connector_state(kv_caches)
        ensure_state_ms = (time.perf_counter() - ensure_state_start) * 1000.0
        impl = envs.VLLM_BAM_DIRECT_PLACEMENT_IMPL.strip().lower()
        if impl not in ("lmcache", "fused"):
            raise ValueError(
                "VLLM_BAM_DIRECT_PLACEMENT_IMPL must be 'lmcache' or 'fused', "
                f"got {impl!r}")
        if impl == "fused" and triton is None:
            raise RuntimeError("Triton is required for fused direct placement")

        build_plan_start = time.perf_counter()
        plan = self._build_plan(
            results=results,
            slot_mapping=slot_mapping,
            chunk_starts=chunk_starts,
        )
        launch_plan = self._build_launch_plan(
            plan=plan,
            launch_start_chunk=launch_start_chunk,
            max_chunks_to_launch=max_chunks_to_launch,
        )
        build_plan_ms = (time.perf_counter() - build_plan_start) * 1000.0
        if not launch_plan.entries:
            return

        warmup_start = time.perf_counter()
        if impl == "fused":
            warmup_executed = self._maybe_warmup_fused(launch_plan, kv_caches)
        else:
            warmup_executed = self._maybe_warmup_merged_refill(launch_plan)
        warmup_ms = (time.perf_counter() - warmup_start) * 1000.0
        prepare_total_ms = (time.perf_counter() - prepare_total_start) * 1000.0
        logger.info(
            "[LMCACHE_BAM_DIRECT_PLACEMENT_PREPARE_PROFILE] impl=%s "
            "launch_start_chunk=%d launch_chunks=%d ensure_state_ms=%.3f "
            "build_plan_ms=%.3f warmup_ms=%.3f warmup_executed=%s "
            "prepare_total_ms=%.3f",
            impl,
            int(launch_start_chunk),
            len(launch_plan.entries),
            ensure_state_ms,
            build_plan_ms,
            warmup_ms,
            str(bool(warmup_executed)).lower(),
            prepare_total_ms,
        )

    def _maybe_warmup_merged_refill(
        self,
        plan: _BaMDirectPlacementPlan,
    ) -> bool:
        """对 merged refill 路径做一次安全预热。

        当前定位结果已经表明，4 个 merged refill step 里只有第一个特别慢，
        后面几个 step 已接近亚毫秒级。这更像 Triton kernel 的首次编译/
        初始化成本，而不是 steady-state 搬运本身的问题。

        因此这里在第一次遇到某组真实 shape/layout 时，主动做一次最小预热：

        - 复用真实 `pages` dtype / device
        - 复用真实 layout
        - 复用真实 token 区间参数

        这样后面的真实计时更接近 steady-state。
        """
        if triton is None or not plan.entries:
            return False

        first_entry = plan.entries[0]
        first_pages = getattr(first_entry.result, "pages", None)
        if first_pages is None:
            # 单测桩或纯控制面验证场景里，result 可能没有真实 CUDA pages。
            # 这类场景无需做 Triton warmup，直接跳过即可。
            return False
        warmup_signature = (
            int(self.layout.num_layers),
            int(self.layout.hidden_dim),
            int(self.layout.page_token_capacity),
            int(self.layout.pages_per_kv_layer),
            int(plan.total_tokens),
            int(first_entry.actual_tokens),
            int(first_pages.device.index
                if first_pages.device.index is not None else -1),
        )
        if (self._merged_refill_warmup_done
                and self._merged_refill_warmup_signature == warmup_signature):
            return False

        logger.info(
            "[LMCACHE_BAM_DIRECT_PLACEMENT_MERGED_REFILL_WARMUP] "
            "total_tokens=%d actual_tokens=%d hidden_dim=%d num_layers=%d",
            int(plan.total_tokens),
            int(first_entry.actual_tokens),
            int(self.layout.hidden_dim),
            int(self.layout.num_layers),
        )
        warmup_out = torch.empty(
            (2, int(self.layout.num_layers), int(plan.total_tokens),
             int(self.layout.hidden_dim)),
            device=first_pages.device,
            dtype=self.layout.dtype,
        )
        refill_pages_to_lmcache_tensor_into(
            first_pages,
            out=warmup_out,
            token_offset=0,
            actual_tokens=int(first_entry.actual_tokens),
            layout=self.layout,
        )
        torch.cuda.synchronize(first_pages.device)
        self._merged_refill_warmup_done = True
        self._merged_refill_warmup_signature = warmup_signature
        return True

    def _maybe_warmup_fused(
        self,
        plan: _BaMDirectPlacementPlan,
        kv_caches: Sequence[torch.Tensor],
    ) -> bool:
        """对 fused direct placement 路径做一次安全预热。

        最新日志已经清楚表明：

        - chunk0 fused step 约 846ms
        - chunk1/2/3 fused step 约 0.1~0.2ms

        这说明当前性能回退并不来自 steady-state 的 page -> vLLM scatter，
        而是首个 Triton kernel 的一次性编译/初始化成本。

        因此这里和 merged refill 一样，在第一次真实 placement 前做一次最小化
        的安全 warmup：

        - 复用真实 pages / dtype / device
        - 复用真实 layout 参数
        - 复用真实 actual_tokens
        - 但写入一个独立的 dummy KV cache，避免污染真实 vLLM cache

        这里尤其不能把 `pages.data_ptr()` 之类“本轮临时 batch buffer 地址”编进
        warmup signature。因为 BaM batch read 往往会为每轮请求分配新的 page
        tensor；如果把这类瞬时地址纳入 signature，那么即使 shape/layout 完全
        没变，warmup 也会被误判成“每次请求都需要重新做一遍”，从而把一次性
        Triton 编译/初始化成本反复记回热路径。
        """
        if triton is None or not plan.entries:
            return False

        first_entry = plan.entries[0]
        first_pages = getattr(first_entry.result, "pages", None)
        if first_pages is None:
            # 单测桩或纯控制面验证场景里，result 可能没有真实 CUDA pages。
            # 这类场景无需做 Triton warmup，直接跳过即可。
            return False
        warmup_signature = (
            int(self.layout.num_layers),
            int(self.layout.hidden_dim),
            int(self.layout.page_token_capacity),
            int(self.layout.pages_per_kv_layer),
            int(first_entry.actual_tokens),
            int(kv_caches[0].shape[1]),
            int(kv_caches[0].shape[2]),
            int(first_pages.device.index
                if first_pages.device.index is not None else -1),
        )
        if (self._fused_warmup_done
                and self._fused_warmup_signature == warmup_signature):
            return False

        logger.info(
            "[LMCACHE_BAM_DIRECT_PLACEMENT_FUSED_WARMUP] "
            "actual_tokens=%d hidden_dim=%d num_layers=%d page_buffer_size=%d",
            int(first_entry.actual_tokens),
            int(self.layout.hidden_dim),
            int(self.layout.num_layers),
            int(self._page_buffer_size),
        )

        # 这里显式分配一个与真实 vLLM KV cache 同形状的 dummy cache。
        # warmup 的目的只是触发 Triton 编译/初始化，不需要真实数据语义。
        dummy_kv_caches = [
            torch.empty_like(kv_cache, device=kv_cache.device)
            for kv_cache in kv_caches
        ]
        dummy_pointer_values = tuple(
            int(dummy_cache.data_ptr()) for dummy_cache in dummy_kv_caches)
        dummy_kv_cache_pointers_gpu = torch.tensor(
            dummy_pointer_values,
            dtype=torch.int64,
            device=dummy_kv_caches[0].device,
        )
        dummy_slot_mapping = torch.arange(
            int(first_entry.actual_tokens),
            dtype=first_entry.slot_mapping.dtype,
            device=first_entry.slot_mapping.device,
        )

        self._launch_fused_pages_to_vllm_cache(
            first_pages,
            dummy_slot_mapping,
            dummy_kv_cache_pointers_gpu,
            actual_tokens=int(first_entry.actual_tokens),
            page_buffer_size=int(self._page_buffer_size),
        )
        torch.cuda.synchronize(first_pages.device)
        self._fused_warmup_done = True
        self._fused_warmup_signature = warmup_signature
        return True

    def _build_plan(
        self,
        *,
        results: Sequence[Any],
        slot_mapping: torch.Tensor,
        chunk_starts: Sequence[int],
    ) -> _BaMDirectPlacementPlan:
        """把输入结果整理成 direct placement 可消费的批计划。

        这一步的目标不是做任何数据搬运，而是把：

        ```text
        result + chunk_start + slot_mapping
        ```

        收敛成一个结构明确的 plan。这样后面的执行阶段就不需要再混杂：

        - 输入校验
        - slot 切片
        - 真正 GPU kernel launch

        代码会更容易读，也更贴近后续 `KVPlacementPlan` 的演进方向。
        """
        entries: list[_BaMDirectPlacementEntry] = []
        total_tokens = 0
        for result, chunk_start in zip(results, chunk_starts):
            actual_tokens = int(result.descriptor.actual_tokens)
            # Triton fused kernel 和 LMCache connector 都按连续 slot 表读取。
            # 如果上游传入的是带 stride 的视图，这里显式收紧成 contiguous。
            chunk_slots = slot_mapping[chunk_start:chunk_start +
                                       actual_tokens].contiguous()
            if chunk_slots.numel() != actual_tokens:
                raise ValueError(
                    "slot_mapping slice is shorter than chunk tokens: "
                    f"start={chunk_start} actual_tokens={actual_tokens} "
                    f"slice={chunk_slots.numel()}")
            if bool((chunk_slots < 0).any().item()):
                raise ValueError(
                    "direct placement does not support negative slot_mapping "
                    f"in a retrieved chunk yet: start={chunk_start} "
                    f"actual_tokens={actual_tokens}")
            entries.append(
                _BaMDirectPlacementEntry(
                    result=result,
                    chunk_start=int(chunk_start),
                    actual_tokens=actual_tokens,
                    slot_mapping=chunk_slots,
                ))
            total_tokens += actual_tokens
        return _BaMDirectPlacementPlan(
            entries=tuple(entries),
            total_tokens=total_tokens,
        )

    def _build_launch_plan(
        self,
        *,
        plan: _BaMDirectPlacementPlan,
        launch_start_chunk: int,
        max_chunks_to_launch: int | None,
    ) -> _BaMDirectPlacementPlan:
        """从完整 placement plan 中裁出本轮真正要 launch 的连续子计划。

        当前主线默认还是“从 plan 开头起，把这轮命中的全部连续 prefix chunks
        都推进到最终 cache”，因此 `launch_start_chunk=0` 且
        `max_chunks_to_launch=None` 或 `<=0` 时保持原行为。

        但为了继续往“两阶段前沿 / 多波次 launch”推进，这里也支持：

        - 从第 `launch_start_chunk` 个 chunk 开始
        - 再裁出至多 `max_chunks_to_launch` 个 chunk

        例如：

        ```text
        [chunk0, chunk1, chunk2, chunk3]
          + launch_start_chunk=1
          + max_chunks_to_launch=2
        -> [chunk1, chunk2]
        ```

        这样后续的 placement / ready / consumable frontier 都会只围绕这段
        子计划推进，而不会误把未 launch 的 chunk 当作本轮已经可消费的前缀。
        """
        total_chunks = len(plan.entries)
        start_chunk = max(int(launch_start_chunk), 0)
        if start_chunk >= total_chunks:
            return _BaMDirectPlacementPlan(entries=(), total_tokens=0)

        if max_chunks_to_launch is None or max_chunks_to_launch <= 0:
            end_chunk = total_chunks
        else:
            end_chunk = min(total_chunks,
                            start_chunk + int(max_chunks_to_launch))

        limited_entries = plan.entries[start_chunk:end_chunk]
        total_tokens = sum(entry.actual_tokens for entry in limited_entries)
        return _BaMDirectPlacementPlan(
            entries=tuple(limited_entries),
            total_tokens=int(total_tokens),
        )

    def _refill_plan_entries(
        self,
        plan: _BaMDirectPlacementPlan,
    ) -> tuple[torch.Tensor, list[tuple[int, int, int, torch.cuda.Event,
                                        torch.cuda.Event]]]:
        """把 plan 中所有 chunk 的 pages 还原成一个合并后的 LMCache KV tensor。

        当前 direct placement v0 仍然保留“pages -> LMCache tensor ->
        connector transfer”这条正确性优先路径，但这里不再为每个 chunk 分配
        一个独立 tensor，而是直接写到一个合并后的 batch tensor：

        ```text
        [chunk0 pages] -> merged[:, :, 0:t0, :]
        [chunk1 pages] -> merged[:, :, t0:t1, :]
        ...
        ```

        这样做的直接收益是：

        1. 去掉逐 chunk 中间 tensor 分配；
        2. 去掉逐 chunk 的额外 Python 列表组织；
        3. 后续可以只做一次 `multi_layer_kv_transfer`。

        这里额外记录每个 chunk refill step 的 CUDA 计时 event，目的是继续
        定位 merged refill 为何比旧路径更慢。这样下轮日志就能直接回答：

        - 是 4 个 step 都慢；
        - 还是只有首个 step 特别慢；
        - 又或者某个 token_offset 区间有异常。
        """
        total_tokens = int(plan.total_tokens)
        if total_tokens <= 0:
            raise ValueError("placement plan must contain at least one token")

        merged = torch.empty(
            (2, int(self.layout.num_layers), total_tokens,
             int(self.layout.hidden_dim)),
            device=plan.entries[0].result.pages.device,
            dtype=self.layout.dtype,
        )
        step_events: list[tuple[int, int, int, torch.cuda.Event,
                                torch.cuda.Event]] = []
        cursor = 0
        for entry in plan.entries:
            logger.info(
                "[LMCACHE_BAM_DIRECT_PLACEMENT_MERGED_REFILL_STEP] "
                "chunk_start=%d token_offset=%d actual_tokens=%d total_tokens=%d",
                entry.chunk_start,
                cursor,
                entry.actual_tokens,
                total_tokens,
            )
            step_start, step_end = self._new_cuda_event_pair()
            step_start.record()
            refill_pages_to_lmcache_tensor_into(
                entry.result.pages,
                out=merged,
                token_offset=cursor,
                actual_tokens=entry.actual_tokens,
                layout=self.layout,
            )
            step_end.record()
            step_events.append(
                (entry.chunk_start, cursor, entry.actual_tokens, step_start,
                 step_end))
            cursor += entry.actual_tokens
        return merged, step_events

    def _lmcache_transfer_plan_entries(
        self,
        kv_tensor: torch.Tensor,
        plan: _BaMDirectPlacementPlan,
        kv_caches: Sequence[torch.Tensor],
    ) -> None:
        """按 plan 把合并后的 KV tensor 一次性写入 vLLM paged KV cache。

        这里把所有 chunk 的局部 slot_mapping 先拼成一段连续 mapping，再只调
        一次 LMCache 官方 connector kernel：

        ```text
        merged_kv_tensor [2, layers, total_tokens, hidden]
          + merged_slot_mapping [total_tokens]
          -> multi_layer_kv_transfer(...)
        ```

        这一步虽然底层 kernel 还没变，但已经把 direct placement 的控制面从：

        ```text
        for each chunk:
          connector transfer once
        ```

        收敛成：

        ```text
        build merged plan
          -> connector transfer once
        ```

        这是向真正 `KVPlacementPlan` 演进时非常关键的一步：先把“怎么组织
        placement”的控制面收成批，再逐步替换底层 kernel。
        """
        merged_slot_mapping = torch.cat(
            [entry.slot_mapping for entry in plan.entries],
            dim=0,
        )
        if int(merged_slot_mapping.numel()) != int(plan.total_tokens):
            raise ValueError(
                "merged slot_mapping token count mismatch: "
                f"slots={merged_slot_mapping.numel()} "
                f"plan_tokens={plan.total_tokens}")
        self._lmcache_transfer_to_vllm_cache(
            kv_tensor,
            merged_slot_mapping,
            kv_caches,
        )

    @staticmethod
    def _new_cuda_event_pair() -> tuple[torch.cuda.Event, torch.cuda.Event]:
        """创建一对计时 event。

        这里不额外创建 stream，沿用当前 PyTorch stream。这样 event 顺序和
        Triton refill / LMCache connector / fused kernel 的实际执行顺序一致。
        """
        return (torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True))

    @staticmethod
    def _sum_cuda_event_ms(
        events: Sequence[tuple[torch.cuda.Event, torch.cuda.Event]],
    ) -> float:
        """汇总 CUDA event 计时。

        调用方已经做过一次 `torch.cuda.synchronize()`，因此这里读取
        elapsed_time 不会再次阻塞。分阶段计时用于定位瓶颈，不改变数据路径。
        """
        return sum(float(start.elapsed_time(end)) for start, end in events)

    def _ensure_lmcache_connector_state(
        self,
        kv_caches: Sequence[torch.Tensor],
    ) -> None:
        """缓存 LMCache connector kernel 需要的 paged KV cache 指针。

        LMCache 的 `multi_layer_kv_transfer` 接收的是每层 `kv_cache` 的
        data_ptr 表，而不是 Python tensor 列表。这里模仿
        `VLLMPagedMemGPUConnectorV2._initialize_pointers()`：

        ```text
        kv_cache_pointers[layer] = kv_caches[layer].data_ptr()
        page_buffer_size = num_blocks * block_size
        ```

        注意 `page_buffer_size` 是“可寻址 token slot 数”，不是字节数。
        """
        pointer_values = tuple(int(kv_cache.data_ptr()) for kv_cache in kv_caches)
        if (self._kv_cache_pointers_cpu is not None
                and pointer_values == self._kv_cache_pointer_values):
            return

        self._kv_cache_pointers_cpu = torch.empty(
            len(kv_caches),
            dtype=torch.int64,
            device="cpu",
            pin_memory=True,
        )
        for layer_id, pointer in enumerate(pointer_values):
            self._kv_cache_pointers_cpu[layer_id] = pointer

        self._kv_cache_pointers_gpu = torch.tensor(
            pointer_values,
            dtype=torch.int64,
            device=kv_caches[0].device,
        )

        first_cache = kv_caches[0]
        if first_cache.dim() != 3 or first_cache.shape[0] != 2:
            raise ValueError(
                "LMCache direct placement expects vLLM V0 paged cache "
                f"[2, num_blocks, block_size * hidden_dim], got "
                f"{tuple(first_cache.shape)}")
        flattened_page_width = int(first_cache.shape[2])
        hidden_dim = int(self.layout.hidden_dim)
        if flattened_page_width % hidden_dim != 0:
            raise ValueError(
                "vLLM paged cache width is not divisible by hidden_dim: "
                f"width={flattened_page_width} hidden_dim={hidden_dim}")
        block_size = flattened_page_width // hidden_dim
        self._block_size = int(block_size)
        self._page_buffer_size = int(first_cache.shape[1]) * int(block_size)
        self._kv_cache_pointer_values = pointer_values

    def _fused_plan_entries_to_vllm_cache(
        self,
        plan: _BaMDirectPlacementPlan,
        kv_caches: Sequence[torch.Tensor],
    ) -> list[tuple[int, int, torch.cuda.Event, torch.cuda.Event]]:
        """把一批 chunk 直接写入 vLLM paged KV cache。

        这里保留“按 chunk 组织计划”的 Python 层，但把 chunk 内部的数据搬运
        收缩成单次 fused kernel launch。这样上层还能清晰看到 chunk 边界，
        下层却不再为每个 K/V、每个 layer 单独 launch。

        返回每个 chunk 的 CUDA event 计时对，目的和 merged refill 一样：
        让真实日志能直接看出 fused 路径是否真的被执行，以及不同 chunk
        的放置成本是否均匀。
        """
        step_events: list[tuple[int, int, torch.cuda.Event, torch.cuda.Event]] = []
        for entry in plan.entries:
            step_start, step_end = self._new_cuda_event_pair()
            step_start.record()
            self._fused_pages_to_vllm_cache(
                entry.result.pages,
                entry.slot_mapping,
                kv_caches,
                actual_tokens=entry.actual_tokens,
            )
            step_end.record()
            step_events.append((entry.chunk_start, entry.actual_tokens,
                                step_start, step_end))
        return step_events

    def _fused_pages_to_vllm_cache(
        self,
        pages: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_caches: Sequence[torch.Tensor],
        *,
        actual_tokens: int,
    ) -> None:
        """实验版 fused direct placement。

        数据通路：

        ```text
        BaM pages [112, 128KB]
          -> Triton kernel
          -> kv_caches[layer][K/V, slot, hidden]
        ```

        这条路径刻意对齐 LMCache `multi_layer_kv_transfer` 的 flat paged cache
        口径，而不是 vLLM `reshape_and_cache` 的 packed-key 口径。也就是说，
        对当前 V0 cache：

        ```text
        kv_caches[layer]: [2, num_blocks, block_size * hidden_dim]
        逻辑视图:         [2, num_blocks * block_size, hidden_dim]
        ```

        当前版本已经把“单个 chunk 内部”收缩成一次 kernel launch：

        - 一个 launch 同时覆盖 K/V 两份数据
        - 一个 launch 同时覆盖所有 layer

        因而 fused 路径的 Python 循环只保留在“逐 chunk”这一层，便于后续继续
        往 batch-level placement plan 演进。
        """
        if triton is None:
            raise RuntimeError("Triton is required for fused direct placement")
        if pages.dtype != torch.uint8 or not pages.is_cuda:
            raise ValueError("pages must be CUDA uint8 tensor")
        expected_shape = (int(self.layout.pages_per_chunk),
                          int(self.layout.page_bytes))
        if tuple(pages.shape) != expected_shape:
            raise ValueError(
                "BaM pages shape mismatch: "
                f"expected={expected_shape}, got={tuple(pages.shape)}")

        if self._kv_cache_pointers_gpu is None:
            raise RuntimeError("kv cache pointer table is not initialized")

        self._launch_fused_pages_to_vllm_cache(
            pages,
            slot_mapping,
            self._kv_cache_pointers_gpu,
            actual_tokens=int(actual_tokens),
            page_buffer_size=int(self._page_buffer_size),
        )

    def _launch_fused_pages_to_vllm_cache(
        self,
        pages: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_pointers_gpu: torch.Tensor,
        *,
        actual_tokens: int,
        page_buffer_size: int,
    ) -> None:
        """执行一次 fused pages -> vLLM paged cache kernel launch。

        这里把“准备 launch 参数”和“真正 launch kernel”抽出来，目的有两个：

        1. 正常 fused 路径和 warmup 路径可以复用同一套 launch 逻辑；
        2. 避免 warmup 为了不污染真实 KV cache 而临时修改对象内部状态。
        """
        pages_typed = pages.view(self.layout.dtype).view(-1)
        total_elements = int(actual_tokens) * int(self.layout.hidden_dim) * 2 * int(
            self.layout.num_layers)
        block_size = 256
        grid = (triton.cdiv(total_elements, block_size), )
        _bam_pages_to_flat_paged_cache_kernel[grid](
            pages_typed,
            kv_cache_pointers_gpu,
            slot_mapping,
            total_elements,
            int(actual_tokens),
            int(self.layout.hidden_dim),
            int(self.layout.page_token_capacity),
            int(self.layout.pages_per_kv_layer),
            int(self.layout.num_layers),
            int(page_buffer_size),
            BLOCK_SIZE=block_size,
        )

    def _lmcache_transfer_to_vllm_cache(
        self,
        kv_tensor: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_caches: Sequence[torch.Tensor],
    ) -> None:
        """复用 LMCache 官方 multi-layer kernel 写入 vLLM paged KV cache。

        输入 `kv_tensor` 的形状是 `[2, layers, tokens, hidden]`，这正是
        LMCache V0 connector kernel 的标准输入。`direction=False` 表示：

        ```text
        LMCache tensor -> vLLM paged KV cache
        ```

        这里仍然需要 CPU launch kernel，但没有 CPU 参与数据搬运；KV 数据从
        BaM pages 到 vLLM paged cache 的两个阶段都在 GPU 上完成。
        """
        if self._kv_cache_pointers_cpu is None:
            raise RuntimeError("kv cache pointer table is not initialized")
        if not kv_tensor.is_cuda:
            raise ValueError("kv_tensor must be CUDA tensor")
        if tuple(kv_tensor.shape[:2]) != (2, int(self.layout.num_layers)):
            raise ValueError(
                "kv_tensor shape mismatch for LMCache transfer: "
                f"shape={tuple(kv_tensor.shape)} "
                f"num_layers={self.layout.num_layers}")

        # 延迟导入，保持 vllm-bam 在未设置 LMCache PYTHONPATH 时仍可 py_compile。
        import lmcache.c_ops as lmc_ops

        lmc_ops.multi_layer_kv_transfer(
            kv_tensor,
            self._kv_cache_pointers_cpu,
            slot_mapping.flatten(),
            kv_caches[0].device,
            self._page_buffer_size,
            False,
        )


def prepare_bam_results_for_vllm_kvcache(
    *,
    results: Sequence[Any],
    layout: Any,
    kv_caches: Sequence[torch.Tensor],
    slot_mapping: torch.Tensor,
    chunk_starts: Sequence[int],
    kv_cache_dtype: str = "auto",
    placer: BaMDirectKVPlacer | None = None,
    launch_start_chunk: int = 0,
    max_chunks_to_launch: int | None = None,
) -> BaMDirectKVPlacer:
    """在真实放置前预先完成 direct placement 的一次性准备工作。

    返回复用的 placer，便于调用方把“预热/准备”和“真正放置”绑定到同一对象上，
    从而跨请求保留 warmup 状态，避免每轮 direct retrieve 都重新触发一次性成本。
    """
    if placer is None:
        placer = BaMDirectKVPlacer(layout=layout, kv_cache_dtype=kv_cache_dtype)
    placer.prepare_for_batch(
        results=results,
        kv_caches=kv_caches,
        slot_mapping=slot_mapping,
        chunk_starts=chunk_starts,
        launch_start_chunk=launch_start_chunk,
        max_chunks_to_launch=max_chunks_to_launch,
    )
    return placer
