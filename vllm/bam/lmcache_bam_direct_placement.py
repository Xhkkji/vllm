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

import os
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

_DIRECT_FRONTIER_STATUS_SUBMITTED = 1
_DIRECT_FRONTIER_STATUS_READ_READY = 2
_DIRECT_FRONTIER_STATUS_CACHE_READY = 3
_DIRECT_FRONTIER_STATUS_CONSUMABLE = 4
_DIRECT_FRONTIER_COL_STATUS = 0
_DIRECT_FRONTIER_COL_LAUNCH = 1
_DIRECT_FRONTIER_COL_READ_READY = 2
_DIRECT_FRONTIER_COL_CACHE_READY = 3
_DIRECT_FRONTIER_COL_CONSUMABLE = 4
_DIRECT_FRONTIER_COL_TOTAL = 5
_DIRECT_FRONTIER_COL_ERROR = 6
_DIRECT_FRONTIER_COL_COUNT = 7


if triton is not None:

    @triton.jit
    def _bam_pages_to_vllm_paged_cache_kernel(
        pages_ptr,
        kv_cache_pointers_ptr,
        slot_mapping_ptr,
        total_elements: tl.constexpr,
        actual_tokens: tl.constexpr,
        hidden_dim: tl.constexpr,
        num_kv_heads: tl.constexpr,
        head_size: tl.constexpr,
        block_size: tl.constexpr,
        pack_size: tl.constexpr,
        page_token_capacity: tl.constexpr,
        pages_per_kv_layer: tl.constexpr,
        num_layers: tl.constexpr,
        page_buffer_size: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """把一个 chunk 的 K/V page 直接写到 vLLM/LMCache 官方 paged KV ABI。

        关键点：

        - 输入 `pages` 仍然是 BaM 自己的连续 row/page 组织
        - 输出必须严格匹配 LMCache `multi_layer_kv_transfer(direction=false)`
          对 vLLM V0 底层 paged buffer 的物理写入 ABI：
          `[2, page_buffer_size, hidden_dim]`
        - `slot_mapping[token]` 给出 token 最终应该落到哪个 physical slot

        这样 direct placement 和当前已验证正确的 materialized 路径使用同一套
        写入语义。读侧如果需要 packed key/value view，由 vLLM/xformers 在同一
        块底层内存上解释；写端不再自己猜 packed key 的 offset。
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
        block_idx = slot // block_size
        block_offset = slot % block_size
        plane_elements = page_buffer_size * hidden_dim
        dst_index = kv_id * plane_elements + slot * hidden_dim + hidden
        dst_offsets = layer_cache_ptr + dst_index
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


@dataclass(frozen=True)
class BaMDirectPlacementFrontierSnapshot:
    """placement 执行器对外暴露的 request-level frontier 快照。

    这层对象的定位，是把当前 placement 阶段真正需要暴露给上层控制面的信息
    收敛成统一接口，而不是让 storage/runtime 继续直接依赖 tracker 内部细节。

    当前字段分两组：

    - `frontier_row`
      一个紧凑、稳定的 request-level frontier ABI：
      `[status, launch, read_ready, cache_ready, consumable, total, error]`
    - `*_frontier_*`
      便于上层直接读写的具名字段，避免到处手工解码列号

    注意这里的 `cache_ready/consumable` 已经是 placement 语义，而不是底层
    BaM native read request 的保留观测列；这正是它和 native KV read frontier
    的语义边界。
    """

    frontier_row: tuple[int, ...]
    launch_frontier_chunks: int
    read_ready_frontier_chunks: int
    staged_ready_frontier_chunks: int
    cache_ready_frontier_chunks: int
    consumable_frontier_chunks: int
    total_chunks: int
    read_ready_frontier_tokens: int
    staged_ready_frontier_tokens: int
    cache_ready_frontier_tokens: int
    consumable_frontier_tokens: int
    error_code: int = 0

    @property
    def launch_chunks(self) -> int:
        """兼容旧调用方的 launch 前缀 chunk 字段名。"""
        return int(self.launch_frontier_chunks)

    @property
    def read_ready_chunks(self) -> int:
        """兼容旧调用方的 read-ready 前缀 chunk 字段名。"""
        return int(self.read_ready_frontier_chunks)

    @property
    def staged_ready_chunks(self) -> int:
        """兼容旧调用方的 staged-ready 前缀 chunk 字段名。"""
        return int(self.staged_ready_frontier_chunks)

    @property
    def cache_ready_chunks(self) -> int:
        """兼容旧调用方的 cache-ready 前缀 chunk 字段名。"""
        return int(self.cache_ready_frontier_chunks)

    @property
    def consumable_chunks(self) -> int:
        """兼容旧调用方的 consumable 前缀 chunk 字段名。"""
        return int(self.consumable_frontier_chunks)

    @property
    def read_ready_tokens(self) -> int:
        """兼容旧调用方的 read-ready token 字段名。"""
        return int(self.read_ready_frontier_tokens)

    @property
    def staged_ready_tokens(self) -> int:
        """兼容旧调用方的 staged-ready token 字段名。"""
        return int(self.staged_ready_frontier_tokens)

    @property
    def cache_ready_tokens(self) -> int:
        """兼容旧调用方的 cache-ready token 字段名。"""
        return int(self.cache_ready_frontier_tokens)

    @property
    def consumable_tokens(self) -> int:
        """兼容旧调用方的 consumable token 字段名。"""
        return int(self.consumable_frontier_tokens)


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

    def mark_chunks_read_ready_upto(self, ready_chunks: int) -> None:
        """把从 batch 开头起的连续若干 chunk 标记为 read-ready。

        这层 helper 的语义刻意绑定到“连续前缀 frontier”，而不是“任意若干个
        chunk”。原因是当前 direct placement 与上层 prefix 语义都只关心：

        - 从 batch 开头起
        - 连续多少个 chunk
        - 已经可以安全继续推进

        后续无论这个 ready frontier 来自：

        - BaM native KV read 的 request-level frontier table
        - direct placement 自己的 GPU 事件推进
        - 更进一步的 GPU-resident runtime state machine

        都可以先收敛到这条统一入口，避免上层到处散落手写 `for` 循环。
        """
        self._mark_ready_prefix(self._read_ready, ready_chunks)

    def mark_all_read_ready(self) -> None:
        """把整个 batch 标记为“pages 已经全部读回”。"""
        self.mark_chunks_read_ready_upto(len(self._read_ready))

    def mark_chunk_staged_ready(self, chunk_index: int) -> None:
        """把一个 chunk 标记为“placement staging 已完成”。"""
        self._staged_ready[chunk_index] = True

    def mark_chunks_staged_ready_upto(self, ready_chunks: int) -> None:
        """把从 batch 开头起的连续若干 chunk 标记为 staged-ready。"""
        self._mark_ready_prefix(self._staged_ready, ready_chunks)

    def mark_all_staged_ready(self) -> None:
        """把整个 batch 标记为“placement staging 已完成”。"""
        self.mark_chunks_staged_ready_upto(len(self._staged_ready))

    def mark_chunk_cache_ready(self, chunk_index: int) -> None:
        """把一个 chunk 标记为“最终 vLLM cache 已可见”。"""
        self._cache_ready[chunk_index] = True

    def mark_chunks_cache_ready_upto(self, ready_chunks: int) -> None:
        """把从 batch 开头起的连续若干 chunk 标记为 cache-ready。"""
        self._mark_ready_prefix(self._cache_ready, ready_chunks)

    def mark_all_cache_ready(self) -> None:
        """把整个 batch 标记为“最终 vLLM cache 已可见”。"""
        self.mark_chunks_cache_ready_upto(len(self._cache_ready))

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

    @staticmethod
    def _mark_ready_prefix(
        ready_flags: list[bool],
        ready_chunks: int,
    ) -> None:
        """把一个 ready flag 数组的连续前缀推进到指定 chunk 数。

        注意这里是“只增不减”的单调推进语义：

        - 已经 ready 的前缀不会被回退
        - 超出数组长度的目标会自动截断

        这样 tracker 就能安全复用到底层 frontier poll、placement event 推进、
        以及同步 finalize 收口这三类不同来源的状态更新中。
        """
        final_ready_chunks = min(max(int(ready_chunks), 0), len(ready_flags))
        for chunk_index in range(final_ready_chunks):
            ready_flags[chunk_index] = True

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
        frontier_table: torch.Tensor | None = None,
    ) -> None:
        self._launched_batch = launched_batch
        self._state_tracker = state_tracker
        self._committed_staged_chunks = 0
        self._committed_cache_chunks = 0
        self._finished = False
        # 第三步的轻量化推进版本里，先把 placement frontier 收敛成：
        #
        # 1. 一张 GPU-visible frontier table
        # 2. 一份 host 侧 mirror
        #
        # 这样做的目标不是立刻把整个 control plane 下沉成 persistent kernel，
        # 而是先把“统一 frontier ABI”稳定下来：
        #
        # - 当前 host 侧可以继续用 mirror 低成本读状态
        # - 后续 GPU-resident frontier / persistent service 可以直接复用这张表
        #
        # frontier row ABI:
        #   [status, launch, read_ready, cache_ready, consumable, total, error]
        self._frontier_table = self._resolve_frontier_table(frontier_table)
        self._frontier_snapshot_mirror: Optional[
            BaMDirectPlacementFrontierSnapshot] = None
        self._publish_frontier_snapshot_from_tracker()

    @property
    def state_tracker(self) -> BaMDirectPlacementStateTracker | None:
        """返回与本次执行绑定的状态跟踪器。"""
        return self._state_tracker

    def snapshot(self) -> Optional[BaMDirectPlacementBatchStateSnapshot]:
        """返回当前 ready 状态快照。"""
        if self._state_tracker is None:
            return None
        return self._state_tracker.snapshot()

    def frontier_snapshot(self) -> Optional[BaMDirectPlacementFrontierSnapshot]:
        """返回当前 placement request-level frontier 快照。

        这层接口是当前“placement 控制面收敛”的关键边界：

        - 上层不再需要直接理解 tracker 里的 read/staged/cache 三组 flag
        - 后续如果 frontier 真正改成 GPU-resident table，也可以只替换这里的
          具体来源，而不必重改 storage/runtime 调用方
        """
        table_snapshot = self._build_frontier_snapshot_from_table()
        if table_snapshot is not None:
            return table_snapshot
        if self._frontier_snapshot_mirror is not None:
            return self._frontier_snapshot_mirror
        self._publish_frontier_snapshot_from_tracker()
        return self._build_frontier_snapshot_from_table(
        ) or self._frontier_snapshot_mirror

    def frontier_table(self) -> Optional[torch.Tensor]:
        """返回当前 execution 持有的 frontier table。

        返回值语义：

        - CUDA 可用且当前 execution 绑定到 CUDA device 时：
          返回 GPU-visible table，后续可被更底层 runtime 直接复用。
        - 单测或无 CUDA 环境：
          返回 CPU fallback table，用来保持 ABI 与状态推进逻辑一致。

        当前显式对齐到底层 KV native frontier 的 ABI 形状：

        ```text
        frontier_table: [7] int64
        ```

        这样可以让“控制面 ABI”先稳定下来，而不把测试环境强绑到真 CUDA 上。
        """
        return self._frontier_table

    def frontier_row_host(self) -> Optional[tuple[int, ...]]:
        """返回 host mirror 中当前可见的 frontier row。"""
        frontier_snapshot = self.frontier_snapshot()
        if frontier_snapshot is None:
            return None
        return frontier_snapshot.frontier_row

    def poll_frontier(self) -> Optional[BaMDirectPlacementFrontierSnapshot]:
        """非阻塞推进一次 ready 状态，并返回统一 frontier 快照。"""
        self.advance_ready()
        return self.frontier_snapshot()

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
        snapshot = self.snapshot()
        self._publish_frontier_snapshot(snapshot)
        return snapshot

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

    def wait_until_contiguous_cache_ready_frontier(
        self,
        target_chunks: int,
        *,
        timeout_s: float | None = None,
    ) -> Optional[BaMDirectPlacementFrontierSnapshot]:
        """等待到目标连续前缀后，直接返回统一 frontier 快照。

        这个 helper 的意义不是增加新语义，而是给 storage/runtime 提供一个更薄、
        更稳定的等待边界：

        - 旧接口返回 batch state snapshot，调用方仍需知道 tracker 细节
        - 新接口直接返回 frontier snapshot，调用方可以只面向 request-level ABI
        """
        batch_snapshot = self.wait_until_contiguous_cache_ready(
            target_chunks,
            timeout_s=timeout_s,
        )
        if batch_snapshot is None:
            return None
        return self._build_frontier_snapshot(batch_snapshot)

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

    def _frontier_value_from_table(self, col: int) -> Optional[int]:
        """读取共享 frontier table 中某一列的当前值。

        这层 helper 的目标很简单：让 execution 内部的等待判断也开始围绕 shared
        frontier table 工作，而不是继续优先读取 tracker / Python 局部变量。

        这样后续如果 frontier 更新权继续往 GPU runtime 下放，wait/poll 这层
        条件判断逻辑也不需要再改一次。
        """
        frontier_row = self._read_frontier_row_from_table()
        if len(frontier_row) <= int(col):
            return None
        return int(frontier_row[int(col)])

    def _is_launched_range_cache_ready(self) -> bool:
        """判断本次 launched batch 负责的 chunk 是否已全部 cache-ready。"""
        # 只有当前 wave 本身从 batch 开头 launch 时，`cache_ready_frontier_chunks`
        # 这列才能直接表达“本 wave 负责的 launched range 是否都 ready”。
        #
        # 对非零 offset 的 wave 来说，这一列仍然是“从 batch 开头起的连续前缀”
        # 语义，例如：
        #   - wave 从 chunk1 开始 launch 1 个 chunk
        #   - chunk1 已 ready，但 chunk0 还没 ready
        #   - 此时 cache_ready_frontier_chunks 仍可能是 0 或 1
        #
        # 它并不能直接等价为“offset=1 这波已经完成”。所以这里刻意只在
        # `chunk_index_offset == 0` 时优先信 shared table；非零 offset 的局部波次
        # 仍回退到 execution 自己维护的 launched-range committed 计数。
        if int(self._launched_batch.chunk_index_offset) == 0:
            table_cache_ready_chunks = self._frontier_value_from_table(
                _DIRECT_FRONTIER_COL_CACHE_READY)
            if table_cache_ready_chunks is not None:
                launch_end_chunk = len(self._launched_batch.plan.entries)
                return int(table_cache_ready_chunks) >= int(launch_end_chunk)
        return (self._committed_cache_chunks >=
                len(self._launched_batch.plan.entries))

    def _snapshot_reaches_contiguous_target(self, target_chunks: int) -> bool:
        """判断当前 snapshot 是否已经达到目标连续前缀长度。"""
        table_consumable_chunks = self._frontier_value_from_table(
            _DIRECT_FRONTIER_COL_CONSUMABLE)
        if table_consumable_chunks is not None:
            return int(table_consumable_chunks) >= int(target_chunks)
        snapshot = self.snapshot()
        return snapshot is not None and snapshot.consumable_chunks >= target_chunks

    def _resolve_wait_timeout_s(self, timeout_s: float | None) -> float:
        """统一解析 event 轮询超时，避免 query 路径异常时陷入无界死循环。"""
        if timeout_s is not None:
            return max(float(timeout_s), 0.0)
        return max(float(envs.VLLM_ENGINE_ITERATION_TIMEOUT_S), 1.0)

    def _build_frontier_snapshot(
        self,
        snapshot: BaMDirectPlacementBatchStateSnapshot,
    ) -> BaMDirectPlacementFrontierSnapshot:
        """把 batch-level ready 状态整理成统一的 request-level frontier 快照。"""
        launch_frontier_chunks = min(
            len(snapshot.chunk_states),
            int(self._launched_batch.chunk_index_offset) +
            len(self._launched_batch.plan.entries),
        )
        status = _DIRECT_FRONTIER_STATUS_SUBMITTED
        if snapshot.read_ready_chunks > 0:
            status = _DIRECT_FRONTIER_STATUS_READ_READY
        if snapshot.cache_ready_chunks > 0:
            status = _DIRECT_FRONTIER_STATUS_CACHE_READY
        if snapshot.consumable_chunks > 0:
            status = _DIRECT_FRONTIER_STATUS_CONSUMABLE
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
        self._publish_frontier_snapshot_from_tracker()
        self._finished = True

    def _allocate_frontier_table(self) -> Optional[torch.Tensor]:
        """为当前 execution 分配一张轻量的 frontier table。

        这里故意只分配 `[7] int64` 一维表，而不是现在就引入更复杂的多行
        request table / page table，原因是第三步当前只需要先把 placement
        request 的统一 frontier ABI 固定下来。

        同时这里有一个非常明确的兼容目标：尽量和底层 native KV read 已经在用的
        frontier table 口径保持一致。这样后续如果把 placement frontier 真正并入
        GPU runtime，就不需要再做一轮 dtype / shape 迁移。

        单测里通常没有真实 CUDA，因此这里允许自动回退到 CPU tensor；
        真正运行在 GPU 环境时，会优先把这张表放到 execution 绑定的 device 上。
        """
        device = self._launched_batch.device
        tensor_device = torch.device("cpu")
        if device.type == "cuda" and torch.cuda.is_available():
            tensor_device = device
        return torch.zeros(
            (_DIRECT_FRONTIER_COL_COUNT,),
            dtype=torch.int64,
            device=tensor_device,
        )

    def _resolve_frontier_table(
        self,
        frontier_table: torch.Tensor | None,
    ) -> Optional[torch.Tensor]:
        """解析 execution 当前应使用的 frontier table。

        当前优先级非常简单：

        - 调用方显式传入时，直接复用这张共享 request frontier table
        - 否则 execution 自己按旧逻辑分配一张独立 table

        这样 request handle 就可以把同一张 GPU-visible frontier table 贯穿：

        ```text
        native read frontier
          -> placement execution frontier
          -> finalize 后的稳定 request frontier
        ```

        后续如果 persistent GPU runtime 需要直接接手更新 frontier，也只需要接管
        这一张共享表，而不需要再跨阶段搬运 ABI。
        """
        if frontier_table is None:
            return self._allocate_frontier_table()
        if frontier_table.ndim != 1 or int(
                frontier_table.shape[0]) < _DIRECT_FRONTIER_COL_COUNT:
            raise ValueError(
                "frontier_table must have shape [>=7], "
                f"got {tuple(frontier_table.shape)}")
        if frontier_table.dtype != torch.int64:
            raise TypeError(
                "frontier_table.dtype must be torch.int64, "
                f"got {frontier_table.dtype}")
        if not frontier_table.is_contiguous():
            raise ValueError("frontier_table must be contiguous")
        return frontier_table

    def _read_frontier_row_from_table(self) -> tuple[int, ...]:
        """读取当前共享 frontier table 的 host 快照。"""
        if self._frontier_table is None:
            return ()
        return tuple(
            int(value) for value in self._frontier_table.detach().cpu().tolist())

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

    def _build_frontier_snapshot_from_table(
        self,
    ) -> Optional[BaMDirectPlacementFrontierSnapshot]:
        """优先基于共享 frontier table 重建 execution 可见的 frontier 快照。

        当前 execution 仍保留 host mirror，但 getter 这里开始把共享 table 当成
        主事实来源。这样后续如果 frontier 更新权进一步下放给 GPU runtime，
        execution 侧读取逻辑不需要再改。

        staged 信息当前还不在 7 列 ABI 中，因此仍从 tracker snapshot 补齐。
        """
        tracker_snapshot = self.snapshot()
        if tracker_snapshot is None:
            return self._frontier_snapshot_mirror
        frontier_row = self._read_frontier_row_from_table()
        if not frontier_row:
            return self._frontier_snapshot_mirror
        descriptor = tracker_snapshot.descriptor
        status = int(frontier_row[_DIRECT_FRONTIER_COL_STATUS])
        launch_frontier_chunks = max(
            int(frontier_row[_DIRECT_FRONTIER_COL_LAUNCH]), 0)
        read_ready_frontier_chunks = max(
            int(frontier_row[_DIRECT_FRONTIER_COL_READ_READY]), 0)
        cache_ready_frontier_chunks = max(
            int(frontier_row[_DIRECT_FRONTIER_COL_CACHE_READY]), 0)
        consumable_frontier_chunks = max(
            int(frontier_row[_DIRECT_FRONTIER_COL_CONSUMABLE]), 0)
        total_chunks = max(
            int(frontier_row[_DIRECT_FRONTIER_COL_TOTAL]),
            len(tracker_snapshot.chunk_states),
        )
        error_code = int(frontier_row[_DIRECT_FRONTIER_COL_ERROR])
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
            staged_ready_frontier_chunks=int(
                tracker_snapshot.staged_ready_chunks),
            cache_ready_frontier_chunks=int(cache_ready_frontier_chunks),
            consumable_frontier_chunks=int(consumable_frontier_chunks),
            total_chunks=int(total_chunks),
            read_ready_frontier_tokens=self._count_descriptor_tokens_upto(
                descriptor, read_ready_frontier_chunks),
            staged_ready_frontier_tokens=int(
                tracker_snapshot.staged_ready_tokens),
            cache_ready_frontier_tokens=self._count_descriptor_tokens_upto(
                descriptor, cache_ready_frontier_chunks),
            consumable_frontier_tokens=self._count_descriptor_tokens_upto(
                descriptor, consumable_frontier_chunks),
            error_code=int(error_code),
        )

    def _publish_frontier_snapshot_from_tracker(self) -> None:
        """从 tracker 重建 frontier snapshot，并同步到 table/mirror。"""
        snapshot = self.snapshot()
        self._publish_frontier_snapshot(snapshot)

    def _publish_frontier_snapshot(
        self,
        snapshot: Optional[BaMDirectPlacementBatchStateSnapshot],
    ) -> None:
        """把最新 frontier 状态同步到 host mirror 与 GPU-visible table。

        这一层是第三步里最关键的“轻量桥接”：

        - host 侧继续通过 mirror 做低成本控制面判断
        - GPU 侧已经能看到同一份 row ABI

        这样后续如果继续做 persistent service / GPU-resident frontier，只需要让
        更底层 runtime 改写这张表，而不用重新设计上层 storage 接口。
        """
        if snapshot is None:
            self._frontier_snapshot_mirror = None
            if self._frontier_table is not None:
                self._frontier_table.zero_()
            return
        self._write_frontier_table_from_snapshot(snapshot)
        # table 现在已经是 request/execution getter 的主事实来源，因此这里把
        # mirror 也尽量改成“由 table 反解出来的快照”，而不是继续把 mirror 当成
        # 主来源。这样后续如果 frontier 更新权继续往 GPU runtime 下放，这里的
        # 读取语义就不需要再次调整。
        self._frontier_snapshot_mirror = (
            self._build_frontier_snapshot_from_table()
            or self._build_frontier_snapshot(snapshot)
        )

    def _write_frontier_table_from_snapshot(
        self,
        snapshot: BaMDirectPlacementBatchStateSnapshot,
    ) -> None:
        """把 snapshot 中真正属于统一 frontier ABI 的列原位写回共享表。

        这里刻意改成“按列原位更新”，而不是每次重新构造一个完整 tensor 再 copy，
        原因有两个：

        1. 当前 getter 已经把 shared frontier table 当成主事实来源；
        2. 后续如果继续往 persistent service / GPU runtime 推进，最自然的形态
           就是持续原位更新这几列，而不是每次重建整行对象。

        staged 信息当前仍然不在 7 列 ABI 中，因此不会写进这张表。
        """
        if self._frontier_table is None:
            return
        launch_frontier_chunks = min(
            len(snapshot.chunk_states),
            int(self._launched_batch.chunk_index_offset) +
            len(self._launched_batch.plan.entries),
        )
        status = _DIRECT_FRONTIER_STATUS_SUBMITTED
        if snapshot.read_ready_chunks > 0:
            status = _DIRECT_FRONTIER_STATUS_READ_READY
        if snapshot.cache_ready_chunks > 0:
            status = _DIRECT_FRONTIER_STATUS_CACHE_READY
        if snapshot.consumable_chunks > 0:
            status = _DIRECT_FRONTIER_STATUS_CONSUMABLE
        # shared frontier table 现在已经越来越接近“由更底层 runtime 持续维护”的
        # 目标形态。因此 host 侧这里回写时必须遵守一个很关键的约束：
        #
        # - 只能单调前进
        # - 不能把 table 已有的更前状态回退掉
        #
        # 否则一旦未来真的出现“GPU runtime 比 host 先更新这张表”的情况，
        # host 再跑一遍旧 snapshot 回写，就会把 frontier 误回退。
        self._frontier_table[_DIRECT_FRONTIER_COL_STATUS] = max(
            int(self._frontier_table[_DIRECT_FRONTIER_COL_STATUS]),
            int(status),
        )
        self._frontier_table[_DIRECT_FRONTIER_COL_LAUNCH] = max(
            int(self._frontier_table[_DIRECT_FRONTIER_COL_LAUNCH]),
            int(launch_frontier_chunks),
        )
        self._frontier_table[_DIRECT_FRONTIER_COL_READ_READY] = max(
            int(self._frontier_table[_DIRECT_FRONTIER_COL_READ_READY]),
            int(snapshot.read_ready_chunks),
        )
        self._frontier_table[_DIRECT_FRONTIER_COL_CACHE_READY] = max(
            int(self._frontier_table[_DIRECT_FRONTIER_COL_CACHE_READY]),
            int(snapshot.cache_ready_chunks),
        )
        self._frontier_table[_DIRECT_FRONTIER_COL_CONSUMABLE] = max(
            int(self._frontier_table[_DIRECT_FRONTIER_COL_CONSUMABLE]),
            int(snapshot.consumable_chunks),
        )
        self._frontier_table[_DIRECT_FRONTIER_COL_TOTAL] = max(
            int(self._frontier_table[_DIRECT_FRONTIER_COL_TOTAL]),
            int(len(snapshot.chunk_states)),
        )
        self._frontier_table[_DIRECT_FRONTIER_COL_ERROR] = max(
            int(self._frontier_table[_DIRECT_FRONTIER_COL_ERROR]),
            0,
        )

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


@dataclass(frozen=True)
class BaMRuntimeDirectPlacementAttachment:
    """给底层 GPU runtime 使用的最小 direct placement 描述符。

    这层对象刻意不包含 `pages` 本身，因为 `pages` 的来源仍由 BaM KV native
    request/runtime 管理。它只描述“当 pages 已经 ready 后，最终该写到哪里”：

    - `slot_mapping`: 当前 request 的 flat slot mapping
    - `chunk_starts`: 每个 chunk 在 slot_mapping 中的起始 token 偏移
    - `kv_cache_pointers_gpu`: 每层 paged KV cache 的 data_ptr 表
    - 若干布局参数：帮助设备侧按和 fused Triton 路径一致的方式做 scatter
    """

    slot_mapping: torch.Tensor
    chunk_starts: torch.Tensor
    kv_cache_pointers_gpu: torch.Tensor
    page_buffer_size: int
    block_size: int
    page_token_capacity: int
    pages_per_kv_layer: int
    num_layers: int
    num_kv_heads: int
    head_size: int
    pack_size: int
    slot_mapping_len: int
    kv_cache_ptrs_len: int


@dataclass(frozen=True)
class BaMRuntimeAttentionMetadataAttachment:
    """给现有 persistent service CTA 使用的 attention metadata workspace。

    这层 attachment 的目标非常收敛：

    - 不再让前台 adapter 在 request finalize 后临时 `torch.cat/pad/rebuild`
      大量 GPU metadata；
    - 改成在 request start 时一次性预分配好本条 sequence 需要的 workspace；
    - 等 BaM persistent service 完成：
        `BaM cache -> vLLM paged KV cache`
      之后，顺手把当前 sequence 对应的 attention metadata 也原地填好；
    - 前台只做“薄封装”和必要的只读校验，不再承担 metadata rebuild 主工作。

    当前这层只覆盖“单条 sequence request”语义，因为 direct retrieve request
    本身就是按 sequence 单独启动的。后续如果要做 batch 级 metadata 收口，
    也可以继续复用这份 per-seq ABI。
    """

    # `full_query_slot_mapping_src` 是“当前 sequence 在本轮原始 model_input 里，
    # 从当前 sequence 起点开始到本轮 total_seq_len 结束”的完整 slot-mapping
    # 切片。
    #
    # 这里刻意保留 full-seq 口径，而不是只传当前 suffix/query 的那一段，原因是
    # 后端 persistent service 需要同时处理两类不同语义：
    #
    # 1. `rebuilt_slot_mapping_dst`
    #    这是 query/suffix 级别的控制面。它会根据最终补回了多少 prefix token，
    #    从 `full_query_slot_mapping_src` 中截取一个后缀写回。
    #
    # 2. `rebuilt_block_table_dst`
    #    这是 full-seq 级别的控制面。即便当前真正要计算的 query 只是一个后缀，
    #    block table 仍然必须描述“到当前 total_seq_len 为止”的整条 sequence
    #    paged-cache block 布局。
    #
    # 因此这份 source tensor 既服务 suffix slot_mapping rebuild，也服务 full-seq
    # block_table rebuild；两者不能混成同一个坐标系。
    full_query_slot_mapping_src: torch.Tensor
    rebuilt_slot_mapping_dst: torch.Tensor

    # `original_block_table_src` 若非空，表示原始 attn metadata 已经给出了可直接
    # 复用的 row；若为空，则 persistent service 按 `full_query_slot_mapping_src`
    # 和 `block_size` 在 GPU 上保守恢复 block_table。
    original_block_table_src: torch.Tensor
    rebuilt_block_table_dst: torch.Tensor

    # 下列 tensor 都是“单条 sequence”的最终 metadata workspace：
    # - `context_lens_dst[0]`      : 本轮 attention 看到的 context_len
    # - `seq_lens_dst[0]`          : 本轮 attention 看到的总 seq_len
    # - `query_start_loc_dst[:2]`  : 固定写成 [0, q_len]
    # - `selected_token_indices_dst`
    # - `metadata_ready_flag`
    #
    # 其中 `metadata_ready_flag` 现在只保留为一个轻量调试/观测标记。
    #
    # 当前 authoritative 的“这条 request 是否已经可以直接消费 runtime
    # metadata”语义，已经收口到 storage/finalize 发布的 request-level
    # completion 主线上；前台不再依赖这里的 `.item()` 结果做硬门槛分叉。
    context_lens_dst: torch.Tensor
    seq_lens_dst: torch.Tensor
    query_start_loc_dst: torch.Tensor
    selected_token_indices_dst: torch.Tensor
    metadata_ready_flag: torch.Tensor

    total_seq_len: int
    vllm_num_computed_tokens: int
    vllm_num_computed_tokens_align: int
    block_size: int
    is_chunk_prefill: bool
    do_sample: bool


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
        self._configured_num_kv_heads = 0
        self._configured_head_size = 0
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
        num_kv_heads: int = 0,
        head_size: int = 0,
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
        if int(num_kv_heads) > 0:
            self._configured_num_kv_heads = int(num_kv_heads)
        if int(head_size) > 0:
            self._configured_head_size = int(head_size)

        launched_batch = self.start_batch(
            results=results,
            kv_caches=kv_caches,
            slot_mapping=slot_mapping,
            chunk_starts=chunk_starts,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
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

    def _resolve_effective_impl(
        self,
        *,
        phase: str,
    ) -> str:
        """解析当前 direct placement 应实际使用的数据面实现。

        当前保留两套 host materialized finalize 数据面：

        - `fused`
          直接把 BaM pages scatter 到最终 paged KV cache
        - `lmcache`
          先 refill 成 merged KV tensor，再走一跳 LMCache connector transfer

        结合最近几轮日志，已经可以比较确定：

        - persistent/runtime 模式下，控制面 prepare 已经跑通
        - 但真正进入 fused launch 时仍会长时间卡住
        - 而当前阶段最重要的目标是先把
          “GPU 后台 poll/read + host materialized finalize”
          这条主线稳定跑通

        因此这里在 persistent/runtime 模式下，显式把 host materialized finalize
        的实现收敛到 `lmcache`。这不是长期终点，而是当前主线下最小、最清晰的
        收敛策略：

        - 不改 request-level 控制面
        - 不改上层调用链
        - 只把最不稳定的 fused 数据面替换成已验证更稳的一跳实现
        """
        impl = envs.VLLM_BAM_DIRECT_PLACEMENT_IMPL.strip().lower()
        if impl not in ("lmcache", "fused"):
            raise ValueError(
                "VLLM_BAM_DIRECT_PLACEMENT_IMPL must be 'lmcache' or 'fused', "
                f"got {impl!r}")
        if impl == "fused" and triton is None:
            raise RuntimeError("Triton is required for fused direct placement")
        if impl == "fused" and self._should_skip_prepare_warmup():
            logger.info(
                "[LMCACHE_BAM_DIRECT_PLACEMENT_IMPL_DEMOTE] "
                "phase=%s configured_impl=fused effective_impl=lmcache "
                "reason=persistent_gpu_service_enabled",
                phase,
            )
            return "lmcache"
        return impl

    def start_batch(
        self,
        *,
        results: Sequence[Any],
        kv_caches: Sequence[torch.Tensor],
        slot_mapping: torch.Tensor,
        chunk_starts: Sequence[int],
        num_kv_heads: int = 0,
        head_size: int = 0,
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
        if int(num_kv_heads) > 0:
            self._configured_num_kv_heads = int(num_kv_heads)
        if int(head_size) > 0:
            self._configured_head_size = int(head_size)
        impl = self._resolve_effective_impl(phase="start")

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
        if self._should_skip_prepare_warmup():
            # `prepare_for_batch()` 已经在 persistent/runtime 模式下显式跳过 warmup。
            # 这里必须复用同一判定，避免：
            #
            # 1. prepare 阶段不 warmup；
            # 2. 但真正 launch wave 时，`start_batch()` 又偷偷跑一遍 fused/merged
            #    warmup；
            #
            # 否则日志上就会表现成：
            #
            # - `FINALIZE_PREPARE_DONE` 已经出现
            # - `FINALIZE_WAVE_BEGIN` 之后又卡在 `FUSED_WARMUP`
            #   或 `MERGED_REFILL_WARMUP`
            #
            # 这正是最新一轮日志已经证明的停点。
            logger.info(
                "[LMCACHE_BAM_DIRECT_PLACEMENT_START_WARMUP_SKIP] "
                "reason=persistent_gpu_service_enabled impl=%s",
                impl,
            )
        elif impl == "fused":
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
        frontier_table: torch.Tensor | None = None,
    ) -> BaMDirectPlacementExecution:
        """把一次已 launch 的 batch 封装成 execution 句柄。"""
        return BaMDirectPlacementExecution(
            launched_batch=launched_batch,
            state_tracker=state_tracker,
            frontier_table=frontier_table,
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
        num_kv_heads: int = 0,
        head_size: int = 0,
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
        if int(num_kv_heads) > 0:
            self._configured_num_kv_heads = int(num_kv_heads)
        if int(head_size) > 0:
            self._configured_head_size = int(head_size)
        ensure_state_ms = (time.perf_counter() - ensure_state_start) * 1000.0
        impl = self._resolve_effective_impl(phase="prepare")

        build_plan_start = time.perf_counter()
        logger.info(
            "[LMCACHE_BAM_DIRECT_PLACEMENT_PREPARE_BUILD_PLAN_BEGIN] "
            "results=%d launch_start_chunk=%d",
            len(results),
            int(launch_start_chunk),
        )
        plan = self._build_plan(
            results=results,
            slot_mapping=slot_mapping,
            chunk_starts=chunk_starts,
        )
        logger.info(
            "[LMCACHE_BAM_DIRECT_PLACEMENT_PREPARE_BUILD_PLAN_DONE] "
            "entries=%d total_tokens=%d",
            len(plan.entries),
            int(plan.total_tokens),
        )
        launch_plan = self._build_launch_plan(
            plan=plan,
            launch_start_chunk=launch_start_chunk,
            max_chunks_to_launch=max_chunks_to_launch,
        )
        logger.info(
            "[LMCACHE_BAM_DIRECT_PLACEMENT_PREPARE_BUILD_LAUNCH_PLAN_DONE] "
            "launch_entries=%d launch_tokens=%d",
            len(launch_plan.entries),
            int(launch_plan.total_tokens),
        )
        build_plan_ms = (time.perf_counter() - build_plan_start) * 1000.0
        if not launch_plan.entries:
            return

        warmup_start = time.perf_counter()
        if self._should_skip_prepare_warmup():
            # 这里显式跳过 warmup，不是为了“偷懒”，而是因为当前主线下
            # GPU worker persistent service 可能已经常驻运行：
            #
            # 1. warmup 本身只用于首轮 Triton/JIT 编译前移；
            # 2. 但一旦它落到默认 stream，再配合同步，就很容易与后台
            #    persistent service 形成隐式串行或设备级等待；
            # 3. 对 materialized finalize 来说，这类一次性优化绝不应该影响
            #    功能正确性，因此在 persistent 模式下直接关掉最稳妥。
            #
            # steady-state 性能优化后续仍可以通过“专用 warmup stream”再做，
            # 但当前先保证主线跑通，避免 request 卡死在 prepare 阶段。
            warmup_executed = False
            logger.info(
                "[LMCACHE_BAM_DIRECT_PLACEMENT_PREPARE_WARMUP_SKIP] "
                "reason=persistent_gpu_service_enabled impl=%s",
                impl,
            )
        elif impl == "fused":
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

    def ensure_kv_cache_pointer_state(
        self,
        *,
        kv_caches: Sequence[torch.Tensor],
    ) -> None:
        """提前准备 fused/materialized finalize 需要的 KV cache 指针状态。

        这层 helper 故意只做一件事：确保 `_ensure_lmcache_connector_state()`
        已完成。

        这么拆出来，是为了把“第一次 GPU 元数据初始化”的时机从：

        - finalize/prepare 阶段

        前移到：

        - direct placement read submit 之前

        当前 persistent service 主线下，这个时序差异非常关键。因为一旦 native
        read request 已经挂进后台 service，再在 finalize 里首次构造：

        - `kv_cache_pointers_gpu`
        - `page_buffer_size`
        - block-size 相关派生状态

        就可能与仍处在活跃状态的后台 service 发生资源争用，表现成
        `FINALIZE_PREPARE_BEGIN` 后长时间无后续日志。

        提前初始化后，finalize 只复用稳定状态，不再在 request 活跃期首次做
        这类 GPU metadata 准备。
        """
        self._ensure_lmcache_connector_state(kv_caches)

    @staticmethod
    def _should_skip_prepare_warmup() -> bool:
        """判断当前这轮 prepare 是否应彻底跳过 warmup。

        当前直接使用环境变量做判定，原因有两个：

        1. 这层 direct placement 组件本身不持有 BaM runtime / kv_store 句柄，
           不应该为了问“后台 service 在不在跑”而反向依赖更底层对象；
        2. 这次要解决的是“persistent 模式下 materialized finalize 卡住”，
           而 persistent/runtime 开关本来就是由脚本和环境变量统一控制的。

        只要启用了 GPU worker runtime 或 persistent service，就把 prepare
        warmup 视作不安全操作并直接跳过。这样不改变真正的数据搬运逻辑，只是
        去掉了首轮编译前移这一步，能最大限度降低与后台常驻 kernel 互相等待的
        风险。
        """

        def _env_enabled(name: str) -> bool:
            value = os.getenv(name)
            if value is None:
                return False
            return value.strip().lower() not in ("", "0", "false", "off", "no")

        return (_env_enabled("GIDS_KV_GPU_WORKER_RUNTIME_ENABLE")
                or _env_enabled("GIDS_KV_GPU_WORKER_PERSISTENT_ENABLE"))

    def _resolve_vllm_paged_kv_layout(
        self,
        *,
        kv_caches: Sequence[torch.Tensor],
        num_kv_heads: int = 0,
        head_size: int = 0,
    ) -> tuple[int, int, int]:
        """解析 direct placement 写最终 paged KV cache 必需的布局参数。

        这里显式收敛三件事：

        - `hidden_dim == num_kv_heads * head_size`
        - `block_size` 已由 `_ensure_lmcache_connector_state()` 推导完成
        - key packed layout 里的 `pack_size = 16 / element_size`

        之前 runtime/host 写端缺的正是这组参数，因此只能错误地退回 flat
        token-row 线性写法。现在统一从这里拿，避免两边再各自猜布局。
        """
        hidden_dim = int(self.layout.hidden_dim)
        num_kv_heads = int(num_kv_heads)
        head_size = int(head_size)
        if num_kv_heads <= 0:
            num_kv_heads = int(self._configured_num_kv_heads)
        if head_size <= 0:
            head_size = int(self._configured_head_size)
        if num_kv_heads <= 0 or head_size <= 0:
            # host 侧 fused 单测/实验路径允许退化成“把整条 hidden 向量视作 1 个 head”。
            # 这样不会影响 runtime 主线的严格校验，但能保持旧控制面调用接口兼容。
            num_kv_heads = 1
            head_size = hidden_dim
        if num_kv_heads <= 0 or head_size <= 0:
            raise ValueError(
                "direct placement requires positive num_kv_heads/head_size, "
                f"got num_kv_heads={num_kv_heads} head_size={head_size}")
        if hidden_dim != num_kv_heads * head_size:
            raise ValueError(
                "direct placement hidden_dim mismatch: "
                f"hidden_dim={hidden_dim} num_kv_heads={num_kv_heads} "
                f"head_size={head_size}")
        if self._block_size <= 0:
            # 一些轻量单测会直接调用 fused warmup/helper，而不会先经过
            # `start_batch()/prepare_for_batch()` 的完整初始化流程。
            # 这里允许按当前 `kv_caches` 的 shape 惰性补齐 block_size，避免把
            # “尚未初始化 pointer state”误判成布局错误。
            self._ensure_lmcache_connector_state(kv_caches)
        if self._block_size <= 0:
            raise ValueError(
                "direct placement block_size has not been initialized")
        pack_size = 16 // int(kv_caches[0].element_size())
        if pack_size <= 0 or head_size % pack_size != 0:
            raise ValueError(
                "direct placement cannot derive valid packed key layout: "
                f"head_size={head_size} pack_size={pack_size}")
        return num_kv_heads, head_size, int(pack_size)

    def build_runtime_direct_placement_attachment(
        self,
        *,
        kv_caches: Sequence[torch.Tensor],
        slot_mapping: torch.Tensor,
        chunk_starts: Sequence[int],
        num_kv_heads: int,
        head_size: int,
    ) -> BaMRuntimeDirectPlacementAttachment:
        """构造给底层 persistent runtime 使用的设备侧 placement 描述符。

        这一步不做任何 pages 搬运，也不 launch placement kernel；它只把 fused
        direct placement 真正需要的 GPU-visible 元信息收成一份稳定结构，供
        BaM persistent service 在 `pages` ready 后继续推进到最终 paged KV cache。
        """
        if not slot_mapping.is_cuda:
            raise ValueError("slot_mapping must be CUDA tensor")
        self._ensure_lmcache_connector_state(kv_caches)
        if self._kv_cache_pointers_gpu is None:
            raise RuntimeError("kv cache pointer table is not initialized")
        (num_kv_heads, head_size,
         pack_size) = self._resolve_vllm_paged_kv_layout(
             kv_caches=kv_caches,
             num_kv_heads=num_kv_heads,
             head_size=head_size,
         )
        return BaMRuntimeDirectPlacementAttachment(
            slot_mapping=slot_mapping.contiguous(),
            chunk_starts=torch.tensor(
                list(int(value) for value in chunk_starts),
                dtype=torch.int64,
                device=slot_mapping.device,
            ),
            kv_cache_pointers_gpu=self._kv_cache_pointers_gpu,
            page_buffer_size=int(self._page_buffer_size),
            block_size=int(self._block_size),
            page_token_capacity=int(self.layout.page_token_capacity),
            pages_per_kv_layer=int(self.layout.pages_per_kv_layer),
            num_layers=int(self.layout.num_layers),
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            pack_size=int(pack_size),
            slot_mapping_len=int(slot_mapping.numel()),
            kv_cache_ptrs_len=int(len(kv_caches)),
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
        self._synchronize_current_cuda_stream(first_pages.device)
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
        (num_kv_heads, head_size,
         pack_size) = self._resolve_vllm_paged_kv_layout(
             kv_caches=kv_caches,
         )

        self._launch_fused_pages_to_vllm_cache(
            first_pages,
            dummy_slot_mapping,
            dummy_kv_cache_pointers_gpu,
            actual_tokens=int(first_entry.actual_tokens),
            page_buffer_size=int(self._page_buffer_size),
            num_kv_heads=int(num_kv_heads),
            head_size=int(head_size),
            block_size=int(self._block_size),
            pack_size=int(pack_size),
        )
        self._synchronize_current_cuda_stream(first_pages.device)
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
            # 这里不要在 CUDA 热路径里做 `.any().item()` 这类 host 同步检查。
            #
            # 当前 persistent service 会常驻在同一张卡上持续推进 native read；
            # 如果 finalize/prepare 阶段再在 `slot_mapping` 上做设备到主机的同步
            # 读取，就可能把前台线程卡在这一步。
            #
            # `slot_mapping` 的非负性本身属于上层 vLLM/LMCache 已经保证的契约，
            # 因此这里把强校验收窄到：
            #
            # - CPU / 单测桩场景：仍保留检查，便于尽早发现非法输入
            # - 真实 CUDA 热路径：相信上层契约，避免无意义同步
            #
            # 这样既保住了主线性能/可运行性，也没有把无效输入默默吞掉到所有场景。
            if (not chunk_slots.is_cuda
                    and bool((chunk_slots < 0).any().item())):
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

    @staticmethod
    def _synchronize_current_cuda_stream(device: torch.device) -> None:
        """只同步当前 launch stream，避免把后台 persistent service 也等住。

        这里不能再使用 `torch.cuda.synchronize(device)` 这类 device-wide 同步。
        原因是当前 KV 主线下，GPU worker persistent service 可能长期常驻在同一张
        卡上的另一条 stream；如果 warmup 这里等待整张卡空闲，就会把：

        - 当前 prepare/warmup 自己刚 launch 的 kernel
        - 与之无关、但本来就设计成持续运行的后台 service kernel

        一起纳入等待范围，最终表现成 materialized/fused finalize 在
        `prepare_for_batch()` 里“看起来像卡住了”。

        warmup 真正需要的语义其实更窄：

        - 只保证当前默认 stream 上这次 launch 的 kernel 已经完成

        因此这里显式收窄到 current stream，同步边界更贴近真实需要。
        """
        torch.cuda.current_stream(device=device).synchronize()

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
        (num_kv_heads, head_size,
         pack_size) = self._resolve_vllm_paged_kv_layout(
             kv_caches=kv_caches,
         )
        for entry in plan.entries:
            step_start, step_end = self._new_cuda_event_pair()
            step_start.record()
            self._fused_pages_to_vllm_cache(
                entry.result.pages,
                entry.slot_mapping,
                kv_caches,
                actual_tokens=entry.actual_tokens,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                pack_size=pack_size,
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
        num_kv_heads: int,
        head_size: int,
        pack_size: int,
    ) -> None:
        """实验版 fused direct placement。

        数据通路：

        ```text
        BaM pages [pages_per_chunk, 128KB]
          -> Triton kernel
          -> vLLM paged KV cache
             key  : [num_blocks, num_kv_heads, head_size/pack, block_size, pack]
             value: [num_blocks, num_kv_heads, head_size, block_size]
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
            num_kv_heads=int(num_kv_heads),
            head_size=int(head_size),
            block_size=int(self._block_size),
            pack_size=int(pack_size),
        )

    def _launch_fused_pages_to_vllm_cache(
        self,
        pages: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_pointers_gpu: torch.Tensor,
        *,
        actual_tokens: int,
        page_buffer_size: int,
        num_kv_heads: int,
        head_size: int,
        block_size: int,
        pack_size: int,
    ) -> None:
        """执行一次 fused pages -> vLLM paged cache kernel launch。

        这里把“准备 launch 参数”和“真正 launch kernel”抽出来，目的有两个：

        1. 正常 fused 路径和 warmup 路径可以复用同一套 launch 逻辑；
        2. 避免 warmup 为了不污染真实 KV cache 而临时修改对象内部状态。
        """
        pages_typed = pages.view(self.layout.dtype).view(-1)
        total_elements = int(actual_tokens) * int(self.layout.hidden_dim) * 2 * int(
            self.layout.num_layers)
        launch_block_size = 256
        grid = (triton.cdiv(total_elements, launch_block_size), )
        _bam_pages_to_vllm_paged_cache_kernel[grid](
            pages_typed,
            kv_cache_pointers_gpu,
            slot_mapping,
            total_elements,
            int(actual_tokens),
            int(self.layout.hidden_dim),
            int(num_kv_heads),
            int(head_size),
            int(block_size),
            int(pack_size),
            int(self.layout.page_token_capacity),
            int(self.layout.pages_per_kv_layer),
            int(self.layout.num_layers),
            int(page_buffer_size),
            BLOCK_SIZE=launch_block_size,
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
