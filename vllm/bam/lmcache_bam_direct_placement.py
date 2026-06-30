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
        kv_cache_ptr,
        slot_mapping_ptr,
        total_elements: tl.constexpr,
        kv_id: tl.constexpr,
        layer_id: tl.constexpr,
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
        token = offsets // hidden_dim
        slot = tl.load(slot_mapping_ptr + token, mask=mask, other=-1)
        valid = mask & (slot >= 0)

        token_page = token // page_token_capacity
        token_in_page = token - token_page * page_token_capacity
        page_id = (kv_id * num_layers * pages_per_kv_layer +
                   layer_id * pages_per_kv_layer + token_page)
        page_width_elems = page_token_capacity * hidden_dim
        src_offsets = page_id * page_width_elems + token_in_page * hidden_dim + hidden

        dst_offsets = (kv_id * page_buffer_size * hidden_dim +
                       slot * hidden_dim + hidden)
        values = tl.load(pages_ptr + src_offsets, mask=valid)
        tl.store(kv_cache_ptr + dst_offsets, values, mask=valid)


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
        self._kv_cache_pointers: torch.Tensor | None = None
        self._kv_cache_pointer_values: tuple[int, ...] = ()
        self._page_buffer_size = 0
        self._block_size = 0
        # merged refill 的第一个真实 step 目前会承担明显的一次性 Triton/JIT
        # 初始化成本。这里记录“当前这组 shape/layout 是否已经预热过”，把这部分
        # 开销尽量前移到首次 placement 入口，而不是落到 request_2 的热路径上。
        self._merged_refill_warmup_done = False
        self._merged_refill_warmup_signature: tuple[int, ...] | None = None

    def place_batch(
        self,
        *,
        results: Sequence[Any],
        kv_caches: Sequence[torch.Tensor],
        slot_mapping: torch.Tensor,
        chunk_starts: Sequence[int],
        num_kv_heads: int,
        head_size: int,
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

        total_start = time.perf_counter()
        placed_tokens = 0
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
        if not plan.entries:
            return BaMDirectPlacementStats(
                chunks=0,
                tokens=0,
                read_ms=0.0,
                refill_ms=0.0,
                transfer_ms=0.0,
                fused_ms=0.0,
                place_ms=0.0,
                total_ms=0.0,
                impl=impl,
            )
        if impl != "fused":
            self._maybe_warmup_merged_refill(plan)

        # PyTorch/Triton/CUDA extension launch 都是异步的。不能用 Python
        # `time.perf_counter()` 包住函数调用来判断真实 GPU 耗时，否则看到的
        # 只是 CPU launch 开销。这里用 CUDA event 记录同一条 stream 上的阶段
        # 边界，最后统一 synchronize 后再读取 elapsed_time。
        refill_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        transfer_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        fused_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        if impl == "fused":
            for entry in plan.entries:
                fused_start, fused_end = self._new_cuda_event_pair()
                fused_start.record()
                self._fused_pages_to_vllm_cache(
                    entry.result.pages,
                    entry.slot_mapping,
                    kv_caches,
                    actual_tokens=entry.actual_tokens,
                )
                fused_end.record()
                fused_events.append((fused_start, fused_end))
        else:
            refill_step_events: list[tuple[int, int, int, torch.cuda.Event,
                                           torch.cuda.Event]] = []
            refill_start, refill_end = self._new_cuda_event_pair()
            refill_start.record()
            kv_tensors, refill_step_events = self._refill_plan_entries(plan)
            refill_end.record()
            refill_events.append((refill_start, refill_end))

            transfer_start, transfer_end = self._new_cuda_event_pair()
            transfer_start.record()
            self._lmcache_transfer_plan_entries(kv_tensors, plan, kv_caches)
            transfer_end.record()
            transfer_events.append((transfer_start, transfer_end))
        placed_tokens = plan.total_tokens

        # 这里同步只用于计时口径，避免把 placement kernel 延迟记到后续 attention。
        # 后续做 layer-wise pipeline 时可以把同步移到更外层，和计算 overlap。
        if placed_tokens > 0:
            torch.cuda.synchronize(slot_mapping.device)
        total_ms = (time.perf_counter() - total_start) * 1000.0
        refill_ms = self._sum_cuda_event_ms(refill_events)
        transfer_ms = self._sum_cuda_event_ms(transfer_events)
        fused_ms = self._sum_cuda_event_ms(fused_events)
        if impl != "fused":
            for chunk_start, token_offset, actual_tokens, step_start, step_end in refill_step_events:
                logger.info(
                    "[LMCACHE_BAM_DIRECT_PLACEMENT_MERGED_REFILL_STEP_DONE] "
                    "chunk_start=%d token_offset=%d actual_tokens=%d step_ms=%.3f",
                    chunk_start,
                    token_offset,
                    actual_tokens,
                    float(step_start.elapsed_time(step_end)),
                )
        place_ms = refill_ms + transfer_ms + fused_ms
        return BaMDirectPlacementStats(
            chunks=len(results),
            tokens=placed_tokens,
            read_ms=0.0,
            refill_ms=refill_ms,
            transfer_ms=transfer_ms,
            fused_ms=fused_ms,
            place_ms=place_ms,
            total_ms=total_ms,
            impl=impl,
        )

    def _maybe_warmup_merged_refill(
        self,
        plan: _BaMDirectPlacementPlan,
    ) -> None:
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
            return

        first_entry = plan.entries[0]
        warmup_signature = (
            int(self.layout.num_layers),
            int(self.layout.hidden_dim),
            int(self.layout.page_token_capacity),
            int(self.layout.pages_per_kv_layer),
            int(plan.total_tokens),
            int(first_entry.actual_tokens),
            int(first_entry.result.pages.data_ptr()),
            int(first_entry.result.pages.device.index
                if first_entry.result.pages.device.index is not None else -1),
        )
        if (self._merged_refill_warmup_done
                and self._merged_refill_warmup_signature == warmup_signature):
            return

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
            device=first_entry.result.pages.device,
            dtype=self.layout.dtype,
        )
        refill_pages_to_lmcache_tensor_into(
            first_entry.result.pages,
            out=warmup_out,
            token_offset=0,
            actual_tokens=int(first_entry.actual_tokens),
            layout=self.layout,
        )
        torch.cuda.synchronize(first_entry.result.pages.device)
        self._merged_refill_warmup_done = True
        self._merged_refill_warmup_signature = warmup_signature

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
        if (self._kv_cache_pointers is not None
                and pointer_values == self._kv_cache_pointer_values):
            return

        self._kv_cache_pointers = torch.empty(
            len(kv_caches),
            dtype=torch.int64,
            device="cpu",
            pin_memory=True,
        )
        for layer_id, pointer in enumerate(pointer_values):
            self._kv_cache_pointers[layer_id] = pointer

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

        第一版每层 K/V 各 launch 一个 kernel，逻辑简单、便于验证；后续可把
        layer/KV 也并进一个更大的 kernel。
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

        pages_typed = pages.view(self.layout.dtype).view(-1)
        total_elements = int(actual_tokens) * int(self.layout.hidden_dim)
        block_size = 256
        grid = (triton.cdiv(total_elements, block_size), )
        for layer_id, kv_cache in enumerate(kv_caches):
            kv_cache_flat = kv_cache.view(self.layout.dtype).view(-1)
            for kv_id in (0, 1):
                _bam_pages_to_flat_paged_cache_kernel[grid](
                    pages_typed,
                    kv_cache_flat,
                    slot_mapping,
                    total_elements,
                    kv_id,
                    layer_id,
                    int(actual_tokens),
                    int(self.layout.hidden_dim),
                    int(self.layout.page_token_capacity),
                    int(self.layout.pages_per_kv_layer),
                    int(self.layout.num_layers),
                    int(self._page_buffer_size),
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
        if self._kv_cache_pointers is None:
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
            self._kv_cache_pointers,
            slot_mapping.flatten(),
            kv_caches[0].device,
            self._page_buffer_size,
            False,
        )


def place_bam_results_to_vllm_kvcache(
    *,
    results: Sequence[Any],
    layout: Any,
    kv_caches: Sequence[torch.Tensor],
    slot_mapping: torch.Tensor,
    chunk_starts: Sequence[int],
    kv_cache_dtype: str = "auto",
    num_kv_heads: int,
    head_size: int,
) -> BaMDirectPlacementStats:
    """函数式入口，方便 storage wrapper 调用。"""
    placer = BaMDirectKVPlacer(layout=layout, kv_cache_dtype=kv_cache_dtype)
    return placer.place_batch(
        results=results,
        kv_caches=kv_caches,
        slot_mapping=slot_mapping,
        chunk_starts=chunk_starts,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
    )
