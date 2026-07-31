# SPDX-License-Identifier: Apache-2.0
"""按 vLLM physical block 组织的 BaM direct KVStore 数据面。

本模块与现有 LMCache/BaM chunk 路径完全独立：不导入 LMCache，不创建 BaM
page cache，也不执行 pack/refill/scatter。它只把一个 vLLM block 展开为当前
真实 paged-KV allocation 中的 layer/K/V fragments，再提交给 BaMDirectKVIO。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch


@dataclass(frozen=True)
class BaMDirectKVLayout:
    """vLLM V0 paged KV allocation 的直接 IO 布局。"""

    num_layers: int
    num_gpu_blocks: int
    fragment_bytes: int
    layer_bytes: int
    logical_block_bytes: int
    element_size: int
    kv_stride_elems: int
    block_stride_elems: int

    @classmethod
    def from_gpu_cache(
        cls, gpu_cache: Sequence[torch.Tensor]
    ) -> "BaMDirectKVLayout":
        if not gpu_cache:
            raise ValueError("gpu_cache must not be empty")

        first = gpu_cache[0]
        if first.dim() != 3 or first.shape[0] != 2:
            raise ValueError(
                "phase-1 direct KVStore requires vLLM V0 layout "
                f"[2, num_blocks, block_elems], got {tuple(first.shape)}"
            )
        if not first.is_cuda or not first.is_contiguous():
            raise ValueError("vLLM KV cache must be contiguous CUDA memory")
        if first.stride(2) != 1 or first.stride(1) != first.shape[2]:
            raise ValueError(
                "each vLLM KV block must be a contiguous fragment, got "
                f"shape={tuple(first.shape)} stride={tuple(first.stride())}"
            )

        element_size = int(first.element_size())
        fragment_bytes = int(first.stride(1) * element_size)
        layer_bytes = int(first.numel() * element_size)
        if fragment_bytes % 4096 != 0:
            raise ValueError(
                "direct KV fragment must be 4KB aligned, got "
                f"fragment_bytes={fragment_bytes}"
            )

        for layer_id, layer_cache in enumerate(gpu_cache):
            if (
                layer_cache.shape != first.shape
                or layer_cache.stride() != first.stride()
                or layer_cache.dtype != first.dtype
                or layer_cache.device != first.device
                or not layer_cache.is_contiguous()
            ):
                raise ValueError(
                    "all KV layers must share one layout; mismatch at "
                    f"layer={layer_id} shape={tuple(layer_cache.shape)} "
                    f"stride={tuple(layer_cache.stride())}"
                )

        num_layers = len(gpu_cache)
        return cls(
            num_layers=num_layers,
            num_gpu_blocks=int(first.shape[1]),
            fragment_bytes=fragment_bytes,
            layer_bytes=layer_bytes,
            logical_block_bytes=num_layers * 2 * fragment_bytes,
            element_size=element_size,
            kv_stride_elems=int(first.stride(0)),
            block_stride_elems=int(first.stride(1)),
        )

    def region_offset(self, *, kv_index: int, block_id: int) -> int:
        if kv_index not in (0, 1):
            raise ValueError(f"kv_index must be 0 or 1, got {kv_index}")
        if block_id < 0 or block_id >= self.num_gpu_blocks:
            raise ValueError(
                f"physical block id {block_id} is outside "
                f"[0, {self.num_gpu_blocks})"
            )
        return (
            kv_index * self.kv_stride_elems
            + block_id * self.block_stride_elems
        ) * self.element_size

    def ssd_fragment_offset(self, *, layer_id: int, kv_index: int) -> int:
        if layer_id < 0 or layer_id >= self.num_layers:
            raise ValueError(f"invalid layer id: {layer_id}")
        if kv_index not in (0, 1):
            raise ValueError(f"kv_index must be 0 or 1, got {kv_index}")
        return (layer_id * 2 + kv_index) * self.fragment_bytes


@dataclass(frozen=True)
class BaMDirectBlockHandle:
    """一个 block batch 和底层 native direct-IO handle 的绑定。"""

    native_handle: Any
    block_count: int
    fragment_count: int
    operation: str


class BaMDirectBlockStore:
    """vLLM block -> direct NVMe fragment request 的唯一转换层。"""

    def __init__(
        self,
        *,
        direct_io: Any,
        gpu_cache: Sequence[torch.Tensor],
        ssd_base_offset: int,
    ) -> None:
        if ssd_base_offset < 0 or ssd_base_offset % 4096 != 0:
            raise ValueError("ssd_base_offset must be non-negative and 4KB aligned")
        self.direct_io = direct_io
        self.gpu_cache = list(gpu_cache)
        self.layout = BaMDirectKVLayout.from_gpu_cache(self.gpu_cache)
        self.ssd_base_offset = int(ssd_base_offset)

        # 每层 tensor 只注册一次。后续所有 block read/write 都通过 region id +
        # byte offset 引用它，不按 request 重复 map/unmap。
        self.region_ids = [
            int(self.direct_io.register_tensor(layer_cache))
            for layer_cache in self.gpu_cache
        ]

    def write_blocks(
        self,
        *,
        gpu_block_ids: Sequence[int],
        storage_block_ids: Sequence[int],
        stream: torch.cuda.Stream | None = None,
    ) -> BaMDirectBlockHandle:
        """直接从 vLLM KV cache 写入 SSD，不经过 pack 或 staging。"""
        return self._submit_blocks(
            operation=1,
            operation_name="write",
            gpu_block_ids=gpu_block_ids,
            storage_block_ids=storage_block_ids,
            stream=stream,
        )

    def read_blocks(
        self,
        *,
        storage_block_ids: Sequence[int],
        gpu_block_ids: Sequence[int],
        stream: torch.cuda.Stream | None = None,
    ) -> BaMDirectBlockHandle:
        """从 SSD 直接恢复到最终 vLLM physical blocks。"""
        return self._submit_blocks(
            operation=0,
            operation_name="read",
            gpu_block_ids=gpu_block_ids,
            storage_block_ids=storage_block_ids,
            stream=stream,
        )

    def poll(self, handle: BaMDirectBlockHandle) -> bool:
        return bool(self.direct_io.poll(handle.native_handle))

    def finish(self, handle: BaMDirectBlockHandle) -> None:
        self.direct_io.finish(handle.native_handle)

    def _submit_blocks(
        self,
        *,
        operation: int,
        operation_name: str,
        gpu_block_ids: Sequence[int],
        storage_block_ids: Sequence[int],
        stream: torch.cuda.Stream | None,
    ) -> BaMDirectBlockHandle:
        if len(gpu_block_ids) != len(storage_block_ids):
            raise ValueError(
                "gpu_block_ids and storage_block_ids must have equal length"
            )
        if not gpu_block_ids:
            raise ValueError("at least one block is required")

        operations: list[int] = []
        ssd_byte_offsets: list[int] = []
        region_ids: list[int] = []
        region_offsets: list[int] = []
        lengths: list[int] = []

        # SSD 记录采用 block-major：一个 storage block 内按 layer0 K/V、
        # layer1 K/V 顺序排列。GPU 端仍保持 vLLM 原始 layer-major allocation；
        # descriptor table 直接描述二者映射，不创建中间连续 tensor。
        for gpu_block_id_raw, storage_block_id_raw in zip(
            gpu_block_ids, storage_block_ids
        ):
            gpu_block_id = int(gpu_block_id_raw)
            storage_block_id = int(storage_block_id_raw)
            if storage_block_id < 0:
                raise ValueError(
                    f"storage block id must be non-negative: {storage_block_id}"
                )
            storage_block_base = (
                self.ssd_base_offset
                + storage_block_id * self.layout.logical_block_bytes
            )
            for layer_id, region_id in enumerate(self.region_ids):
                for kv_index in (0, 1):
                    operations.append(operation)
                    ssd_byte_offsets.append(
                        storage_block_base
                        + self.layout.ssd_fragment_offset(
                            layer_id=layer_id, kv_index=kv_index
                        )
                    )
                    region_ids.append(region_id)
                    region_offsets.append(
                        self.layout.region_offset(
                            kv_index=kv_index, block_id=gpu_block_id
                        )
                    )
                    lengths.append(self.layout.fragment_bytes)

        if len(operations) > self.direct_io.request_capacity:
            raise ValueError(
                f"direct batch requires {len(operations)} requests, but "
                f"capacity is {self.direct_io.request_capacity}"
            )
        native_handle = self.direct_io.submit(
            operations=operations,
            ssd_byte_offsets=ssd_byte_offsets,
            region_ids=region_ids,
            region_offsets=region_offsets,
            lengths=lengths,
            stream=stream,
        )
        return BaMDirectBlockHandle(
            native_handle=native_handle,
            block_count=len(gpu_block_ids),
            fragment_count=len(operations),
            operation=operation_name,
        )
