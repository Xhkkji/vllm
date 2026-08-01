# SPDX-License-Identifier: Apache-2.0
"""按 vLLM physical block 组织的 BaM direct KVStore 数据面。

本模块与现有 LMCache/BaM chunk 路径完全独立：不导入 LMCache，不创建 BaM
page cache，也不执行 pack/refill/scatter。它只把一个 vLLM block 展开为当前
真实 paged-KV allocation 中的 layer/K/V fragments，再提交给 BaMDirectKVIO。
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import torch

import vllm.envs as envs
from vllm.logger import init_logger


logger = init_logger(__name__)

GPU_DMA_ALIGNMENT = 64 * 1024
SSD_IO_ALIGNMENT = 4096
DIRECT_IO_REQUEST_CAPACITY = 1024
DIRECT_IO_REGION_CAPACITY = 64
DIRECT_IO_QUEUE_DEPTH = 4096
DIRECT_IO_NUM_QUEUES = 128
DIRECT_IO_POLL_TIMEOUT_S = 30.0
DIRECT_IO_SSD_TAIL_GUARD_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class BaMDirectKVLayout:
    """【BaM KVStore 直通调用链】描述 vLLM V0 paged KV 的 IO 布局。"""

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
        """【BaM KVStore 直通调用链】从真实 KV tensors 固化 block stride。"""
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
        """【BaM KVStore 直通调用链】计算 block fragment 的 layer-region 偏移。"""
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
        """【BaM KVStore 直通调用链】计算 block-major SSD 记录内的偏移。"""
        if layer_id < 0 or layer_id >= self.num_layers:
            raise ValueError(f"invalid layer id: {layer_id}")
        if kv_index not in (0, 1):
            raise ValueError(f"kv_index must be 0 or 1, got {kv_index}")
        return (layer_id * 2 + kv_index) * self.fragment_bytes


@dataclass(frozen=True)
class BaMDirectBlockHandle:
    """【BaM KVStore 直通调用链】block batch 与 native handle 的绑定。"""

    native_handle: Any
    block_count: int
    fragment_count: int
    operation: str


class BaMDirectBlockStore:
    """【BaM KVStore 直通调用链】block 到 NVMe fragment 的唯一转换层。"""

    def __init__(
        self,
        *,
        direct_io: Any,
        gpu_cache: Sequence[torch.Tensor],
        ssd_base_offset: int,
        dma_regions: Sequence[torch.Tensor] | None = None,
    ) -> None:
        """【BaM KVStore 直通调用链】注册 vLLM 持有的 KV CUDA regions。

        ``gpu_cache`` 是 attention 真正读取的 tensor；``dma_regions`` 是覆盖这些
        tensor 的 64KB 对齐 allocation。合成测试中二者可以是同一对象，真实
        vLLM 中后者允许保留最多一个 DMA page 的尾部 padding，而不会改变
        attention 看到的 shape/stride。
        """
        if ssd_base_offset < 0 or ssd_base_offset % SSD_IO_ALIGNMENT != 0:
            raise ValueError("ssd_base_offset must be non-negative and 4KB aligned")
        self.direct_io = direct_io
        self.gpu_cache = list(gpu_cache)
        self.layout = BaMDirectKVLayout.from_gpu_cache(self.gpu_cache)
        self.ssd_base_offset = int(ssd_base_offset)

        if dma_regions is None:
            dma_regions = self.gpu_cache
        if len(dma_regions) != len(self.gpu_cache):
            raise ValueError("dma_regions and gpu_cache must have equal length")
        self.dma_regions = list(dma_regions)
        self.region_base_offsets: list[int] = []
        for layer_id, (layer_cache, dma_region) in enumerate(
                zip(self.gpu_cache, self.dma_regions)):
            if (not dma_region.is_cuda or not dma_region.is_contiguous()
                    or dma_region.device != layer_cache.device):
                raise ValueError(
                    "DMA region must be contiguous CUDA memory at layer "
                    f"{layer_id}")
            layer_begin = int(layer_cache.data_ptr())
            layer_end = layer_begin + int(layer_cache.numel()
                                          * layer_cache.element_size())
            region_begin = int(dma_region.data_ptr())
            region_end = region_begin + int(dma_region.numel()
                                             * dma_region.element_size())
            if layer_begin < region_begin or layer_end > region_end:
                raise ValueError(
                    "DMA region does not cover KV layer allocation at layer "
                    f"{layer_id}")
            self.region_base_offsets.append(layer_begin - region_begin)

        # 每层 tensor 只注册一次。后续所有 block read/write 都通过 region id +
        # byte offset 引用它，不按 request 重复 map/unmap。
        self.region_ids = [
            int(self.direct_io.register_tensor(dma_region))
            for dma_region in self.dma_regions
        ]

    def write_blocks(
        self,
        *,
        gpu_block_ids: Sequence[int],
        storage_block_ids: Sequence[int],
        stream: torch.cuda.Stream | None = None,
    ) -> BaMDirectBlockHandle:
        """【BaM KVStore 直通调用链】从 vLLM KV cache 直接写入 SSD。"""
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
        """【BaM KVStore 直通调用链】从 SSD 直接恢复到 vLLM blocks。"""
        return self._submit_blocks(
            operation=0,
            operation_name="read",
            gpu_block_ids=gpu_block_ids,
            storage_block_ids=storage_block_ids,
            stream=stream,
        )

    def poll(self, handle: BaMDirectBlockHandle) -> bool:
        """【BaM KVStore 直通调用链】只检查 GPU 已发布的 batch ready。"""
        return bool(self.direct_io.poll(handle.native_handle))

    def finish(self, handle: BaMDirectBlockHandle) -> None:
        """【BaM KVStore 直通调用链】在 ready 后回收 direct batch slot。"""
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
        """【BaM KVStore 直通调用链】展开 block 并无阻塞发布 GPU submit。"""
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
                        self.region_base_offsets[layer_id]
                        + self.layout.region_offset(
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


def allocate_aligned_kv_region(
    *,
    allocation_shape: Sequence[int],
    stride_order: Sequence[int],
    dtype: torch.dtype,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """【BaM KVStore 直通调用链】分配 DMA 对齐且保持 vLLM shape 的 KV layer。

    返回 ``(owner, dma_region, kv_layer)``：

    - ``owner`` 负责持有完整 CUDA allocation；
    - ``dma_region`` 是 64KB 对齐并向上补齐到 64KB 的注册范围；
    - ``kv_layer`` 与原 CacheEngine 分配结果保持相同 shape/stride，attention
      仍按 vLLM 原生 paged layout 使用它。

    这里没有 payload staging。padding 只用于满足 GPUDirect registration 的
    allocation 边界要求，SSD 数据仍然直接落到 ``kv_layer``。
    """
    if device != "cuda":
        raise ValueError("aligned direct KV allocation only supports CUDA")
    element_size = int(torch.empty((), dtype=dtype).element_size())
    logical_bytes = element_size
    for dimension in allocation_shape:
        logical_bytes *= int(dimension)
    region_bytes = (
        (logical_bytes + GPU_DMA_ALIGNMENT - 1) // GPU_DMA_ALIGNMENT
        * GPU_DMA_ALIGNMENT
    )

    owner = torch.empty(
        region_bytes + GPU_DMA_ALIGNMENT, dtype=torch.uint8, device=device
    )
    aligned_offset = (-int(owner.data_ptr())) % GPU_DMA_ALIGNMENT
    dma_region = owner.narrow(0, aligned_offset, region_bytes)
    logical_region = dma_region.narrow(0, 0, logical_bytes)
    kv_layer = logical_region.view(dtype).view(tuple(allocation_shape))
    kv_layer = kv_layer.permute(*tuple(stride_order))
    kv_layer.zero_()

    if int(dma_region.data_ptr()) % GPU_DMA_ALIGNMENT != 0:
        raise AssertionError("failed to create 64KB-aligned KV DMA region")
    return owner, dma_region, kv_layer


def _import_bam_direct_kv_io() -> Any:
    """【BaM KVStore 直通调用链】按实验环境定位 direct-IO Python binding。"""
    candidate_paths: list[Path] = []
    if envs.VLLM_BAM_IMPORT_PATH:
        candidate_paths.append(Path(envs.VLLM_BAM_IMPORT_PATH))
    candidate_paths.append(
        Path(__file__).resolve().parents[3] / "BaM_IOStack" / "gids_module"
    )

    errors: list[str] = []
    for path in candidate_paths:
        path_string = str(path)
        if path.exists() and path_string not in sys.path:
            sys.path.insert(0, path_string)
        try:
            module = importlib.import_module("bam_direct_kv_io")
            return module.BaMDirectKVIO
        except Exception as exc:  # pragma: no cover - 依赖部署环境
            errors.append(f"{path_string}: {type(exc).__name__}: {exc}")
    raise ImportError("Failed to import BaMDirectKVIO:\n" + "\n".join(errors))


def _parse_direct_ssd_list() -> tuple[int, ...]:
    """【BaM KVStore 直通调用链】复用现有 BaM SSD 选择配置。"""
    value = envs.VLLM_BAM_SSD_LIST
    if value is None or not value.strip():
        return (int(envs.VLLM_BAM_CTRL_IDX), )
    parsed = tuple(int(item.strip()) for item in value.split(",")
                   if item.strip())
    if len(parsed) != 1:
        raise ValueError("phase-1 direct KVStore requires exactly one SSD")
    return parsed


class BaMVLLMDirectKVStore:
    """【BaM KVStore 直通调用链】连接 CacheEngine swap 与 direct 数据面。

    本类只解释 vLLM scheduler 已经生成的 ``src_to_dst`` block mapping。
    GPU submit、NVMe CQ polling 和 DMA 均由 ``BaMDirectKVIO`` 完成；CPU 侧循环
    只读取 GPU worker 发布的 request 状态，ready 后才返回 CacheEngine 继续
    attention。
    """

    def __init__(
        self,
        *,
        gpu_cache: Sequence[torch.Tensor],
        dma_regions: Sequence[torch.Tensor],
        num_storage_blocks: int,
    ) -> None:
        """【BaM KVStore 直通调用链】绑定真实 vLLM cache 和 SSD block 空间。"""
        if num_storage_blocks <= 0:
            raise ValueError("direct KVStore requires positive storage blocks")
        direct_io_class = _import_bam_direct_kv_io()
        self.direct_io = direct_io_class(
            ssd_list=_parse_direct_ssd_list(),
            request_capacity=DIRECT_IO_REQUEST_CAPACITY,
            region_capacity=DIRECT_IO_REGION_CAPACITY,
            queue_depth=DIRECT_IO_QUEUE_DEPTH,
            num_queues=DIRECT_IO_NUM_QUEUES,
        )
        try:
            layout = BaMDirectKVLayout.from_gpu_cache(gpu_cache)
            storage_bytes = int(num_storage_blocks) * layout.logical_block_bytes
            namespace_bytes = int(self.direct_io.namespace_size_bytes)
            required_bytes = storage_bytes + DIRECT_IO_SSD_TAIL_GUARD_BYTES
            if required_bytes > namespace_bytes:
                raise ValueError(
                    "direct KVStore SSD namespace is too small: "
                    f"required={required_bytes}, available={namespace_bytes}"
                )
            # 独立 direct backend 在 namespace 尾部区域保留连续 block 空间，
            # 同时留出 64MB 尾部 guard。该布局与已经验证稳定的 direct block
            # roundtrip 一致，既不覆盖旧 row/page baseline 从低地址开始使用的
            # 数据，也不把 NVMe command 压到 namespace 最后一个 LBA 边界。
            ssd_base_offset = (
                (namespace_bytes - required_bytes) // SSD_IO_ALIGNMENT
                * SSD_IO_ALIGNMENT
            )
            self.block_store = BaMDirectBlockStore(
                direct_io=self.direct_io,
                gpu_cache=gpu_cache,
                dma_regions=dma_regions,
                ssd_base_offset=ssd_base_offset,
            )
        except Exception:
            self.direct_io.close()
            raise

        self.num_storage_blocks = int(num_storage_blocks)
        fragments_per_block = self.block_store.layout.num_layers * 2
        self.max_blocks_per_batch = max(
            1, int(self.direct_io.request_capacity) // fragments_per_block
        )
        logger.info(
            "[BAM_DIRECT_KVSTORE] initialized layers=%d gpu_blocks=%d "
            "storage_blocks=%d logical_block_bytes=%d ssd_base_offset=%d "
            "max_blocks_per_batch=%d bam_page_cache=0 staging=0 refill=0",
            self.block_store.layout.num_layers,
            self.block_store.layout.num_gpu_blocks,
            self.num_storage_blocks,
            self.block_store.layout.logical_block_bytes,
            self.block_store.ssd_base_offset,
            self.max_blocks_per_batch,
        )

    def swap_out(self, src_to_dst: torch.Tensor) -> None:
        """【BaM KVStore 直通调用链】执行 GPU block -> SSD storage block。"""
        self._transfer_mapping(src_to_dst, operation="write")

    def swap_in(self, src_to_dst: torch.Tensor) -> None:
        """【BaM KVStore 直通调用链】执行 SSD storage block -> GPU block。"""
        self._transfer_mapping(src_to_dst, operation="read")

    def close(self) -> None:
        """【BaM KVStore 直通调用链】停止 persistent worker 并解除 DMA map。"""
        direct_io = getattr(self, "direct_io", None)
        if direct_io is not None:
            direct_io.close()
            self.direct_io = None

    def _transfer_mapping(
        self, src_to_dst: torch.Tensor, *, operation: str
    ) -> None:
        """【BaM KVStore 直通调用链】分批 submit、检查 ready 并完成请求。"""
        if src_to_dst.numel() == 0:
            return
        transfer_start = time.perf_counter()
        mappings = src_to_dst.to(device="cpu", dtype=torch.int64).tolist()
        if envs.VLLM_V0_SWAP_TRACE:
            logger.info(
                "[BAM_DIRECT_KVSTORE] op=%s phase=submit blocks=%d",
                operation,
                len(mappings),
            )
        for begin in range(0, len(mappings), self.max_blocks_per_batch):
            batch = mappings[begin:begin + self.max_blocks_per_batch]
            source_ids = [int(mapping[0]) for mapping in batch]
            destination_ids = [int(mapping[1]) for mapping in batch]
            if operation == "write":
                self._validate_storage_block_ids(destination_ids)
                handle = self.block_store.write_blocks(
                    gpu_block_ids=source_ids,
                    storage_block_ids=destination_ids,
                )
            elif operation == "read":
                self._validate_storage_block_ids(source_ids)
                handle = self.block_store.read_blocks(
                    storage_block_ids=source_ids,
                    gpu_block_ids=destination_ids,
                )
            else:
                raise ValueError(f"unsupported direct KV operation: {operation}")
            self._wait_until_ready(handle)
            self.block_store.finish(handle)
        if envs.VLLM_V0_SWAP_TRACE:
            logger.info(
                "[BAM_DIRECT_KVSTORE] op=%s phase=done blocks=%d "
                "elapsed_ms=%.3f",
                operation,
                len(mappings),
                (time.perf_counter() - transfer_start) * 1000.0,
            )

    def _wait_until_ready(self, handle: BaMDirectBlockHandle) -> None:
        """【BaM KVStore 直通调用链】CPU 仅检查 GPU request-ready flag。"""
        deadline = time.monotonic() + DIRECT_IO_POLL_TIMEOUT_S
        while not self.block_store.poll(handle):
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "BaM direct KVStore request did not become ready within "
                    f"{DIRECT_IO_POLL_TIMEOUT_S}s operation={handle.operation} "
                    f"blocks={handle.block_count}"
                )
            time.sleep(0.001)

    def _validate_storage_block_ids(self, block_ids: Sequence[int]) -> None:
        """【BaM KVStore 直通调用链】阻止 scheduler id 越过 SSD extent。"""
        for block_id in block_ids:
            if block_id < 0 or block_id >= self.num_storage_blocks:
                raise ValueError(
                    f"storage block id {block_id} is outside "
                    f"[0, {self.num_storage_blocks})"
                )

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            # CUDA runtime 退出阶段不能从析构函数继续抛异常。
            pass
