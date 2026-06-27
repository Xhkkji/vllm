# SPDX-License-Identifier: Apache-2.0
"""chunk 粒度的原生 GDS/cuFile baseline store。"""

from __future__ import annotations

import time
from collections import OrderedDict

import torch

from vllm.bam.gds_baseline.chunk_store_base import (ChunkMetadata,
                                                    ChunkTransferResult)
from vllm.bam.gds_baseline.cufile_context import CuFileSlab
from vllm.logger import init_logger

logger = init_logger(__name__)

_ALIGNMENT = 4096


def _align_up(value: int, alignment: int = _ALIGNMENT) -> int:
    return (int(value) + alignment - 1) // alignment * alignment


class GDSChunkStore:
    """用 LMCache V1 风格 cuFile 数据面实现的 chunk store。

    这里使用一个顺序分配的 slab 文件，而不是 LMCache V1 的完整目录分片、
    evictor 和 StorageManager。这样它可以在当前 V100 + vLLM V0 环境里作为
    数据面 baseline 单独 replay，不影响现有 BaM / LMCache 主路径。
    """

    backend_name = "gds"

    def __init__(self,
                 slab_path: str,
                 slab_bytes: int,
                 device: str = "cuda:0",
                 use_direct_io: bool = True) -> None:
        self.device = torch.device(device)
        self.slab = CuFileSlab(slab_path, slab_bytes, use_direct_io=use_direct_io)
        self._next_offset = 0
        self._metadata: "OrderedDict[str, ChunkMetadata]" = OrderedDict()

    def put_chunk(self, chunk_hash: str, tensor: torch.Tensor,
                  actual_tokens: int) -> ChunkTransferResult:
        tensor = self._prepare_tensor(tensor)
        nbytes = int(tensor.numel() * tensor.element_size())
        offset = self._allocate(chunk_hash, nbytes, tensor.shape, tensor.dtype,
                                actual_tokens)

        start = time.perf_counter()
        bytes_done = self.slab.transfer(tensor, offset, "write")
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        result = ChunkTransferResult("gds", "write", chunk_hash, bytes_done,
                                     elapsed_ms)
        logger.info(
            "[GDS_WRITE] chunk_hash=%s offset=%d nbytes=%d actual_tokens=%d "
            "elapsed_ms=%.3f bw_gib_s=%.3f",
            chunk_hash[:16],
            offset,
            bytes_done,
            actual_tokens,
            result.elapsed_ms,
            result.bw_gib_s,
        )
        return result

    def get_chunk(self, chunk_hash: str,
                  out_tensor: torch.Tensor) -> ChunkTransferResult:
        metadata = self._metadata.get(chunk_hash)
        if metadata is None:
            raise KeyError(f"GDS chunk not found: {chunk_hash}")

        out_tensor = self._prepare_tensor(out_tensor)
        if tuple(out_tensor.shape) != tuple(metadata.shape):
            raise ValueError(
                "GDS output tensor shape mismatch: "
                f"expected={tuple(metadata.shape)} got={tuple(out_tensor.shape)}")
        if out_tensor.dtype != metadata.dtype:
            raise ValueError(
                "GDS output tensor dtype mismatch: "
                f"expected={metadata.dtype} got={out_tensor.dtype}")

        start = time.perf_counter()
        bytes_done = self.slab.transfer(out_tensor, metadata.offset, "read")
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        result = ChunkTransferResult("gds", "read", chunk_hash, bytes_done,
                                     elapsed_ms)
        logger.info(
            "[GDS_READ] chunk_hash=%s offset=%d nbytes=%d actual_tokens=%d "
            "elapsed_ms=%.3f bw_gib_s=%.3f",
            chunk_hash[:16],
            metadata.offset,
            bytes_done,
            metadata.actual_tokens,
            result.elapsed_ms,
            result.bw_gib_s,
        )
        return result

    def close(self) -> None:
        self.slab.close()

    def _prepare_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if not tensor.is_cuda or tensor.device != self.device:
            tensor = tensor.to(device=self.device, non_blocking=False)
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        return tensor

    def _allocate(self, chunk_hash: str, nbytes: int, shape: torch.Size,
                  dtype: torch.dtype, actual_tokens: int) -> int:
        existing = self._metadata.get(chunk_hash)
        if existing is not None:
            if existing.nbytes != nbytes:
                raise ValueError(
                    f"cannot overwrite chunk with different size: {chunk_hash}")
            self._metadata.move_to_end(chunk_hash)
            return existing.offset

        offset = _align_up(self._next_offset)
        self._next_offset = offset + _align_up(nbytes)
        metadata = ChunkMetadata(
            chunk_hash=chunk_hash,
            offset=offset,
            nbytes=nbytes,
            shape=torch.Size(shape),
            dtype=dtype,
            actual_tokens=int(actual_tokens),
        )
        self._metadata[chunk_hash] = metadata
        return offset
