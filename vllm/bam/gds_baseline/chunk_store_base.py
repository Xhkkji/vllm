# SPDX-License-Identifier: Apache-2.0
"""BaM / GDS trace replay 共用的 chunk store 接口。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class ChunkMetadata:
    """单个 KV chunk 的最小元数据。

    这里不引入 LMCache V1 的完整 metadata/allocator，只保留 replay 对比
    必需的信息，避免 GDS baseline 反向污染当前 V0 主路径。
    """

    chunk_hash: str
    offset: int
    nbytes: int
    shape: torch.Size
    dtype: torch.dtype
    actual_tokens: int


@dataclass(frozen=True)
class ChunkTransferResult:
    """一次 chunk 读写的统计结果。"""

    backend: str
    op: str
    chunk_hash: str
    nbytes: int
    elapsed_ms: float

    @property
    def bw_gib_s(self) -> float:
        if self.elapsed_ms <= 0:
            return 0.0
        return (self.nbytes / (self.elapsed_ms / 1000.0)) / (1024**3)


class KVChunkStore(Protocol):
    """统一 BaM 和 GDS 的 chunk 粒度接口。"""

    backend_name: str

    def put_chunk(self, chunk_hash: str, tensor: torch.Tensor,
                  actual_tokens: int) -> ChunkTransferResult:
        ...

    def get_chunk(self, chunk_hash: str,
                  out_tensor: torch.Tensor) -> ChunkTransferResult:
        ...

    def close(self) -> None:
        ...
