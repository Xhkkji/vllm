# SPDX-License-Identifier: Apache-2.0
"""BaM 对照用的原生 GDS/cuFile 数据面 baseline。

这个包刻意独立于当前 LMCache V0 主路径：

- 不修改 vLLM V0 scheduler / connector
- 不修改当前已经跑通的 LMCache + BaM prefer-load
- 只提供 chunk 级 put/get 接口，便于 replay 真实 LMCache chunk 负载
"""

from vllm.bam.gds_baseline.chunk_store_base import (ChunkMetadata,
                                                    ChunkTransferResult,
                                                    KVChunkStore)
from vllm.bam.gds_baseline.gds_chunk_store import GDSChunkStore
from vllm.bam.gds_baseline.lmcache_style_gds_store import (
    LMCacheStyleGDSChunkStore)

__all__ = [
    "ChunkMetadata",
    "ChunkTransferResult",
    "GDSChunkStore",
    "KVChunkStore",
    "LMCacheStyleGDSChunkStore",
]
