# SPDX-License-Identifier: Apache-2.0
"""LMCache <-> BaM 适配层的最小骨架。

这里先不直接改动 LMCache repo 内部逻辑，只在 vllm-bam 里把后续会用到的
接口和命名整理出来。当前它主要承担两件事：

1. 明确后续主线不会继续依附于 V0 swap 入口。
2. 给未来的 LMCache shadow-store / prefer-load 接入预留一个简单落点。
"""

from dataclasses import dataclass
from typing import Optional

from vllm.bam.block_store import BaMBlockStore


@dataclass(frozen=True)
class BaMLayerCacheKey:
    """LMCache 侧单层 KV 片段在 BaM 中的逻辑 key。

    这里先保留最小字段集合，后续真正接 LMCache 时，再根据 chunk/request
    的真实生命周期补充序列化和映射策略。
    """

    request_id: str
    chunk_id: int
    layer_name: str


class LMCacheBaMAdapter:
    """LMCache 主线下的 BaM 适配层骨架。"""

    def __init__(self, block_store: BaMBlockStore) -> None:
        self.block_store = block_store

    def describe_status(self) -> str:
        # 当前还未真正接入 LMCache save/load，只先保留一个清晰的占位对象，
        # 避免后续又把主逻辑塞回 worker/cache_engine.py。
        return ("LMCacheBaMAdapter is not wired into LMCache yet; "
                "current mainline still uses the shared BaM data-plane core.")

    def lookup(self, key: BaMLayerCacheKey) -> Optional[int]:
        # 后续这里会变成 “LMCache key -> BaM logical block id” 的映射查询。
        _ = key
        return None
