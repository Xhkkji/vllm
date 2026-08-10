# SPDX-License-Identifier: Apache-2.0

"""BaM MDS 专属 prefix 控制面的最小公共逻辑。

这里刻意不保存物理 block，也不调用 MDS。物理资源的所有权仍归
``SelfAttnBlockSpaceManager``，异步传输生命周期仍归 ``AsyncKVScheduler``。
本模块只定义 prefix 的匹配语义，避免把 BaM 的实验逻辑塞进 vLLM 原生
Scheduler：只匹配从 token 0 开始的连续完整 block，hash 与 V0 prefix cache
完全一致。

第一版索引依附于当前 engine 内的 CPU/storage block allocator，因此不支持
engine 重启后恢复索引。MDS 中的数据仍在 SSD 上，但没有 token hash 到 storage
block id 的持久化元数据时，不能安全地把它当作命中。
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from vllm.core.block.prefix_caching_block import PrefixCachingBlock


def compute_full_prefix_block_hashes(
    token_ids: Sequence[int],
    block_size: int,
    extra_hash: Optional[int],
) -> List[int]:
    """按 vLLM V0 语义计算完整 prefix block 的链式 hash。

    尾部不足一个 block 的 token 不进入索引。每个 block hash 都包含前驱
    hash，因此只有从第一个 block 开始连续相同的请求才可能连续命中。
    """
    hashes: List[int] = []
    previous_hash = PrefixCachingBlock._none_hash
    num_full_blocks = len(token_ids) // block_size
    for block_index in range(num_full_blocks):
        begin = block_index * block_size
        block_tokens = token_ids[begin:begin + block_size]
        block_hash = PrefixCachingBlock.hash_block_tokens(
            is_first_block=block_index == 0,
            prev_block_hash=previous_hash,
            cur_block_token_ids=list(block_tokens),
            extra_hash=extra_hash,
        )
        hashes.append(block_hash)
        previous_hash = block_hash
    return hashes
