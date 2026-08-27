# SPDX-License-Identifier: Apache-2.0

"""GranuleKV prefix matching semantics shared by the scheduler and allocator.

This module only computes the vLLM-compatible chained hash for complete prefix
blocks. It owns no physical storage and does not call the GranuleKV transport.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from vllm.core.block.prefix_caching_block import PrefixCachingBlock


def compute_full_prefix_block_hashes(
    token_ids: Sequence[int],
    block_size: int,
    extra_hash: Optional[int],
) -> List[int]:
    """Compute chained hashes for complete blocks starting at token zero."""
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
