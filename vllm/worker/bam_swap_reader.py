# SPDX-License-Identifier: Apache-2.0
"""兼容旧导入路径。

BaMSwapReader 继续保留，方便现有 V0 swap 实验脚本不改也能跑。
内部实现已经切到通用的 BaMBlockReader。
"""

from typing import List

import torch

from vllm.bam.block_reader import BaMBlockReader
from vllm.bam.block_store import BaMBlockStore


class BaMSwapReader(BaMBlockReader):

    def __init__(self, block_store: BaMBlockStore, dtype: torch.dtype) -> None:
        super().__init__(block_store, dtype)

    def swap_in(self, gpu_cache: List[torch.Tensor],
                cpu_cache: List[torch.Tensor],
                src_to_dst: torch.Tensor) -> None:
        if src_to_dst.numel() == 0:
            return
        self.load_blocks(gpu_cache, src_to_dst[:, 0], src_to_dst[:, 1],
                         reference_cache=cpu_cache)
