# SPDX-License-Identifier: Apache-2.0
from typing import List

import torch

import vllm.envs as envs
from vllm.bam.kv_layout import BaMKVLayout
from vllm.bam.row_store_loader import (import_bam_row_store,
                                       parse_optional_int_list)
from vllm.logger import init_logger

logger = init_logger(__name__)


class BaMBlockStore:
    """共享的 BaM KV block 存储后端。

    这里故意用更中性的 “block” 语义，而不是继续把接口绑死在
    `cpu_block_id` 上。当前 V0 swap 依然会把目标 block 当成 CPU block，
    但后续 LMCache 路径可以把它替换成自己的逻辑 block id。
    """

    def __init__(self, gpu_cache: List[torch.Tensor],
                 num_blocks: int) -> None:
        self.layout = BaMKVLayout.from_gpu_cache(gpu_cache)
        self.device = gpu_cache[0].device
        self.num_blocks = int(num_blocks)

        if self.num_blocks <= 0:
            raise ValueError(
                f"num_blocks must be positive, got {self.num_blocks}")

        bam_row_store_cls = import_bam_row_store()
        ssd_list = parse_optional_int_list(envs.VLLM_BAM_SSD_LIST)
        self.row_store = bam_row_store_cls(
            row_bytes=self.layout.row_bytes,
            num_rows=self.layout.num_layers * self.num_blocks,
            cache_size_mb=envs.VLLM_BAM_CACHE_SIZE_MB,
            num_ssd=envs.VLLM_BAM_NUM_SSD,
            ssd_list=ssd_list,
            ctrl_idx=envs.VLLM_BAM_CTRL_IDX,
        )

        logger.info(
            "[BAM_BLOCK_STORE] initialized row_bytes=%d num_layers=%d "
            "num_blocks=%d total_rows=%d cache_size_mb=%d num_ssd=%d",
            self.layout.row_bytes,
            self.layout.num_layers,
            self.num_blocks,
            self.layout.num_layers * self.num_blocks,
            envs.VLLM_BAM_CACHE_SIZE_MB,
            envs.VLLM_BAM_NUM_SSD,
        )

    def make_block_row_ids(self, block_ids: torch.Tensor) -> torch.Tensor:
        num_blocks = int(block_ids.shape[0])
        layer_offsets = torch.arange(self.layout.num_layers,
                                     device=self.device,
                                     dtype=torch.long)
        # 读写都采用 block-major 排列：
        # [block0_layer0, block0_layer1, ..., block1_layer0, ...]
        return (block_ids.repeat_interleave(self.layout.num_layers) *
                self.layout.num_layers + layer_offsets.repeat(num_blocks))

    def make_block_row_offset(self, block_id: int, start_layer: int = 0) -> int:
        return self.layout.row_offset(block_id, start_layer)
