# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import List

import torch


@dataclass(frozen=True)
class BaMKVLayout:
    """描述当前 vLLM KV block 在 BaM 侧的行布局。"""

    num_layers: int
    row_bytes: int
    block_bytes: int

    @classmethod
    def from_gpu_cache(cls, gpu_cache: List[torch.Tensor]) -> "BaMKVLayout":
        if not gpu_cache:
            raise ValueError("gpu_cache is empty")

        sample_layer = gpu_cache[0]
        if sample_layer.dim() != 3 or sample_layer.shape[0] != 2:
            raise ValueError(
                "Only the V0 paged-attention KV layout is supported, "
                f"got shape={tuple(sample_layer.shape)}")

        row_bytes = int(sample_layer[:, 0, :].numel() *
                        sample_layer.element_size())
        num_layers = len(gpu_cache)
        return cls(
            num_layers=num_layers,
            row_bytes=row_bytes,
            block_bytes=num_layers * row_bytes,
        )

    def row_offset(self, block_id: int, layer_id: int) -> int:
        """block-major 布局下，一个逻辑 block 对应连续 num_layers 行。"""
        return int(block_id) * self.num_layers + int(layer_id)
