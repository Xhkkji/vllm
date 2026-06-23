# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional

import torch

from vllm.bam.kv_layout import BaMKVLayout


class BaMBlockCodec:
    """负责 KV block 与 BaM row bytes 之间的互转。

    这里不关心调度来源是 V0 swap 还是 LMCache，只处理：
    - 怎么把 vLLM 的 paged KV block 打包成 BaM 行
    - 怎么把 BaM 行恢复成 vLLM 需要的 KV block
    - 怎么做最小正确性校验
    """

    def __init__(self, layout: BaMKVLayout, dtype: torch.dtype) -> None:
        self.layout = layout
        self.dtype = dtype

    def pack_layer_rows(self, layer_cache: torch.Tensor,
                        block_ids: torch.Tensor) -> torch.Tensor:
        # 先按 block gather，再把 [2, N, hidden] 变成 [N, row_bytes]。
        packed = layer_cache.index_select(1, block_ids)
        packed = packed.permute(1, 0, 2).contiguous().view(block_ids.shape[0],
                                                           -1)
        return packed.view(torch.uint8).view(block_ids.shape[0], -1)

    def pack_single_block_rows(self, gpu_cache: List[torch.Tensor],
                               block_id: int) -> torch.Tensor:
        # 一个 vLLM block 在 SSD 上对应 num_layers 行，每层一行。
        rows = []
        block_index = torch.tensor([block_id],
                                   device=gpu_cache[0].device,
                                   dtype=torch.long)
        for layer_cache in gpu_cache:
            rows.append(self.pack_layer_rows(layer_cache, block_index))
        return torch.cat(rows, dim=0)

    def restore_layer_blocks(self, layer_rows: torch.Tensor) -> torch.Tensor:
        # 每一行都对应一个 layer 的完整 [K, V] block，这里把 uint8 原始字节
        # 重新解释为模型 dtype，再恢复成 vLLM 的 [2, N, hidden] 布局。
        layer_blocks = layer_rows.contiguous().view(self.dtype)
        layer_blocks = layer_blocks.view(layer_rows.shape[0], 2, -1)
        return layer_blocks.permute(1, 0, 2).contiguous()

    def verify_layer_restore(self, layer_id: int, expected_layer_cache: torch.Tensor,
                             restored_layer_cache: torch.Tensor,
                             expected_block_ids: torch.Tensor,
                             restored_block_ids: torch.Tensor,
                             verify_count: int) -> None:
        if verify_count <= 0:
            return

        verify_expected_block_ids = expected_block_ids[:verify_count].to("cpu")
        verify_restored_block_ids = restored_block_ids[:verify_count]
        expected_blocks = expected_layer_cache.index_select(
            1, verify_expected_block_ids).contiguous()
        restored_blocks = restored_layer_cache.index_select(
            1, verify_restored_block_ids).to("cpu").contiguous()

        # KV cache 里可能存在 NaN 位型。底层搬运校验按原始字节比较更可靠。
        expected_bytes = expected_blocks.view(torch.uint8)
        restored_bytes = restored_blocks.view(torch.uint8)
        if torch.equal(expected_bytes, restored_bytes):
            return

        is_allclose = torch.allclose(expected_blocks,
                                     restored_blocks,
                                     equal_nan=True)
        byte_mismatch_count = int(
            torch.count_nonzero(expected_bytes != restored_bytes).item())
        mismatch_by_mapping = (expected_bytes != restored_bytes).view(
            2, verify_count, -1).any(dim=(0, 2))
        first_bad_idx = int(
            torch.nonzero(mismatch_by_mapping, as_tuple=False)[0].item())
        first_bad_expected_block = int(
            verify_expected_block_ids[first_bad_idx].item())
        first_bad_restored_block = int(
            verify_restored_block_ids[first_bad_idx].detach().cpu().item())
        nan_mismatch_count = int(
            torch.count_nonzero(torch.isnan(expected_blocks) !=
                                torch.isnan(restored_blocks)).item())
        max_abs_diff = (
            expected_blocks.to(torch.float32) -
            restored_blocks.to(torch.float32)).abs().nan_to_num(
                nan=0.0).max().item()
        raise RuntimeError(
            "[BAM_VERIFY] mismatch detected: "
            f"layer={layer_id} checked_blocks={verify_count} "
            f"first_bad_mapping_idx={first_bad_idx} "
            f"first_bad_expected_block={first_bad_expected_block} "
            f"first_bad_restored_block={first_bad_restored_block} "
            f"allclose={int(is_allclose)} "
            f"byte_mismatch_count={byte_mismatch_count} "
            f"nan_mismatch_count={nan_mismatch_count} "
            f"max_abs_diff={max_abs_diff}")

    @staticmethod
    def get_verify_count(num_mappings: int, verify_enabled: bool,
                         verify_blocks: int) -> int:
        if not verify_enabled:
            return 0
        if verify_blocks <= 0:
            return num_mappings
        return min(num_mappings, verify_blocks)

    @staticmethod
    def get_verify_mode(verify_count: int,
                        total_count: int) -> Optional[str]:
        if verify_count <= 0:
            return None
        return "full" if verify_count == total_count else "sample"
