# SPDX-License-Identifier: Apache-2.0
import time
from typing import List

import torch

import vllm.envs as envs
from vllm.bam.block_codec import BaMBlockCodec
from vllm.bam.block_store import BaMBlockStore
from vllm.logger import init_logger

logger = init_logger(__name__)


class BaMBlockWriter:
    """把 vLLM KV block 写入 BaM。

    当前 V0 swap 路径会把它当成 shadow writer 用，但这里本身只关心：
    - 从哪些源 block 取数据
    - 要写到哪些逻辑目标 block
    """

    def __init__(self, block_store: BaMBlockStore, dtype: torch.dtype) -> None:
        self.block_store = block_store
        self.layout = block_store.layout
        self.codec = BaMBlockCodec(self.layout, dtype)
        self._debug_events = 0

    def _log_mapping_debug(self, src_block_ids: torch.Tensor,
                           dst_block_ids: torch.Tensor) -> None:
        if not envs.VLLM_BAM_SWAPIN_VERIFY or self._debug_events >= 4:
            return

        src_blocks = src_block_ids.detach().cpu()
        dst_blocks = dst_block_ids.detach().cpu()
        mapping_pairs = torch.stack([src_blocks, dst_blocks], dim=1)
        logger.info(
            "[BAM_WRITE_DEBUG] event=%d mappings=%d "
            "unique_src=%d unique_dst=%d duplicate_src=%d duplicate_dst=%d "
            "sample_src_to_dst=%s",
            self._debug_events,
            mapping_pairs.shape[0],
            int(torch.unique(src_blocks).numel()),
            int(torch.unique(dst_blocks).numel()),
            int(mapping_pairs.shape[0] - torch.unique(src_blocks).numel()),
            int(mapping_pairs.shape[0] - torch.unique(dst_blocks).numel()),
            mapping_pairs[:8].tolist(),
        )
        self._debug_events += 1

    def store_blocks(self, gpu_cache: List[torch.Tensor],
                     src_block_ids: torch.Tensor,
                     dst_block_ids: torch.Tensor) -> None:
        if src_block_ids.numel() == 0:
            return
        if src_block_ids.shape != dst_block_ids.shape:
            raise ValueError(
                "src_block_ids and dst_block_ids must have the same shape")

        self._log_mapping_debug(src_block_ids, dst_block_ids)

        start = time.perf_counter()
        total_rows = 0
        src_block_ids_cpu = src_block_ids.detach().cpu().tolist()
        dst_block_ids_cpu = dst_block_ids.detach().cpu().tolist()
        for src_block_id, dst_block_id in zip(src_block_ids_cpu,
                                              dst_block_ids_cpu):
            rows = self.codec.pack_single_block_rows(gpu_cache, src_block_id)
            row_offset = self.block_store.make_block_row_offset(dst_block_id)
            self.block_store.row_store.store_rows(rows, row_offset)
            total_rows += rows.shape[0]

        elapsed_s = time.perf_counter() - start
        total_bytes = src_block_ids.shape[0] * self.layout.block_bytes
        gib_per_s = (total_bytes / elapsed_s /
                     (1024**3)) if elapsed_s > 0 else 0.0
        logger.info(
            "[BAM_WRITE] mappings=%d rows=%d block_bytes=%d "
            "total_bytes=%d elapsed_ms=%.3f bw_gib_s=%.3f",
            src_block_ids.shape[0],
            total_rows,
            self.layout.block_bytes,
            total_bytes,
            elapsed_s * 1000,
            gib_per_s,
        )
