# SPDX-License-Identifier: Apache-2.0
import time
from typing import List, Optional

import torch

import vllm.envs as envs
from vllm.bam.block_codec import BaMBlockCodec
from vllm.bam.block_store import BaMBlockStore
from vllm.logger import init_logger

logger = init_logger(__name__)


class BaMBlockReader:
    """从 BaM 读取 KV block，并回填到 vLLM GPU cache。"""

    def __init__(self, block_store: BaMBlockStore, dtype: torch.dtype) -> None:
        self.block_store = block_store
        self.layout = block_store.layout
        self.device = block_store.device
        self.codec = BaMBlockCodec(self.layout, dtype)
        self.verify_enabled = envs.VLLM_BAM_SWAPIN_VERIFY
        self.verify_blocks = envs.VLLM_BAM_SWAPIN_VERIFY_BLOCKS
        self._debug_events = 0

    def _log_mapping_debug(self, src_block_ids: torch.Tensor,
                           dst_block_ids: torch.Tensor) -> None:
        if not self.verify_enabled or self._debug_events >= 4:
            return

        src_blocks = src_block_ids.detach().cpu()
        dst_blocks = dst_block_ids.detach().cpu()
        mapping_pairs = torch.stack([src_blocks, dst_blocks], dim=1)
        logger.info(
            "[BAM_READ_DEBUG] event=%d mappings=%d "
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

    def _load_block_rows(self, src_block_ids: torch.Tensor) -> torch.Tensor:
        row_ids = self.block_store.make_block_row_ids(src_block_ids)
        rows = torch.empty((int(row_ids.shape[0]), self.layout.row_bytes),
                           device=self.device,
                           dtype=torch.uint8)
        self.block_store.row_store.load_rows(row_ids, rows)
        return rows

    def load_blocks(self,
                    gpu_cache: List[torch.Tensor],
                    src_block_ids: torch.Tensor,
                    dst_block_ids: torch.Tensor,
                    reference_cache: Optional[List[torch.Tensor]] = None) -> None:
        if src_block_ids.numel() == 0:
            return
        if src_block_ids.shape != dst_block_ids.shape:
            raise ValueError(
                "src_block_ids and dst_block_ids must have the same shape")

        start = time.perf_counter()
        src_block_ids = src_block_ids.to(device=self.device, dtype=torch.long)
        dst_block_ids = dst_block_ids.to(device=self.device, dtype=torch.long)
        verify_count = self.codec.get_verify_count(
            int(src_block_ids.shape[0]), self.verify_enabled,
            self.verify_blocks)
        self._log_mapping_debug(src_block_ids, dst_block_ids)

        rows = self._load_block_rows(src_block_ids)
        rows = rows.view(int(src_block_ids.shape[0]), self.layout.num_layers,
                         self.layout.row_bytes)

        for layer_id, layer_cache in enumerate(gpu_cache):
            layer_blocks = self.codec.restore_layer_blocks(rows[:, layer_id, :])
            layer_cache.index_copy_(1, dst_block_ids, layer_blocks)
            if reference_cache is not None:
                self.codec.verify_layer_restore(layer_id,
                                               reference_cache[layer_id],
                                               layer_cache,
                                               src_block_ids,
                                               dst_block_ids,
                                               verify_count)

        elapsed_s = time.perf_counter() - start
        total_bytes = src_block_ids.shape[0] * self.layout.block_bytes
        gib_per_s = (total_bytes / elapsed_s /
                     (1024**3)) if elapsed_s > 0 else 0.0
        verify_mode = self.codec.get_verify_mode(verify_count,
                                                 int(src_block_ids.shape[0]))
        if verify_mode is not None:
            logger.info(
                "[BAM_VERIFY] mode=%s checked_blocks=%d "
                "checked_layers=%d checked_layer_blocks=%d exact=1",
                verify_mode,
                verify_count,
                self.layout.num_layers,
                verify_count * self.layout.num_layers,
            )
        logger.info(
            "[BAM_READ] mappings=%d rows=%d block_bytes=%d total_bytes=%d "
            "elapsed_ms=%.3f bw_gib_s=%.3f",
            src_block_ids.shape[0],
            rows.shape[0],
            self.layout.block_bytes,
            total_bytes,
            elapsed_s * 1000,
            gib_per_s,
        )
