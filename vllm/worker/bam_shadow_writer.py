# SPDX-License-Identifier: Apache-2.0
import importlib
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.worker.bam_kv_layout import BaMKVLayout

logger = init_logger(__name__)


def _parse_optional_int_list(value: Optional[str]) -> Optional[List[int]]:
    if value is None or value.strip() == "":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _candidate_import_paths() -> Iterable[Path]:
    # 优先使用用户显式指定的 BaM Python 路径。
    if envs.VLLM_BAM_IMPORT_PATH:
        yield Path(envs.VLLM_BAM_IMPORT_PATH)

    llm_inference_dir = Path(__file__).resolve().parents[3]
    bam_gids_dir = llm_inference_dir / "BaM_IOStack" / "gids_module"
    yield bam_gids_dir

    if bam_gids_dir.exists():
        for build_dir in sorted(bam_gids_dir.glob("build*")):
            yield build_dir


def _import_bam_row_store():
    errors: List[str] = []
    for path in _candidate_import_paths():
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)
        try:
            module = importlib.import_module("bam_row_store")
            return module.BaMRowStore
        except Exception as exc:  # pragma: no cover - 这里只做环境探测
            errors.append(f"{path_str}: {type(exc).__name__}: {exc}")

    try:
        module = importlib.import_module("bam_row_store")
        return module.BaMRowStore
    except Exception as exc:
        errors.append(f"default sys.path: {type(exc).__name__}: {exc}")

    raise ImportError("Failed to import bam_row_store. Tried paths:\n" +
                      "\n".join(errors))


class BaMShadowWriter:
    """把 vLLM swap_out 的 GPU KV block 影子写到 BaM SSD。"""

    def __init__(self, gpu_cache: List[torch.Tensor], num_cpu_blocks: int) -> None:
        self.layout = BaMKVLayout.from_gpu_cache(gpu_cache)
        self.num_cpu_blocks = int(num_cpu_blocks)

        if self.num_cpu_blocks <= 0:
            raise ValueError(
                f"num_cpu_blocks must be positive, got {self.num_cpu_blocks}")

        bam_row_store_cls = _import_bam_row_store()
        ssd_list = _parse_optional_int_list(envs.VLLM_BAM_SSD_LIST)
        self.store = bam_row_store_cls(
            row_bytes=self.layout.row_bytes,
            num_rows=self.layout.num_layers * self.num_cpu_blocks,
            cache_size_mb=envs.VLLM_BAM_CACHE_SIZE_MB,
            num_ssd=envs.VLLM_BAM_NUM_SSD,
            ssd_list=ssd_list,
            ctrl_idx=envs.VLLM_BAM_CTRL_IDX,
        )

        logger.info(
            "[BAM_SHADOW] initialized row_bytes=%d num_layers=%d "
            "num_cpu_blocks=%d total_rows=%d cache_size_mb=%d num_ssd=%d",
            self.layout.row_bytes,
            self.layout.num_layers,
            self.num_cpu_blocks,
            self.layout.num_layers * self.num_cpu_blocks,
            envs.VLLM_BAM_CACHE_SIZE_MB,
            envs.VLLM_BAM_NUM_SSD,
        )

    def _pack_layer_rows(self, layer_cache: torch.Tensor,
                         gpu_block_ids: torch.Tensor) -> torch.Tensor:
        # 先按 block gather，再把 [2, N, hidden] 变成 [N, row_bytes]。
        packed = layer_cache.index_select(1, gpu_block_ids)
        packed = packed.permute(1, 0, 2).contiguous().view(gpu_block_ids.shape[0],
                                                           -1)
        return packed.view(torch.uint8).view(gpu_block_ids.shape[0], -1)

    def _pack_single_block_rows(self, gpu_cache: List[torch.Tensor],
                                gpu_block_id: int) -> torch.Tensor:
        # 一个 vLLM block 在 SSD 上对应 num_layers 行，每层一行 64KB。
        rows = []
        gpu_block_index = torch.tensor([gpu_block_id],
                                       device=gpu_cache[0].device,
                                       dtype=torch.long)
        for layer_cache in gpu_cache:
            row = self._pack_layer_rows(layer_cache, gpu_block_index)
            rows.append(row)
        return torch.cat(rows, dim=0)

    def on_swap_out(self, gpu_cache: List[torch.Tensor],
                    src_to_dst: torch.Tensor) -> None:
        if src_to_dst.numel() == 0:
            return

        start = time.perf_counter()
        total_rows = 0
        for mapping in src_to_dst.tolist():
            gpu_block_id, cpu_block_id = int(mapping[0]), int(mapping[1])
            rows = self._pack_single_block_rows(gpu_cache, gpu_block_id)
            row_offset = self.layout.row_offset(cpu_block_id, 0)
            self.store.store_rows(rows, row_offset)
            total_rows += rows.shape[0]

        elapsed_s = time.perf_counter() - start
        total_bytes = src_to_dst.shape[0] * self.layout.block_bytes
        gib_per_s = (total_bytes / elapsed_s / (1024**3)) if elapsed_s > 0 else 0.0
        logger.info(
            "[BAM_SHADOW] swap_out_shadow mappings=%d rows=%d block_bytes=%d "
            "total_bytes=%d elapsed_ms=%.3f bw_gib_s=%.3f",
            src_to_dst.shape[0],
            total_rows,
            self.layout.block_bytes,
            total_bytes,
            elapsed_s * 1000,
            gib_per_s,
        )
