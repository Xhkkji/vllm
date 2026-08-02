# SPDX-License-Identifier: Apache-2.0
"""vLLM V0 与 BaM_IOStack MDSClient 之间的同步 KVStore connector。"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
from typing import Sequence

import torch

import vllm.envs as envs
from vllm.bam.mds.kv_layout import VLLMKVLayout
from vllm.logger import init_logger


logger = init_logger(__name__)


class BaMMDSConnector:
    """只解释 vLLM KV layout/mapping；IPC 和 request 状态由 MDSClient 持有。"""

    def __init__(
        self,
        *,
        allocation_shape: Sequence[int],
        stride_order: Sequence[int],
        dtype: torch.dtype,
        device_index: int,
        num_layers: int,
        num_gpu_blocks: int,
        num_storage_blocks: int,
    ) -> None:
        if not envs.VLLM_BAM_MDS_IOSTACK_ROOT:
            raise ValueError("VLLM_BAM_MDS_IOSTACK_ROOT is required")
        if not envs.VLLM_BAM_MDS_CONTROL_DIR:
            raise ValueError("VLLM_BAM_MDS_CONTROL_DIR is required")

        self.iostack_root = Path(envs.VLLM_BAM_MDS_IOSTACK_ROOT).resolve()
        gids_module = self.iostack_root / "gids_module"
        if str(gids_module) not in sys.path:
            sys.path.insert(0, str(gids_module))
        client_module = importlib.import_module("bam_mds.client")

        self.layout = VLLMKVLayout.from_allocation(
            allocation_shape=allocation_shape,
            stride_order=stride_order,
            dtype=dtype,
            num_layers=num_layers,
            num_gpu_blocks=num_gpu_blocks,
            num_storage_blocks=num_storage_blocks,
            device_index=device_index)
        self.bridge = self._load_torch_bridge()
        self.client = client_module.MDSClient(
            control_dir=Path(envs.VLLM_BAM_MDS_CONTROL_DIR).resolve(),
            cuda_library=self._resolve_cuda_library(),
            device_index=device_index,
            timeout_seconds=float(envs.VLLM_BAM_MDS_TIMEOUT_SECONDS))
        self.max_blocks_per_batch = 0
        self.gpu_cache: list[torch.Tensor] = []
        try:
            self.gpu_cache = self._allocate_and_wrap()
        except Exception:
            self.client.publish_error()
            self.client.close_imports()
            raise

    def _resolve_cuda_library(self) -> Path:
        if envs.VLLM_BAM_MDS_CUDA_IPC_LIBRARY:
            return Path(envs.VLLM_BAM_MDS_CUDA_IPC_LIBRARY).resolve()
        return (self.iostack_root / "gids_module" / "bam_mds" / "build"
                / "libmds_cuda_ipc.so")

    def _load_torch_bridge(self):
        if envs.VLLM_BAM_MDS_TORCH_BRIDGE_DIR:
            bridge_dir = Path(envs.VLLM_BAM_MDS_TORCH_BRIDGE_DIR).resolve()
        else:
            bridge_dir = Path(__file__).resolve().parent / "build" / "torch_bridge"
        if str(bridge_dir) not in sys.path:
            sys.path.insert(0, str(bridge_dir))
        return importlib.import_module("mds_torch_bridge")

    def _allocate_and_wrap(self) -> list[torch.Tensor]:
        result = self.client.allocate(
            self.layout.allocation_payload(client_pid=os.getpid()))
        self.layout.validate_manifest(result.manifest)
        self.max_blocks_per_batch = int(
            result.manifest["max_blocks_per_batch"])
        tensors: list[torch.Tensor] = []
        for region in result.regions:
            tensor = self.bridge.tensor_from_cuda_ptr(
                region.region_ptr,
                list(self.layout.tensor_shape),
                list(self.layout.tensor_strides),
                self.layout.dtype_name,
                self.layout.device_index)
            if (tensor.data_ptr() != region.region_ptr
                    or tuple(tensor.shape) != self.layout.tensor_shape
                    or tuple(tensor.stride()) != self.layout.tensor_strides
                    or region.region_bytes != self.layout.region_bytes):
                raise RuntimeError("imported MDS tensor metadata mismatch")
            tensors.append(tensor)

        # daemon 使用 cudaMalloc，必须由 client 显式恢复原生 CacheEngine 的
        # torch.zeros 语义，尤其是 null block 和 warmup 前的初始状态。
        for tensor in tensors:
            tensor.zero_()
        torch.cuda.synchronize(self.layout.device_index)
        logger.info(
            "[BAM_MDS] imported daemon KV cache layers=%d gpu_blocks=%d "
            "storage_blocks=%d fragment_bytes=%d max_blocks_per_batch=%d",
            self.layout.num_layers, self.layout.num_gpu_blocks,
            self.layout.num_storage_blocks, self.layout.fragment_bytes,
            self.max_blocks_per_batch)
        return tensors

    def start(self) -> None:
        self.client.start()
        logger.info("[BAM_MDS] synchronous service enabled after model warmup")

    def swap_out(self, src_to_dst: torch.Tensor) -> None:
        self._transfer_mapping(src_to_dst, operation="write")

    def swap_in(self, src_to_dst: torch.Tensor) -> None:
        self._transfer_mapping(src_to_dst, operation="read")

    def _transfer_mapping(self, src_to_dst: torch.Tensor, *, operation: str) -> None:
        if src_to_dst.numel() == 0:
            return
        mappings = src_to_dst.to(device="cpu", dtype=torch.int64).tolist()
        for begin in range(0, len(mappings), self.max_blocks_per_batch):
            batch = mappings[begin:begin + self.max_blocks_per_batch]
            source_ids = [int(mapping[0]) for mapping in batch]
            destination_ids = [int(mapping[1]) for mapping in batch]
            if operation == "write":
                gpu_ids, storage_ids = source_ids, destination_ids
            elif operation == "read":
                storage_ids, gpu_ids = source_ids, destination_ids
            else:
                raise ValueError(f"unsupported MDS operation: {operation}")
            payload = {
                "gpu_block_ids": gpu_ids,
                "storage_block_ids": storage_ids,
            }
            if operation == "write":
                self.client.submit_write(payload)
            else:
                self.client.submit_read(payload)
