# SPDX-License-Identifier: Apache-2.0
"""vLLM V0 与 BaM_IOStack resident MDS 之间的 KVStore connector。"""

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
        self._pending_read_handle = None
        self._pending_read_mapping: tuple[tuple[int, int], ...] | None = None
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
        logger.info(
            "[BAM_MDS] resident async service enabled after model warmup")

    def swap_out(self, src_to_dst: torch.Tensor) -> None:
        self._transfer_mapping(src_to_dst, operation="write")

    def swap_in(self, src_to_dst: torch.Tensor) -> None:
        self._transfer_mapping(src_to_dst, operation="read")

    def swap_in_async(self, src_to_dst: torch.Tensor) -> bool:
        """推进一个跨 engine iteration 存活的 logical swap-in。

        同一个 scheduler output 被 defer 后会在下一轮重试，因此这里用完整 block
        mapping 识别 live request，严禁重复提交。MDS daemon 内部负责按 native
        descriptor 容量拆批；只有所有 batch 完成后，本函数才返回 True。
        """
        if src_to_dst.numel() == 0:
            if self._pending_read_handle is not None:
                raise RuntimeError(
                    "empty mapping cannot replace a pending MDS read")
            return True
        mapping, payload = self._mapping_payload(src_to_dst, operation="read")
        if self._pending_read_handle is None:
            self._pending_read_handle = self.client.submit_read_async(payload)
            self._pending_read_mapping = mapping
            logger.debug(
                "[BAM_MDS] submitted async swap-in request_id=%d blocks=%d",
                self._pending_read_handle.request_id,
                len(mapping),
            )
        elif mapping != self._pending_read_mapping:
            raise RuntimeError(
                "MDS swap-in mapping changed while a request was in flight")

        if not self.client.poll(self._pending_read_handle):
            return False

        request_id = self._pending_read_handle.request_id
        io_elapsed_ns = self.client.finish(self._pending_read_handle)
        self._pending_read_handle = None
        self._pending_read_mapping = None
        logger.info(
            "[BAM_MDS] async swap-in done request_id=%d blocks=%d "
            "io_elapsed_ms=%.3f",
            request_id,
            len(mapping),
            io_elapsed_ns / 1.0e6,
        )
        return True

    def _transfer_mapping(self, src_to_dst: torch.Tensor, *, operation: str) -> None:
        if src_to_dst.numel() == 0:
            return
        if self._pending_read_handle is not None:
            raise RuntimeError("cannot run a blocking MDS transfer while read is pending")
        _, payload = self._mapping_payload(src_to_dst, operation=operation)
        if operation == "write":
            self.client.submit_write(payload)
        else:
            self.client.submit_read(payload)

    @staticmethod
    def _mapping_payload(
        src_to_dst: torch.Tensor,
        *,
        operation: str,
    ) -> tuple[tuple[tuple[int, int], ...], dict[str, list[int]]]:
        mappings = tuple(
            (int(source), int(destination))
            for source, destination in src_to_dst.to(
                device="cpu", dtype=torch.int64).tolist())
        source_ids = [mapping[0] for mapping in mappings]
        destination_ids = [mapping[1] for mapping in mappings]
        if operation == "write":
            gpu_ids, storage_ids = source_ids, destination_ids
        elif operation == "read":
            storage_ids, gpu_ids = source_ids, destination_ids
        else:
            raise ValueError(f"unsupported MDS operation: {operation}")
        return mappings, {
            "gpu_block_ids": gpu_ids,
            "storage_block_ids": storage_ids,
        }
