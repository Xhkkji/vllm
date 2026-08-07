# SPDX-License-Identifier: Apache-2.0
"""vLLM V0 与 BaM_IOStack resident MDS 之间的 KVStore connector。"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Sequence

import torch

import vllm.envs as envs
from vllm.bam.mds.kv_layout import VLLMKVLayout
from vllm.logger import init_logger


logger = init_logger(__name__)


@dataclass
class _PendingTransfer:
    """Connector 对一笔 logical transfer 的本地控制面记录。"""

    handle: object
    mapping: tuple[tuple[int, int], ...]
    operation: str
    submitted_at_ns: int


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
        self.descriptor_pool_capacity = 0
        self.max_blocks_per_pool = 0
        self.max_in_flight = 0
        self.gpu_cache: list[torch.Tensor] = []
        # scheduler request_id 与底层 MDS handle 属于两个命名空间；每个
        # scheduler request 独立保存 mapping 和 handle，completion 可乱序。
        self._pending_transfers: dict[str, _PendingTransfer] = {}
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
        self.descriptor_pool_capacity = int(
            result.manifest["descriptor_pool_capacity"])
        self.max_blocks_per_pool = int(
            result.manifest["max_blocks_per_pool"])
        self.max_in_flight = int(result.manifest["request_slot_count"])
        if self.max_in_flight != envs.VLLM_BAM_MDS_MAX_IN_FLIGHT:
            raise RuntimeError(
                "MDS request-slot mismatch: daemon="
                f"{self.max_in_flight}, vLLM="
                f"{envs.VLLM_BAM_MDS_MAX_IN_FLIGHT}")
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
            "storage_blocks=%d fragment_bytes=%d pool_capacity=%d "
            "max_blocks_per_pool=%d max_in_flight=%d",
            self.layout.num_layers, self.layout.num_gpu_blocks,
            self.layout.num_storage_blocks, self.layout.fragment_bytes,
            self.descriptor_pool_capacity, self.max_blocks_per_pool,
            self.max_in_flight)
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

        这是原有 DeferredModelExecution 路径的兼容入口。新的
        AsyncKVScheduler 使用显式 request_id 调用通用 transfer 接口；旧
        DeferredModelExecution 路径仍使用固定 legacy request_id，并保证
        同一份 read mapping 不会被重复提交。
        """
        legacy_request_id = "legacy-deferred-swap-in"
        if "legacy-deferred-swap-in" not in self._pending_transfers:
            if self.submit_transfer_async(
                    legacy_request_id, src_to_dst, operation="read"):
                return True
        return self.poll_transfer_async(legacy_request_id)

    def submit_transfer_async(self, scheduler_request_id: str,
                              src_to_dst: torch.Tensor, *,
                              operation: str) -> bool:
        """提交一笔 MDS logical read/write，不等待完成。

        返回 True 仅表示 mapping 为空、无需 I/O；正常提交返回 False。MDS
        request table 负责 slot backpressure，本层只维护 request identity。
        """
        if not scheduler_request_id:
            raise ValueError("scheduler_request_id must not be empty")
        if operation not in ("read", "write"):
            raise ValueError(f"unsupported async MDS operation: {operation}")
        if src_to_dst.numel() == 0:
            return True
        if scheduler_request_id in self._pending_transfers:
            pending = self._pending_transfers[scheduler_request_id]
            mapping, _ = self._mapping_payload(src_to_dst,
                                                operation=operation)
            if (mapping != pending.mapping
                    or operation != pending.operation):
                raise RuntimeError(
                    "MDS transfer changed while request was in flight")
            return False

        mapping, payload = self._mapping_payload(src_to_dst,
                                                 operation=operation)
        submit = (self.client.submit_read_async
                  if operation == "read" else self.client.submit_write_async)
        handle = submit(payload)
        submitted_at_ns = time.monotonic_ns()
        self._pending_transfers[scheduler_request_id] = _PendingTransfer(
            handle=handle,
            mapping=mapping,
            operation=operation,
            submitted_at_ns=submitted_at_ns)
        if envs.VLLM_V0_SWAP_TRACE:
            logger.info(
                "[V0_SWAP_TRACE][AsyncKV][Connector] phase=submit "
                "operation=%s scheduler_request_id=%s mds_request_id=%d "
                "submit_monotonic_ns=%d blocks=%d",
                operation,
                scheduler_request_id,
                handle.request_id,
                submitted_at_ns,
                len(mapping),
            )
        logger.debug(
            "[BAM_MDS] submitted async %s scheduler_request_id=%s "
            "mds_request_id=%d blocks=%d",
            operation,
            scheduler_request_id,
            handle.request_id,
            len(mapping),
        )
        return False

    def poll_transfer_async(self, scheduler_request_id: str) -> bool:
        """非阻塞查询指定 scheduler request 的 MDS 完成状态。"""
        pending = self._pending_transfers.get(scheduler_request_id)
        if pending is None:
            raise RuntimeError(
                "cannot poll MDS transfer without a pending request")

        try:
            if not self.client.poll(pending.handle):
                return False
        except Exception:
            # completion 已经失败，后续不能再用同一个 scheduler request
            # 轮询；释放 client/connector identity，由 Scheduler abort block
            # reservation。daemon 若进入全局 ERROR 会阻止后续 submit。
            self.client.discard(pending.handle)
            del self._pending_transfers[scheduler_request_id]
            raise

        request_id = pending.handle.request_id
        mapping = pending.mapping
        operation = pending.operation
        observed_done_ns = time.monotonic_ns()
        io_elapsed_ns = self.client.finish(pending.handle)
        del self._pending_transfers[scheduler_request_id]
        if envs.VLLM_V0_SWAP_TRACE:
            logger.info(
                "[V0_SWAP_TRACE][AsyncKV][Connector] phase=ready "
                "operation=%s scheduler_request_id=%s mds_request_id=%d "
                "control_done_observed_monotonic_ns=%d "
                "submit_to_control_done_ms=%.3f io_elapsed_ms=%.3f",
                operation,
                scheduler_request_id,
                request_id,
                observed_done_ns,
                (observed_done_ns - pending.submitted_at_ns) / 1.0e6,
                io_elapsed_ns / 1.0e6,
            )
        logger.info(
            "[BAM_MDS] async %s done scheduler_request_id=%s "
            "mds_request_id=%d blocks=%d io_elapsed_ms=%.3f",
            operation,
            scheduler_request_id,
            request_id,
            len(mapping),
            io_elapsed_ns / 1.0e6,
        )
        return True

    def _transfer_mapping(self, src_to_dst: torch.Tensor, *, operation: str) -> None:
        if src_to_dst.numel() == 0:
            return
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
