# SPDX-License-Identifier: Apache-2.0
"""vLLM V0 与 BaM_IOStack resident MDS 之间的 KVStore connector。"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
import time
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
        # scheduler request_id 与底层 MDS handle 的 request_id 属于两个
        # 命名空间。前者用于 Engine/Worker 事件关联，后者由 MDSClient
        # 写入控制槽；必须同时保存，不能用其中一个替代另一个。
        self._pending_read_scheduler_request_id: str | None = None
        self._pending_read_submitted_at_ns: int | None = None
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

        这是原有 DeferredModelExecution 路径的兼容入口。新的
        AsyncKVScheduler 使用显式 request_id 调用 ``submit_swap_in_async``
        和 ``poll_swap_in_async``；旧路径则使用固定 legacy request_id，仍然
        保证同一份 mapping 不会被重复提交。
        """
        legacy_request_id = "legacy-deferred-swap-in"
        if self._pending_read_handle is None:
            if self.submit_swap_in_async(legacy_request_id, src_to_dst):
                return True
        return self.poll_swap_in_async(legacy_request_id)

    def submit_swap_in_async(self, scheduler_request_id: str,
                             src_to_dst: torch.Tensor) -> bool:
        """只提交一次 MDS logical read，不等待完成。

        返回 True 仅表示 mapping 为空、无需 I/O；正常提交返回 False。
        当前 resident MDS 是单槽协议，因此已有 live request 时只接受完全
        相同的 scheduler request_id 和 block mapping，其他请求立即报错，
        防止新请求覆盖正在使用的控制槽。
        """
        if not scheduler_request_id:
            raise ValueError("scheduler_request_id must not be empty")
        if src_to_dst.numel() == 0:
            if self._pending_read_handle is not None:
                raise RuntimeError(
                    "empty mapping cannot replace a pending MDS read")
            return True

        mapping, payload = self._mapping_payload(src_to_dst, operation="read")
        if self._pending_read_handle is not None:
            if (scheduler_request_id !=
                    self._pending_read_scheduler_request_id):
                raise RuntimeError(
                    "MDS single-slot read already belongs to another "
                    "scheduler request")
            if mapping != self._pending_read_mapping:
                raise RuntimeError(
                    "MDS swap-in mapping changed while a request was in "
                    "flight")
            return False

        self._pending_read_handle = self.client.submit_read_async(payload)
        self._pending_read_mapping = mapping
        self._pending_read_scheduler_request_id = scheduler_request_id
        self._pending_read_submitted_at_ns = time.monotonic_ns()
        if envs.VLLM_V0_SWAP_TRACE:
            logger.info(
                "[V0_SWAP_TRACE][AsyncKV][Connector] phase=submit "
                "scheduler_request_id=%s mds_request_id=%d "
                "submit_monotonic_ns=%d blocks=%d",
                scheduler_request_id,
                self._pending_read_handle.request_id,
                self._pending_read_submitted_at_ns,
                len(mapping),
            )
        logger.debug(
            "[BAM_MDS] submitted async swap-in scheduler_request_id=%s "
            "mds_request_id=%d blocks=%d",
            scheduler_request_id,
            self._pending_read_handle.request_id,
            len(mapping),
        )
        return False

    def poll_swap_in_async(self, scheduler_request_id: str) -> bool:
        """非阻塞查询指定 scheduler request 的 MDS 完成状态。

        PENDING 时返回 False，不修改控制槽；DONE 时调用 MDSClient.finish
        归还单槽，并清除 connector 内的三项身份记录。只有本方法返回 True
        后，Scheduler 才能把对应 sequence group 提升到 running。
        """
        if self._pending_read_handle is None:
            raise RuntimeError(
                "cannot poll MDS swap-in without a pending request")
        if scheduler_request_id != self._pending_read_scheduler_request_id:
            raise RuntimeError(
                "MDS swap-in scheduler request identity mismatch")

        if not self.client.poll(self._pending_read_handle):
            return False

        request_id = self._pending_read_handle.request_id
        mapping = self._pending_read_mapping
        assert mapping is not None
        observed_done_ns = time.monotonic_ns()
        io_elapsed_ns = self.client.finish(self._pending_read_handle)
        self._pending_read_handle = None
        self._pending_read_mapping = None
        self._pending_read_scheduler_request_id = None
        submitted_at_ns = self._pending_read_submitted_at_ns
        self._pending_read_submitted_at_ns = None
        if envs.VLLM_V0_SWAP_TRACE:
            logger.info(
                "[V0_SWAP_TRACE][AsyncKV][Connector] phase=ready "
                "scheduler_request_id=%s mds_request_id=%d "
                "control_done_observed_monotonic_ns=%d "
                "submit_to_control_done_ms=%.3f io_elapsed_ms=%.3f",
                scheduler_request_id,
                request_id,
                observed_done_ns,
                ((observed_done_ns - submitted_at_ns) / 1.0e6
                 if submitted_at_ns is not None else -1.0),
                io_elapsed_ns / 1.0e6,
            )
        logger.info(
            "[BAM_MDS] async swap-in done scheduler_request_id=%s "
            "mds_request_id=%d blocks=%d io_elapsed_ms=%.3f",
            scheduler_request_id,
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
