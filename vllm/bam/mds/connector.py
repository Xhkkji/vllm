# SPDX-License-Identifier: Apache-2.0
"""vLLM V0 与 BaM_IOStack resident MDS 之间的 KVStore connector。"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any, Optional, Sequence, Tuple

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
    layer_range: Optional[Tuple[int, int]]
    submitted_at_ns: int
    prefetch_plan_id: Optional[str] = None


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
        num_gpu_regions: int,
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
            num_gpu_regions=num_gpu_regions,
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
        # request_id -> (plan_id, mapping, operation, layer_range)。模板登记不
        # 占 MDS slot，model progress 只能激活这里已经存在的 request。
        self._prefetch_templates: dict[
            str, tuple[str, tuple[tuple[int, int], ...], str,
                       Optional[Tuple[int, int]]]] = {}
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
        physical_regions: list[torch.Tensor] = []
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
            physical_regions.append(tensor)

        # daemon 使用 cudaMalloc，必须由 client 显式恢复原生 CacheEngine 的
        # torch.zeros 语义，尤其是 null block 和 warmup 前的初始状态。
        for tensor in physical_regions:
            tensor.zero_()
        torch.cuda.synchronize(self.layout.device_index)
        # Attention 仍按真实模型层索引 KV tensor；工作集模式把多个模型层绑定
        # 到同一个环形 region。MDS 会在每个 layer barrier 前写入该层数据，
        # 因而模型和 attention backend 都无需理解 region 生命周期。
        tensors = [
            physical_regions[layer % self.layout.num_gpu_regions]
            for layer in range(self.layout.num_layers)
        ]
        logger.info(
            "[BAM_MDS] imported daemon KV cache layers=%d gpu_regions=%d "
            "gpu_blocks=%d "
            "storage_blocks=%d fragment_bytes=%d pool_capacity=%d "
            "max_blocks_per_pool=%d max_in_flight=%d",
            self.layout.num_layers, self.layout.num_gpu_regions,
            self.layout.num_gpu_blocks,
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
                              operation: str,
                              layer_range: Optional[Tuple[int, int]] = None
                              ) -> bool:
        """提交一笔 MDS logical read/write，不等待完成。

        返回 True 仅表示 mapping 为空、无需 I/O；正常提交返回 False。MDS
        request table 负责 slot backpressure，本层只维护 request identity。
        """
        if not scheduler_request_id:
            raise ValueError("scheduler_request_id must not be empty")
        if operation not in ("read", "write"):
            raise ValueError(f"unsupported async MDS operation: {operation}")
        layer_range = self._validate_layer_range(layer_range)
        if src_to_dst.numel() == 0:
            return True
        if scheduler_request_id in self._pending_transfers:
            pending = self._pending_transfers[scheduler_request_id]
            mapping, _ = self._mapping_payload(src_to_dst,
                                                operation=operation,
                                                layer_range=layer_range)
            if (mapping != pending.mapping
                    or operation != pending.operation
                    or layer_range != pending.layer_range):
                raise RuntimeError(
                    "MDS transfer changed while request was in flight")
            return False

        mapping, payload = self._mapping_payload(src_to_dst,
                                                 operation=operation,
                                                 layer_range=layer_range)
        submit = (self.client.submit_read_async
                  if operation == "read" else self.client.submit_write_async)
        handle = submit(payload)
        submitted_at_ns = time.monotonic_ns()
        self._pending_transfers[scheduler_request_id] = _PendingTransfer(
            handle=handle,
            mapping=mapping,
            operation=operation,
            layer_range=layer_range,
            submitted_at_ns=submitted_at_ns)
        if envs.VLLM_V0_SWAP_TRACE:
            logger.info(
                "[V0_SWAP_TRACE][AsyncKV][Connector] phase=submit "
                "operation=%s scheduler_request_id=%s mds_request_id=%d "
                "submit_monotonic_ns=%d blocks=%d layer_range=%s",
                operation,
                scheduler_request_id,
                handle.request_id,
                submitted_at_ns,
                len(mapping),
                layer_range,
            )
        logger.debug(
            "[BAM_MDS] submitted async %s scheduler_request_id=%s "
            "mds_request_id=%d blocks=%d layer_range=%s",
            operation,
            scheduler_request_id,
            handle.request_id,
            len(mapping),
            layer_range,
        )
        return False

    def stage_prefetch_plan(
        self,
        plan_id: str,
        units: Sequence[tuple[str, torch.Tensor, str,
                              Optional[Tuple[int, int]]]],
    ) -> None:
        """一次登记完整 plan 的 descriptor templates，不提交 I/O。"""
        staged: dict[str, tuple[dict[str, Any], str]] = {}
        for request_id, mapping_tensor, operation, layer_range in units:
            if operation not in ("read", "write"):
                raise ValueError(f"unsupported MDS operation: {operation}")
            layer_range = self._validate_layer_range(layer_range)
            mapping, payload = self._mapping_payload(
                mapping_tensor, operation=operation, layer_range=layer_range)
            if request_id in self._prefetch_templates:
                raise RuntimeError(f"duplicate staged prefetch unit: {request_id}")
            staged[request_id] = (payload, operation)
            self._prefetch_templates[request_id] = (
                plan_id, mapping, operation, layer_range)
        try:
            self.client.register_prefetch_plan(plan_id, staged)
        except Exception:
            for request_id in staged:
                self._prefetch_templates.pop(request_id, None)
            raise
        if envs.VLLM_V0_SWAP_TRACE:
            logger.info(
                "[BAM_MDS_PREFETCH][Connector] phase=plan_staged "
                "plan_id=%s units=%d",
                plan_id,
                len(staged),
            )

    def activate_prefetch_transfer_async(
        self,
        plan_id: str,
        scheduler_request_id: str,
        src_to_dst: torch.Tensor,
        *,
        operation: str,
        layer_range: Optional[Tuple[int, int]],
    ) -> bool:
        """激活一个已 stage unit；禁止 forward 临时更改 mapping。"""
        template = self._prefetch_templates.get(scheduler_request_id)
        if template is None or template[0] != plan_id:
            raise RuntimeError("unknown staged prefetch unit")
        layer_range = self._validate_layer_range(layer_range)
        mapping, _ = self._mapping_payload(src_to_dst,
                                            operation=operation,
                                            layer_range=layer_range)
        if (mapping, operation, layer_range) != template[1:]:
            raise RuntimeError("prefetch unit changed after plan stage")
        try:
            handle = self.client.activate_prefetch_units(
                plan_id, (scheduler_request_id, ))[0]
        except Exception:
            # submit 失败发生在 unit 尚未拥有 active handle 时，可以安全删除
            # 模板；Scheduler 会把该 unit 作为 ERROR 推进父事务。
            self.client.discard_prefetch_units(plan_id,
                                               (scheduler_request_id, ))
            del self._prefetch_templates[scheduler_request_id]
            self._maybe_release_prefetch_plan(plan_id)
            raise
        self._pending_transfers[scheduler_request_id] = _PendingTransfer(
            handle=handle,
            mapping=mapping,
            operation=operation,
            layer_range=layer_range,
            submitted_at_ns=time.monotonic_ns(),
            prefetch_plan_id=plan_id,
        )
        return False

    def discard_staged_prefetch_units(
        self,
        scheduler_request_ids: Sequence[str],
    ) -> None:
        by_plan: dict[str, list[str]] = {}
        for request_id in scheduler_request_ids:
            template = self._prefetch_templates.get(request_id)
            if template is None:
                raise RuntimeError(f"unknown staged prefetch unit: {request_id}")
            by_plan.setdefault(template[0], []).append(request_id)
        for plan_id, request_ids in by_plan.items():
            self.client.discard_prefetch_units(plan_id, tuple(request_ids))
            for request_id in request_ids:
                del self._prefetch_templates[request_id]
            self._maybe_release_prefetch_plan(plan_id)

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
            if pending.prefetch_plan_id is None:
                self.client.discard(pending.handle)
            else:
                self.client.fail_prefetch_unit(pending.prefetch_plan_id,
                                               scheduler_request_id)
                del self._prefetch_templates[scheduler_request_id]
                self._maybe_release_prefetch_plan(
                    pending.prefetch_plan_id)
            del self._pending_transfers[scheduler_request_id]
            raise

        request_id = pending.handle.request_id
        mapping = pending.mapping
        operation = pending.operation
        observed_done_ns = time.monotonic_ns()
        if pending.prefetch_plan_id is None:
            io_elapsed_ns = self.client.finish(pending.handle)
        else:
            io_elapsed_ns = self.client.finish_prefetch_unit(
                pending.prefetch_plan_id, scheduler_request_id)
        del self._pending_transfers[scheduler_request_id]
        if pending.prefetch_plan_id is not None:
            del self._prefetch_templates[scheduler_request_id]
            self._maybe_release_prefetch_plan(pending.prefetch_plan_id)
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
            "mds_request_id=%d blocks=%d layer_range=%s io_elapsed_ms=%.3f",
            operation,
            scheduler_request_id,
            request_id,
            len(mapping),
            pending.layer_range,
            io_elapsed_ns / 1.0e6,
        )
        return True

    def _maybe_release_prefetch_plan(self, plan_id: str) -> None:
        if any(template[0] == plan_id
               for template in self._prefetch_templates.values()):
            return
        self.client.release_prefetch_plan(plan_id)
        if envs.VLLM_V0_SWAP_TRACE:
            logger.info(
                "[BAM_MDS_PREFETCH][Connector] phase=plan_released "
                "plan_id=%s",
                plan_id,
            )

    def _transfer_mapping(self, src_to_dst: torch.Tensor, *, operation: str) -> None:
        if src_to_dst.numel() == 0:
            return
        _, payload = self._mapping_payload(src_to_dst, operation=operation)
        if operation == "write":
            self.client.submit_write(payload)
        else:
            self.client.submit_read(payload)

    def _mapping_payload(
        self,
        src_to_dst: torch.Tensor,
        *,
        operation: str,
        layer_range: Optional[Tuple[int, int]] = None,
    ) -> tuple[tuple[tuple[int, int], ...], dict[str, Any]]:
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
        payload: dict[str, Any] = {
            "gpu_block_ids": gpu_ids,
            "storage_block_ids": storage_ids,
        }
        if layer_range is not None:
            payload["layer_start"] = layer_range[0]
            payload["layer_end"] = layer_range[1]
            # layer window 起点按 region pool 取模。region 数是 window 大小的
            # 整数倍，因此一个 window 不会在内部回绕；后端会据此覆盖已消费槽。
            num_gpu_regions = getattr(self.layout, "num_gpu_regions",
                                      self.layout.num_layers)
            if num_gpu_regions < self.layout.num_layers:
                payload["gpu_region_start"] = (
                    layer_range[0] % num_gpu_regions)
        elif (getattr(getattr(self, "layout", None), "num_gpu_regions",
                      getattr(getattr(self, "layout", None), "num_layers",
                              0))
              < getattr(getattr(self, "layout", None), "num_layers", 0)):
            raise RuntimeError(
                "layer working-set mode only supports layer-ranged MDS I/O")
        return mappings, payload

    def _validate_layer_range(
        self,
        layer_range: Optional[Tuple[int, int]],
    ) -> Optional[Tuple[int, int]]:
        """尽早拒绝越界 window，避免 daemon 已开始部分 DMA 后才报错。"""
        if layer_range is None:
            return None
        start_layer, end_layer = (int(layer_range[0]), int(layer_range[1]))
        if not 0 <= start_layer < end_layer <= self.layout.num_layers:
            raise ValueError(
                "MDS layer range is outside local KV cache: "
                f"[{start_layer}, {end_layer}) vs {self.layout.num_layers}")
        return start_layer, end_layer
