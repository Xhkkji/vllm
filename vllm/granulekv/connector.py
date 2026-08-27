# SPDX-License-Identifier: Apache-2.0
"""vLLM control-plane connector for the independent GranuleKV I/O path."""

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
from vllm.granulekv.kv_layout import GranuleKVLayout
from vllm.logger import init_logger


logger = init_logger(__name__)


@dataclass
class _PendingTransfer:
    handle: object
    mapping: tuple[tuple[int, int], ...]
    operation: str
    layer_range: Optional[Tuple[int, int]]
    submitted_at_ns: int
    prefetch_plan_id: Optional[str] = None


class GranuleKVConnector:
    """Translate vLLM block requests into GranuleKV control-plane requests."""

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
        if not envs.VLLM_GRANULEKV_IOSTACK_ROOT:
            raise ValueError("VLLM_GRANULEKV_IOSTACK_ROOT is required")
        if not envs.VLLM_GRANULEKV_CONTROL_DIR:
            raise ValueError("VLLM_GRANULEKV_CONTROL_DIR is required")

        self.iostack_root = Path(envs.VLLM_GRANULEKV_IOSTACK_ROOT).resolve()
        gids_module = self.iostack_root / "gids_module"
        if str(gids_module) not in sys.path:
            sys.path.insert(0, str(gids_module))
        client_module = importlib.import_module("granulekv.client")
        self.layout = GranuleKVLayout.from_allocation(
            allocation_shape=allocation_shape,
            stride_order=stride_order,
            dtype=dtype,
            num_layers=num_layers,
            num_gpu_regions=num_gpu_regions,
            num_gpu_blocks=num_gpu_blocks,
            num_storage_blocks=num_storage_blocks,
            device_index=device_index,
        )
        self.bridge = self._load_torch_bridge()
        self.client = client_module.GranuleKVClient(
            control_dir=Path(envs.VLLM_GRANULEKV_CONTROL_DIR).resolve(),
            cuda_library=self._resolve_cuda_library(),
            device_index=device_index,
            timeout_seconds=float(envs.VLLM_GRANULEKV_TIMEOUT_SECONDS),
        )
        self.descriptor_pool_capacity = 0
        self.max_blocks_per_pool = 0
        self.max_in_flight = 0
        self.gpu_cache: list[torch.Tensor] = []
        self._pending_transfers: dict[str, _PendingTransfer] = {}
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
        if envs.VLLM_GRANULEKV_CUDA_IPC_LIBRARY:
            return Path(envs.VLLM_GRANULEKV_CUDA_IPC_LIBRARY).resolve()
        return (self.iostack_root / "gids_module" / "build"
                / "libgranulekv_cuda_ipc.so")

    def _load_torch_bridge(self):
        if envs.VLLM_GRANULEKV_TORCH_BRIDGE_DIR:
            bridge_dir = Path(envs.VLLM_GRANULEKV_TORCH_BRIDGE_DIR).resolve()
        else:
            bridge_dir = Path(__file__).resolve().parent / "build" / "torch_bridge"
        if str(bridge_dir) not in sys.path:
            sys.path.insert(0, str(bridge_dir))
        return importlib.import_module("granulekv_torch_bridge")

    def _allocate_and_wrap(self) -> list[torch.Tensor]:
        result = self.client.allocate(
            self.layout.allocation_payload(client_pid=os.getpid()))
        self.layout.validate_manifest(result.manifest)
        self.descriptor_pool_capacity = int(
            result.manifest["descriptor_pool_capacity"])
        self.max_blocks_per_pool = int(result.manifest["max_blocks_per_pool"])
        self.max_in_flight = int(result.manifest["request_slot_count"])
        if self.max_in_flight != envs.VLLM_GRANULEKV_MAX_IN_FLIGHT:
            raise RuntimeError(
                "GranuleKV request-slot mismatch: daemon="
                f"{self.max_in_flight}, vLLM={envs.VLLM_GRANULEKV_MAX_IN_FLIGHT}")
        physical_regions: list[torch.Tensor] = []
        for region in result.regions:
            tensor = self.bridge.tensor_from_cuda_ptr(
                region.region_ptr,
                list(self.layout.tensor_shape),
                list(self.layout.tensor_strides),
                self.layout.dtype_name,
                self.layout.device_index,
            )
            if (tensor.data_ptr() != region.region_ptr
                    or tuple(tensor.shape) != self.layout.tensor_shape
                    or tuple(tensor.stride()) != self.layout.tensor_strides
                    or region.region_bytes != self.layout.region_bytes):
                raise RuntimeError("imported GranuleKV tensor metadata mismatch")
            physical_regions.append(tensor)
        for tensor in physical_regions:
            tensor.zero_()
        torch.cuda.synchronize(self.layout.device_index)
        tensors = [
            physical_regions[layer % self.layout.num_gpu_regions]
            for layer in range(self.layout.num_layers)
        ]
        logger.info(
            "[GRANULEKV] imported KV cache layers=%d gpu_regions=%d "
            "gpu_blocks=%d storage_blocks=%d fragment_bytes=%d "
            "pool_capacity=%d max_blocks_per_pool=%d max_in_flight=%d",
            self.layout.num_layers, self.layout.num_gpu_regions,
            self.layout.num_gpu_blocks, self.layout.num_storage_blocks,
            self.layout.fragment_bytes, self.descriptor_pool_capacity,
            self.max_blocks_per_pool, self.max_in_flight)
        return tensors

    def start(self) -> None:
        self.client.start()
        logger.info("[GRANULEKV] resident async service enabled after model warmup")

    def swap_out(self, src_to_dst: torch.Tensor) -> None:
        self._transfer_mapping(src_to_dst, operation="write")

    def swap_in(self, src_to_dst: torch.Tensor) -> None:
        self._transfer_mapping(src_to_dst, operation="read")

    def swap_in_async(self, src_to_dst: torch.Tensor) -> bool:
        request_id = "legacy-deferred-swap-in"
        if request_id not in self._pending_transfers:
            if self.submit_transfer_async(request_id, src_to_dst, operation="read"):
                return True
        return self.poll_transfer_async(request_id)

    def submit_transfer_async(
        self,
        scheduler_request_id: str,
        src_to_dst: torch.Tensor,
        *,
        operation: str,
        layer_range: Optional[Tuple[int, int]] = None,
    ) -> bool:
        if not scheduler_request_id:
            raise ValueError("scheduler_request_id must not be empty")
        if operation not in ("read", "write"):
            raise ValueError(f"unsupported GranuleKV operation: {operation}")
        layer_range = self._validate_layer_range(layer_range)
        if src_to_dst.numel() == 0:
            return True
        if scheduler_request_id in self._pending_transfers:
            pending = self._pending_transfers[scheduler_request_id]
            mapping, _ = self._mapping_payload(
                src_to_dst, operation=operation, layer_range=layer_range)
            if (mapping != pending.mapping or operation != pending.operation
                    or layer_range != pending.layer_range):
                raise RuntimeError("GranuleKV transfer changed while in flight")
            return False
        mapping, payload = self._mapping_payload(
            src_to_dst, operation=operation, layer_range=layer_range)
        submit = (self.client.submit_read_async
                  if operation == "read" else self.client.submit_write_async)
        handle = submit(payload)
        self._pending_transfers[scheduler_request_id] = _PendingTransfer(
            handle=handle, mapping=mapping, operation=operation,
            layer_range=layer_range, submitted_at_ns=time.monotonic_ns())
        return False

    def stage_prefetch_plan(
        self,
        plan_id: str,
        units: Sequence[tuple[str, torch.Tensor, str,
                              Optional[Tuple[int, int]]]],
    ) -> None:
        staged: dict[str, tuple[dict[str, Any], str]] = {}
        for request_id, mapping_tensor, operation, layer_range in units:
            if operation not in ("read", "write"):
                raise ValueError(f"unsupported GranuleKV operation: {operation}")
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

    def activate_prefetch_transfer_async(
        self,
        plan_id: str,
        scheduler_request_id: str,
        src_to_dst: torch.Tensor,
        *,
        operation: str,
        layer_range: Optional[Tuple[int, int]],
    ) -> bool:
        template = self._prefetch_templates.get(scheduler_request_id)
        if template is None or template[0] != plan_id:
            raise RuntimeError("unknown staged GranuleKV prefetch unit")
        layer_range = self._validate_layer_range(layer_range)
        mapping, _ = self._mapping_payload(
            src_to_dst, operation=operation, layer_range=layer_range)
        if (mapping, operation, layer_range) != template[1:]:
            raise RuntimeError("GranuleKV prefetch unit changed after staging")
        try:
            handle = self.client.activate_prefetch_units(
                plan_id, (scheduler_request_id,))[0]
        except Exception:
            self.client.discard_prefetch_units(
                plan_id, (scheduler_request_id,))
            del self._prefetch_templates[scheduler_request_id]
            self._maybe_release_prefetch_plan(plan_id)
            raise
        self._pending_transfers[scheduler_request_id] = _PendingTransfer(
            handle=handle, mapping=mapping, operation=operation,
            layer_range=layer_range, submitted_at_ns=time.monotonic_ns(),
            prefetch_plan_id=plan_id)
        return False

    def discard_staged_prefetch_units(self, scheduler_request_ids: Sequence[str]) -> None:
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
        pending = self._pending_transfers.get(scheduler_request_id)
        if pending is None:
            raise RuntimeError("cannot poll GranuleKV transfer without a pending request")
        try:
            if not self.client.poll(pending.handle):
                return False
        except Exception:
            if pending.prefetch_plan_id is None:
                self.client.discard(pending.handle)
            else:
                self.client.fail_prefetch_unit(
                    pending.prefetch_plan_id, scheduler_request_id)
                del self._prefetch_templates[scheduler_request_id]
                self._maybe_release_prefetch_plan(pending.prefetch_plan_id)
            del self._pending_transfers[scheduler_request_id]
            raise
        if pending.prefetch_plan_id is None:
            self.client.finish(pending.handle)
        else:
            self.client.finish_prefetch_unit(
                pending.prefetch_plan_id, scheduler_request_id)
        del self._pending_transfers[scheduler_request_id]
        if pending.prefetch_plan_id is not None:
            del self._prefetch_templates[scheduler_request_id]
            self._maybe_release_prefetch_plan(pending.prefetch_plan_id)
        return True

    def _maybe_release_prefetch_plan(self, plan_id: str) -> None:
        if any(template[0] == plan_id
               for template in self._prefetch_templates.values()):
            return
        self.client.release_prefetch_plan(plan_id)

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
            raise ValueError(f"unsupported GranuleKV operation: {operation}")
        payload: dict[str, Any] = {
            "gpu_block_ids": gpu_ids,
            "storage_block_ids": storage_ids,
        }
        if layer_range is not None:
            payload["layer_start"], payload["layer_end"] = layer_range
            if self.layout.num_gpu_regions < self.layout.num_layers:
                payload["gpu_region_start"] = (
                    layer_range[0] % self.layout.num_gpu_regions)
        elif self.layout.num_gpu_regions < self.layout.num_layers:
            raise RuntimeError(
                "layer working-set mode only supports layer-ranged GranuleKV I/O")
        return mappings, payload

    def _validate_layer_range(
        self, layer_range: Optional[Tuple[int, int]]
    ) -> Optional[Tuple[int, int]]:
        if layer_range is None:
            return None
        start_layer, end_layer = (int(layer_range[0]), int(layer_range[1]))
        if not 0 <= start_layer < end_layer <= self.layout.num_layers:
            raise ValueError(
                "GranuleKV layer range is outside local KV cache: "
                f"[{start_layer}, {end_layer}) vs {self.layout.num_layers}")
        return start_layer, end_layer
