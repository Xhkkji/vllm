# SPDX-License-Identifier: Apache-2.0
"""vLLM V0 与独立 BaM MDS daemon 之间的最小同步 connector。"""

from __future__ import annotations

import base64
import importlib
import os
from pathlib import Path
import sys
import time
from typing import Sequence

import torch

import vllm.envs as envs
from vllm.logger import init_logger


logger = init_logger(__name__)


_DTYPE_NAMES = {
    torch.uint8: "uint8",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}


def _contiguous_strides(shape: Sequence[int]) -> tuple[int, ...]:
    stride = 1
    reversed_strides: list[int] = []
    for size in reversed(shape):
        reversed_strides.append(stride)
        stride *= int(size)
    return tuple(reversed(reversed_strides))


class BaMMDSConnector:
    """持有 imported CUDA handles，并保持 CacheEngine 同步 swap 语义。

    vLLM 只知道 physical block mapping。layer/K/V fragment 展开、NVMe offset、
    region id 和 CQ polling 全部留在 daemon，避免把 BaM runtime 再带回 compute
    进程。首版每次只允许一个 batch in flight。
    """

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
        self.slot = None
        self.imported_ptrs: list[int] = []
        self.gpu_cache: list[torch.Tensor] = []
        if not envs.VLLM_BAM_MDS_IOSTACK_ROOT:
            raise ValueError("VLLM_BAM_MDS_IOSTACK_ROOT is required")
        if not envs.VLLM_BAM_MDS_CONTROL_DIR:
            raise ValueError("VLLM_BAM_MDS_CONTROL_DIR is required")
        if dtype not in _DTYPE_NAMES:
            raise ValueError(f"unsupported MDS KV dtype: {dtype}")
        if num_storage_blocks <= 0:
            raise ValueError("MDS KVStore requires positive CPU/storage blocks")

        self.iostack_root = Path(envs.VLLM_BAM_MDS_IOSTACK_ROOT).resolve()
        gids_module = self.iostack_root / "gids_module"
        if str(gids_module) not in sys.path:
            sys.path.insert(0, str(gids_module))

        # 共享协议从 BaM_IOStack 导入，vLLM 侧不复制 slot 状态或 JSON schema。
        self.protocol = importlib.import_module("bam_mds.protocol")
        cuda_ipc_module = importlib.import_module("bam_mds.cuda_ipc")
        cuda_library = self._resolve_cuda_library()
        self.cuda = cuda_ipc_module.CudaIpc(cuda_library)
        self.cuda.set_device(device_index)
        self.bridge = self._load_torch_bridge()

        self.control_dir = Path(envs.VLLM_BAM_MDS_CONTROL_DIR).resolve()
        self.timeout_seconds = float(envs.VLLM_BAM_MDS_TIMEOUT_SECONDS)
        self.started = False
        self.generation = 0
        self.max_blocks_per_batch = 0

        try:
            self.gpu_cache = self._allocate_and_import(
                allocation_shape=tuple(int(value)
                                       for value in allocation_shape),
                stride_order=tuple(int(value) for value in stride_order),
                dtype=dtype,
                device_index=device_index,
                num_layers=num_layers,
                num_gpu_blocks=num_gpu_blocks,
                num_storage_blocks=num_storage_blocks,
            )
        except Exception:
            self._publish_client_error()
            self._close_imports_after_tensor_release()
            raise

    def _resolve_cuda_library(self) -> Path:
        if envs.VLLM_BAM_MDS_CUDA_IPC_LIBRARY:
            return Path(envs.VLLM_BAM_MDS_CUDA_IPC_LIBRARY).resolve()
        return (self.iostack_root / "vllm_evaluation" / "mds_poc" / "phase3"
                / "build" / "libmds_cuda_ipc.so")

    def _load_torch_bridge(self):
        if envs.VLLM_BAM_MDS_TORCH_BRIDGE_DIR:
            bridge_dir = Path(envs.VLLM_BAM_MDS_TORCH_BRIDGE_DIR).resolve()
        else:
            bridge_dir = (self.iostack_root / "vllm_evaluation" / "mds_poc"
                          / "phase3" / "build" / "torch_bridge")
        if str(bridge_dir) not in sys.path:
            sys.path.insert(0, str(bridge_dir))
        return importlib.import_module("mds_torch_bridge")

    def _allocate_and_import(
        self,
        *,
        allocation_shape: tuple[int, ...],
        stride_order: tuple[int, ...],
        dtype: torch.dtype,
        device_index: int,
        num_layers: int,
        num_gpu_blocks: int,
        num_storage_blocks: int,
    ) -> list[torch.Tensor]:
        if sorted(stride_order) != list(range(len(allocation_shape))):
            raise ValueError("invalid KV cache stride order")

        # 必须精确复现原生分配的 ``zeros(allocation_shape).permute(order)``。
        # 这里仅做整数 stride 计算，不在 vLLM 进程额外分配同尺寸 CUDA tensor。
        allocation_strides = _contiguous_strides(allocation_shape)
        tensor_shape = tuple(allocation_shape[index]
                             for index in stride_order)
        tensor_strides = tuple(allocation_strides[index]
                               for index in stride_order)
        if len(tensor_shape) != 3 or tensor_shape[0] != 2:
            raise ValueError(
                f"Phase 4A requires V0 [2,N,E] KV layout, got {tensor_shape}")
        if tensor_strides[2] != 1:
            raise ValueError("MDS KV fragment must be contiguous")

        element_size = torch.empty((), dtype=dtype).element_size()
        region_elements = 1 + sum(
            (size - 1) * stride
            for size, stride in zip(tensor_shape, tensor_strides))
        region_bytes = region_elements * element_size
        fragment_bytes = tensor_strides[1] * element_size
        if fragment_bytes % 4096 != 0:
            raise ValueError("MDS KV fragment must be 4KB aligned")

        control_path = self.control_dir / self.protocol.CONTROL_FILE
        self.protocol.wait_for_path(control_path, self.timeout_seconds)
        self.slot = self.protocol.ControlSlot.open_existing(control_path)
        self.slot.wait_for_states((self.protocol.STATE_DAEMON_READY,),
                                  self.timeout_seconds)
        self.protocol.atomic_write_json(
            self.control_dir / self.protocol.ALLOCATION_REQUEST_FILE, {
                "protocol_version": self.protocol.MDS_PROTOCOL_VERSION,
                "client_pid": os.getpid(),
                "device_index": device_index,
                "num_layers": num_layers,
                "num_gpu_blocks": num_gpu_blocks,
                "num_storage_blocks": num_storage_blocks,
                "region_bytes": region_bytes,
                "element_size": element_size,
                "fragment_bytes": fragment_bytes,
                "kv_stride_elems": tensor_strides[0],
                "block_stride_elems": tensor_strides[1],
                "logical_block_bytes": num_layers * 2 * fragment_bytes,
                "tensor_shape": list(tensor_shape),
                "tensor_strides": list(tensor_strides),
                "dtype": _DTYPE_NAMES[dtype],
            })
        self.slot.update(state=self.protocol.STATE_ALLOC_REQUESTED)
        ready = self.slot.wait_for_states(
            (self.protocol.STATE_CACHE_READY, self.protocol.STATE_ERROR),
            self.timeout_seconds)
        if ready.state == self.protocol.STATE_ERROR:
            raise RuntimeError(
                f"MDS daemon allocation failed: error={ready.error_code}")

        manifest = self.protocol.read_json(
            self.control_dir / self.protocol.ALLOCATION_MANIFEST_FILE)
        if (tuple(manifest["tensor_shape"]) != tensor_shape
                or tuple(manifest["tensor_strides"]) != tensor_strides
                or manifest["dtype"] != _DTYPE_NAMES[dtype]):
            raise RuntimeError("MDS allocation manifest layout mismatch")
        regions = manifest["regions"]
        if len(regions) != num_layers:
            raise RuntimeError("MDS allocation manifest layer count mismatch")
        self.max_blocks_per_batch = int(manifest["max_blocks_per_batch"])

        tensors: list[torch.Tensor] = []
        for layer_id, region in enumerate(regions):
            if int(region["layer_id"]) != layer_id:
                raise RuntimeError("MDS region order mismatch")
            handle = base64.b64decode(region["handle_b64"], validate=True)
            imported_ptr = int(self.cuda.open_handle(handle))
            self.imported_ptrs.append(imported_ptr)
            region_ptr = imported_ptr + int(region["region_offset"])
            tensor = self.bridge.tensor_from_cuda_ptr(
                region_ptr,
                list(tensor_shape),
                list(tensor_strides),
                _DTYPE_NAMES[dtype],
                device_index,
            )
            if (tensor.data_ptr() != region_ptr
                    or tuple(tensor.shape) != tensor_shape
                    or tuple(tensor.stride()) != tensor_strides):
                raise RuntimeError("imported MDS tensor metadata mismatch")
            tensors.append(tensor)

        # 原生 CacheEngine 用 torch.zeros 分配全部层；MDS allocation 是 cudaMalloc，
        # 因此 client 必须显式清零，保留 null block 和 warmup 的原有语义。
        for tensor in tensors:
            tensor.zero_()
        torch.cuda.synchronize(device_index)
        self.slot.update(state=self.protocol.STATE_CACHE_IMPORTED)
        logger.info(
            "[BAM_MDS] imported daemon KV cache layers=%d gpu_blocks=%d "
            "storage_blocks=%d fragment_bytes=%d max_blocks_per_batch=%d",
            num_layers, num_gpu_blocks, num_storage_blocks, fragment_bytes,
            self.max_blocks_per_batch)
        return tensors

    def start(self) -> None:
        """模型 warmup/CUDA graph capture 完成后才允许 daemon 接收 I/O。"""
        if self.started:
            return
        assert self.slot is not None
        current = self.slot.read()
        if current.state != self.protocol.STATE_CACHE_IMPORTED:
            raise RuntimeError(
                f"cannot start MDS service from state {current.state}")
        self.slot.update(state=self.protocol.STATE_SERVICE_READY)
        self.started = True
        logger.info("[BAM_MDS] synchronous service enabled after model warmup")

    def swap_out(self, src_to_dst: torch.Tensor) -> None:
        self._transfer_mapping(src_to_dst, operation=self.protocol.OP_WRITE)

    def swap_in(self, src_to_dst: torch.Tensor) -> None:
        self._transfer_mapping(src_to_dst, operation=self.protocol.OP_READ)

    def _transfer_mapping(self, src_to_dst: torch.Tensor, *, operation: int) -> None:
        if src_to_dst.numel() == 0:
            return
        if not self.started or self.slot is None:
            raise RuntimeError("MDS service is not initialized after warmup")
        mappings = src_to_dst.to(device="cpu", dtype=torch.int64).tolist()
        for begin in range(0, len(mappings), self.max_blocks_per_batch):
            batch = mappings[begin:begin + self.max_blocks_per_batch]
            source_ids = [int(mapping[0]) for mapping in batch]
            destination_ids = [int(mapping[1]) for mapping in batch]
            if operation == self.protocol.OP_WRITE:
                gpu_ids, storage_ids = source_ids, destination_ids
            else:
                storage_ids, gpu_ids = source_ids, destination_ids

            self.generation += 1
            request_id = time.time_ns() & 0xFFFFFFFFFFFFFFFF or 1
            self.protocol.atomic_write_json(
                self.control_dir / self.protocol.BATCH_REQUEST_FILE, {
                    "protocol_version": self.protocol.MDS_PROTOCOL_VERSION,
                    "request_id": request_id,
                    "generation": self.generation,
                    "gpu_block_ids": gpu_ids,
                    "storage_block_ids": storage_ids,
                })
            self.slot.update(state=self.protocol.STATE_SUBMITTED,
                             operation=operation,
                             request_id=request_id,
                             generation=self.generation,
                             io_elapsed_ns=0,
                             error_code=self.protocol.ERROR_NONE)
            completion = self.slot.wait_for_states(
                (self.protocol.STATE_DONE, self.protocol.STATE_ERROR),
                self.timeout_seconds)
            if completion.state == self.protocol.STATE_ERROR:
                raise RuntimeError(
                    f"MDS batch failed: error={completion.error_code}")
            if (completion.request_id != request_id
                    or completion.generation != self.generation):
                raise RuntimeError("MDS completion id/generation mismatch")
            self.slot.update(state=self.protocol.STATE_SERVICE_READY)

    def _publish_client_error(self) -> None:
        if self.slot is None or not hasattr(self, "protocol"):
            return
        try:
            self.slot.update(state=self.protocol.STATE_ERROR,
                             error_code=self.protocol.ERROR_PROTOCOL)
        except Exception:
            pass

    def _close_imports_after_tensor_release(self) -> None:
        # 仅在构造失败、局部 tensor 已离开作用域时使用。正常运行期由 vLLM 进程
        # 退出关闭 IPC mappings，runner 随后再停止 owner daemon。
        for pointer in reversed(getattr(self, "imported_ptrs", [])):
            try:
                self.cuda.close_handle(pointer)
            except Exception:
                pass
        self.imported_ptrs.clear()
        if self.slot is not None:
            self.slot.close()
            self.slot = None
