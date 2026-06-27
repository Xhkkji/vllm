# SPDX-License-Identifier: Apache-2.0
"""贴近 LMCache V1 GdsBackend 文件组织的 chunk store。

它复用 LMCache V1 的核心组织方式：

- `<gds_path>/<hash[:2]>/<hash[2:4]>/<quoted-key>.kvcache.safetensors`
- 文件前 4KB 存 safetensors-like metadata
- payload 从 offset 4096 开始
- 额外写一份 `.metadata` 文件，便于后续扫描/复用

这不是完整迁移 LMCache V1 StorageManager，而是把 V1 GDS 数据面整理成
当前 V100 + vLLM V0 能单独启用的可选 backend。
"""

from __future__ import annotations

import ctypes
import json
import mmap
import os
import struct
import tempfile
import time
import urllib.parse
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional

import torch

from vllm.bam.gds_baseline.chunk_store_base import ChunkTransferResult
from vllm.logger import init_logger

logger = init_logger(__name__)

_METADATA_FILE_SUFFIX = ".metadata"
_DATA_FILE_SUFFIX = ".kvcache.safetensors"
_METADATA_VERSION = 1
_METADATA_MAX_SIZE = 4096
_CUFILE_ALIGNMENT = 4096

_TORCH_DTYPES = {
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.float32: "F32",
    torch.float64: "F64",
    torch.uint8: "U8",
    torch.int8: "I8",
    torch.int16: "I16",
    torch.int32: "I32",
    torch.int64: "I64",
}
_TORCH_DTYPES_INVERSE = {value: key for key, value in _TORCH_DTYPES.items()}


def _align_up(value: int, alignment: int = _CUFILE_ALIGNMENT) -> int:
    return (int(value) + alignment - 1) // alignment * alignment


def _pack_metadata(tensor: torch.Tensor, fmt: str, **extra_metadata: Any) -> bytes:
    """复刻 LMCache V1 GdsBackend 的 4KB metadata 组织。"""
    if tensor.dtype not in _TORCH_DTYPES:
        raise RuntimeError(f"unhandled dtype {tensor.dtype}")

    data_size = tensor.numel() * tensor.element_size()
    tensor_meta = {
        "dtype": _TORCH_DTYPES[tensor.dtype],
        "shape": list(tensor.size()),
        "data_offsets": [0, data_size],
        # 官方 LMCache V1 写入的是 MemoryFormat.value；这里用字符串保持解耦。
        "fmt": fmt,
        "__metadata__": extra_metadata,
    }
    meta = {"kvcache": tensor_meta}
    encoded = json.dumps(meta).encode("utf-8")
    if len(encoded) > _METADATA_MAX_SIZE - 8:
        raise ValueError("GDS metadata is too large")
    encoded += b" " * (_METADATA_MAX_SIZE - 8 - len(encoded))
    return struct.pack("<Q", len(encoded)) + encoded


def _unpack_metadata(
        buffer: bytes) -> tuple[torch.Size, torch.dtype, int, str, dict[str, Any]]:
    meta_len = struct.unpack("<Q", buffer[:8])[0]
    decoded = buffer[8:8 + meta_len].rstrip(b" ")
    meta = json.loads(decoded.decode("utf-8"))
    tensor_meta = meta["kvcache"]
    dtype = _TORCH_DTYPES_INVERSE[tensor_meta["dtype"]]
    shape = torch.Size(tensor_meta["shape"])
    nbytes = int(tensor_meta["data_offsets"][1] - tensor_meta["data_offsets"][0])
    fmt = str(tensor_meta.get("fmt", "KV_2LTD"))
    return shape, dtype, nbytes, fmt, tensor_meta["__metadata__"]


def _extract_chunk_hash(key: Any) -> str:
    chunk_hash = getattr(key, "chunk_hash", None)
    if isinstance(chunk_hash, str) and chunk_hash:
        return chunk_hash
    if isinstance(key, str) and key:
        return key
    raise ValueError(f"Invalid chunk key: {key!r}")


def _key_to_string(key: Any) -> str:
    to_string = getattr(key, "to_string", None)
    if callable(to_string):
        return str(to_string())
    return _extract_chunk_hash(key)


@dataclass(frozen=True)
class LMCacheStyleGDSMetadata:
    """单个 chunk 在 LMCache-style GDS 里的元数据。

    `path` 指向 `.kvcache.safetensors` 文件。
    `nbytes` 是 payload 字节数，不包含前 4KB metadata。
    """

    path: str
    nbytes: int
    shape: torch.Size
    dtype: torch.dtype
    actual_tokens: int
    fmt: str


@dataclass(frozen=True)
class LMCacheStyleGDSConfig:
    """LMCache-style GDS 的最小配置。

    这个 config 只服务 GDS baseline/wrapper，不侵入 LMCache 原生 SSD 或 BaM。
    字段命名尽量贴近 LMCache V1 GdsBackend，便于后续继续迁移。
    """

    gds_path: str
    device: str
    use_gds: bool = True
    use_direct_io: bool = True
    fmt: str = "KV_2LTD"
    use_registered_buffer: bool = False
    registered_buffer_size: int = 0


class _RegisteredGDSBuffer:
    """V1 CuFileMemoryAllocator 的轻量替代：预注册一块 GPU staging buffer。

    官方 V1 会通过 CuFileMemoryAllocator 分配并注册 GPU buffer，再用
    `base_pointer + dev_offset` 做 cuFile 读写。这里不迁移 allocator，
    只在 GDS store 内提供一个可选 staging buffer，保持接口简单、路径解耦。
    """

    def __init__(self, device: torch.device, min_bytes: int) -> None:
        self.device = device
        self.raw: Optional[torch.Tensor] = None
        self.buffer: Optional[torch.Tensor] = None
        self.nbytes = 0
        if min_bytes > 0:
            self.resize(min_bytes)

    def resize(self, min_bytes: int) -> None:
        min_bytes = _align_up(min_bytes)
        if self.buffer is not None and self.nbytes >= min_bytes:
            return
        self.close()

        # 多申请 4KB，手动切到 4KB 对齐地址，贴近 V1 allocator 的
        # `align_bytes=4096`。
        raw = torch.empty(min_bytes + _CUFILE_ALIGNMENT,
                          device=self.device,
                          dtype=torch.uint8)
        offset = (-raw.data_ptr()) % _CUFILE_ALIGNMENT
        buffer = raw[offset:offset + min_bytes]

        from cufile.bindings import cuFileBufRegister

        cuFileBufRegister(ctypes.c_void_p(buffer.data_ptr()), min_bytes, flags=0)
        self.raw = raw
        self.buffer = buffer
        self.nbytes = min_bytes
        logger.info(
            "[LMCACHE_GDS] registered staging buffer nbytes=%d ptr=0x%x",
            self.nbytes,
            buffer.data_ptr(),
        )

    def view(self, nbytes: int) -> torch.Tensor:
        self.resize(nbytes)
        assert self.buffer is not None
        return self.buffer[:nbytes]

    def close(self) -> None:
        if self.buffer is None:
            return
        # 确保没有 DMA 仍引用这块 staging buffer。
        torch.cuda.synchronize(self.device)
        from cufile.bindings import cuFileBufDeregister

        try:
            cuFileBufDeregister(ctypes.c_void_p(self.buffer.data_ptr()))
        finally:
            self.raw = None
            self.buffer = None
            self.nbytes = 0


class LMCacheStyleGDSChunkStore:
    """LMCache V1-style GDS 文件 backend。

    `use_gds=True` 时使用原生 `cufile.CuFile`。
    当前环境没有 cufile bindings 时会在首次读写时报错；
    这能明确区分“GDS 环境未准备好”和 BaM 路径问题。
    """

    backend_name = "lmcache_style_gds"

    def __init__(self,
                 gds_path: Optional[str] = None,
                 device: Optional[str] = None,
                 use_gds: bool = True,
                 use_direct_io: bool = True,
                 fmt: str = "KV_2LTD",
                 use_registered_buffer: bool = False,
                 registered_buffer_size: int = 0,
                 config: Optional[LMCacheStyleGDSConfig] = None) -> None:
        if config is None:
            if gds_path is None or device is None:
                raise ValueError("gds_path and device are required")
            config = LMCacheStyleGDSConfig(
                gds_path=gds_path,
                device=device,
                use_gds=use_gds,
                use_direct_io=use_direct_io,
                fmt=fmt,
                use_registered_buffer=use_registered_buffer,
                registered_buffer_size=registered_buffer_size,
            )
        self.config = config
        self.gds_path = config.gds_path
        self.device = torch.device(config.device)
        self.use_gds = bool(config.use_gds)
        self.use_direct_io = bool(config.use_direct_io)
        self.fmt = config.fmt
        self.use_registered_buffer = bool(config.use_registered_buffer)
        self._registered_buffer: Optional[_RegisteredGDSBuffer] = None
        self.hot_cache: "OrderedDict[str, LMCacheStyleGDSMetadata]" = OrderedDict()
        os.makedirs(self.gds_path, exist_ok=True)
        if self.use_gds and self.use_registered_buffer:
            self._registered_buffer = _RegisteredGDSBuffer(
                self.device, config.registered_buffer_size)
        logger.info(
            "[LMCACHE_GDS] initialized gds_path=%s device=%s use_gds=%s "
            "use_direct_io=%s fmt=%s use_registered_buffer=%s "
            "registered_buffer_size=%d",
            self.gds_path,
            self.device,
            self.use_gds,
            self.use_direct_io,
            self.fmt,
            self.use_registered_buffer,
            config.registered_buffer_size,
        )

    def put_chunk(self,
                  key: Any,
                  tensor: torch.Tensor,
                  actual_tokens: Optional[int] = None) -> ChunkTransferResult:
        # 写入数据组织：
        #   1. tensor 先整理到 CUDA contiguous buffer
        #   2. 写 4KB metadata
        #   3. payload 从 offset 4096 开始写入
        #   4. 最后落成 `<hash[:2]>/<hash[2:4]>/<key>.kvcache.safetensors`
        chunk_hash = _extract_chunk_hash(key)
        actual_tokens = (int(actual_tokens) if actual_tokens is not None else
                         int(tensor.shape[2]) if tensor.dim() >= 3 else 0)
        tensor = self._prepare_tensor(tensor)
        path = self._key_to_path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        start = time.perf_counter()
        metadata = _pack_metadata(
            tensor,
            fmt=self.fmt,
            lmcache_version=str(_METADATA_VERSION),
            actual_tokens=actual_tokens,
            chunk_hash=chunk_hash,
        )
        self._save_tensor(path, tensor, metadata)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        item = LMCacheStyleGDSMetadata(
            path=path,
            nbytes=int(tensor.numel() * tensor.element_size()),
            shape=torch.Size(tensor.shape),
            dtype=tensor.dtype,
            actual_tokens=actual_tokens,
            fmt=self.fmt,
        )
        self.hot_cache[chunk_hash] = item

        result = ChunkTransferResult(
            backend=self.backend_name,
            op="write",
            chunk_hash=chunk_hash,
            nbytes=item.nbytes,
            elapsed_ms=elapsed_ms,
        )
        logger.info(
            "[LMCACHE_GDS_WRITE] chunk_hash=%s path=%s nbytes=%d "
            "actual_tokens=%d elapsed_ms=%.3f bw_gib_s=%.3f",
            chunk_hash[:16],
            path,
            item.nbytes,
            actual_tokens,
            result.elapsed_ms,
            result.bw_gib_s,
        )
        return result

    def load_chunk_tensor(self, key: Any) -> Optional[torch.Tensor]:
        # 这里的读顺序与写顺序相反：
        #   1. 从 metadata 文件恢复 shape/dtype/nbytes
        #   2. 为 payload 分配同形状 CUDA tensor
        #   3. 用 cuFile 或 POSIX fallback 把 payload 读回
        chunk_hash = _extract_chunk_hash(key)
        metadata = self._lookup_metadata(key)
        if metadata is None:
            return None

        tensor = torch.empty(metadata.shape, device=self.device, dtype=metadata.dtype)
        start = time.perf_counter()
        bytes_done = self._load_tensor(metadata.path, tensor)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if bytes_done != metadata.nbytes:
            raise RuntimeError(
                f"GDS read size mismatch: got={bytes_done} expected={metadata.nbytes}")

        result = ChunkTransferResult(
            backend=self.backend_name,
            op="read",
            chunk_hash=chunk_hash,
            nbytes=metadata.nbytes,
            elapsed_ms=elapsed_ms,
        )
        logger.info(
            "[LMCACHE_GDS_READ] chunk_hash=%s path=%s nbytes=%d "
            "actual_tokens=%d elapsed_ms=%.3f bw_gib_s=%.3f",
            chunk_hash[:16],
            metadata.path,
            metadata.nbytes,
            metadata.actual_tokens,
            result.elapsed_ms,
            result.bw_gib_s,
        )
        return tensor

    def get_chunk(self, key: Any, out_tensor: torch.Tensor) -> ChunkTransferResult:
        """统一 replay 接口：读出 chunk 并 copy 到调用方提供的 tensor。"""
        chunk_hash = _extract_chunk_hash(key)
        start = time.perf_counter()
        tensor = self.load_chunk_tensor(key)
        if tensor is None:
            raise KeyError(f"LMCache-style GDS chunk not found: {chunk_hash}")
        out_tensor.copy_(tensor.to(device=out_tensor.device, non_blocking=False))
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return ChunkTransferResult(
            backend=self.backend_name,
            op="read",
            chunk_hash=chunk_hash,
            nbytes=int(out_tensor.numel() * out_tensor.element_size()),
            elapsed_ms=elapsed_ms,
        )

    def get_chunk_metadata(self, key: Any) -> Optional[LMCacheStyleGDSMetadata]:
        return self._lookup_metadata(key)

    def close(self) -> None:
        if self._registered_buffer is not None:
            self._registered_buffer.close()
            self._registered_buffer = None

    def _prepare_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if not tensor.is_cuda or tensor.device != self.device:
            # V0 LMCache 当前常给 CPU buffer；这里显式 staging 到 CUDA。
            tensor = tensor.to(device=self.device, non_blocking=False)
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        return tensor

    def _key_to_path(self, key: Any) -> str:
        chunk_hash = _extract_chunk_hash(key)
        key_str = urllib.parse.quote(_key_to_string(key), safe="")
        l1_dir = chunk_hash[:2]
        l2_dir = chunk_hash[2:4]
        return os.path.join(self.gds_path, l1_dir, l2_dir,
                            key_str + _DATA_FILE_SUFFIX)

    def _lookup_metadata(self, key: Any) -> Optional[LMCacheStyleGDSMetadata]:
        chunk_hash = _extract_chunk_hash(key)
        item = self.hot_cache.get(chunk_hash)
        if item is not None:
            self.hot_cache.move_to_end(chunk_hash)
            return item

        path = self._key_to_path(key)
        metadata_path = path + _METADATA_FILE_SUFFIX
        if not os.path.exists(metadata_path):
            return None

        with open(metadata_path, "rb") as f:
            metadata_blob = f.read(_METADATA_MAX_SIZE)
        shape, dtype, nbytes, fmt, extra = _unpack_metadata(metadata_blob)
        actual_tokens = int(extra.get("actual_tokens", shape[2] if len(shape) > 2 else 0))
        item = LMCacheStyleGDSMetadata(
            path=path,
            nbytes=nbytes,
            shape=shape,
            dtype=dtype,
            actual_tokens=actual_tokens,
            fmt=fmt,
        )
        self.hot_cache[chunk_hash] = item
        return item

    def _save_tensor(self, path: str, tensor: torch.Tensor, metadata: bytes) -> None:
        # GDS 文件写入按“临时文件 -> 原子替换”组织，避免半写状态被读到。
        fd, tmp_path = tempfile.mkstemp(prefix=".tmp", dir=os.path.dirname(path))
        os.close(fd)
        try:
            with open(tmp_path, "wb") as f:
                f.write(metadata)

            if self.use_gds:
                self._gds_write(tmp_path, tensor)
            else:
                self._posix_write(tmp_path, tensor)

            os.replace(tmp_path, path)
            metadata_tmp = path + ".metadata.tmp"
            with open(metadata_tmp, "wb") as f:
                f.write(metadata)
            os.replace(metadata_tmp, path + _METADATA_FILE_SUFFIX)
        except Exception:
            try:
                os.remove(tmp_path)
            except FileNotFoundError:
                pass
            raise

    def _load_tensor(self, path: str, tensor: torch.Tensor) -> int:
        # 读路径在 GDS / POSIX 两种实现间切换，便于对照测试文件组织本身。
        if self.use_gds:
            return self._gds_read(path, tensor)
        return self._posix_read(path, tensor)

    def _gds_write(self, path: str, tensor: torch.Tensor) -> None:
        import cufile

        # cuFile 的同步 API 不感知 torch 当前 stream。
        # 写前必须确认 GPU 数据已就绪。
        torch.cuda.synchronize(tensor.device)
        nbytes = tensor.numel() * tensor.element_size()
        transfer_tensor = self._prepare_gds_transfer_tensor(tensor, "write")
        addr = ctypes.c_void_p(transfer_tensor.data_ptr())
        with cufile.CuFile(path, "r+", use_direct_io=self.use_direct_io) as f:
            bytes_done = int(f.write(
                addr,
                nbytes,
                file_offset=_METADATA_MAX_SIZE,
                dev_offset=0,
            ))
        if bytes_done != nbytes:
            raise RuntimeError(
                f"GDS write size mismatch: got={bytes_done} expected={nbytes}")

    def _gds_read(self, path: str, tensor: torch.Tensor) -> int:
        import cufile

        nbytes = tensor.numel() * tensor.element_size()
        transfer_tensor = self._prepare_gds_transfer_tensor(tensor, "read")
        addr = ctypes.c_void_p(transfer_tensor.data_ptr())
        with cufile.CuFile(path, "r", use_direct_io=self.use_direct_io) as f:
            bytes_done = int(f.read(
                addr,
                nbytes,
                file_offset=_METADATA_MAX_SIZE,
                dev_offset=0,
        ))
        # 读后同步，确保后续 torch 校验/copy 看到的是 DMA 完成后的内容。
        torch.cuda.synchronize(tensor.device)
        tensor_bytes = tensor.view(torch.uint8).reshape(-1)
        if transfer_tensor.data_ptr() != tensor_bytes.data_ptr():
            tensor_bytes.copy_(transfer_tensor[:nbytes], non_blocking=False)
        return bytes_done

    def _prepare_gds_transfer_tensor(self, tensor: torch.Tensor,
                                     op: str) -> torch.Tensor:
        """返回实际交给 cuFile 的 GPU byte buffer。

        默认直接使用传入 tensor；启用 registered buffer 时，先把数据写入
        预注册 staging buffer，贴近 V1 的 CuFileMemoryAllocator 数据组织。
        """
        nbytes = tensor.numel() * tensor.element_size()
        tensor_bytes = tensor.view(torch.uint8).reshape(-1)
        if self._registered_buffer is None:
            return tensor_bytes

        staging = self._registered_buffer.view(nbytes)
        if op == "write":
            staging.copy_(tensor_bytes, non_blocking=False)
            torch.cuda.synchronize(tensor.device)
        return staging

    def _posix_write(self, path: str, tensor: torch.Tensor) -> None:
        payload = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        with open(path, "r+b") as f:
            f.seek(_METADATA_MAX_SIZE)
            f.write(payload)

    def _posix_read(self, path: str, tensor: torch.Tensor) -> int:
        nbytes = tensor.numel() * tensor.element_size()
        fd = os.open(path, os.O_RDONLY)
        try:
            file_size = os.fstat(fd).st_size
            mm = mmap.mmap(fd, file_size, prot=mmap.PROT_READ,
                           flags=mmap.MAP_PRIVATE)
            payload = mm[_METADATA_MAX_SIZE:_METADATA_MAX_SIZE + nbytes]
            cpu_tensor = torch.frombuffer(bytearray(payload),
                                          dtype=torch.uint8).view(tensor.dtype)
            cpu_tensor = cpu_tensor.view(tensor.shape)
            tensor.copy_(cpu_tensor.to(device=tensor.device), non_blocking=False)
            mm.close()
            return nbytes
        finally:
            os.close(fd)
