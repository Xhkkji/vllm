# SPDX-License-Identifier: Apache-2.0
"""从 LMCache V1 GDS 路径抽出的最小 cuFile 数据面。

这份代码的目标是复用 LMCache V1 的原生 GDS/cuFile 思路，而不是迁移
LMCache V1 的 StorageManager。这里仅负责：

- 创建一个 O_DIRECT slab 文件
- 注册 CUDA buffer / CUDA stream
- 对单个连续 CUDA tensor 发起 cuFileReadAsync/cuFileWriteAsync

put/get 在返回前会同步当前 CUDA stream。这个同步口径和当前 BaM 同步读写
baseline 对齐，后续如果要测队列并发，可以在这一层继续扩展。
"""

from __future__ import annotations

import ctypes
import os
import threading
from dataclasses import dataclass
from typing import Any, Literal

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_CUFILE_ALIGNMENT = 4096
_MAX_CUFILE_REGION = 16 * 1024 * 1024
_STREAM_REGISTER_FLAGS = 0x7
_driver_opened = False
_driver_lock = threading.Lock()


def _declare_signatures() -> None:
    """声明 cuFile async C API 的 ctypes 签名。"""
    from cufile.bindings import CUfileError, libcufile

    if getattr(libcufile.cuFileReadAsync, "argtypes", None):
        return

    libcufile.cuFileReadAsync.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_void_p,
    ]
    libcufile.cuFileReadAsync.restype = CUfileError

    libcufile.cuFileWriteAsync.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_void_p,
    ]
    libcufile.cuFileWriteAsync.restype = CUfileError

    libcufile.cuFileStreamRegister.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    libcufile.cuFileStreamRegister.restype = CUfileError

    libcufile.cuFileStreamDeregister.argtypes = [ctypes.c_void_p]
    libcufile.cuFileStreamDeregister.restype = CUfileError


def _ensure_driver_open() -> None:
    """按需打开 cuFile driver。"""
    global _driver_opened
    with _driver_lock:
        if _driver_opened:
            return
        from cufile.bindings import cuFileDriverOpen

        cuFileDriverOpen()
        _declare_signatures()
        _driver_opened = True


def _check(err: Any, op: str) -> None:
    if err.err != 0:
        raise RuntimeError(
            f"{op} failed: cuFileError(err={err.err}, cu_err={err.cu_err})")


def _align_up(value: int, alignment: int = _CUFILE_ALIGNMENT) -> int:
    return (int(value) + alignment - 1) // alignment * alignment


@dataclass
class _Submission:
    """保存 cuFile async 调用需要延长生命周期的 ctypes 参数。"""

    size: ctypes.c_size_t
    file_offset: ctypes.c_int64
    buf_offset: ctypes.c_int64
    bytes_done: ctypes.c_int64

    @classmethod
    def create(cls, size: int, file_offset: int,
               buf_offset: int) -> "_Submission":
        return cls(
            size=ctypes.c_size_t(size),
            file_offset=ctypes.c_int64(file_offset),
            buf_offset=ctypes.c_int64(buf_offset),
            bytes_done=ctypes.c_int64(0),
        )


class CuFileSlab:
    """一个 slab 文件上的最小 GDS 读写上下文。"""

    def __init__(self, path: str, slab_bytes: int, use_direct_io: bool = True) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CuFileSlab requires CUDA")

        self.path = path
        self.slab_bytes = _align_up(slab_bytes)
        self.use_direct_io = bool(use_direct_io)
        self._fd = -1
        self._handle: Any = None
        self._registered_streams: set[int] = set()
        self._open()

    def _open(self) -> None:
        _ensure_driver_open()

        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        creator_fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o644)
        try:
            os.posix_fallocate(creator_fd, 0, self.slab_bytes)
        finally:
            os.close(creator_fd)

        flags = os.O_RDWR
        if self.use_direct_io:
            flags |= os.O_DIRECT
        self._fd = os.open(self.path, flags)
        try:
            from cufile.bindings import cuFileHandleRegister

            self._handle = cuFileHandleRegister(self._fd)
        except Exception:
            os.close(self._fd)
            self._fd = -1
            raise

        logger.info(
            "[GDS_BASELINE] slab opened path=%s slab_bytes=%d use_direct_io=%s",
            self.path,
            self.slab_bytes,
            self.use_direct_io,
        )

    def close(self) -> None:
        for raw_stream in list(self._registered_streams):
            self._deregister_stream(raw_stream)
        self._registered_streams.clear()

        if self._fd >= 0:
            from cufile.bindings import cuFileHandleDeregister

            try:
                cuFileHandleDeregister(self._handle)
            finally:
                os.close(self._fd)
                self._fd = -1

    def transfer(self, tensor: torch.Tensor, file_offset: int,
                 direction: Literal["read", "write"]) -> int:
        """在当前 CUDA stream 上同步完成一次 tensor <-> slab 传输。"""
        if not tensor.is_cuda:
            raise ValueError("GDS transfer tensor must be CUDA tensor")
        if not tensor.is_contiguous():
            raise ValueError("GDS transfer tensor must be contiguous")

        view = tensor.view(torch.uint8).reshape(-1)
        nbytes = int(view.numel())
        if nbytes == 0:
            return 0
        if file_offset % _CUFILE_ALIGNMENT != 0 or nbytes % _CUFILE_ALIGNMENT != 0:
            raise ValueError(
                "GDS O_DIRECT transfer requires 4KB aligned offset and size: "
                f"offset={file_offset}, nbytes={nbytes}")
        if file_offset + nbytes > self.slab_bytes:
            raise ValueError(
                f"GDS slab overflow: offset={file_offset}, nbytes={nbytes}, "
                f"slab_bytes={self.slab_bytes}")

        stream = torch.cuda.current_stream(device=tensor.device)
        raw_stream = int(stream.cuda_stream)
        self._register_stream(raw_stream)

        registered_regions: list[torch.Tensor] = []
        submissions: list[_Submission] = []
        try:
            pos = 0
            while pos < nbytes:
                seg_len = min(_MAX_CUFILE_REGION, nbytes - pos)
                region = view[pos:pos + seg_len]
                self._register_buffer(region)
                registered_regions.append(region)

                sub = _Submission.create(
                    size=seg_len,
                    file_offset=file_offset + pos,
                    buf_offset=0,
                )
                self._submit(region.data_ptr(), sub, raw_stream, direction)
                submissions.append(sub)
                pos += seg_len

            stream.synchronize()
            bytes_done = sum(int(sub.bytes_done.value) for sub in submissions)
            if bytes_done != nbytes:
                raise RuntimeError(
                    f"GDS {direction} transferred {bytes_done} bytes, expected {nbytes}")
            return bytes_done
        finally:
            # 同步后再注销 buffer，避免 DMA 仍引用已注销区域。
            for region in reversed(registered_regions):
                self._deregister_buffer(region)

    def _submit(self, buf_base: int, sub: _Submission, raw_stream: int,
                direction: Literal["read", "write"]) -> None:
        from cufile.bindings import libcufile

        fn = (libcufile.cuFileReadAsync
              if direction == "read" else libcufile.cuFileWriteAsync)
        _check(
            fn(
                self._handle,
                ctypes.c_void_p(buf_base),
                ctypes.byref(sub.size),
                ctypes.byref(sub.file_offset),
                ctypes.byref(sub.buf_offset),
                ctypes.byref(sub.bytes_done),
                ctypes.c_void_p(raw_stream),
            ),
            f"cuFile{direction.title()}Async",
        )

    def _register_buffer(self, tensor: torch.Tensor) -> None:
        from cufile.bindings import libcufile

        _check(
            libcufile.cuFileBufRegister(
                ctypes.c_void_p(tensor.data_ptr()),
                ctypes.c_size_t(tensor.numel() * tensor.element_size()),
                ctypes.c_int(0),
            ),
            "cuFileBufRegister",
        )

    def _deregister_buffer(self, tensor: torch.Tensor) -> None:
        from cufile.bindings import libcufile

        _check(
            libcufile.cuFileBufDeregister(ctypes.c_void_p(tensor.data_ptr())),
            "cuFileBufDeregister",
        )

    def _register_stream(self, raw_stream: int) -> None:
        if raw_stream in self._registered_streams:
            return
        from cufile.bindings import libcufile

        _check(
            libcufile.cuFileStreamRegister(
                ctypes.c_void_p(raw_stream), _STREAM_REGISTER_FLAGS),
            "cuFileStreamRegister",
        )
        self._registered_streams.add(raw_stream)

    def _deregister_stream(self, raw_stream: int) -> None:
        from cufile.bindings import libcufile

        _check(
            libcufile.cuFileStreamDeregister(ctypes.c_void_p(raw_stream)),
            "cuFileStreamDeregister",
        )
