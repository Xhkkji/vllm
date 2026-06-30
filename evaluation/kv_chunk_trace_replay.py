#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""用同一批 KV chunk 负载对比 BaM 和原生 GDS/cuFile 数据面。"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay KV chunk writes/reads on BaM and native GDS/cuFile.")
    parser.add_argument("--backend",
                        choices=("bam", "bam_prefetch", "bam_prefetch_batch",
                                 "bam_kv_fast_path",
                                 "bam_kv_fast_path_batch", "bam_cold_read",
                                 "gds", "lmcache_gds", "all"),
                        default="all",
                        help="要测试的 backend；all 默认只跑主线 bam + lmcache_gds")
    parser.add_argument("--trace-jsonl",
                        type=Path,
                        default=None,
                        help="可选真实 chunk trace；为空则生成合成 trace")
    parser.add_argument("--num-chunks",
                        type=int,
                        default=8,
                        help="合成 trace 的 chunk 数，默认 8")
    parser.add_argument("--num-layers",
                        type=int,
                        default=28,
                        help="KV chunk layer 数，默认 28")
    parser.add_argument("--slot-num-tokens",
                        type=int,
                        default=256,
                        help="每个 chunk 的固定 token 槽位，默认 256")
    parser.add_argument("--hidden-dim",
                        type=int,
                        default=512,
                        help="KV hidden dim，默认 512")
    parser.add_argument("--dtype",
                        choices=("float16", "float32"),
                        default="float16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gds-slab-path",
                        default="/tmp/vllm-bam-gds-baseline/lmcache_gds_slab.bin")
    parser.add_argument("--gds-slab-gb",
                        type=float,
                        default=4.0,
                        help="GDS slab 文件大小，默认 4GB")
    parser.add_argument("--lmcache-gds-path",
                        default="/tmp/vllm-bam-lmcache-gds",
                        help="LMCache-style GDS 文件根目录")
    parser.add_argument("--lmcache-gds-use-posix",
                        action="store_true",
                        help="不用 cuFile，走 POSIX fallback 调试 LMCache-style 文件组织")
    parser.add_argument("--lmcache-gds-use-direct-io",
                        action=argparse.BooleanOptionalAction,
                        default=True,
                        help="是否使用 O_DIRECT，默认开启")
    parser.add_argument("--lmcache-gds-fmt",
                        default="KV_2LTD",
                        help="写入 metadata 的 MemoryFormat.value，默认 KV_2LTD")
    parser.add_argument("--lmcache-gds-use-registered-buffer",
                        action="store_true",
                        help="使用 V1-like 预注册 GPU staging buffer")
    parser.add_argument("--lmcache-gds-registered-buffer-mb",
                        type=int,
                        default=0,
                        help="预注册 staging buffer 大小，0 表示按 chunk 懒分配")
    parser.add_argument("--no-verify",
                        action="store_true",
                        help="关闭读回 exact_equal 校验")
    parser.add_argument("--bam-cold-manifest",
                        type=Path,
                        default=None,
                        help="BaM two-process cold-read 的 manifest 路径")
    parser.add_argument("--summary-warmup-samples",
                        type=int,
                        default=1,
                        help="summary 中额外统计稳定段时跳过的前置样本数，默认跳过 1 个")
    parser.add_argument("--batch-prefetch-warmup",
                        action=argparse.BooleanOptionalAction,
                        default=True,
                        help="bam_prefetch_batch 正式计时前先用一个 chunk 预热 refill/JIT，默认开启")
    return parser.parse_args()


def dtype_from_name(name: str):
    import torch

    return getattr(torch, name)


class DummyKey:
    def __init__(self, chunk_hash: str) -> None:
        self.chunk_hash = chunk_hash


class BaMReplayStore:
    """把现有 LMCacheBaMStore 包成统一 replay 接口。"""

    backend_name = "bam"

    def __init__(self,
                 shape: tuple[int, ...],
                 dtype: Any,
                 manifest_path: Path | None = None) -> None:
        import torch
        from vllm.bam.lmcache_bam_storage import LMCacheBaMStore

        self.store = LMCacheBaMStore.from_kv_shape(torch.Size(shape), dtype)
        self.manifest_path = manifest_path
        self.manifest_entries: list[dict[str, Any]] = []
        self.shape = tuple(int(v) for v in shape)
        self.dtype_name = str(dtype).replace("torch.", "")

    def put_chunk(self, chunk_hash: str, tensor, actual_tokens: int):
        # replay 里的写入顺序：
        #   1. 生成合成 KV tensor
        #   2. 交给 BaM store 写入
        #   3. 记录耗时，不把造数时间算进 put
        from vllm.bam.gds_baseline.chunk_store_base import ChunkTransferResult
        import time

        # BaM 当前 store 会自己记录 actual_tokens，这里要求输入已经是固定槽位。
        _ = actual_tokens
        start = time.perf_counter()
        self.store.store_chunk(DummyKey(chunk_hash), tensor)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        nbytes = int(tensor.numel() * tensor.element_size())
        return ChunkTransferResult("bam", "write", chunk_hash, nbytes, elapsed_ms)

    def record_manifest_entry(self, chunk_hash: str, salt: int) -> None:
        # cold-read 用的 manifest 只记录元数据，不记录真实 tensor。
        # 这样新进程只靠 slot_id/page_offset 就能把相同 chunk 重新读出来。
        if self.manifest_path is None:
            return
        metadata = self.store.get_chunk_metadata(DummyKey(chunk_hash))
        if metadata is None:
            raise KeyError(f"missing BaM metadata after write: {chunk_hash}")
        self.manifest_entries.append({
            "chunk_hash": chunk_hash,
            "slot_id": metadata.slot_id,
            "page_offset": metadata.page_offset,
            "actual_tokens": metadata.actual_tokens,
            "shape": list(metadata.shape),
            "dtype": self.dtype_name,
            "salt": int(salt),
        })

    def get_chunk(self, chunk_hash: str, out_tensor):
        # replay 的读路径：
        #   1. 根据 chunk_hash 找到对应 backend 的 chunk
        #   2. 读回 tensor
        #   3. copy 到调用方给定的 out_tensor
        from vllm.bam.gds_baseline.chunk_store_base import ChunkTransferResult
        import time

        start = time.perf_counter()
        tensor = self.store.load_chunk_tensor(DummyKey(chunk_hash))
        if tensor is None:
            raise KeyError(f"BaM chunk not found: {chunk_hash}")
        out_tensor.copy_(tensor.to(device=out_tensor.device, non_blocking=False))
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        nbytes = int(out_tensor.numel() * out_tensor.element_size())
        return ChunkTransferResult("bam", "read", chunk_hash, nbytes, elapsed_ms)

    def close(self) -> None:
        if self.manifest_path is None:
            return
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "shape": list(self.shape),
            "dtype": self.dtype_name,
            "entries": self.manifest_entries,
        }
        self.manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")


class BaMPrefetchReplayStore(BaMReplayStore):
    """显式测试 BaM page-level rowctx prefetch 中间层。"""

    backend_name = "bam_prefetch"

    def get_chunk(self, chunk_hash: str, out_tensor):
        from vllm.bam.gds_baseline.chunk_store_base import ChunkTransferResult
        import time

        start = time.perf_counter()
        tensor = self.store.load_chunk_tensor_prefetch(DummyKey(chunk_hash))
        if tensor is None:
            raise KeyError(f"BaM chunk not found: {chunk_hash}")
        out_tensor.copy_(tensor.to(device=out_tensor.device, non_blocking=False))
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        nbytes = int(out_tensor.numel() * out_tensor.element_size())
        return ChunkTransferResult("bam_prefetch", "read", chunk_hash, nbytes,
                                   elapsed_ms)


class BaMPrefetchBatchReplayStore(BaMPrefetchReplayStore):
    """批量测试 BaM page-level prefetch 请求表。

    单 chunk `bam_prefetch` 的读路径是：

      chunk_hash -> metadata -> page_ids -> submit/poll/complete -> refill

    这个 backend 把多个 read chunk 合成一批：

      [chunk_hash_0, chunk_hash_1, ...]
        -> 一批 GPU page_ids 请求表
        -> 批量 submit 到 BaM rowctx
        -> 按 FIFO complete
        -> 逐个 refill / verify

    当前统计使用“批量总耗时 / chunk 数”的摊销口径，目的是观察 batch
    submit 后的整体吞吐，而不是把它误解成每个 chunk 的独立延迟。
    """

    backend_name = "bam_prefetch_batch"

    def __init__(self,
                 shape: tuple[int, ...],
                 dtype: Any,
                 manifest_path: Path | None = None) -> None:
        super().__init__(shape, dtype, manifest_path=manifest_path)
        self._warmup_done = False

    def _warmup_refill_once(self, entry: Any, args: argparse.Namespace) -> None:
        """用一个 chunk 做不计时预热，主要吃掉 Triton refill 首次 JIT。

        之前 batch 结果里 `refill_ms` 接近 456ms，说明第一次运行主要不是
        BaM IO，而是 Triton kernel 编译。这里复用完全相同的 batch 读取接口，
        但只读一个 chunk，且不把结果加入 summary。

        这一步仍会经过：
          page_ids -> submit/poll/complete -> `[112,128KB]` -> Triton refill

        所以正式 batch 的统计会更接近稳态数据通路。
        """
        import time
        import torch

        key = DummyKey(entry.chunk_hash)
        start = time.perf_counter()
        tensors = self.store.load_chunk_tensors_prefetch_batch([key])
        tensor = tensors.get(entry.chunk_hash)
        if tensor is None:
            raise KeyError(f"BaM chunk not found during warmup: {entry.chunk_hash}")
        if tensor.is_cuda:
            torch.cuda.synchronize(tensor.device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        print(
            f"[bam_prefetch_batch warmup] chunk_hash={entry.chunk_hash[:16]} "
            f"elapsed_ms={elapsed_ms:.3f} not_counted=1")

    def get_chunks(self, entries: list[Any], args: argparse.Namespace):
        from vllm.bam.gds_baseline.chunk_store_base import ChunkTransferResult
        import time
        import torch

        if not entries:
            return [], {}

        if args.batch_prefetch_warmup and not self._warmup_done:
            self._warmup_refill_once(entries[0], args)
            self._warmup_done = True

        keys = [DummyKey(entry.chunk_hash) for entry in entries]
        start = time.perf_counter()
        tensors = self.store.load_chunk_tensors_prefetch_batch(keys)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        amortized_ms = elapsed_ms / len(entries)

        outputs = {}
        results = []
        for entry in entries:
            tensor = tensors.get(entry.chunk_hash)
            if tensor is None:
                raise KeyError(f"BaM chunk not found: {entry.chunk_hash}")
            out = torch.empty(entry.shape,
                              device=args.device,
                              dtype=entry.torch_dtype)
            out.copy_(tensor.to(device=out.device, non_blocking=False))
            outputs[entry.chunk_hash] = out
            nbytes = int(out.numel() * out.element_size())
            results.append(
                ChunkTransferResult("bam_prefetch_batch", "read",
                                    entry.chunk_hash, nbytes, amortized_ms))

        print(
            f"[bam_prefetch_batch read_batch] chunks={len(entries)} "
            f"elapsed_ms={elapsed_ms:.3f} "
            f"amortized_ms={amortized_ms:.3f}")
        return results, outputs


class BaMKVFastPathReplayStore(BaMReplayStore):
    """测试 KVCache 专用 fast path 的单 chunk 读取。

    与 `bam_prefetch` 的底层第一阶段类似，当前仍复用 BaM rowctx；区别在于
    上层接口已经从通用 page prefetch pipeline 切到 KV descriptor：

      chunk_hash -> metadata -> BaMKVRequest -> [112, 128KB] -> KV tensor
    """

    backend_name = "bam_kv_fast_path"

    def get_chunk(self, chunk_hash: str, out_tensor):
        from vllm.bam.gds_baseline.chunk_store_base import ChunkTransferResult
        import time

        start = time.perf_counter()
        tensor = self.store.load_chunk_tensor_kv_fast_path(DummyKey(chunk_hash))
        if tensor is None:
            raise KeyError(f"BaM chunk not found: {chunk_hash}")
        out_tensor.copy_(tensor.to(device=out_tensor.device, non_blocking=False))
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        nbytes = int(out_tensor.numel() * out_tensor.element_size())
        return ChunkTransferResult("bam_kv_fast_path", "read", chunk_hash,
                                   nbytes, elapsed_ms)


class BaMKVFastPathBatchReplayStore(BaMKVFastPathReplayStore):
    """批量测试 KVCache 专用 fast path。

    这是下一阶段 BaM 修改的第一块试金石：同一批 chunk 不再通过
    通用 feature/page prefetch 抽象提交，而是通过 KV chunk descriptor 批量
    进入 BaM KV store。当前仍按 rowctx FIFO 完成，后续可以把内部替换成
    GPU-visible queue / persistent worker。
    """

    backend_name = "bam_kv_fast_path_batch"

    def __init__(self,
                 shape: tuple[int, ...],
                 dtype: Any,
                 manifest_path: Path | None = None) -> None:
        super().__init__(shape, dtype, manifest_path=manifest_path)
        self._warmup_done = False

    def _warmup_refill_once(self, entry: Any, args: argparse.Namespace) -> None:
        """用同一个 KV fast path batch 接口预热一次 refill/JIT。

        之前 `bam_kv_fast_path_batch` 的正式统计几乎全是首次 Triton refill JIT：

        ```text
        total_ms ~= refill_ms ~= 300-500ms
        ```

        这里先用 1 个 chunk 走完整 KV descriptor batch read + refill，但不加入
        summary。这样正式 batch 的数据更接近稳态 BaM KV fast path。
        """
        import time
        import torch

        key = DummyKey(entry.chunk_hash)
        start = time.perf_counter()
        tensors = self.store.load_chunk_tensors_kv_fast_path_batch([key])
        tensor = tensors.get(entry.chunk_hash)
        if tensor is None:
            raise KeyError(
                f"BaM KV fast path chunk not found during warmup: {entry.chunk_hash}"
            )
        if tensor.is_cuda:
            torch.cuda.synchronize(tensor.device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        print(
            f"[bam_kv_fast_path_batch warmup] chunk_hash={entry.chunk_hash[:16]} "
            f"elapsed_ms={elapsed_ms:.3f} not_counted=1")

    def get_chunks(self, entries: list[Any], args: argparse.Namespace):
        from vllm.bam.gds_baseline.chunk_store_base import ChunkTransferResult
        import time
        import torch

        if not entries:
            return [], {}

        if args.batch_prefetch_warmup and not self._warmup_done:
            self._warmup_refill_once(entries[0], args)
            self._warmup_done = True

        keys = [DummyKey(entry.chunk_hash) for entry in entries]
        start = time.perf_counter()
        tensors = self.store.load_chunk_tensors_kv_fast_path_batch(keys)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        amortized_ms = elapsed_ms / len(entries)

        outputs = {}
        results = []
        for entry in entries:
            tensor = tensors.get(entry.chunk_hash)
            if tensor is None:
                raise KeyError(f"BaM chunk not found: {entry.chunk_hash}")
            out = torch.empty(entry.shape,
                              device=args.device,
                              dtype=entry.torch_dtype)
            out.copy_(tensor.to(device=out.device, non_blocking=False))
            outputs[entry.chunk_hash] = out
            nbytes = int(out.numel() * out.element_size())
            results.append(
                ChunkTransferResult("bam_kv_fast_path_batch", "read",
                                    entry.chunk_hash, nbytes, amortized_ms))

        print(
            f"[bam_kv_fast_path_batch read_batch] chunks={len(entries)} "
            f"elapsed_ms={elapsed_ms:.3f} "
            f"amortized_ms={amortized_ms:.3f}")
        return results, outputs


class BaMColdReadReplayStore(BaMReplayStore):
    """新进程只注册 metadata 后读 BaM，避免命中写进程的 GPU page cache。"""

    backend_name = "bam_cold_read"

    def __init__(self, shape: tuple[int, ...], dtype: Any,
                 manifest_path: Path) -> None:
        super().__init__(shape, dtype)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.salts: dict[str, int] = {}
        for item in payload["entries"]:
            self.salts[item["chunk_hash"]] = int(item["salt"])
            self.store.register_existing_chunk(
                DummyKey(item["chunk_hash"]),
                slot_id=int(item["slot_id"]),
                page_offset=int(item["page_offset"]),
                actual_tokens=int(item["actual_tokens"]),
                shape=torch_size_from_list(item["shape"]),
                dtype=dtype,
            )

    def expected_tensor(self, chunk_hash: str, shape: tuple[int, ...], dtype,
                        device: str):
        return make_tensor(shape, dtype, device, salt=self.salts[chunk_hash])


def torch_size_from_list(shape: list[int]):
    import torch

    return torch.Size(int(v) for v in shape)


def make_synthetic_trace(args: argparse.Namespace):
    from vllm.bam.gds_baseline.trace_schema import ChunkTraceEntry

    shape = (2, args.num_layers, args.slot_num_tokens, args.hidden_dim)
    writes = [
        ChunkTraceEntry("write", f"replay_chunk_{idx:04d}", shape, args.dtype,
                        args.slot_num_tokens)
        for idx in range(args.num_chunks)
    ]
    reads = [
        ChunkTraceEntry("read", f"replay_chunk_{idx:04d}", shape, args.dtype,
                        args.slot_num_tokens)
        for idx in range(args.num_chunks)
    ]
    return writes + reads


def make_tensor(shape: tuple[int, ...], dtype, device: str, salt: int):
    # 合成数据按固定规则生成，保证读回后可以做 exact_equal 校验。
    # 这里返回的 tensor 形状仍然是 `[2, num_layers, slot_tokens, hidden_dim]`。
    import torch

    # 数据生成不计入 put/get 计时；固定模式便于读回做 exact_equal。
    numel = 1
    for dim in shape:
        numel *= dim
    tensor = torch.arange(numel, device=device, dtype=torch.int32)
    tensor = (tensor + salt).remainder(2048).view(shape)
    return tensor.to(dtype)


def ensure_bam_replay_cache_capacity(entries: list[Any],
                                     args: argparse.Namespace) -> None:
    """当前 BaM replay 先写后读；

    校验模式下先避免踩小 page-cache 替换路径，保证读回对照更稳定。
    """
    if args.no_verify:
        return

    first = entries[0]
    dtype_size = first.torch_dtype.itemsize
    page_bytes = 128 * 1024
    _, num_layers, slot_num_tokens, hidden_dim = first.shape
    kv_layer_bytes = int(slot_num_tokens * hidden_dim * dtype_size)
    pages_per_kv_layer = (kv_layer_bytes + page_bytes - 1) // page_bytes
    pages_per_chunk = int(2 * num_layers * pages_per_kv_layer)
    written_chunks = {entry.chunk_hash for entry in entries if entry.op == "write"}

    cache_size_mb = int(os.environ.get("VLLM_BAM_CACHE_SIZE_MB", "64"))
    cache_pages = cache_size_mb * 1024 * 1024 // page_bytes
    required_pages = pages_per_chunk * len(written_chunks)
    if cache_pages < required_pages:
        required_mb = (required_pages * page_bytes + 1024 * 1024 - 1) // (
            1024 * 1024)
        raise ValueError(
            "BaM replay verify needs page cache to cover all written chunks: "
            f"cache={cache_size_mb}MB/{cache_pages} pages, "
            f"required={required_mb}MB/{required_pages} pages, "
            f"pages_per_chunk={pages_per_chunk}, "
            f"num_write_chunks={len(written_chunks)}. "
            "Increase VLLM_BAM_CACHE_SIZE_MB or reduce NUM_CHUNKS.")


def _print_summary(label: str, results: list[Any]) -> None:
    if not results:
        print(f"[{label}] no samples")
        return
    elapsed = [r.elapsed_ms for r in results]
    bw = [r.bw_gib_s for r in results]
    print(f"[{label}] samples={len(results)}")
    print(f"[{label}] mean_ms={statistics.mean(elapsed):.3f}")
    print(f"[{label}] median_ms={statistics.median(elapsed):.3f}")
    print(f"[{label}] min_ms={min(elapsed):.3f}")
    print(f"[{label}] max_ms={max(elapsed):.3f}")
    print(f"[{label}] mean_bw_gib_s={statistics.mean(bw):.3f}")
    print(f"[{label}] median_bw_gib_s={statistics.median(bw):.3f}")


def summarize(label: str, results: list[Any], warmup_samples: int) -> None:
    """打印全量样本和稳定段样本。

    BaM/GDS replay 的第一条样本经常包含：
    - Triton 首次 JIT
    - BaM page-cache 首块初始化
    - cuFile / 文件路径首次打开开销

    所以这里保留全量统计，同时额外输出 `.steady`，默认跳过第 1 条。
    """
    _print_summary(label, results)

    warmup_samples = max(0, int(warmup_samples))
    if warmup_samples == 0:
        return
    if len(results) <= warmup_samples:
        print(
            f"[{label}.steady] no samples after warmup_skip={warmup_samples}")
        return

    steady_results = results[warmup_samples:]
    print(f"[{label}.steady] warmup_skip={warmup_samples}")
    _print_summary(f"{label}.steady", steady_results)


def build_store(backend: str, shape: tuple[int, ...], dtype,
                args: argparse.Namespace):
    if backend == "bam":
        return BaMReplayStore(shape, dtype, manifest_path=args.bam_cold_manifest)
    if backend == "bam_prefetch":
        return BaMPrefetchReplayStore(shape,
                                      dtype,
                                      manifest_path=args.bam_cold_manifest)
    if backend == "bam_prefetch_batch":
        return BaMPrefetchBatchReplayStore(
            shape,
            dtype,
            manifest_path=args.bam_cold_manifest,
        )
    if backend == "bam_kv_fast_path":
        os.environ.setdefault("VLLM_BAM_LMCACHE_READ_MODE", "prefetch")
        os.environ.setdefault("VLLM_BAM_KV_FAST_PATH", "1")
        return BaMKVFastPathReplayStore(
            shape,
            dtype,
            manifest_path=args.bam_cold_manifest,
        )
    if backend == "bam_kv_fast_path_batch":
        os.environ.setdefault("VLLM_BAM_LMCACHE_READ_MODE", "prefetch")
        os.environ.setdefault("VLLM_BAM_KV_FAST_PATH", "1")
        return BaMKVFastPathBatchReplayStore(
            shape,
            dtype,
            manifest_path=args.bam_cold_manifest,
        )
    if backend == "bam_cold_read":
        if args.bam_cold_manifest is None:
            raise ValueError("--bam-cold-manifest is required for bam_cold_read")
        return BaMColdReadReplayStore(shape, dtype, args.bam_cold_manifest)
    if backend == "gds":
        from vllm.bam.gds_baseline.gds_chunk_store import GDSChunkStore

        return GDSChunkStore(
            slab_path=args.gds_slab_path,
            slab_bytes=int(args.gds_slab_gb * (1024**3)),
            device=args.device,
            use_direct_io=True,
        )
    if backend == "lmcache_gds":
        from vllm.bam.gds_baseline.lmcache_style_gds_store import (
            LMCacheStyleGDSChunkStore, LMCacheStyleGDSConfig)

        config = LMCacheStyleGDSConfig(
            gds_path=args.lmcache_gds_path,
            device=args.device,
            use_gds=not args.lmcache_gds_use_posix,
            use_direct_io=args.lmcache_gds_use_direct_io,
            fmt=args.lmcache_gds_fmt,
            use_registered_buffer=args.lmcache_gds_use_registered_buffer,
            registered_buffer_size=(
                args.lmcache_gds_registered_buffer_mb * 1024 * 1024),
        )
        return LMCacheStyleGDSChunkStore(config=config)
    raise ValueError(f"unknown backend: {backend}")


def replay_backend(backend: str, entries: list[Any],
                   args: argparse.Namespace) -> None:
    import torch

    first = entries[0]
    dtype = first.torch_dtype
    if backend in ("bam", "bam_prefetch", "bam_prefetch_batch",
                   "bam_kv_fast_path", "bam_kv_fast_path_batch"):
        ensure_bam_replay_cache_capacity(entries, args)
    store = build_store(backend, first.shape, dtype, args)
    expected: dict[str, torch.Tensor] = {}
    write_results = []
    read_results = []

    try:
        # replay 的整体形态是：
        #   write entries -> 可选记录 expected
        #   read entries  -> backend.get_chunk()
        #   校验与统计   -> exact_equal / latency / bw
        print("=" * 80)
        print(f"backend={backend}")
        print(f"num_entries={len(entries)}")
        print(f"shape={first.shape}")
        print(f"dtype={dtype}")
        print("=" * 80)

        pending_batch_reads: list[Any] = []

        def flush_pending_batch_reads() -> None:
            """提交并校验当前累积的一批 read entries。

            只有 batch backend 会走这里。其他 backend 仍保持原来的
            单 chunk 读写流程，避免 batch 实验影响已有 baseline。
            """
            nonlocal pending_batch_reads
            if not pending_batch_reads:
                return
            if not isinstance(
                    store,
                (BaMPrefetchBatchReplayStore, BaMKVFastPathBatchReplayStore)):
                raise RuntimeError("pending batch reads require batch store")

            results, outputs = store.get_chunks(pending_batch_reads, args)
            read_results.extend(results)
            for entry, result in zip(pending_batch_reads, results):
                out = outputs[entry.chunk_hash]
                if not args.no_verify:
                    ref = expected.get(entry.chunk_hash)
                    if ref is None:
                        raise KeyError(
                            f"read before write in trace: {entry.chunk_hash}")
                    exact = bool(torch.equal(out, ref))
                    if not exact:
                        diff = (out - ref).abs().max().item()
                        raise RuntimeError(
                            f"{backend} verify failed chunk={entry.chunk_hash} "
                            f"max_abs_diff={diff}")

                print(
                    f"[{backend} {entry.op}] chunk_hash={entry.chunk_hash[:16]} "
                    f"elapsed_ms={result.elapsed_ms:.3f} "
                    f"bw_gib_s={result.bw_gib_s:.3f}")
            pending_batch_reads = []

        for idx, entry in enumerate(entries):
            if tuple(entry.shape) != tuple(first.shape):
                raise ValueError("this simple replay currently expects one fixed shape")

            if entry.op == "write":
                flush_pending_batch_reads()
                tensor = make_tensor(entry.shape, dtype, args.device, salt=idx)
                # 合成数据由 CUDA kernel 生成；
                # 这里先同步，避免从未完成的 GPU 写入中读取数据，
                # 同时也避免把造数开销混进 put 计时。
                if tensor.is_cuda:
                    torch.cuda.synchronize(tensor.device)
                expected_tensor = None
                if not args.no_verify:
                    # expected 只用于后续 correctness 对照，应该在 BaM 写之前准备好。
                    # 这样如果 BaM store/flush 把 CUDA context 打坏，错误会定位到
                    # BaM 写后同步，而不是误报在 expected.clone() 上。
                    expected_tensor = tensor.detach().clone()
                result = store.put_chunk(entry.chunk_hash, tensor,
                                         entry.actual_tokens)
                if tensor.is_cuda:
                    # BaM C++/CUDA 扩展内部有自己的 synchronize，但 pybind 不一定
                    # 会把 device error 抛回 Python。这里再同步一次，把底层写/flush
                    # 的非法访存尽量归因到 write entry。
                    torch.cuda.synchronize(tensor.device)
                write_results.append(result)
                if isinstance(store, BaMReplayStore):
                    store.record_manifest_entry(entry.chunk_hash, idx)
                if expected_tensor is not None:
                    expected[entry.chunk_hash] = expected_tensor
            else:
                if isinstance(
                        store,
                    (BaMPrefetchBatchReplayStore, BaMKVFastPathBatchReplayStore)):
                    pending_batch_reads.append(entry)
                    continue

                out = torch.empty(entry.shape, device=args.device, dtype=dtype)
                result = store.get_chunk(entry.chunk_hash, out)
                read_results.append(result)
                if not args.no_verify:
                    ref = expected.get(entry.chunk_hash)
                    if ref is None and isinstance(store, BaMColdReadReplayStore):
                        ref = store.expected_tensor(entry.chunk_hash, entry.shape,
                                                    dtype, args.device)
                    if ref is None:
                        raise KeyError(
                            f"read before write in trace: {entry.chunk_hash}")
                    exact = bool(torch.equal(out, ref))
                    if not exact:
                        diff = (out - ref).abs().max().item()
                        raise RuntimeError(
                            f"{backend} verify failed chunk={entry.chunk_hash} "
                            f"max_abs_diff={diff}")

                print(
                    f"[{backend} {entry.op}] chunk_hash={entry.chunk_hash[:16]} "
                    f"elapsed_ms={result.elapsed_ms:.3f} "
                    f"bw_gib_s={result.bw_gib_s:.3f}")

        flush_pending_batch_reads()

        summarize(f"{backend}.write", write_results,
                  args.summary_warmup_samples)
        summarize(f"{backend}.read", read_results,
                  args.summary_warmup_samples)
    finally:
        store.close()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("LMCACHE_CHUNK_SIZE", str(args.slot_num_tokens))

    if args.trace_jsonl is None:
        entries = make_synthetic_trace(args)
    else:
        from vllm.bam.gds_baseline.trace_schema import read_trace

        entries = read_trace(args.trace_jsonl)
    if not entries:
        raise ValueError("empty trace")

    if args.backend == "bam_cold_read":
        entries = [entry for entry in entries if entry.op == "read"]

    # 主线对比是 BaM vs LMCache-style GDS。
    # raw slab GDS 是底层调试 backend，需要时单独跑，避免它的 cuFile
    # 限制挡住主实验。
    backends = ["bam", "lmcache_gds"] if args.backend == "all" else [
        args.backend
    ]
    for backend in backends:
        replay_backend(backend, entries, args)


if __name__ == "__main__":
    main()
