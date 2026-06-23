#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""测量同进程内 BaM 顺序写 dummy chunk 的首块/稳态开销。"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="顺序写 dummy LMCache KV chunk 到 BaM，拆开测首块、第二块和稳态。"
    )
    parser.add_argument("--num-iters",
                        type=int,
                        default=12,
                        help="总写入轮数，默认 12")
    parser.add_argument("--num-layers",
                        type=int,
                        default=28,
                        help="KV chunk 的 layer 数，默认 28")
    parser.add_argument("--slot-num-tokens",
                        type=int,
                        default=256,
                        help="每个 chunk 的 token 槽位，默认 256")
    parser.add_argument("--hidden-dim",
                        type=int,
                        default=512,
                        help="KV chunk 最后一维 hidden 大小，默认 512")
    parser.add_argument("--dtype",
                        default="float16",
                        choices=("float16", "float32"),
                        help="dummy chunk 的 dtype，默认 float16")
    parser.add_argument("--ctrl-idx",
                        type=int,
                        default=0,
                        help="BaM 控制 GPU index，默认 0")
    return parser.parse_args()


def summarize(label: str, values_ms: list[float], chunk_bytes: int) -> None:
    if not values_ms:
        print(f"[{label}] no samples")
        return

    mean_ms = statistics.mean(values_ms)
    median_ms = statistics.median(values_ms)
    min_ms = min(values_ms)
    max_ms = max(values_ms)
    gib_per_s = (chunk_bytes / (mean_ms / 1000.0)) / (1024**3)

    print(f"[{label}]")
    print(f"  samples={len(values_ms)}")
    print(f"  mean_ms={mean_ms:.3f}")
    print(f"  median_ms={median_ms:.3f}")
    print(f"  min_ms={min_ms:.3f}")
    print(f"  max_ms={max_ms:.3f}")
    print(f"  approx_bw_gib_s={gib_per_s:.3f}")


def main() -> None:
    args = parse_args()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for BaM write microbench")

    os.environ.setdefault("VLLM_BAM_IMPORT_PATH",
                          "/home/xhk/llm-inference/BaM_IOStack/gids_module")
    os.environ.setdefault("VLLM_BAM_CACHE_SIZE_MB", "64")
    os.environ.setdefault("VLLM_BAM_NUM_SSD", "1")
    os.environ.setdefault("VLLM_BAM_SSD_LIST", "0")
    os.environ.setdefault("VLLM_BAM_CTRL_IDX", str(args.ctrl_idx))
    os.environ.setdefault("LMCACHE_CHUNK_SIZE", str(args.slot_num_tokens))

    from vllm.bam.lmcache_bam_storage import LMCacheBaMAdapter

    dtype = getattr(torch, args.dtype)
    shape = (2, args.num_layers, args.slot_num_tokens, args.hidden_dim)
    device = torch.device(f"cuda:{args.ctrl_idx}")

    # 固定同一个 dummy chunk，避免把随机生成开销混进来。
    tensor = torch.arange(
        int(torch.tensor(shape).prod().item()),
        device=device,
        dtype=torch.int32,
    ).view(shape).to(dtype)

    adapter = LMCacheBaMAdapter.from_kv_shape(torch.Size(shape), dtype)

    class DummyKey:
        def __init__(self, chunk_hash: str) -> None:
            self.chunk_hash = chunk_hash

    chunk_bytes = int(tensor.numel() * tensor.element_size())
    elapsed_ms_list: list[float] = []

    print("=" * 80)
    print("LMCache BaM 128KB write microbench")
    print(f"shape={shape}")
    print(f"dtype={dtype}")
    print(f"chunk_bytes={chunk_bytes}")
    print(f"chunk_size_mib={chunk_bytes / (1024**2):.3f}")
    print(f"num_iters={args.num_iters}")
    print("=" * 80)

    for idx in range(args.num_iters):
        # 每轮使用新的 chunk hash，模拟真实请求顺序占用新槽位。
        chunk_hash = f"dummy_chunk_seq_{idx:04d}"
        key = DummyKey(chunk_hash)

        torch.cuda.synchronize(device)
        start = time.perf_counter()
        adapter.store_chunk(key, tensor)
        torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        elapsed_ms_list.append(elapsed_ms)
        bw_gib_s = (chunk_bytes / (elapsed_ms / 1000.0)) / (1024**3)
        print(
            f"[iter {idx:02d}] chunk_hash={chunk_hash} elapsed_ms={elapsed_ms:.3f} "
            f"bw_gib_s={bw_gib_s:.3f}"
        )

    print("=" * 80)
    summarize("first_write", elapsed_ms_list[:1], chunk_bytes)
    summarize("second_write", elapsed_ms_list[1:2], chunk_bytes)
    summarize("steady_state", elapsed_ms_list[2:], chunk_bytes)
    summarize("all_writes", elapsed_ms_list, chunk_bytes)
    print("=" * 80)


if __name__ == "__main__":
    main()
