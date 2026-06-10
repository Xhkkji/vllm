#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""直接测量 vLLM V0 CacheEngine 的 CPU swap 基线。"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Iterable, Sequence

# 允许直接以脚本路径运行，无需预先 `pip install -e .`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="直接对 vLLM V0 CacheEngine 做 swap_in/swap_out microbenchmark。"
    )
    parser.add_argument("model", help="模型名或本地模型路径")
    parser.add_argument("--tokenizer",
                        default=None,
                        help="可选 tokenizer 路径，默认与模型相同")
    parser.add_argument("--dtype",
                        default="half",
                        help="传给 vLLM 的 dtype，V100 建议 half")
    parser.add_argument("--max-model-len",
                        type=int,
                        default=8192,
                        help="传给 vLLM 的 max_model_len")
    parser.add_argument("--gpu-memory-utilization",
                        type=float,
                        default=0.6,
                        help="vLLM 的 gpu_memory_utilization")
    parser.add_argument("--swap-space",
                        type=float,
                        default=8.0,
                        help="vLLM 的 swap_space，单位 GiB")
    parser.add_argument("--tensor-parallel-size",
                        type=int,
                        default=1,
                        help="张量并行数，V100 单卡实验通常保持 1")
    parser.add_argument("--device",
                        default="cuda",
                        help="显式指定 vLLM 设备类型，默认 cuda")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="随机种子")
    parser.add_argument("--trust-remote-code",
                        action="store_true",
                        help="是否信任远程自定义代码")
    parser.add_argument("--warmup-iters",
                        type=int,
                        default=2,
                        help="每个 batch size 的预热轮数")
    parser.add_argument("--repeat-iters",
                        type=int,
                        default=5,
                        help="每个 batch size 的正式测量轮数")
    parser.add_argument(
        "--batch-sizes",
        default="1,2,4,8,16,32,64,128,256,512,1024,2048,4096",
        help="逗号分隔的 mappings 数量列表，例如 64,256,1024",
    )
    parser.add_argument(
        "--ops",
        default="swap_out,swap_in,round_trip",
        help="逗号分隔的测试项，可选 swap_out,swap_in,round_trip",
    )
    parser.add_argument("--disable-async-output-proc",
                        action="store_true",
                        default=True,
                        help="关闭 async output processor，避免平台兼容问题")
    return parser.parse_args()


def parse_csv_ints(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError("batch sizes 不能为空")
    return sorted(set(values))


def parse_csv_strs(raw: str) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("ops 不能为空")
    invalid = sorted(set(values) - {"swap_out", "swap_in", "round_trip"})
    if invalid:
        raise ValueError(f"不支持的 op: {invalid}")
    return values


def percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("percentile() received empty values")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return sorted_values[low]
    ratio = pos - low
    return sorted_values[low] * (1 - ratio) + sorted_values[high] * ratio


def format_bytes(num_bytes: float) -> str:
    gib = num_bytes / (1024**3)
    mib = num_bytes / (1024**2)
    if gib >= 1:
        return f"{gib:.2f} GiB"
    return f"{mib:.2f} MiB"


def build_mapping(num_mappings: int) -> torch.Tensor:
    import torch

    # 用一一对应的 block 映射，避免额外引入复杂访问模式。
    pairs = [[idx, idx] for idx in range(num_mappings)]
    return torch.tensor(pairs, dtype=torch.int64, device="cpu")


def run_single_op(cache_engine, op: str, mapping: torch.Tensor) -> None:
    if op == "swap_out":
        cache_engine.swap_out(mapping)
    elif op == "swap_in":
        cache_engine.swap_in(mapping)
    elif op == "round_trip":
        cache_engine.swap_out(mapping)
        cache_engine.swap_in(mapping)
    else:
        raise ValueError(f"unexpected op: {op}")


def benchmark_op(cache_engine, op: str, mapping: torch.Tensor, warmup_iters: int,
                 repeat_iters: int) -> list[float]:
    import torch

    for _ in range(warmup_iters):
        torch.cuda.synchronize()
        run_single_op(cache_engine, op, mapping)
        torch.cuda.synchronize()

    elapsed_ms_list: list[float] = []
    for _ in range(repeat_iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        run_single_op(cache_engine, op, mapping)
        torch.cuda.synchronize()
        elapsed_ms_list.append((time.perf_counter() - start) * 1000.0)
    return elapsed_ms_list


def main() -> None:
    args = parse_args()

    # 这个脚本是纯 swap microbenchmark，不需要打开 trace 日志。
    os.environ["VLLM_USE_V1"] = "0"
    os.environ["VLLM_V0_SWAP_TRACE"] = "0"

    import torch
    from vllm import LLM
    from vllm.worker.cache_engine import CacheEngine

    batch_sizes = parse_csv_ints(args.batch_sizes)
    ops = parse_csv_strs(args.ops)
    tokenizer_name = args.tokenizer or args.model

    print("=" * 80)
    print("V0 swap microbenchmark")
    print(f"model={args.model}")
    print(f"tokenizer={tokenizer_name}")
    print(f"dtype={args.dtype}")
    print(f"max_model_len={args.max_model_len}")
    print(f"gpu_memory_utilization={args.gpu_memory_utilization}")
    print(f"swap_space={args.swap_space} GiB")
    print(f"ops={ops}")
    print(f"batch_sizes={batch_sizes}")
    print(f"warmup_iters={args.warmup_iters}")
    print(f"repeat_iters={args.repeat_iters}")
    print("=" * 80)

    # 这里只测 swap 后端，不关心 cudagraph，强制 eager 能明显减少初始化噪音。
    llm = LLM(
        model=args.model,
        tokenizer=tokenizer_name,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        swap_space=args.swap_space,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        disable_async_output_proc=args.disable_async_output_proc,
        preemption_mode="swap",
    )

    worker = llm.llm_engine.model_executor.driver_worker.worker
    cache_engine = worker.cache_engine[0]

    max_mappings = min(cache_engine.num_gpu_blocks, cache_engine.num_cpu_blocks)
    block_bytes = CacheEngine.get_cache_block_size(cache_engine.cache_config,
                                                   cache_engine.model_config,
                                                   cache_engine.parallel_config)

    print(f"num_gpu_blocks={cache_engine.num_gpu_blocks}")
    print(f"num_cpu_blocks={cache_engine.num_cpu_blocks}")
    print(f"max_mappings={max_mappings}")
    print(f"block_size_tokens={cache_engine.block_size}")
    print(f"block_bytes={block_bytes} ({format_bytes(block_bytes)})")
    print("=" * 80)

    valid_batch_sizes = [size for size in batch_sizes if size <= max_mappings]
    skipped_batch_sizes = [size for size in batch_sizes if size > max_mappings]
    if skipped_batch_sizes:
        print(f"skipped_batch_sizes={skipped_batch_sizes}")
        print("=" * 80)
    if not valid_batch_sizes:
        raise RuntimeError("没有可执行的 batch size，全部超过了可用 mappings 上限。")

    # 先触发一次最小 round trip，确保 CPU/GPU cache 都被实际访问过。
    bootstrap_mapping = build_mapping(min(8, max_mappings))
    benchmark_op(cache_engine, "round_trip", bootstrap_mapping, warmup_iters=1,
                 repeat_iters=1)

    for op in ops:
        print(f"[{op}]")
        for batch_size in valid_batch_sizes:
            mapping = build_mapping(batch_size)
            elapsed_ms_list = benchmark_op(cache_engine, op, mapping,
                                           args.warmup_iters,
                                           args.repeat_iters)
            sorted_elapsed = sorted(elapsed_ms_list)
            total_bytes = batch_size * block_bytes
            if op == "round_trip":
                total_bytes *= 2
            avg_elapsed = mean(sorted_elapsed)
            ms_per_block = avg_elapsed / batch_size
            gib_per_sec = (
                total_bytes / (avg_elapsed / 1000.0) / (1024**3)
                if avg_elapsed > 0 else math.inf)
            print(
                f"  mappings={batch_size:<5d} "
                f"avg_ms={avg_elapsed:>9.3f} "
                f"p50_ms={percentile(sorted_elapsed, 0.50):>9.3f} "
                f"p95_ms={percentile(sorted_elapsed, 0.95):>9.3f} "
                f"ms_per_block={ms_per_block:>9.6f} "
                f"bw_gib_s={gib_per_sec:>7.3f}")
        print()


if __name__ == "__main__":
    main()
