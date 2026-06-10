#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""解析 V0 swap trace 日志，统计当前 CPU swap 基线。"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable, Sequence


LOG_GLOB_PATTERNS = (
    "v0_swap_trace*.log",
    "v0_swap_trace_*_*.log",
)

TRACE_RE = re.compile(
    r"\[V0_SWAP_TRACE\]\[CacheEngine\] "
    r"op=(?P<op>swap_in|swap_out) "
    r"mappings=(?P<mappings>\d+) "
    r"block_size=(?P<block_size>\d+) "
    r"block_bytes=(?P<block_bytes>\d+) "
    r"total_bytes=(?P<total_bytes>\d+) "
    r"num_attention_layers=(?P<num_attention_layers>\d+) "
    r"elapsed_ms=(?P<elapsed_ms>\d+(?:\.\d+)?)")


@dataclass(frozen=True)
class SwapEvent:
    op: str
    mappings: int
    block_size: int
    block_bytes: int
    total_bytes: int
    num_attention_layers: int
    elapsed_ms: float
    line_no: int

    @property
    def ms_per_block(self) -> float:
        return self.elapsed_ms / self.mappings

    @property
    def gib_per_sec(self) -> float:
        seconds = self.elapsed_ms / 1000.0
        if seconds <= 0:
            return math.inf
        return self.total_bytes / seconds / (1024**3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="统计 vLLM V0 CPU swap 日志中的 block 搬运基线。")
    parser.add_argument(
        "logfile",
        nargs="?",
        default=None,
        help="可选，指定日志文件；默认自动选择 evaluation/logs 下最新的 "
        "v0_swap_trace 日志",
    )
    parser.add_argument("--log-dir",
                        default="/home/xhk/llm-inference/vllm/evaluation/logs",
                        help="日志目录，默认是 evaluation/logs")
    parser.add_argument("--top-batches",
                        type=int,
                        default=8,
                        help="按出现次数展示前多少种 batch mappings")
    return parser.parse_args()


def find_latest_log(log_dir: Path) -> Path:
    candidates: list[Path] = []
    for pattern in LOG_GLOB_PATTERNS:
        candidates.extend(log_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"在 {log_dir} 下没有找到 v0_swap_trace 日志。")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_events(log_path: Path) -> list[SwapEvent]:
    events: list[SwapEvent] = []
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            match = TRACE_RE.search(line)
            if not match:
                continue
            events.append(
                SwapEvent(
                    op=match.group("op"),
                    mappings=int(match.group("mappings")),
                    block_size=int(match.group("block_size")),
                    block_bytes=int(match.group("block_bytes")),
                    total_bytes=int(match.group("total_bytes")),
                    num_attention_layers=int(
                        match.group("num_attention_layers")),
                    elapsed_ms=float(match.group("elapsed_ms")),
                    line_no=line_no,
                ))
    return events


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


def summarize_op(op: str, events: Sequence[SwapEvent], top_batches: int) -> None:
    total_mappings = sum(event.mappings for event in events)
    total_bytes = sum(event.total_bytes for event in events)
    total_elapsed_ms = sum(event.elapsed_ms for event in events)
    weighted_ms_per_block = total_elapsed_ms / total_mappings
    weighted_gib_per_sec = (
        total_bytes / (total_elapsed_ms / 1000.0) / (1024**3)
        if total_elapsed_ms > 0 else math.inf)

    event_ms = sorted(event.elapsed_ms for event in events)
    per_block_ms = sorted(event.ms_per_block for event in events)
    per_event_bw = sorted(event.gib_per_sec for event in events)

    print(f"[{op}]")
    print(f"  events={len(events)}")
    print(f"  total_mappings={total_mappings}")
    print(f"  total_bytes={format_bytes(total_bytes)}")
    print(f"  total_elapsed_ms={total_elapsed_ms:.3f}")
    print(f"  weighted_ms_per_block={weighted_ms_per_block:.6f}")
    print(f"  weighted_gib_per_sec={weighted_gib_per_sec:.3f}")
    print(f"  event_elapsed_ms_avg={mean(event_ms):.3f}")
    print(f"  event_elapsed_ms_p50={percentile(event_ms, 0.50):.3f}")
    print(f"  event_elapsed_ms_p95={percentile(event_ms, 0.95):.3f}")
    print(f"  per_block_ms_avg={mean(per_block_ms):.6f}")
    print(f"  per_block_ms_p50={percentile(per_block_ms, 0.50):.6f}")
    print(f"  per_block_ms_p95={percentile(per_block_ms, 0.95):.6f}")
    print(f"  event_bw_gib_s_avg={mean(per_event_bw):.3f}")
    print(f"  event_bw_gib_s_p50={percentile(per_event_bw, 0.50):.3f}")
    print(f"  event_bw_gib_s_p95={percentile(per_event_bw, 0.95):.3f}")

    batch_counter = Counter(event.mappings for event in events)
    batch_to_events: dict[int, list[SwapEvent]] = defaultdict(list)
    for event in events:
        batch_to_events[event.mappings].append(event)

    print("  common_batches:")
    for mappings, count in batch_counter.most_common(top_batches):
        grouped_events = batch_to_events[mappings]
        avg_elapsed = mean(event.elapsed_ms for event in grouped_events)
        avg_per_block = mean(event.ms_per_block for event in grouped_events)
        avg_bw = mean(event.gib_per_sec for event in grouped_events)
        print(
            "    "
            f"mappings={mappings:<5d} count={count:<4d} "
            f"avg_elapsed_ms={avg_elapsed:.3f} "
            f"avg_ms_per_block={avg_per_block:.6f} "
            f"avg_bw_gib_s={avg_bw:.3f}")


def print_roundtrip_hint(events_by_op: dict[str, list[SwapEvent]]) -> None:
    swap_in_events = events_by_op.get("swap_in", [])
    swap_out_events = events_by_op.get("swap_out", [])
    if not swap_in_events or not swap_out_events:
        return

    in_total_mappings = sum(event.mappings for event in swap_in_events)
    out_total_mappings = sum(event.mappings for event in swap_out_events)
    in_total_elapsed = sum(event.elapsed_ms for event in swap_in_events)
    out_total_elapsed = sum(event.elapsed_ms for event in swap_out_events)

    in_per_block = in_total_elapsed / in_total_mappings
    out_per_block = out_total_elapsed / out_total_mappings
    print("[round_trip_hint]")
    print(f"  swap_out_ms_per_block={out_per_block:.6f}")
    print(f"  swap_in_ms_per_block={in_per_block:.6f}")
    print(f"  swap_round_trip_ms_per_block={out_per_block + in_per_block:.6f}")
    print(
        "  note=这是按日志中所有事件加权后的每 block 往返估计，"
        "更接近真实批量搬运成本，不等同于单 block microbenchmark。")


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir)
    log_path = Path(args.logfile) if args.logfile else find_latest_log(log_dir)
    if not log_path.exists():
        raise FileNotFoundError(f"日志文件不存在: {log_path}")

    events = parse_events(log_path)
    if not events:
        raise RuntimeError(f"没有在日志中找到 CacheEngine swap trace: {log_path}")

    ops = {event.op for event in events}
    block_bytes = {event.block_bytes for event in events}
    block_sizes = {event.block_size for event in events}
    layer_counts = {event.num_attention_layers for event in events}

    print("=" * 80)
    print("V0 CPU swap baseline")
    print(f"log={log_path}")
    print(f"events={len(events)}")
    print(f"ops={sorted(ops)}")
    print(f"block_size_tokens={sorted(block_sizes)}")
    print(f"block_bytes={sorted(block_bytes)}")
    print(f"num_attention_layers={sorted(layer_counts)}")
    print("=" * 80)

    events_by_op: dict[str, list[SwapEvent]] = defaultdict(list)
    for event in events:
        events_by_op[event.op].append(event)

    for op in ("swap_out", "swap_in"):
        if op in events_by_op:
            summarize_op(op, events_by_op[op], args.top_batches)
            print()

    print_roundtrip_hint(events_by_op)
    print("=" * 80)


if __name__ == "__main__":
    main()
