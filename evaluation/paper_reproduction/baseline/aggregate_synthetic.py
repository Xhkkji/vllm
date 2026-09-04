#!/usr/bin/env python3
"""Validate and aggregate one real GranuleKV sync/async transport run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_markers(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.is_file():
        raise RuntimeError(f"missing daemon log: {path}")
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("tag") == "GranuleKV_BATCH_DONE":
            records.append(record)
    return records


def validate_backend(result_dir: Path, backend: str) -> dict[str, Any]:
    backend_dir = result_dir / backend
    summary_path = backend_dir / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"missing summary: {summary_path}")
    summary = json.loads(summary_path.read_text())
    if not summary.get("verified"):
        raise RuntimeError(f"payload verification failed for {backend}")

    markers = read_markers(backend_dir / "daemon.log")
    writes = [item for item in markers if item.get("operation") == "write"]
    reads = [item for item in markers if item.get("operation") == "read"]
    if not writes or not reads:
        raise RuntimeError(
            f"missing real SSD markers for {backend}: "
            f"writes={len(writes)} reads={len(reads)}")
    expected_reads = int(summary.get("windows", 1))
    if len(reads) != expected_reads:
        raise RuntimeError(
            f"incomplete read marker set for {backend}: "
            f"expected={expected_reads} actual={len(reads)}")
    last_read = reads[-1]
    if (int(last_read.get("active_logical_transfers", 0)) != 0
            or int(last_read.get("active_transfer_count", 0)) != 0):
        raise RuntimeError(
            f"active GranuleKV transfer remains after {backend}: {last_read}")

    summary = dict(summary)
    summary.update({
        "write_markers": len(writes),
        "read_markers": len(reads),
        "physical_write_io_ms": sum(
            float(item.get("io_elapsed_ms", 0.0)) for item in writes),
        "physical_read_io_ms": sum(
            float(item.get("io_elapsed_ms", 0.0)) for item in reads),
        "marker_validation": "pass",
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    sync = validate_backend(result_dir, "sync")
    asynchronous = validate_backend(result_dir, "async")

    sync_total = float(sync["total_ms"])
    async_total = float(asynchronous["total_ms"])
    speedup = sync_total / async_total if async_total else 0.0
    reduction = ((sync_total - async_total) / sync_total
                 if sync_total else 0.0)
    payload = {
        "schema_version": 1,
        "experiment": "granulekv_synthetic_transport",
        "sync": sync,
        "async": asynchronous,
        "async_vs_sync_speedup": speedup,
        "async_vs_sync_time_reduction": reduction,
        "marker_validation": "pass",
    }
    (result_dir / "comparison.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    overlap = float(asynchronous.get("overlap_ms", 0.0))
    lines = [
        "# GranuleKV transport baseline",
        "",
        "Both rows use the same GPU payload: GPU -> SSD write, GPU clear, "
        "SSD -> GPU restore, and CUDA compute. The daemon markers are "
        "required and are not synthesized from host timing.",
        "",
        "| Backend | Write markers | Read markers | Physical write ms | "
        "Physical read ms | Read span ms | Compute ms | I/O-compute "
        "overlap ms | Total ms | Verified |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, row in (("GranuleKV sync", sync),
                      ("GranuleKV async", asynchronous)):
        lines.append(
            f"| {name} | {row['write_markers']} | {row['read_markers']} | "
            f"{row['physical_write_io_ms']:.3f} | "
            f"{row['physical_read_io_ms']:.3f} | "
            f"{float(row.get('read_wall_ms', 0.0)):.3f} | "
            f"{float(row.get('compute_ms', 0.0)):.3f} | "
            f"{overlap if name.endswith('async') else 0.0:.3f} | "
            f"{float(row['total_ms']):.3f} | yes |")
    lines.extend([
        "",
        f"Async/sync speedup: **{speedup:.3f}x**",
        f"Async/sync end-to-end reduction: **{reduction * 100:.2f}%**",
        "",
        "`overlap_ms` is the intersection of the recorded request "
        "submit-to-ready spans and CUDA compute spans. It demonstrates "
        "schedule-level overlap; it is not a model throughput claim.",
    ])
    (result_dir / "RESULTS.md").write_text("\n".join(lines) + "\n",
                                          encoding="utf-8")
    print(f"GRANULEKV_SYNTHETIC_AGGREGATE_PASS results={result_dir}")


if __name__ == "__main__":
    main()
