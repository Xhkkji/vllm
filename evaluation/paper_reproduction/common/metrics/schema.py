"""Stable, paper-independent metric schema."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

METRIC_FIELDS = (
    "strategy", "selector_recall", "attention_output_error", "task_accuracy",
    "prefetch_precision", "prefetch_recall", "speculative_hit",
    "speculative_miss", "wasted_bytes", "correction_read_bytes",
    "descriptor_count", "average_descriptor_bytes", "ssd_io_us",
    "first_window_ready_us", "gpu_stall_us", "barrier_wait_us",
    "io_compute_overlap", "hbm_working_set_bytes", "ttft_us", "tpot_us",
    "e2e_latency_us", "throughput", "fallback_count", "active_requests",
)


@dataclass
class MetricRecord:
    strategy: str
    selector_recall: Optional[float] = None
    attention_output_error: Optional[float] = None
    task_accuracy: Optional[float] = None
    prefetch_precision: Optional[float] = None
    prefetch_recall: Optional[float] = None
    speculative_hit: Optional[int] = None
    speculative_miss: Optional[int] = None
    wasted_bytes: Optional[int] = None
    correction_read_bytes: Optional[int] = None
    descriptor_count: Optional[int] = None
    average_descriptor_bytes: Optional[float] = None
    ssd_io_us: Optional[float] = None
    first_window_ready_us: Optional[float] = None
    gpu_stall_us: Optional[float] = None
    barrier_wait_us: Optional[float] = None
    io_compute_overlap: Optional[float] = None
    hbm_working_set_bytes: Optional[int] = None
    ttft_us: Optional[float] = None
    tpot_us: Optional[float] = None
    e2e_latency_us: Optional[float] = None
    throughput: Optional[float] = None
    fallback_count: int = 0
    active_requests: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2) + "\n")

    def write_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS)
            writer.writeheader()
            writer.writerow(self.as_dict())
