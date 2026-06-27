# SPDX-License-Identifier: Apache-2.0
"""KV chunk trace 的轻量 JSONL 格式。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal

import torch


@dataclass(frozen=True)
class ChunkTraceEntry:
    op: Literal["write", "read"]
    chunk_hash: str
    shape: tuple[int, ...]
    dtype: str
    actual_tokens: int

    @classmethod
    def from_json(cls, line: str) -> "ChunkTraceEntry":
        data = json.loads(line)
        return cls(
            op=data["op"],
            chunk_hash=data["chunk_hash"],
            shape=tuple(int(v) for v in data["shape"]),
            dtype=data["dtype"],
            actual_tokens=int(data["actual_tokens"]),
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @property
    def torch_dtype(self) -> torch.dtype:
        if self.dtype == "float16":
            return torch.float16
        if self.dtype == "float32":
            return torch.float32
        raise ValueError(f"unsupported dtype in trace: {self.dtype}")


def read_trace(path: Path) -> list[ChunkTraceEntry]:
    entries: list[ChunkTraceEntry] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(ChunkTraceEntry.from_json(line))
    return entries


def write_trace(path: Path, entries: Iterable[ChunkTraceEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(entry.to_json() + "\n")
