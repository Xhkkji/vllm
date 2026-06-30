# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch

from vllm.bam.lmcache_bam_storage import BaMChunkMetadata, LMCacheBaMStore


@dataclass(frozen=True)
class _FakeDescriptor:
    total_bytes: int


@dataclass(frozen=True)
class _FakeStats:
    submit_ms: float = 0.11
    poll_ms: float = 0.22
    poll_iters: int = 1
    get_ms: float = 0.33
    executor_name: str = "gpu_worker"
    worker_backend: str = "kv_cq_service_v1"


@dataclass(frozen=True)
class _FakeReadResult:
    descriptor: _FakeDescriptor
    stats: _FakeStats


class _FakeTokenDatabase:

    def __init__(self, entries: list[tuple[int, int, Any]]) -> None:
        self._entries = list(entries)

    def process_tokens(self, tokens: torch.Tensor,
                       mask: torch.Tensor | None):
        del tokens, mask
        yield from self._entries


def _make_store() -> LMCacheBaMStore:
    """构造一个最小可测的 BaMStore 实例。

    这里不走真实 `from_kv_shape()`，避免测试依赖 BaM 设备初始化。
    我们只保留 direct placement / metadata 相关字段。
    """
    layout = type(
        "Layout", (),
        {
            "page_bytes": 128 * 1024,
            "pages_per_chunk": 112,
            "num_layers": 2,
            "hidden_dim": 16,
            "slot_num_tokens": 256,
            "page_token_capacity": 128,
            "pages_per_kv_layer": 28,
            "dtype": torch.float16,
        })()
    return LMCacheBaMStore(
        row_store=object(),
        layout=layout,
        chunk_capacity=16,
        base_row_offset=0,
    )


def _register_chunk(store: LMCacheBaMStore, chunk_hash: str, slot_id: int) -> Any:
    key = type("Key", (), {"chunk_hash": chunk_hash})()
    store.register_existing_chunk(
        key,
        slot_id=slot_id,
        page_offset=slot_id * store.layout.pages_per_chunk,
        actual_tokens=256,
        shape=torch.Size([2, 2, 256, 16]),
        dtype=torch.float16,
    )
    return key


def test_collect_direct_placement_entries_stops_at_first_bam_miss():
    """前缀路径上只要 BaM 有一个 chunk 缺失，就必须立刻停止。

    这是 LMCache prefix 语义的关键约束：后面的 chunk 即使也在 BaM 中，
    也不能越过中间缺口继续 direct placement。
    """
    store = _make_store()
    key_a = _register_chunk(store, "chunk-a", slot_id=0)
    key_c = _register_chunk(store, "chunk-c", slot_id=2)
    key_b = type("Key", (), {"chunk_hash": "chunk-b"})()

    token_db = _FakeTokenDatabase([
        (0, 256, key_a),
        (256, 512, key_b),
        (512, 768, key_c),
    ])

    entries = store._collect_direct_placement_entries(
        token_database=token_db,
        tokens=torch.arange(768, dtype=torch.int64),
        mask=None,
    )

    assert [(start, end, entry_key.chunk_hash)
            for start, end, entry_key in entries] == [(0, 256, "chunk-a")]


def test_direct_place_chunks_returns_none_when_bam_has_no_prefix_hits(
    monkeypatch: pytest.MonkeyPatch,
):
    """BaM 0 命中时必须返回 None，交给 LMCache 原始 retrieve 回退。

    这里特意不返回“全 False mask”，因为那会让上层误以为 direct placement
    已经完整执行完成，从而吞掉后续的 LMCache SSD/native fallback。
    """
    monkeypatch.setenv("VLLM_BAM_KV_FAST_PATH", "1")
    store = _make_store()
    key = type("Key", (), {"chunk_hash": "miss"})()

    token_db = _FakeTokenDatabase([(0, 256, key)])
    ret = store.direct_place_chunks_to_vllm_kvcache(
        token_database=token_db,
        tokens=torch.arange(256, dtype=torch.int64),
        mask=None,
        kv_caches=[],
        slot_mapping=torch.arange(256, dtype=torch.int64),
        num_kv_heads=1,
        head_size=16,
    )

    assert ret is None


def test_direct_place_chunks_builds_ret_mask_and_calls_placer(
    monkeypatch: pytest.MonkeyPatch,
):
    """BaM 命中多个连续 chunks 时，应返回正确的局部 ret_mask。

    这个测试同时验证两件事：

    1. `chunk_starts` 会按 token_database 的局部坐标保留下来。
    2. `ret_mask` 会只覆盖当前 direct placement 实际命中的 chunk 区间。
    """
    monkeypatch.setenv("VLLM_BAM_KV_FAST_PATH", "1")

    store = _make_store()
    key_a = _register_chunk(store, "chunk-a", slot_id=0)
    key_b = _register_chunk(store, "chunk-b", slot_id=1)

    fake_results = [
        _FakeReadResult(
            descriptor=_FakeDescriptor(total_bytes=1024),
            stats=_FakeStats(),
        ),
        _FakeReadResult(
            descriptor=_FakeDescriptor(total_bytes=1024),
            stats=_FakeStats(),
        ),
    ]

    observed: dict[str, Any] = {}

    def _fake_read(keys: list[Any]) -> list[_FakeReadResult]:
        observed["read_keys"] = [key.chunk_hash for key in keys]
        return fake_results

    def _fake_place(**kwargs):
        observed["chunk_starts"] = list(kwargs["chunk_starts"])
        observed["slot_mapping"] = kwargs["slot_mapping"]
        return type(
            "PlaceStats", (),
            {
                "impl": "lmcache",
                "refill_ms": 1.0,
                "transfer_ms": 2.0,
                "fused_ms": 0.0,
                "place_ms": 3.0,
            })()

    monkeypatch.setattr(
        store,
        "read_chunk_pages_kv_fast_path_batch",
        _fake_read,
    )
    monkeypatch.setattr(
        "vllm.bam.lmcache_bam_storage.place_bam_results_to_vllm_kvcache",
        _fake_place,
    )

    token_db = _FakeTokenDatabase([
        (0, 256, key_a),
        (256, 512, key_b),
    ])
    ret = store.direct_place_chunks_to_vllm_kvcache(
        token_database=token_db,
        tokens=torch.arange(512, dtype=torch.int64),
        mask=None,
        kv_caches=[torch.zeros((2, 8, 16), dtype=torch.float16)],
        slot_mapping=torch.arange(512, dtype=torch.int64),
        num_kv_heads=1,
        head_size=16,
    )

    assert observed["read_keys"] == ["chunk-a", "chunk-b"]
    assert observed["chunk_starts"] == [0, 256]
    assert ret is not None
    assert ret.dtype == torch.bool
    assert ret.device.type == "cpu"
    assert bool(ret.all().item()) is True
