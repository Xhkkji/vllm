# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch

from vllm.bam.lmcache_bam_direct_placement import (
    BaMDirectKVPlacer, BaMDirectPlacementBatchDescriptor,
    BaMDirectPlacementChunkDescriptor, BaMDirectPlacementStateTracker)
from vllm.bam.lmcache_bam_storage import BaMChunkMetadata, LMCacheBaMStore


@dataclass(frozen=True)
class _FakeDescriptor:
    total_bytes: int
    actual_tokens: int = 256


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


@dataclass(frozen=True)
class _FakeDirectPlacementDescriptor:
    actual_tokens: int
    total_bytes: int = 1024


@dataclass(frozen=True)
class _FakeDirectPlacementResult:
    descriptor: _FakeDirectPlacementDescriptor
    pages: Any = None


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

    class _FakeExecution:

        def __init__(self, state_tracker):
            self._state_tracker = state_tracker

        def advance_ready(self):
            return self._state_tracker.snapshot()

        def wait(self):
            self._state_tracker.mark_all_staged_ready()
            self._state_tracker.mark_all_cache_ready()
            return type(
                "PlaceStats", (),
                {
                    "impl": "lmcache",
                    "refill_ms": 1.0,
                    "transfer_ms": 2.0,
                    "fused_ms": 0.0,
                    "place_ms": 3.0,
                })(), self._state_tracker.snapshot()

    class _FakePlacer:

        def prepare_for_batch(self, **kwargs):
            del kwargs
            return None

        def start_batch(self, **kwargs):
            observed["chunk_starts"] = list(kwargs["chunk_starts"])
            observed["slot_mapping"] = kwargs["slot_mapping"]
            return object()

        def execution_from_launched_batch(self, *, launched_batch, state_tracker):
            del launched_batch
            return _FakeExecution(state_tracker)

        def log_launched_batch_step_timings(self, launched_batch):
            del launched_batch
            return None

    monkeypatch.setattr(
        store,
        "read_chunk_pages_kv_fast_path_batch",
        _fake_read,
    )
    monkeypatch.setattr(
        store,
        "_ensure_direct_kv_placer",
        lambda kv_cache_dtype: _FakePlacer(),
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
        slot_mapping=_FakeCudaTensor(torch.arange(512, dtype=torch.int64)),
        num_kv_heads=1,
        head_size=16,
    )

    assert observed["read_keys"] == ["chunk-a", "chunk-b"]
    assert observed["chunk_starts"] == [0, 256]
    assert ret is not None
    assert ret.dtype == torch.bool
    assert ret.device.type == "cpu"
    assert bool(ret.all().item()) is True

    snapshot = store.get_last_direct_placement_state_snapshot()
    assert snapshot is not None
    assert [chunk.descriptor.chunk_hash
            for chunk in snapshot.chunk_states] == ["chunk-a", "chunk-b"]
    assert snapshot.read_ready_chunks == 2
    assert snapshot.staged_ready_chunks == 2
    assert snapshot.cache_ready_chunks == 2
    assert snapshot.read_ready_tokens == 512
    assert snapshot.cache_ready_tokens == 512
    assert snapshot.consumable_chunks == 2
    assert snapshot.consumable_tokens == 512


def test_direct_placement_tracker_only_exposes_contiguous_cache_ready_prefix():
    """即使后面的 chunk 已 ready，也只能暴露从开头连续 cache-ready 的前缀。

    这是后续“chunk_ready -> chunk_consumable”最关键的约束：

    - chunk0 ready, chunk1 not ready, chunk2 ready
    - 真正可消费的前缀只能到 chunk0 结束

    否则上层会把中间存在空洞的 prefix 当成完整上下文使用，破坏 prefix 语义。
    """
    tracker = BaMDirectPlacementStateTracker(
        BaMDirectPlacementBatchDescriptor(
            chunks=(
                BaMDirectPlacementChunkDescriptor(
                    chunk_hash="chunk-a",
                    chunk_start=0,
                    chunk_end=256,
                    actual_tokens=256,
                    total_bytes=1024,
                ),
                BaMDirectPlacementChunkDescriptor(
                    chunk_hash="chunk-b",
                    chunk_start=256,
                    chunk_end=512,
                    actual_tokens=256,
                    total_bytes=1024,
                ),
                BaMDirectPlacementChunkDescriptor(
                    chunk_hash="chunk-c",
                    chunk_start=512,
                    chunk_end=768,
                    actual_tokens=256,
                    total_bytes=1024,
                ),
            ),
            total_tokens=768,
            total_bytes=3072,
        ))
    tracker.mark_all_read_ready()
    tracker.mark_chunk_staged_ready(0)
    tracker.mark_chunk_cache_ready(0)
    tracker.mark_chunk_staged_ready(2)
    tracker.mark_chunk_cache_ready(2)

    snapshot = tracker.snapshot()
    assert snapshot.cache_ready_chunks == 2
    assert snapshot.cache_ready_tokens == 512
    assert snapshot.consumable_chunks == 1
    assert snapshot.consumable_tokens == 256


def test_direct_place_chunks_returns_all_prefix_hits_when_single_wave_ready(
    monkeypatch: pytest.MonkeyPatch,
):
    """单波主线下，连续 prefix 命中多少，就应返回多少可消费前缀。"""
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

    def _fake_read(keys: list[Any]) -> list[_FakeReadResult]:
        del keys
        return fake_results

    class _FakeExecution:

        def __init__(self, state_tracker):
            self._state_tracker = state_tracker

        def advance_ready(self):
            return self._state_tracker.snapshot()

        def wait(self):
            self._state_tracker.mark_chunk_staged_ready(0)
            self._state_tracker.mark_chunk_cache_ready(0)
            self._state_tracker.mark_chunk_staged_ready(1)
            self._state_tracker.mark_chunk_cache_ready(1)
            return type(
                "PlaceStats", (),
                {
                    "impl": "fused",
                    "refill_ms": 0.0,
                    "transfer_ms": 0.0,
                    "fused_ms": 1.0,
                    "place_ms": 1.0,
                })(), self._state_tracker.snapshot()

    class _FakePlacer:

        def prepare_for_batch(self, **kwargs):
            del kwargs
            return None

        def start_batch(self, **kwargs):
            del kwargs
            return object()

        def execution_from_launched_batch(self, *, launched_batch, state_tracker):
            del launched_batch
            return _FakeExecution(state_tracker)

        def log_launched_batch_step_timings(self, launched_batch):
            del launched_batch
            return None

    monkeypatch.setattr(
        store,
        "read_chunk_pages_kv_fast_path_batch",
        _fake_read,
    )
    monkeypatch.setattr(
        store,
        "_ensure_direct_kv_placer",
        lambda kv_cache_dtype: _FakePlacer(),
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
        slot_mapping=_FakeCudaTensor(torch.arange(512, dtype=torch.int64)),
        num_kv_heads=1,
        head_size=16,
    )

    assert ret is not None
    assert ret.dtype == torch.bool
    assert ret.device.type == "cpu"
    assert bool(ret.all().item()) is True

    snapshot = store.get_last_direct_placement_state_snapshot()
    assert snapshot is not None
    assert snapshot.cache_ready_chunks == 2
    assert snapshot.cache_ready_tokens == 512
    assert snapshot.consumable_chunks == 2
    assert snapshot.consumable_tokens == 512
    assert int(ret.sum().item()) == snapshot.consumable_tokens


def test_direct_place_chunks_launches_all_prefix_hits_in_single_wave(
    monkeypatch: pytest.MonkeyPatch,
):
    """当前主线应把整段连续 prefix 一次性作为单波 launch 目标。"""
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
        del keys
        return fake_results

    class _FakeExecution:

        def __init__(self, state_tracker):
            self._state_tracker = state_tracker

        def advance_ready(self):
            return self._state_tracker.snapshot()

        def wait(self):
            self._state_tracker.mark_chunk_staged_ready(0)
            self._state_tracker.mark_chunk_cache_ready(0)
            self._state_tracker.mark_chunk_staged_ready(1)
            self._state_tracker.mark_chunk_cache_ready(1)
            return type(
                "PlaceStats", (),
                {
                    "impl": "fused",
                    "refill_ms": 0.0,
                    "transfer_ms": 0.0,
                    "fused_ms": 1.0,
                    "place_ms": 1.0,
                })(), self._state_tracker.snapshot()

    class _FakePlacer:

        def prepare_for_batch(self, **kwargs):
            observed["prepare_max_chunks_to_launch"] = kwargs[
                "max_chunks_to_launch"]
            return None

        def start_batch(self, **kwargs):
            observed["start_max_chunks_to_launch"] = kwargs[
                "max_chunks_to_launch"]
            return object()

        def execution_from_launched_batch(self, *, launched_batch, state_tracker):
            del launched_batch
            return _FakeExecution(state_tracker)

        def log_launched_batch_step_timings(self, launched_batch):
            del launched_batch
            return None

    monkeypatch.setattr(
        store,
        "read_chunk_pages_kv_fast_path_batch",
        _fake_read,
    )
    monkeypatch.setattr(
        store,
        "_ensure_direct_kv_placer",
        lambda kv_cache_dtype: _FakePlacer(),
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
        slot_mapping=_FakeCudaTensor(torch.arange(512, dtype=torch.int64)),
        num_kv_heads=1,
        head_size=16,
    )

    assert observed["prepare_max_chunks_to_launch"] == 2
    assert observed["start_max_chunks_to_launch"] == 2
    assert ret is not None
    assert bool(ret.all().item()) is True

    snapshot = store.get_last_direct_placement_state_snapshot()
    assert snapshot is not None
    assert snapshot.cache_ready_chunks == 2
    assert snapshot.consumable_chunks == 2
    assert snapshot.consumable_tokens == 512


def test_direct_place_chunks_finalize_runs_single_wave_only(
    monkeypatch: pytest.MonkeyPatch,
):
    """当前 direct placement finalize 只保留一次 prepare/start/wait 主线。"""
    monkeypatch.setenv("VLLM_BAM_KV_FAST_PATH", "1")

    store = _make_store()
    key_a = _register_chunk(store, "chunk-a", slot_id=0)
    key_b = _register_chunk(store, "chunk-b", slot_id=1)
    key_c = _register_chunk(store, "chunk-c", slot_id=2)

    fake_results = [
        _FakeReadResult(
            descriptor=_FakeDescriptor(total_bytes=1024),
            stats=_FakeStats(),
        ),
        _FakeReadResult(
            descriptor=_FakeDescriptor(total_bytes=1024),
            stats=_FakeStats(),
        ),
        _FakeReadResult(
            descriptor=_FakeDescriptor(total_bytes=1024),
            stats=_FakeStats(),
        ),
    ]
    observed: dict[str, Any] = {
        "prepare_calls": [],
        "start_calls": [],
    }

    def _fake_read(keys: list[Any]) -> list[_FakeReadResult]:
        del keys
        return fake_results

    class _FakeExecution:

        def __init__(self, state_tracker, launched_batch):
            self._state_tracker = state_tracker
            self._launched_batch = launched_batch

        def advance_ready(self):
            return self._state_tracker.snapshot()

        def wait(self):
            chunk_offset = self._launched_batch["launch_start_chunk"]
            chunk_count = self._launched_batch["launch_chunk_count"]
            for chunk_index in range(chunk_offset, chunk_offset + chunk_count):
                self._state_tracker.mark_chunk_staged_ready(chunk_index)
                self._state_tracker.mark_chunk_cache_ready(chunk_index)
            return type(
                "PlaceStats", (),
                {
                    "impl": "fused",
                    "refill_ms": 0.0,
                    "transfer_ms": 0.0,
                    "fused_ms": 1.0,
                    "place_ms": 1.0,
                })(), self._state_tracker.snapshot()

    class _FakePlacer:

        def prepare_for_batch(self, **kwargs):
            observed["prepare_calls"].append(
                (kwargs["launch_start_chunk"], kwargs["max_chunks_to_launch"]))
            return None

        def start_batch(self, **kwargs):
            observed["start_calls"].append(
                (kwargs["launch_start_chunk"], kwargs["max_chunks_to_launch"]))
            launch_chunk_count = kwargs["max_chunks_to_launch"]
            if launch_chunk_count is None or launch_chunk_count <= 0:
                launch_chunk_count = len(kwargs["results"]) - kwargs[
                    "launch_start_chunk"]
            return {
                "launch_start_chunk": kwargs["launch_start_chunk"],
                "launch_chunk_count": int(launch_chunk_count),
            }

        def execution_from_launched_batch(self, *, launched_batch, state_tracker):
            return _FakeExecution(state_tracker, launched_batch)

        def log_launched_batch_step_timings(self, launched_batch):
            del launched_batch
            return None

    monkeypatch.setattr(
        store,
        "read_chunk_pages_kv_fast_path_batch",
        _fake_read,
    )
    monkeypatch.setattr(
        store,
        "_ensure_direct_kv_placer",
        lambda kv_cache_dtype: _FakePlacer(),
    )

    token_db = _FakeTokenDatabase([
        (0, 256, key_a),
        (256, 512, key_b),
        (512, 768, key_c),
    ])
    ret = store.direct_place_chunks_to_vllm_kvcache(
        token_database=token_db,
        tokens=torch.arange(768, dtype=torch.int64),
        mask=None,
        kv_caches=[torch.zeros((2, 8, 16), dtype=torch.float16)],
        slot_mapping=_FakeCudaTensor(torch.arange(768, dtype=torch.int64)),
        num_kv_heads=1,
        head_size=16,
    )

    assert observed["prepare_calls"] == [(0, 3)]
    assert observed["start_calls"] == [(0, 3)]
    assert ret is not None
    assert bool(ret.all().item()) is True

    snapshot = store.get_last_direct_placement_state_snapshot()
    assert snapshot is not None
    assert snapshot.cache_ready_chunks == 3
    assert snapshot.cache_ready_tokens == 768
    assert snapshot.consumable_chunks == 3
    assert snapshot.consumable_tokens == 768
    assert int(ret.sum().item()) == 768


def test_direct_place_chunks_ret_mask_matches_consumable_tokens(
    monkeypatch: pytest.MonkeyPatch,
):
    """正常主线路径下，ret_mask token 数必须等于 consumable_tokens。

    这条约束非常关键，因为：

    - `ret_mask` 决定 LMCache / vLLM 以为恢复了多少 prefix
    - `consumable_tokens` 决定底层当前真正可以安全消费多少 prefix

    两者一旦不一致，就说明上层推理语义和底层 cache-ready 语义已经脱节。
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

    def _fake_read(keys: list[Any]) -> list[_FakeReadResult]:
        del keys
        return fake_results

    class _FakeExecution:

        def __init__(self, state_tracker):
            self._state_tracker = state_tracker

        def advance_ready(self):
            return self._state_tracker.snapshot()

        def wait_until_contiguous_cache_ready(self, target_chunks):
            del target_chunks
            self._state_tracker.mark_chunk_staged_ready(0)
            self._state_tracker.mark_chunk_cache_ready(0)
            self._state_tracker.mark_chunk_staged_ready(1)
            self._state_tracker.mark_chunk_cache_ready(1)
            return self._state_tracker.snapshot()

        def _build_stats(self):
            return type(
                "PlaceStats", (),
                {
                    "impl": "fused",
                    "refill_ms": 0.0,
                    "transfer_ms": 0.0,
                    "fused_ms": 1.0,
                    "place_ms": 1.0,
                })()

    class _FakePlacer:

        def prepare_for_batch(self, **kwargs):
            del kwargs
            return None

        def start_batch(self, **kwargs):
            del kwargs
            return object()

        def execution_from_launched_batch(self, *, launched_batch, state_tracker):
            del launched_batch
            return _FakeExecution(state_tracker)

        def log_launched_batch_step_timings(self, launched_batch):
            del launched_batch
            return None

    monkeypatch.setattr(
        store,
        "read_chunk_pages_kv_fast_path_batch",
        _fake_read,
    )
    monkeypatch.setattr(
        store,
        "_ensure_direct_kv_placer",
        lambda kv_cache_dtype: _FakePlacer(),
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
        slot_mapping=_FakeCudaTensor(torch.arange(512, dtype=torch.int64)),
        num_kv_heads=1,
        head_size=16,
    )

    snapshot = store.get_last_direct_placement_state_snapshot()
    assert ret is not None
    assert snapshot is not None
    assert int(ret.sum().item()) == snapshot.consumable_tokens == 512


def test_direct_placement_request_handle_start_poll_finalize(
    monkeypatch: pytest.MonkeyPatch,
):
    """request 级 handle 应能显式走通 start / poll / finalize 三段。

    这条测试不是重复验证 direct placement 数据面，而是专门保护这次新增的
    request 控制面边界：

    ```text
    start request
      -> poll request
      -> finalize request
    ```

    这样后续如果继续把 handle 往 runtime 上提，就不会因为 wrapper 仍可用而忽略
    掉 request 级接口已经被改坏。
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

    def _fake_read(keys: list[Any]) -> list[_FakeReadResult]:
        del keys
        return fake_results

    class _FakeExecution:

        def __init__(self, state_tracker):
            self._state_tracker = state_tracker

        def advance_ready(self):
            return self._state_tracker.snapshot()

        def wait_until_contiguous_cache_ready(self, target_chunks):
            del target_chunks
            self._state_tracker.mark_chunk_staged_ready(0)
            self._state_tracker.mark_chunk_cache_ready(0)
            self._state_tracker.mark_chunk_staged_ready(1)
            self._state_tracker.mark_chunk_cache_ready(1)
            return self._state_tracker.snapshot()

        def _build_stats(self):
            return type(
                "PlaceStats", (),
                {
                    "impl": "fused",
                    "refill_ms": 0.0,
                    "transfer_ms": 0.0,
                    "fused_ms": 1.0,
                    "place_ms": 1.0,
                })()

    class _FakePlacer:

        def prepare_for_batch(self, **kwargs):
            del kwargs
            return None

        def start_batch(self, **kwargs):
            del kwargs
            return object()

        def execution_from_launched_batch(self, *, launched_batch, state_tracker):
            del launched_batch
            return _FakeExecution(state_tracker)

        def log_launched_batch_step_timings(self, launched_batch):
            del launched_batch
            return None

    monkeypatch.setattr(
        store,
        "read_chunk_pages_kv_fast_path_batch",
        _fake_read,
    )
    monkeypatch.setattr(
        store,
        "_ensure_direct_kv_placer",
        lambda kv_cache_dtype: _FakePlacer(),
    )

    token_db = _FakeTokenDatabase([
        (0, 256, key_a),
        (256, 512, key_b),
    ])
    in_flight_request = store.start_direct_placement_request(
        token_database=token_db,
        tokens=torch.arange(512, dtype=torch.int64),
        mask=None,
        kv_caches=[torch.zeros((2, 8, 16), dtype=torch.float16)],
        slot_mapping=_FakeCudaTensor(torch.arange(512, dtype=torch.int64)),
        kv_cache_dtype="auto",
    )

    assert in_flight_request is not None
    poll_snapshot = store.poll_direct_placement_request(
        in_flight_request=in_flight_request,
    )
    assert poll_snapshot.consumable_chunks == 0
    assert poll_snapshot.consumable_tokens == 0

    ret = store.finalize_direct_placement_request(
        in_flight_request=in_flight_request,
    )
    snapshot = store.get_last_direct_placement_state_snapshot()
    assert snapshot is not None
    assert int(ret.sum().item()) == 512
    assert snapshot.consumable_chunks == 2
    assert snapshot.consumable_tokens == 512


def test_direct_placement_request_poll_is_observe_only_before_finalize(
    monkeypatch: pytest.MonkeyPatch,
):
    """request-handle 主线下，poll 只观察 native read frontier，不触发 placement。"""
    monkeypatch.setenv("VLLM_BAM_KV_FAST_PATH", "1")

    store = _make_store()
    key_a = _register_chunk(store, "chunk-a", slot_id=0)
    key_b = _register_chunk(store, "chunk-b", slot_id=1)

    fake_handle = object()

    def _fake_submit(keys: list[Any]):
        del keys
        return fake_handle

    def _fake_poll(handle: Any):
        assert handle is fake_handle
        return type(
            "PollSnapshot", (),
            {
                "ready": False,
                "poll_iters": 1,
                "host_status": 1,
                "launch_frontier_chunks": 2,
                "read_ready_frontier_chunks": 1,
                "cache_ready_frontier_chunks": 0,
                "consumable_frontier_chunks": 0,
                "total_chunks": 2,
                "error_code": 0,
            },
        )()

    observed: dict[str, int] = {
        "prepare_calls": 0,
        "start_batch_calls": 0,
    }

    class _FakePlacer:

        def prepare_for_batch(self, **kwargs):
            del kwargs
            observed["prepare_calls"] += 1
            return None

        def start_batch(self, **kwargs):
            del kwargs
            observed["start_batch_calls"] += 1
            return object()

        def log_launched_batch_step_timings(self, launched_batch):
            del launched_batch
            return None

    monkeypatch.setattr(
        store,
        "submit_chunk_pages_kv_fast_path_batch_request",
        _fake_submit,
    )
    monkeypatch.setattr(
        store,
        "poll_chunk_pages_kv_fast_path_batch_request",
        _fake_poll,
    )
    monkeypatch.setattr(
        store,
        "_ensure_direct_kv_placer",
        lambda kv_cache_dtype: _FakePlacer(),
    )

    token_db = _FakeTokenDatabase([
        (0, 256, key_a),
        (256, 512, key_b),
    ])
    in_flight_request = store.start_direct_placement_request(
        token_database=token_db,
        tokens=torch.arange(512, dtype=torch.int64),
        mask=None,
        kv_caches=[torch.zeros((2, 8, 16), dtype=torch.float16)],
        slot_mapping=_FakeCudaTensor(torch.arange(512, dtype=torch.int64)),
        kv_cache_dtype="auto",
    )

    assert in_flight_request is not None
    assert observed["prepare_calls"] == 0
    assert observed["start_batch_calls"] == 0

    poll_snapshot = store.poll_direct_placement_request(
        in_flight_request=in_flight_request,
    )
    assert observed["prepare_calls"] == 0
    assert observed["start_batch_calls"] == 0
    assert poll_snapshot.read_ready_chunks == 1
    assert poll_snapshot.read_ready_tokens == 256
    assert poll_snapshot.consumable_chunks == 0
    assert poll_snapshot.consumable_tokens == 0


def test_direct_placement_runtime_attached_finalize_uses_cleanup_only_path(
    monkeypatch: pytest.MonkeyPatch,
):
    """runtime-direct-placement 已接管数据搬运时，finalize 不应再物化 results。

    这条用例保护当前最新主线的关键语义：

    - GPU persistent service 已经把数据写到最终 paged KV cache
    - host finalize 只做 cleanup-only 收尾
    - 不再回到 `consume_chunk_pages_kv_fast_path_batch_request()` 构造 pages results
    """
    monkeypatch.setenv("VLLM_BAM_KV_FAST_PATH", "1")
    monkeypatch.setenv("VLLM_BAM_DIRECT_PLACEMENT_RUNTIME_ONE_COPY", "1")

    store = _make_store()
    key_a = _register_chunk(store, "chunk-a", slot_id=0)
    key_b = _register_chunk(store, "chunk-b", slot_id=1)

    fake_handle = object()
    observed = {
        "runtime_finalize_calls": 0,
        "materialize_consume_calls": 0,
    }

    def _fake_submit(keys: list[Any]):
        assert keys == [key_a, key_b]
        return fake_handle

    def _fake_attach(handle: Any, **kwargs):
        assert handle is fake_handle
        del kwargs
        return True

    def _fake_runtime_finalize(handle: Any, *, timeout_s: float | None = None):
        assert handle is fake_handle
        assert timeout_s is not None
        observed["runtime_finalize_calls"] += 1
        return True

    def _fake_consume(handle: Any, *, timeout_s: float | None = None):
        del handle, timeout_s
        observed["materialize_consume_calls"] += 1
        raise AssertionError("runtime-direct-placement should not materialize results")

    class _FakePlacer:

        def build_runtime_direct_placement_attachment(self, **kwargs):
            slot_mapping = kwargs["slot_mapping"]
            chunk_starts = kwargs["chunk_starts"]
            return type(
                "RuntimeAttachment", (),
                {
                    "slot_mapping": slot_mapping,
                    "chunk_starts": torch.tensor(chunk_starts, dtype=torch.int64),
                    "kv_cache_pointers_gpu": torch.zeros((1, ), dtype=torch.int64),
                    "page_buffer_size": 8,
                    "block_size": 2,
                    "page_token_capacity": 128,
                    "pages_per_kv_layer": 2,
                    "num_layers": 1,
                    "num_kv_heads": 4,
                    "head_size": 8,
                    "pack_size": 8,
                },
            )()

    monkeypatch.setattr(
        store,
        "submit_chunk_pages_kv_fast_path_batch_request",
        _fake_submit,
    )
    monkeypatch.setattr(
        store,
        "attach_chunk_pages_kv_fast_path_batch_request_runtime_direct_placement",
        _fake_attach,
    )
    monkeypatch.setattr(
        store,
        "finalize_chunk_pages_kv_fast_path_batch_request_runtime_direct_placement",
        _fake_runtime_finalize,
    )
    monkeypatch.setattr(
        store,
        "consume_chunk_pages_kv_fast_path_batch_request",
        _fake_consume,
    )
    monkeypatch.setattr(
        store,
        "_ensure_direct_kv_placer",
        lambda kv_cache_dtype: _FakePlacer(),
    )

    token_db = _FakeTokenDatabase([
        (0, 256, key_a),
        (256, 512, key_b),
    ])
    in_flight_request = store.start_direct_placement_request(
        token_database=token_db,
        tokens=torch.arange(512, dtype=torch.int64),
        mask=None,
        kv_caches=[torch.zeros((2, 8, 16), dtype=torch.float16)],
        slot_mapping=_FakeCudaTensor(torch.arange(512, dtype=torch.int64)),
        kv_cache_dtype="auto",
    )

    assert in_flight_request is not None
    assert in_flight_request.runtime_direct_placement_attached is True

    ret = store.finalize_direct_placement_request(
        in_flight_request=in_flight_request,
    )
    snapshot = store.get_last_direct_placement_state_snapshot()

    assert observed["runtime_finalize_calls"] == 1
    assert observed["materialize_consume_calls"] == 0
    assert int(ret.sum().item()) == 512
    assert snapshot is not None
    assert snapshot.consumable_chunks == 2
    assert snapshot.consumable_tokens == 512


def test_runtime_one_copy_required_fails_fast_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
):
    """strict 主线下若未打开 one-copy，应在 start 阶段直接失败。"""
    monkeypatch.setenv("VLLM_BAM_KV_FAST_PATH", "1")
    monkeypatch.setenv("GIDS_KV_GPU_WORKER_RUNTIME_ENABLE", "1")
    monkeypatch.setenv("GIDS_KV_GPU_WORKER_PERSISTENT_ENABLE", "1")
    monkeypatch.setenv("VLLM_BAM_DIRECT_PLACEMENT_RUNTIME_ONE_COPY", "0")
    monkeypatch.setenv(
        "VLLM_BAM_DIRECT_PLACEMENT_REQUIRE_RUNTIME_ONE_COPY",
        "1",
    )

    store = _make_store()
    key_a = _register_chunk(store, "chunk-a", slot_id=0)
    key_b = _register_chunk(store, "chunk-b", slot_id=1)

    fake_handle = object()

    def _fake_submit(keys: list[Any]):
        assert keys == [key_a, key_b]
        return fake_handle

    class _FakePlacer:

        def build_runtime_direct_placement_attachment(self, **kwargs):
            raise AssertionError(
                "strict disabled path should fail before building attachment")

    monkeypatch.setattr(
        store,
        "submit_chunk_pages_kv_fast_path_batch_request",
        _fake_submit,
    )
    monkeypatch.setattr(
        store,
        "_ensure_direct_kv_placer",
        lambda kv_cache_dtype: _FakePlacer(),
    )

    token_db = _FakeTokenDatabase([
        (0, 256, key_a),
        (256, 512, key_b),
    ])
    with pytest.raises(
            RuntimeError,
            match="requires VLLM_BAM_DIRECT_PLACEMENT_RUNTIME_ONE_COPY=1"):
        store.start_direct_placement_request(
            token_database=token_db,
            tokens=torch.arange(512, dtype=torch.int64),
            mask=None,
            kv_caches=[torch.zeros((2, 8, 16), dtype=torch.float16)],
            slot_mapping=_FakeCudaTensor(torch.arange(512, dtype=torch.int64)),
            kv_cache_dtype="auto",
        )


def test_runtime_one_copy_required_fails_fast_when_attach_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    """strict 主线下 attach 失败应立即报错，而不是退回旧 host finalize。"""
    monkeypatch.setenv("VLLM_BAM_KV_FAST_PATH", "1")
    monkeypatch.setenv("GIDS_KV_GPU_WORKER_RUNTIME_ENABLE", "1")
    monkeypatch.setenv("GIDS_KV_GPU_WORKER_PERSISTENT_ENABLE", "1")
    monkeypatch.setenv("VLLM_BAM_DIRECT_PLACEMENT_RUNTIME_ONE_COPY", "1")
    monkeypatch.setenv(
        "VLLM_BAM_DIRECT_PLACEMENT_REQUIRE_RUNTIME_ONE_COPY",
        "1",
    )

    store = _make_store()
    key_a = _register_chunk(store, "chunk-a", slot_id=0)
    key_b = _register_chunk(store, "chunk-b", slot_id=1)

    fake_handle = object()

    def _fake_submit(keys: list[Any]):
        assert keys == [key_a, key_b]
        return fake_handle

    def _fake_attach(handle: Any, **kwargs):
        assert handle is fake_handle
        del kwargs
        return False

    class _FakePlacer:

        def build_runtime_direct_placement_attachment(self, **kwargs):
            slot_mapping = kwargs["slot_mapping"]
            chunk_starts = kwargs["chunk_starts"]
            return type(
                "RuntimeAttachment", (),
                {
                    "slot_mapping": slot_mapping,
                    "chunk_starts": torch.tensor(chunk_starts, dtype=torch.int64),
                    "kv_cache_pointers_gpu": torch.zeros((1, ), dtype=torch.int64),
                    "page_buffer_size": 8,
                    "block_size": 2,
                    "page_token_capacity": 128,
                    "pages_per_kv_layer": 2,
                    "num_layers": 1,
                    "num_kv_heads": 4,
                    "head_size": 8,
                    "pack_size": 8,
                },
            )()

    monkeypatch.setattr(
        store,
        "submit_chunk_pages_kv_fast_path_batch_request",
        _fake_submit,
    )
    monkeypatch.setattr(
        store,
        "attach_chunk_pages_kv_fast_path_batch_request_runtime_direct_placement",
        _fake_attach,
    )
    monkeypatch.setattr(
        store,
        "_ensure_direct_kv_placer",
        lambda kv_cache_dtype: _FakePlacer(),
    )

    token_db = _FakeTokenDatabase([
        (0, 256, key_a),
        (256, 512, key_b),
    ])
    with pytest.raises(
            RuntimeError,
            match="requires successful runtime one-copy attach"):
        store.start_direct_placement_request(
            token_database=token_db,
            tokens=torch.arange(512, dtype=torch.int64),
            mask=None,
            kv_caches=[torch.zeros((2, 8, 16), dtype=torch.float16)],
            slot_mapping=_FakeCudaTensor(torch.arange(512, dtype=torch.int64)),
            kv_cache_dtype="auto",
        )


def test_direct_placement_request_poll_surfaces_runtime_consumable_frontier(
    monkeypatch: pytest.MonkeyPatch,
):
    """runtime direct placement attach 后，poll 应直接暴露 GPU 发布的 consumable。

    这条用例保护当前目标主线的关键语义：

    - CPU 调度热路径只依赖 `poll()` 返回值
    - 一旦 GPU 后台已经把前缀写到最终 KV cache，`consumable_chunks`
      就应立刻在 poll snapshot 里可见
    - 不需要再额外高频调用 `get_frontier()` 才能看见最新可计算边界
    """
    monkeypatch.setenv("VLLM_BAM_KV_FAST_PATH", "1")
    monkeypatch.setenv("VLLM_BAM_DIRECT_PLACEMENT_RUNTIME_ONE_COPY", "1")

    store = _make_store()
    key_a = _register_chunk(store, "chunk-a", slot_id=0)
    key_b = _register_chunk(store, "chunk-b", slot_id=1)

    fake_handle = object()

    def _fake_submit(keys: list[Any]):
        assert keys == [key_a, key_b]
        return fake_handle

    def _fake_attach(handle: Any, **kwargs):
        assert handle is fake_handle
        del kwargs
        return True

    def _fake_poll(handle: Any):
        assert handle is fake_handle
        return type(
            "PollSnapshot", (),
            {
                "ready": False,
                "poll_iters": 3,
                "host_status": 2,
                "launch_frontier_chunks": 2,
                "read_ready_frontier_chunks": 2,
                "cache_ready_frontier_chunks": 1,
                "consumable_frontier_chunks": 1,
                "total_chunks": 2,
                "error_code": 0,
            },
        )()

    class _FakePlacer:

        def build_runtime_direct_placement_attachment(self, **kwargs):
            slot_mapping = kwargs["slot_mapping"]
            chunk_starts = kwargs["chunk_starts"]
            return type(
                "RuntimeAttachment", (),
                {
                    "slot_mapping": slot_mapping,
                    "chunk_starts": torch.tensor(chunk_starts, dtype=torch.int64),
                    "kv_cache_pointers_gpu": torch.zeros((1, ), dtype=torch.int64),
                    "page_buffer_size": 8,
                    "block_size": 2,
                    "page_token_capacity": 128,
                    "pages_per_kv_layer": 2,
                    "num_layers": 1,
                    "num_kv_heads": 4,
                    "head_size": 8,
                    "pack_size": 8,
                },
            )()

    monkeypatch.setattr(
        store,
        "submit_chunk_pages_kv_fast_path_batch_request",
        _fake_submit,
    )
    monkeypatch.setattr(
        store,
        "attach_chunk_pages_kv_fast_path_batch_request_runtime_direct_placement",
        _fake_attach,
    )
    monkeypatch.setattr(
        store,
        "poll_chunk_pages_kv_fast_path_batch_request",
        _fake_poll,
    )
    monkeypatch.setattr(
        store,
        "_ensure_direct_kv_placer",
        lambda kv_cache_dtype: _FakePlacer(),
    )

    token_db = _FakeTokenDatabase([
        (0, 256, key_a),
        (256, 512, key_b),
    ])
    in_flight_request = store.start_direct_placement_request(
        token_database=token_db,
        tokens=torch.arange(512, dtype=torch.int64),
        mask=None,
        kv_caches=[torch.zeros((2, 8, 16), dtype=torch.float16)],
        slot_mapping=_FakeCudaTensor(torch.arange(512, dtype=torch.int64)),
        kv_cache_dtype="auto",
    )

    assert in_flight_request is not None
    assert in_flight_request.runtime_direct_placement_attached is True

    poll_snapshot = store.poll_direct_placement_request(
        in_flight_request=in_flight_request,
    )
    frontier_snapshot = store.get_direct_placement_request_frontier(
        in_flight_request=in_flight_request,
    )

    assert poll_snapshot.read_ready_chunks == 2
    assert poll_snapshot.cache_ready_chunks == 1
    assert poll_snapshot.consumable_chunks == 1
    assert poll_snapshot.read_ready_tokens == 512
    assert poll_snapshot.cache_ready_tokens == 256
    assert poll_snapshot.consumable_tokens == 256
    assert frontier_snapshot.cache_ready_frontier_chunks == 1
    assert frontier_snapshot.consumable_frontier_chunks == 1


def test_runtime_direct_placement_attach_can_be_explicitly_disabled(
    monkeypatch: pytest.MonkeyPatch,
):
    """默认主线允许显式跳过 runtime one-copy attach。

    当前正式主线优先保证：

    - GPU worker / persistent service 继续负责后台 poll/read
    - 但不稳定的 runtime one-copy scatter 不应默认偷偷打开

    因此一旦显式关闭这条实验主线，storage 应直接返回 `(False, None)`，
    让后续自然走已验证正确的 materialized/fused finalize。
    """
    monkeypatch.setenv("VLLM_BAM_DIRECT_PLACEMENT_RUNTIME_ONE_COPY", "0")

    store = _make_store()

    class _FakePlacer:

        def build_runtime_direct_placement_attachment(self, **kwargs):
            raise AssertionError(
                "runtime one-copy disabled path should not build attachment")

    attached, attachment = store._try_attach_runtime_direct_placement(
        kv_read_handle=object(),
        direct_placer=_FakePlacer(),
        kv_caches=[torch.zeros((2, 8, 16), dtype=torch.float16)],
        slot_mapping=_FakeCudaTensor(torch.arange(256, dtype=torch.int64)),
        chunk_starts=[0],
    )

    assert attached is False
    assert attachment is None


def test_direct_placement_request_handle_prefers_frontier_v2_interfaces(
    monkeypatch: pytest.MonkeyPatch,
):
    """finalize 阶段应优先消费 execution 导出的 frontier v2 等待接口。"""
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

    def _fake_read(keys: list[Any]) -> list[_FakeReadResult]:
        del keys
        return fake_results

    observed = {
        "wait_frontier_v2_calls": 0,
        "legacy_advance_ready_calls": 0,
        "legacy_wait_calls": 0,
    }

    def _make_frontier_snapshot(
        *,
        launch_chunks: int,
        read_ready_chunks: int,
        staged_ready_chunks: int,
        cache_ready_chunks: int,
        consumable_chunks: int,
        total_chunks: int,
    ):
        token_unit = 256
        return type(
            "FrontierSnapshot", (),
            {
                "frontier_row": (
                    4 if consumable_chunks > 0 else
                    3 if cache_ready_chunks > 0 else
                    2 if read_ready_chunks > 0 else 1,
                    launch_chunks,
                    read_ready_chunks,
                    cache_ready_chunks,
                    consumable_chunks,
                    total_chunks,
                    0,
                ),
                "launch_frontier_chunks": launch_chunks,
                "read_ready_frontier_chunks": read_ready_chunks,
                "staged_ready_frontier_chunks": staged_ready_chunks,
                "cache_ready_frontier_chunks": cache_ready_chunks,
                "consumable_frontier_chunks": consumable_chunks,
                "total_chunks": total_chunks,
                "read_ready_frontier_tokens": read_ready_chunks * token_unit,
                "staged_ready_frontier_tokens": staged_ready_chunks * token_unit,
                "cache_ready_frontier_tokens": cache_ready_chunks * token_unit,
                "consumable_frontier_tokens": consumable_chunks * token_unit,
                "error_code": 0,
            },
        )()

    class _FakeExecution:

        def __init__(self, state_tracker):
            self._state_tracker = state_tracker

        def wait_until_contiguous_cache_ready_frontier(self, target_chunks):
            assert target_chunks == 2
            observed["wait_frontier_v2_calls"] += 1
            self._state_tracker.mark_chunks_read_ready_upto(2)
            self._state_tracker.mark_chunks_staged_ready_upto(2)
            self._state_tracker.mark_chunks_cache_ready_upto(2)
            return _make_frontier_snapshot(
                launch_chunks=2,
                read_ready_chunks=2,
                staged_ready_chunks=2,
                cache_ready_chunks=2,
                consumable_chunks=2,
                total_chunks=2,
            )

        def advance_ready(self):
            observed["legacy_advance_ready_calls"] += 1
            return self._state_tracker.snapshot()

        def wait_until_contiguous_cache_ready(self, target_chunks):
            del target_chunks
            observed["legacy_wait_calls"] += 1
            self._state_tracker.mark_chunks_staged_ready_upto(2)
            self._state_tracker.mark_chunks_cache_ready_upto(2)
            return self._state_tracker.snapshot()

        def get_stats(self):
            return type(
                "PlaceStats", (),
                {
                    "impl": "fused",
                    "refill_ms": 0.0,
                    "transfer_ms": 0.0,
                    "fused_ms": 1.0,
                    "place_ms": 1.0,
                })()

    class _FakePlacer:

        def prepare_for_batch(self, **kwargs):
            del kwargs
            return None

        def start_batch(self, **kwargs):
            del kwargs
            return object()

        def execution_from_launched_batch(self, *, launched_batch, state_tracker):
            del launched_batch
            return _FakeExecution(state_tracker)

        def log_launched_batch_step_timings(self, launched_batch):
            del launched_batch
            return None

    monkeypatch.setattr(
        store,
        "read_chunk_pages_kv_fast_path_batch",
        _fake_read,
    )
    monkeypatch.setattr(
        store,
        "_ensure_direct_kv_placer",
        lambda kv_cache_dtype: _FakePlacer(),
    )

    token_db = _FakeTokenDatabase([
        (0, 256, key_a),
        (256, 512, key_b),
    ])
    in_flight_request = store.start_direct_placement_request(
        token_database=token_db,
        tokens=torch.arange(512, dtype=torch.int64),
        mask=None,
        kv_caches=[torch.zeros((2, 8, 16), dtype=torch.float16)],
        slot_mapping=_FakeCudaTensor(torch.arange(512, dtype=torch.int64)),
        kv_cache_dtype="auto",
    )

    assert in_flight_request is not None

    first_poll_snapshot = store.poll_direct_placement_request(
        in_flight_request=in_flight_request,
    )
    assert first_poll_snapshot.consumable_chunks == 0

    second_poll_snapshot = store.poll_direct_placement_request(
        in_flight_request=in_flight_request,
    )
    assert observed["legacy_advance_ready_calls"] == 0
    assert second_poll_snapshot.consumable_chunks == 0
    assert second_poll_snapshot.consumable_tokens == 0

    ret = store.finalize_direct_placement_request(
        in_flight_request=in_flight_request,
    )
    snapshot = store.get_last_direct_placement_state_snapshot()
    assert observed["wait_frontier_v2_calls"] == 1
    assert observed["legacy_advance_ready_calls"] == 1
    assert observed["legacy_wait_calls"] == 0
    assert int(ret.sum().item()) == 512
    assert snapshot is not None
    assert snapshot.consumable_chunks == 2


def test_direct_placement_fused_path_uses_plan_entries(
    monkeypatch: pytest.MonkeyPatch,
):
    """fused 路径应按 plan 逐 chunk 调度，而不是退回逐层/KV 的旧组织方式。

    这个测试不依赖真实 CUDA/Triton，只验证控制面是否已经收口到：

    ```text
    build plan
      -> fused plan executor
      -> 每个 entry 一次 fused chunk placement
    ```

    这样后续继续把“逐 chunk”收缩成 batch kernel 时，只需要替换执行函数，
    而不会把上层 direct placement 入口重新改散。
    """
    monkeypatch.setenv("VLLM_BAM_DIRECT_PLACEMENT_IMPL", "fused")

    layout = type(
        "Layout", (),
        {
            "page_bytes": 128 * 1024,
            "pages_per_chunk": 112,
            "num_layers": 2,
            "hidden_dim": 16,
            "slot_num_tokens": 256,
            "page_token_capacity": 128,
            "pages_per_kv_layer": 1,
            "dtype": torch.float16,
        })()
    placer = BaMDirectKVPlacer(layout=layout)

    # 避免测试依赖真实 CUDA event / synchronize。
    monkeypatch.setattr(
        "vllm.bam.lmcache_bam_direct_placement.torch.cuda.synchronize",
        lambda device=None: None,
    )
    monkeypatch.setattr(
        "vllm.bam.lmcache_bam_direct_placement.BaMDirectKVPlacer._new_cuda_event_pair",
        staticmethod(lambda: (_FakeEvent(), _FakeEvent())),
    )

    observed: dict[str, Any] = {}

    def _fake_ensure_state(kv_caches):
        observed["ensure_state_layers"] = len(kv_caches)

    def _fake_fused_plan(plan, kv_caches):
        observed["chunk_tokens"] = [entry.actual_tokens for entry in plan.entries]
        observed["chunk_starts"] = [entry.chunk_start for entry in plan.entries]
        observed["slot_slices"] = [
            entry.slot_mapping.detach().cpu().tolist() for entry in plan.entries
        ]
        observed["kv_cache_layers"] = len(kv_caches)
        return [
            (entry.chunk_start, entry.actual_tokens, _FakeEvent(), _FakeEvent())
            for entry in plan.entries
        ]

    monkeypatch.setattr(placer, "_ensure_lmcache_connector_state", _fake_ensure_state)
    monkeypatch.setattr(placer, "_fused_plan_entries_to_vllm_cache", _fake_fused_plan)

    results = [
        _FakeDirectPlacementResult(
            descriptor=_FakeDirectPlacementDescriptor(actual_tokens=256)),
        _FakeDirectPlacementResult(
            descriptor=_FakeDirectPlacementDescriptor(actual_tokens=128)),
    ]
    state_tracker = BaMDirectPlacementStateTracker(
        BaMDirectPlacementBatchDescriptor(
            chunks=(
                BaMDirectPlacementChunkDescriptor(
                    chunk_hash="chunk-a",
                    chunk_start=0,
                    chunk_end=256,
                    actual_tokens=256,
                    total_bytes=1024,
                ),
                BaMDirectPlacementChunkDescriptor(
                    chunk_hash="chunk-b",
                    chunk_start=256,
                    chunk_end=384,
                    actual_tokens=128,
                    total_bytes=1024,
                ),
            ),
            total_tokens=384,
            total_bytes=2048,
        ))
    kv_caches = [torch.zeros((2, 8, 16), dtype=torch.float16) for _ in range(2)]
    slot_mapping = torch.arange(512, dtype=torch.int64)
    fake_cuda_slot_mapping = _FakeCudaTensor(slot_mapping)

    stats = placer.place_batch(
        results=results,
        kv_caches=kv_caches,
        slot_mapping=fake_cuda_slot_mapping,
        chunk_starts=[0, 256],
        state_tracker=state_tracker,
    )

    assert observed["ensure_state_layers"] == 2
    assert observed["kv_cache_layers"] == 2
    assert observed["chunk_tokens"] == [256, 128]
    assert observed["chunk_starts"] == [0, 256]
    assert observed["slot_slices"][0] == list(range(256))
    assert observed["slot_slices"][1] == list(range(256, 384))
    assert stats.impl == "fused"
    assert stats.tokens == 384
    assert stats.fused_ms == 0.0
    snapshot = state_tracker.snapshot()
    assert snapshot.staged_ready_chunks == 2
    assert snapshot.cache_ready_chunks == 2
    assert snapshot.cache_ready_tokens == 384
    assert snapshot.consumable_chunks == 2
    assert snapshot.consumable_tokens == 384


def test_fused_warmup_uses_current_stream_sync_helper(
    monkeypatch: pytest.MonkeyPatch,
):
    """fused warmup 不应再做 device-wide synchronize。

    当前主线里 GPU worker persistent service 可能长期常驻。若 warmup 继续调用
    `torch.cuda.synchronize(device)`，就会把后台 service 也纳入等待范围，导致
    `prepare_for_batch()` 卡住。

    这条测试不依赖真实 CUDA，只保护控制面语义：

    - `_maybe_warmup_fused()` 完成一次 launch 后
    - 应调用新的“只同步当前 stream”的 helper
    - 而不是重新把 device-wide synchronize 漏回主线
    """
    monkeypatch.setattr(
        "vllm.bam.lmcache_bam_direct_placement.triton",
        object(),
    )

    layout = type(
        "Layout", (),
        {
            "page_bytes": 128 * 1024,
            "pages_per_chunk": 112,
            "num_layers": 2,
            "hidden_dim": 16,
            "slot_num_tokens": 256,
            "page_token_capacity": 128,
            "pages_per_kv_layer": 1,
            "dtype": torch.float16,
        })()
    placer = BaMDirectKVPlacer(layout=layout)
    placer._page_buffer_size = 16

    observed = {"launch": 0, "sync": 0}

    monkeypatch.setattr(
        placer,
        "_launch_fused_pages_to_vllm_cache",
        lambda *args, **kwargs: observed.__setitem__("launch",
                                                     observed["launch"] + 1),
    )
    monkeypatch.setattr(
        placer,
        "_synchronize_current_cuda_stream",
        lambda device: observed.__setitem__("sync", observed["sync"] + 1),
    )

    fake_pages = type("FakePages", (), {"device": torch.device("cpu")})()
    plan = type(
        "Plan", (),
        {
            "entries": (
                type(
                    "Entry", (),
                    {
                        "result": type("Result", (), {"pages": fake_pages})(),
                        "actual_tokens": 256,
                        "slot_mapping": torch.arange(256, dtype=torch.int64),
                    },
                )(),
            ),
        },
    )()
    kv_caches = [torch.zeros((2, 8, 256), dtype=torch.float16)]

    assert placer._maybe_warmup_fused(plan, kv_caches) is True
    assert observed["launch"] == 1
    assert observed["sync"] == 1


def test_merged_refill_warmup_uses_current_stream_sync_helper(
    monkeypatch: pytest.MonkeyPatch,
):
    """merged refill warmup 也只应同步当前 stream。"""
    monkeypatch.setattr(
        "vllm.bam.lmcache_bam_direct_placement.triton",
        object(),
    )

    layout = type(
        "Layout", (),
        {
            "page_bytes": 128 * 1024,
            "pages_per_chunk": 112,
            "num_layers": 2,
            "hidden_dim": 16,
            "slot_num_tokens": 256,
            "page_token_capacity": 128,
            "pages_per_kv_layer": 1,
            "dtype": torch.float16,
        })()
    placer = BaMDirectKVPlacer(layout=layout)

    observed = {"refill": 0, "sync": 0}

    monkeypatch.setattr(
        "vllm.bam.lmcache_bam_direct_placement.refill_pages_to_lmcache_tensor_into",
        lambda *args, **kwargs: observed.__setitem__("refill",
                                                     observed["refill"] + 1),
    )
    monkeypatch.setattr(
        placer,
        "_synchronize_current_cuda_stream",
        lambda device: observed.__setitem__("sync", observed["sync"] + 1),
    )

    fake_pages = type("FakePages", (), {"device": torch.device("cpu")})()
    plan = type(
        "Plan", (),
        {
            "entries": (
                type(
                    "Entry", (),
                    {
                        "result": type("Result", (), {"pages": fake_pages})(),
                        "actual_tokens": 256,
                    },
                )(),
            ),
            "total_tokens": 256,
        },
    )()

    assert placer._maybe_warmup_merged_refill(plan) is True
    assert observed["refill"] == 1
    assert observed["sync"] == 1


def test_prepare_for_batch_skips_warmup_when_persistent_service_enabled(
    monkeypatch: pytest.MonkeyPatch,
):
    """persistent service 模式下，prepare 不应再触发 warmup。

    这条测试保护当前修复的核心语义：

    - GPU worker runtime / persistent service 已启用
    - materialized finalize 仍会调用 `prepare_for_batch()`
    - 但它只能构建 plan，不能再进去做 warmup + 同步

    否则就可能因为默认 stream 与后台常驻 service 的隐式同步关系，
    把请求卡死在 prepare 阶段。
    """
    monkeypatch.setenv("GIDS_KV_GPU_WORKER_PERSISTENT_ENABLE", "1")
    monkeypatch.setenv("VLLM_BAM_DIRECT_PLACEMENT_IMPL", "fused")

    layout = type(
        "Layout", (),
        {
            "page_bytes": 128 * 1024,
            "pages_per_chunk": 112,
            "num_layers": 2,
            "hidden_dim": 16,
            "slot_num_tokens": 256,
            "page_token_capacity": 128,
            "pages_per_kv_layer": 1,
            "dtype": torch.float16,
        })()
    placer = BaMDirectKVPlacer(layout=layout)

    observed = {"fused": 0, "merged": 0}

    monkeypatch.setattr(
        placer,
        "_ensure_lmcache_connector_state",
        lambda kv_caches: None,
    )
    monkeypatch.setattr(
        placer,
        "_build_plan",
        lambda **kwargs: type(
            "Plan", (),
            {
                "entries": (object(),),
                "total_tokens": 256,
            },
        )(),
    )
    monkeypatch.setattr(
        placer,
        "_build_launch_plan",
        lambda **kwargs: type(
            "LaunchPlan", (),
            {
                "entries": (object(),),
                "total_tokens": 256,
            },
        )(),
    )
    monkeypatch.setattr(
        placer,
        "_maybe_warmup_fused",
        lambda *args, **kwargs: observed.__setitem__("fused",
                                                     observed["fused"] + 1),
    )
    monkeypatch.setattr(
        placer,
        "_maybe_warmup_merged_refill",
        lambda *args, **kwargs: observed.__setitem__("merged",
                                                     observed["merged"] + 1),
    )

    placer.prepare_for_batch(
        results=[object()],
        kv_caches=[torch.zeros((2, 8, 256), dtype=torch.float16)],
        slot_mapping=_FakeCudaTensor(torch.arange(256, dtype=torch.int64)),
        chunk_starts=[0],
    )

    assert observed["fused"] == 0
    assert observed["merged"] == 0


def test_build_plan_keeps_negative_slot_validation_for_cpu_inputs():
    """CPU 场景仍应保留负 slot 校验。"""
    layout = type(
        "Layout", (),
        {
            "page_bytes": 128 * 1024,
            "pages_per_chunk": 112,
            "num_layers": 2,
            "hidden_dim": 16,
            "slot_num_tokens": 256,
            "page_token_capacity": 128,
            "pages_per_kv_layer": 1,
            "dtype": torch.float16,
        })()
    placer = BaMDirectKVPlacer(layout=layout)

    results = [
        _FakeDirectPlacementResult(
            descriptor=_FakeDirectPlacementDescriptor(actual_tokens=4)),
    ]
    slot_mapping = torch.tensor([0, 1, -1, 3], dtype=torch.int64)

    with pytest.raises(ValueError, match="negative slot_mapping"):
        placer._build_plan(
            results=results,
            slot_mapping=slot_mapping,
            chunk_starts=[0],
        )


def test_start_batch_demotes_fused_to_lmcache_when_persistent_service_enabled(
    monkeypatch: pytest.MonkeyPatch,
):
    """persistent service 模式下，start_batch 应收敛到更稳的 lmcache 实现。"""
    monkeypatch.setenv("GIDS_KV_GPU_WORKER_PERSISTENT_ENABLE", "1")
    monkeypatch.setenv("VLLM_BAM_DIRECT_PLACEMENT_IMPL", "fused")

    layout = type(
        "Layout", (),
        {
            "page_bytes": 128 * 1024,
            "pages_per_chunk": 112,
            "num_layers": 2,
            "hidden_dim": 16,
            "slot_num_tokens": 256,
            "page_token_capacity": 128,
            "pages_per_kv_layer": 1,
            "dtype": torch.float16,
        })()
    placer = BaMDirectKVPlacer(layout=layout)

    observed = {"warmup": 0, "fused": 0, "refill": 0, "transfer": 0}

    monkeypatch.setattr(
        placer,
        "_ensure_lmcache_connector_state",
        lambda kv_caches: None,
    )
    monkeypatch.setattr(
        placer,
        "_build_plan",
        lambda **kwargs: type(
            "Plan", (),
            {
                "entries": (
                    type(
                        "Entry", (),
                        {
                            "result": type(
                                "Result", (),
                                {
                                    "pages": torch.zeros(
                                        (112, 128 * 1024),
                                        dtype=torch.uint8,
                                    ),
                                },
                            )(),
                            "chunk_start": 0,
                            "actual_tokens": 256,
                            "slot_mapping": torch.arange(
                                256, dtype=torch.int64),
                        },
                    )(),
                ),
                "total_tokens": 256,
            },
        )(),
    )
    monkeypatch.setattr(
        placer,
        "_build_launch_plan",
        lambda **kwargs: kwargs["plan"],
    )
    monkeypatch.setattr(
        placer,
        "_maybe_warmup_fused",
        lambda *args, **kwargs: observed.__setitem__("warmup",
                                                     observed["warmup"] + 1),
    )
    monkeypatch.setattr(
        placer,
        "_new_cuda_event_pair",
        staticmethod(lambda: (_FakeEvent(), _FakeEvent())),
    )
    monkeypatch.setattr(
        placer,
        "_fused_plan_entries_to_vllm_cache",
        lambda *args, **kwargs: observed.__setitem__("fused",
                                                     observed["fused"] + 1)
        or [],
    )
    monkeypatch.setattr(
        placer,
        "_refill_plan_entries",
        lambda *args, **kwargs: (
            observed.__setitem__("refill", observed["refill"] + 1)
            or torch.zeros((2, 2, 256, 16), dtype=torch.float16),
            [],
        ),
    )
    monkeypatch.setattr(
        placer,
        "_lmcache_transfer_plan_entries",
        lambda *args, **kwargs: observed.__setitem__("transfer",
                                                     observed["transfer"] + 1),
    )

    launched = placer.start_batch(
        results=[object()],
        kv_caches=[torch.zeros((2, 8, 16), dtype=torch.float16)],
        slot_mapping=_FakeCudaTensor(torch.arange(256, dtype=torch.int64)),
        chunk_starts=[0],
    )

    assert launched.impl == "lmcache"
    assert observed["warmup"] == 0
    assert observed["fused"] == 0
    assert observed["refill"] == 1
    assert observed["transfer"] == 1


def test_direct_placement_second_wave_does_not_expose_noncontiguous_prefix(
    monkeypatch: pytest.MonkeyPatch,
):
    """第二波从非零 chunk 开始时，不能把中间仍未 ready 的前缀错误暴露出去。

    这个测试对应“两阶段前沿”里的关键安全约束：

    - wave0 还没 launch / 还没 ready
    - wave1 单独把 chunk1 放进 cache

    此时 `cache_ready_chunks` 会增长，但 `consumable_chunks` 必须仍然为 0，
    因为 chunk0 这个前缀空洞还没被填上。
    """
    monkeypatch.setenv("VLLM_BAM_DIRECT_PLACEMENT_IMPL", "fused")

    layout = type(
        "Layout", (),
        {
            "page_bytes": 128 * 1024,
            "pages_per_chunk": 112,
            "num_layers": 2,
            "hidden_dim": 16,
            "slot_num_tokens": 256,
            "page_token_capacity": 128,
            "pages_per_kv_layer": 1,
            "dtype": torch.float16,
        })()
    placer = BaMDirectKVPlacer(layout=layout)

    monkeypatch.setattr(
        "vllm.bam.lmcache_bam_direct_placement.torch.cuda.synchronize",
        lambda device=None: None,
    )
    monkeypatch.setattr(
        "vllm.bam.lmcache_bam_direct_placement.BaMDirectKVPlacer._new_cuda_event_pair",
        staticmethod(lambda: (_FakeEvent(), _FakeEvent())),
    )

    observed: dict[str, Any] = {}

    def _fake_ensure_state(kv_caches):
        observed["ensure_state_layers"] = len(kv_caches)

    def _fake_fused_plan(plan, kv_caches):
        observed["chunk_starts"] = [entry.chunk_start for entry in plan.entries]
        observed["chunk_tokens"] = [entry.actual_tokens for entry in plan.entries]
        observed["kv_cache_layers"] = len(kv_caches)
        return [
            (entry.chunk_start, entry.actual_tokens, _FakeEvent(), _FakeEvent())
            for entry in plan.entries
        ]

    monkeypatch.setattr(placer, "_ensure_lmcache_connector_state", _fake_ensure_state)
    monkeypatch.setattr(placer, "_fused_plan_entries_to_vllm_cache", _fake_fused_plan)

    results = [
        _FakeDirectPlacementResult(
            descriptor=_FakeDirectPlacementDescriptor(actual_tokens=256)),
        _FakeDirectPlacementResult(
            descriptor=_FakeDirectPlacementDescriptor(actual_tokens=256)),
        _FakeDirectPlacementResult(
            descriptor=_FakeDirectPlacementDescriptor(actual_tokens=256)),
    ]
    state_tracker = BaMDirectPlacementStateTracker(
        BaMDirectPlacementBatchDescriptor(
            chunks=(
                BaMDirectPlacementChunkDescriptor(
                    chunk_hash="chunk-a",
                    chunk_start=0,
                    chunk_end=256,
                    actual_tokens=256,
                    total_bytes=1024,
                ),
                BaMDirectPlacementChunkDescriptor(
                    chunk_hash="chunk-b",
                    chunk_start=256,
                    chunk_end=512,
                    actual_tokens=256,
                    total_bytes=1024,
                ),
                BaMDirectPlacementChunkDescriptor(
                    chunk_hash="chunk-c",
                    chunk_start=512,
                    chunk_end=768,
                    actual_tokens=256,
                    total_bytes=1024,
                ),
            ),
            total_tokens=768,
            total_bytes=3072,
        ))
    kv_caches = [torch.zeros((2, 8, 16), dtype=torch.float16) for _ in range(2)]
    fake_cuda_slot_mapping = _FakeCudaTensor(torch.arange(768, dtype=torch.int64))

    launched_batch = placer.start_batch(
        results=results,
        kv_caches=kv_caches,
        slot_mapping=fake_cuda_slot_mapping,
        chunk_starts=[0, 256, 512],
        launch_start_chunk=1,
        max_chunks_to_launch=1,
    )
    execution = placer.execution_from_launched_batch(
        launched_batch=launched_batch,
        state_tracker=state_tracker,
    )
    stats, snapshot = execution.wait()

    assert observed["ensure_state_layers"] == 2
    assert observed["kv_cache_layers"] == 2
    assert observed["chunk_starts"] == [256]
    assert observed["chunk_tokens"] == [256]
    assert stats.tokens == 256
    assert snapshot is not None
    assert snapshot.cache_ready_chunks == 1
    assert snapshot.cache_ready_tokens == 256
    assert snapshot.consumable_chunks == 0
    assert snapshot.consumable_tokens == 0


def test_direct_placement_two_wave_execution_advances_contiguous_frontier(
    monkeypatch: pytest.MonkeyPatch,
):
    """两波 placement 串起来后，contiguous frontier 应按顺序自然增长。

    这条语义是后续把“先放一小段前缀、再补后续 wave”接到更高层消费逻辑前的
    控制面基础：

    - 第一波只放 chunk0，frontier=1
    - 第二波再放 chunk1，frontier=2

    也就是说，后续 wave 不需要重建 tracker，只要继续基于同一个 tracker
    推进即可。
    """
    monkeypatch.setenv("VLLM_BAM_DIRECT_PLACEMENT_IMPL", "fused")

    layout = type(
        "Layout", (),
        {
            "page_bytes": 128 * 1024,
            "pages_per_chunk": 112,
            "num_layers": 2,
            "hidden_dim": 16,
            "slot_num_tokens": 256,
            "page_token_capacity": 128,
            "pages_per_kv_layer": 1,
            "dtype": torch.float16,
        })()
    placer = BaMDirectKVPlacer(layout=layout)

    monkeypatch.setattr(
        "vllm.bam.lmcache_bam_direct_placement.torch.cuda.synchronize",
        lambda device=None: None,
    )
    monkeypatch.setattr(
        "vllm.bam.lmcache_bam_direct_placement.BaMDirectKVPlacer._new_cuda_event_pair",
        staticmethod(lambda: (_FakeEvent(), _FakeEvent())),
    )

    observed_calls: list[list[int]] = []

    def _fake_ensure_state(kv_caches):
        del kv_caches
        return None

    def _fake_fused_plan(plan, kv_caches):
        del kv_caches
        observed_calls.append([entry.chunk_start for entry in plan.entries])
        return [
            (entry.chunk_start, entry.actual_tokens, _FakeEvent(), _FakeEvent())
            for entry in plan.entries
        ]

    monkeypatch.setattr(placer, "_ensure_lmcache_connector_state", _fake_ensure_state)
    monkeypatch.setattr(placer, "_fused_plan_entries_to_vllm_cache", _fake_fused_plan)

    results = [
        _FakeDirectPlacementResult(
            descriptor=_FakeDirectPlacementDescriptor(actual_tokens=256)),
        _FakeDirectPlacementResult(
            descriptor=_FakeDirectPlacementDescriptor(actual_tokens=256)),
        _FakeDirectPlacementResult(
            descriptor=_FakeDirectPlacementDescriptor(actual_tokens=256)),
    ]
    state_tracker = BaMDirectPlacementStateTracker(
        BaMDirectPlacementBatchDescriptor(
            chunks=(
                BaMDirectPlacementChunkDescriptor(
                    chunk_hash="chunk-a",
                    chunk_start=0,
                    chunk_end=256,
                    actual_tokens=256,
                    total_bytes=1024,
                ),
                BaMDirectPlacementChunkDescriptor(
                    chunk_hash="chunk-b",
                    chunk_start=256,
                    chunk_end=512,
                    actual_tokens=256,
                    total_bytes=1024,
                ),
                BaMDirectPlacementChunkDescriptor(
                    chunk_hash="chunk-c",
                    chunk_start=512,
                    chunk_end=768,
                    actual_tokens=256,
                    total_bytes=1024,
                ),
            ),
            total_tokens=768,
            total_bytes=3072,
        ))
    kv_caches = [torch.zeros((2, 8, 16), dtype=torch.float16) for _ in range(2)]
    fake_cuda_slot_mapping = _FakeCudaTensor(torch.arange(768, dtype=torch.int64))

    first_wave = placer.start_batch(
        results=results,
        kv_caches=kv_caches,
        slot_mapping=fake_cuda_slot_mapping,
        chunk_starts=[0, 256, 512],
        launch_start_chunk=0,
        max_chunks_to_launch=1,
    )
    first_execution = placer.execution_from_launched_batch(
        launched_batch=first_wave,
        state_tracker=state_tracker,
    )
    _stats1, snapshot1 = first_execution.wait()

    assert snapshot1 is not None
    assert snapshot1.cache_ready_chunks == 1
    assert snapshot1.consumable_chunks == 1
    assert snapshot1.consumable_tokens == 256

    second_wave = placer.start_batch(
        results=results,
        kv_caches=kv_caches,
        slot_mapping=fake_cuda_slot_mapping,
        chunk_starts=[0, 256, 512],
        launch_start_chunk=1,
        max_chunks_to_launch=1,
    )
    second_execution = placer.execution_from_launched_batch(
        launched_batch=second_wave,
        state_tracker=state_tracker,
    )
    _stats2, snapshot2 = second_execution.wait()

    assert observed_calls == [[0], [256]]
    assert snapshot2 is not None
    assert snapshot2.cache_ready_chunks == 2
    assert snapshot2.cache_ready_tokens == 512
    assert snapshot2.consumable_chunks == 2
    assert snapshot2.consumable_tokens == 512


def test_direct_placement_wait_avoids_device_wide_synchronize(
    monkeypatch: pytest.MonkeyPatch,
):
    """wait() 应只等待本 wave 的 event，而不是做整卡 synchronize。

    这条测试对应当前 GPU-initiated 主线里一个很重要的收口要求：

    - direct placement 只应等待“这次自己 launch 的那一波”
    - 不应顺手把当前设备上其它无关 CUDA 工作也一起 block 住
    """
    monkeypatch.setenv("VLLM_BAM_DIRECT_PLACEMENT_IMPL", "fused")

    layout = type(
        "Layout", (),
        {
            "page_bytes": 128 * 1024,
            "pages_per_chunk": 112,
            "num_layers": 2,
            "hidden_dim": 16,
            "slot_num_tokens": 256,
            "page_token_capacity": 128,
            "pages_per_kv_layer": 1,
            "dtype": torch.float16,
        })()
    placer = BaMDirectKVPlacer(layout=layout)

    def _unexpected_sync(device=None):
        del device
        raise AssertionError("wait() should not call torch.cuda.synchronize")

    monkeypatch.setattr(
        "vllm.bam.lmcache_bam_direct_placement.torch.cuda.synchronize",
        _unexpected_sync,
    )
    monkeypatch.setattr(
        "vllm.bam.lmcache_bam_direct_placement.BaMDirectKVPlacer._new_cuda_event_pair",
        staticmethod(lambda: (_FakeQueryEvent(), _FakeQueryEvent())),
    )
    monkeypatch.setattr(
        placer,
        "_ensure_lmcache_connector_state",
        lambda kv_caches: None,
    )
    monkeypatch.setattr(
        placer,
        "_fused_plan_entries_to_vllm_cache",
        lambda plan, kv_caches: [
            (entry.chunk_start, entry.actual_tokens, _FakeQueryEvent(),
             _FakeQueryEvent()) for entry in plan.entries
        ],
    )

    results = [
        _FakeDirectPlacementResult(
            descriptor=_FakeDirectPlacementDescriptor(actual_tokens=256)),
    ]
    state_tracker = BaMDirectPlacementStateTracker(
        BaMDirectPlacementBatchDescriptor(
            chunks=(
                BaMDirectPlacementChunkDescriptor(
                    chunk_hash="chunk-a",
                    chunk_start=0,
                    chunk_end=256,
                    actual_tokens=256,
                    total_bytes=1024,
                ),
            ),
            total_tokens=256,
            total_bytes=1024,
        ))
    kv_caches = [torch.zeros((2, 8, 16), dtype=torch.float16) for _ in range(2)]
    fake_cuda_slot_mapping = _FakeCudaTensor(torch.arange(256, dtype=torch.int64))

    launched_batch = placer.start_batch(
        results=results,
        kv_caches=kv_caches,
        slot_mapping=fake_cuda_slot_mapping,
        chunk_starts=[0],
    )
    execution = placer.execution_from_launched_batch(
        launched_batch=launched_batch,
        state_tracker=state_tracker,
    )
    stats, snapshot = execution.wait()

    assert stats.tokens == 256
    assert snapshot is not None
    assert snapshot.cache_ready_chunks == 1
    assert snapshot.consumable_chunks == 1


def test_direct_placement_wait_until_contiguous_cache_ready_stops_at_target(
    monkeypatch: pytest.MonkeyPatch,
):
    """只等到目标连续前缀 ready 后就应返回，不必顺手等完整批次。

    这里故意让 chunk0 立刻 ready、chunk1 长时间不 ready，用来验证：

    - `wait_until_contiguous_cache_ready(1)` 会在前缀 1 个 chunk ready 后立即返回
    - 不会继续死等后面的 chunk1
    """
    monkeypatch.setenv("VLLM_BAM_DIRECT_PLACEMENT_IMPL", "fused")

    layout = type(
        "Layout", (),
        {
            "page_bytes": 128 * 1024,
            "pages_per_chunk": 112,
            "num_layers": 2,
            "hidden_dim": 16,
            "slot_num_tokens": 256,
            "page_token_capacity": 128,
            "pages_per_kv_layer": 1,
            "dtype": torch.float16,
        })()
    placer = BaMDirectKVPlacer(layout=layout)

    delayed_chunk_event = _FakeQueryEvent(ready_after_queries=10_000)

    monkeypatch.setattr(
        placer,
        "_ensure_lmcache_connector_state",
        lambda kv_caches: None,
    )
    monkeypatch.setattr(
        "vllm.bam.lmcache_bam_direct_placement.BaMDirectKVPlacer._new_cuda_event_pair",
        staticmethod(lambda: (_FakeQueryEvent(), _FakeQueryEvent())),
    )
    monkeypatch.setattr(
        placer,
        "_fused_plan_entries_to_vllm_cache",
        lambda plan, kv_caches: [
            (plan.entries[0].chunk_start, plan.entries[0].actual_tokens,
             _FakeQueryEvent(), _FakeQueryEvent()),
            (plan.entries[1].chunk_start, plan.entries[1].actual_tokens,
             _FakeQueryEvent(), delayed_chunk_event),
        ],
    )

    results = [
        _FakeDirectPlacementResult(
            descriptor=_FakeDirectPlacementDescriptor(actual_tokens=256)),
        _FakeDirectPlacementResult(
            descriptor=_FakeDirectPlacementDescriptor(actual_tokens=256)),
    ]
    state_tracker = BaMDirectPlacementStateTracker(
        BaMDirectPlacementBatchDescriptor(
            chunks=(
                BaMDirectPlacementChunkDescriptor(
                    chunk_hash="chunk-a",
                    chunk_start=0,
                    chunk_end=256,
                    actual_tokens=256,
                    total_bytes=1024,
                ),
                BaMDirectPlacementChunkDescriptor(
                    chunk_hash="chunk-b",
                    chunk_start=256,
                    chunk_end=512,
                    actual_tokens=256,
                    total_bytes=1024,
                ),
            ),
            total_tokens=512,
            total_bytes=2048,
        ))
    kv_caches = [torch.zeros((2, 8, 16), dtype=torch.float16) for _ in range(2)]
    fake_cuda_slot_mapping = _FakeCudaTensor(torch.arange(512, dtype=torch.int64))

    launched_batch = placer.start_batch(
        results=results,
        kv_caches=kv_caches,
        slot_mapping=fake_cuda_slot_mapping,
        chunk_starts=[0, 256],
    )
    execution = placer.execution_from_launched_batch(
        launched_batch=launched_batch,
        state_tracker=state_tracker,
    )
    snapshot = execution.wait_until_contiguous_cache_ready(1, timeout_s=0.01)
    frontier_snapshot = execution.frontier_snapshot()

    assert snapshot is not None
    assert snapshot.cache_ready_chunks == 1
    assert snapshot.consumable_chunks == 1
    assert snapshot.consumable_tokens == 256
    assert frontier_snapshot is not None
    assert frontier_snapshot.launch_frontier_chunks == 2
    assert frontier_snapshot.cache_ready_frontier_chunks == 1
    assert frontier_snapshot.consumable_frontier_chunks == 1
    assert frontier_snapshot.consumable_frontier_tokens == 256
    assert delayed_chunk_event.query_calls >= 1
    assert delayed_chunk_event.query() is False


def test_direct_placement_execution_frontier_table_initializes_from_tracker(
    monkeypatch: pytest.MonkeyPatch,
):
    """execution 创建时，应立刻生成统一 frontier table 与 host mirror。

    这是第三步轻量版的重要约束：

    - 即使后面还没有真正做 GPU-resident persistent placement runtime
    - execution 也应该在创建时就暴露稳定的 frontier ABI

    这样 storage/runtime 才能围绕统一 request-level frontier 收敛，而不是继续
    依赖 tracker 的内部细节。
    """
    monkeypatch.setenv("VLLM_BAM_DIRECT_PLACEMENT_IMPL", "fused")

    layout = type(
        "Layout", (),
        {
            "page_bytes": 128 * 1024,
            "pages_per_chunk": 112,
            "num_layers": 2,
            "hidden_dim": 16,
            "slot_num_tokens": 256,
            "page_token_capacity": 128,
            "pages_per_kv_layer": 1,
            "dtype": torch.float16,
        })()
    placer = BaMDirectKVPlacer(layout=layout)

    monkeypatch.setattr(
        placer,
        "_ensure_lmcache_connector_state",
        lambda kv_caches: None,
    )
    monkeypatch.setattr(
        "vllm.bam.lmcache_bam_direct_placement.BaMDirectKVPlacer._new_cuda_event_pair",
        staticmethod(lambda: (_FakeQueryEvent(), _FakeQueryEvent())),
    )
    monkeypatch.setattr(
        placer,
        "_fused_plan_entries_to_vllm_cache",
        lambda plan, kv_caches: [
            (entry.chunk_start, entry.actual_tokens, _FakeQueryEvent(),
             _FakeQueryEvent(ready_after_queries=1))
            for entry in plan.entries
        ],
    )

    state_tracker = BaMDirectPlacementStateTracker(
        BaMDirectPlacementBatchDescriptor(
            chunks=(
                BaMDirectPlacementChunkDescriptor(
                    chunk_hash="chunk-a",
                    chunk_start=0,
                    chunk_end=256,
                    actual_tokens=256,
                    total_bytes=1024,
                ),
                BaMDirectPlacementChunkDescriptor(
                    chunk_hash="chunk-b",
                    chunk_start=256,
                    chunk_end=512,
                    actual_tokens=256,
                    total_bytes=1024,
                ),
            ),
            total_tokens=512,
            total_bytes=2048,
        ))

    launched_batch = placer.start_batch(
        results=[
            _FakeDirectPlacementResult(
                descriptor=_FakeDirectPlacementDescriptor(actual_tokens=256)),
            _FakeDirectPlacementResult(
                descriptor=_FakeDirectPlacementDescriptor(actual_tokens=256)),
        ],
        kv_caches=[torch.zeros((2, 8, 16), dtype=torch.float16) for _ in range(2)],
        slot_mapping=_FakeCudaTensor(torch.arange(512, dtype=torch.int64)),
        chunk_starts=[0, 256],
    )
    execution = placer.execution_from_launched_batch(
        launched_batch=launched_batch,
        state_tracker=state_tracker,
    )

    frontier_snapshot = execution.frontier_snapshot()
    frontier_table = execution.frontier_table()

    assert frontier_snapshot is not None
    assert frontier_snapshot.frontier_row == (1, 2, 0, 0, 0, 2, 0)
    assert execution.frontier_row_host() == (1, 2, 0, 0, 0, 2, 0)
    assert frontier_table is not None
    assert tuple(int(v) for v in frontier_table.tolist()) == (1, 2, 0, 0, 0, 2, 0)


def test_direct_placement_execution_frontier_table_tracks_ready_progress(
    monkeypatch: pytest.MonkeyPatch,
):
    """execution 推进 ready 后，frontier table 与 host mirror 应同步前进。"""
    monkeypatch.setenv("VLLM_BAM_DIRECT_PLACEMENT_IMPL", "fused")

    layout = type(
        "Layout", (),
        {
            "page_bytes": 128 * 1024,
            "pages_per_chunk": 112,
            "num_layers": 2,
            "hidden_dim": 16,
            "slot_num_tokens": 256,
            "page_token_capacity": 128,
            "pages_per_kv_layer": 1,
            "dtype": torch.float16,
        })()
    placer = BaMDirectKVPlacer(layout=layout)

    delayed_second_chunk = _FakeQueryEvent(ready_after_queries=8)
    monkeypatch.setattr(
        placer,
        "_ensure_lmcache_connector_state",
        lambda kv_caches: None,
    )
    monkeypatch.setattr(
        "vllm.bam.lmcache_bam_direct_placement.BaMDirectKVPlacer._new_cuda_event_pair",
        staticmethod(lambda: (_FakeQueryEvent(), _FakeQueryEvent())),
    )
    monkeypatch.setattr(
        placer,
        "_fused_plan_entries_to_vllm_cache",
        lambda plan, kv_caches: [
            (plan.entries[0].chunk_start, plan.entries[0].actual_tokens,
             _FakeQueryEvent(), _FakeQueryEvent()),
            (plan.entries[1].chunk_start, plan.entries[1].actual_tokens,
             _FakeQueryEvent(), delayed_second_chunk),
        ],
    )

    state_tracker = BaMDirectPlacementStateTracker(
        BaMDirectPlacementBatchDescriptor(
            chunks=(
                BaMDirectPlacementChunkDescriptor(
                    chunk_hash="chunk-a",
                    chunk_start=0,
                    chunk_end=256,
                    actual_tokens=256,
                    total_bytes=1024,
                ),
                BaMDirectPlacementChunkDescriptor(
                    chunk_hash="chunk-b",
                    chunk_start=256,
                    chunk_end=512,
                    actual_tokens=256,
                    total_bytes=1024,
                ),
            ),
            total_tokens=512,
            total_bytes=2048,
        ))

    launched_batch = placer.start_batch(
        results=[
            _FakeDirectPlacementResult(
                descriptor=_FakeDirectPlacementDescriptor(actual_tokens=256)),
            _FakeDirectPlacementResult(
                descriptor=_FakeDirectPlacementDescriptor(actual_tokens=256)),
        ],
        kv_caches=[torch.zeros((2, 8, 16), dtype=torch.float16) for _ in range(2)],
        slot_mapping=_FakeCudaTensor(torch.arange(512, dtype=torch.int64)),
        chunk_starts=[0, 256],
    )
    execution = placer.execution_from_launched_batch(
        launched_batch=launched_batch,
        state_tracker=state_tracker,
    )

    snapshot = execution.wait_until_contiguous_cache_ready(1, timeout_s=0.01)
    frontier_snapshot = execution.frontier_snapshot()
    frontier_table = execution.frontier_table()

    assert snapshot is not None
    assert snapshot.consumable_chunks == 1
    assert frontier_snapshot is not None
    assert frontier_snapshot.frontier_row == (4, 2, 0, 1, 1, 2, 0)
    assert execution.frontier_row_host() == (4, 2, 0, 1, 1, 2, 0)
    assert frontier_table is not None
    assert tuple(int(v) for v in frontier_table.tolist()) == (4, 2, 0, 1, 1, 2, 0)
    assert delayed_second_chunk.query() is False


def test_direct_placement_execution_frontier_snapshot_prefers_shared_table(
    monkeypatch: pytest.MonkeyPatch,
):
    """execution 的 frontier getter 应优先解码共享 frontier table。

    这条测试保护的是当前第三步里很关键的一层收敛：

    - frontier table 不只是“同步写一下”
    - execution 自己的 getter 也开始把它当成主事实来源

    这样后续如果 frontier 更新权进一步下放给 GPU runtime，这层读取逻辑就不用
    再改一次。
    """
    monkeypatch.setenv("VLLM_BAM_DIRECT_PLACEMENT_IMPL", "fused")

    layout = type(
        "Layout", (),
        {
            "page_bytes": 128 * 1024,
            "pages_per_chunk": 112,
            "num_layers": 2,
            "hidden_dim": 16,
            "slot_num_tokens": 256,
            "page_token_capacity": 128,
            "pages_per_kv_layer": 1,
            "dtype": torch.float16,
        })()
    placer = BaMDirectKVPlacer(layout=layout)

    monkeypatch.setattr(
        placer,
        "_ensure_lmcache_connector_state",
        lambda kv_caches: None,
    )
    monkeypatch.setattr(
        "vllm.bam.lmcache_bam_direct_placement.BaMDirectKVPlacer._new_cuda_event_pair",
        staticmethod(lambda: (_FakeQueryEvent(), _FakeQueryEvent())),
    )
    monkeypatch.setattr(
        placer,
        "_fused_plan_entries_to_vllm_cache",
        lambda plan, kv_caches: [
            (entry.chunk_start, entry.actual_tokens, _FakeQueryEvent(),
             _FakeQueryEvent(ready_after_queries=9)) for entry in plan.entries
        ],
    )

    state_tracker = BaMDirectPlacementStateTracker(
        BaMDirectPlacementBatchDescriptor(
            chunks=(
                BaMDirectPlacementChunkDescriptor(
                    chunk_hash="chunk-a",
                    chunk_start=0,
                    chunk_end=256,
                    actual_tokens=256,
                    total_bytes=1024,
                ),
                BaMDirectPlacementChunkDescriptor(
                    chunk_hash="chunk-b",
                    chunk_start=256,
                    chunk_end=512,
                    actual_tokens=256,
                    total_bytes=1024,
                ),
            ),
            total_tokens=512,
            total_bytes=2048,
        ))

    launched_batch = placer.start_batch(
        results=[
            _FakeDirectPlacementResult(
                descriptor=_FakeDirectPlacementDescriptor(actual_tokens=256)),
            _FakeDirectPlacementResult(
                descriptor=_FakeDirectPlacementDescriptor(actual_tokens=256)),
        ],
        kv_caches=[torch.zeros((2, 8, 16), dtype=torch.float16) for _ in range(2)],
        slot_mapping=_FakeCudaTensor(torch.arange(512, dtype=torch.int64)),
        chunk_starts=[0, 256],
    )
    execution = placer.execution_from_launched_batch(
        launched_batch=launched_batch,
        state_tracker=state_tracker,
    )

    # 这里不推进 tracker，只直接改共享 table，验证 getter 已经优先解码 table。
    frontier_table = execution.frontier_table()
    assert frontier_table is not None
    frontier_table.copy_(torch.tensor([4, 2, 1, 1, 1, 2, 0], dtype=torch.int64))

    frontier_snapshot = execution.frontier_snapshot()

    assert frontier_snapshot is not None
    assert frontier_snapshot.frontier_row == (4, 2, 1, 1, 1, 2, 0)
    assert frontier_snapshot.read_ready_frontier_tokens == 256
    assert frontier_snapshot.cache_ready_frontier_tokens == 256
    assert frontier_snapshot.consumable_frontier_tokens == 256
    # staged 仍然来自 tracker，因为当前 7 列 ABI 里还没有 staged 列。
    assert frontier_snapshot.staged_ready_frontier_chunks == 0


def test_direct_placement_execution_updates_shared_table_in_place(
    monkeypatch: pytest.MonkeyPatch,
):
    """execution 推进 ready 时，应原位更新共享 frontier table 的 ABI 列。

    这条测试保护的是当前第三步里的一个实现收敛点：

    - frontier table 已经是 getter 的主事实来源
    - 因此 execution 在推进 ready 时，也应该尽量原位维护这几列

    这样后续如果换成 persistent runtime 直接维护同一张表，这层形态就不需要再改。
    """
    monkeypatch.setenv("VLLM_BAM_DIRECT_PLACEMENT_IMPL", "fused")

    layout = type(
        "Layout", (),
        {
            "page_bytes": 128 * 1024,
            "pages_per_chunk": 112,
            "num_layers": 2,
            "hidden_dim": 16,
            "slot_num_tokens": 256,
            "page_token_capacity": 128,
            "pages_per_kv_layer": 1,
            "dtype": torch.float16,
        })()
    placer = BaMDirectKVPlacer(layout=layout)

    delayed_second_chunk = _FakeQueryEvent(ready_after_queries=8)
    monkeypatch.setattr(
        placer,
        "_ensure_lmcache_connector_state",
        lambda kv_caches: None,
    )
    monkeypatch.setattr(
        "vllm.bam.lmcache_bam_direct_placement.BaMDirectKVPlacer._new_cuda_event_pair",
        staticmethod(lambda: (_FakeQueryEvent(), _FakeQueryEvent())),
    )
    monkeypatch.setattr(
        placer,
        "_fused_plan_entries_to_vllm_cache",
        lambda plan, kv_caches: [
            (plan.entries[0].chunk_start, plan.entries[0].actual_tokens,
             _FakeQueryEvent(), _FakeQueryEvent()),
            (plan.entries[1].chunk_start, plan.entries[1].actual_tokens,
             _FakeQueryEvent(), delayed_second_chunk),
        ],
    )

    shared_frontier_table = torch.tensor(
        [1, 2, 0, 0, 0, 2, 77],
        dtype=torch.int64,
    )
    original_storage_ptr = int(shared_frontier_table.data_ptr())

    state_tracker = BaMDirectPlacementStateTracker(
        BaMDirectPlacementBatchDescriptor(
            chunks=(
                BaMDirectPlacementChunkDescriptor(
                    chunk_hash="chunk-a",
                    chunk_start=0,
                    chunk_end=256,
                    actual_tokens=256,
                    total_bytes=1024,
                ),
                BaMDirectPlacementChunkDescriptor(
                    chunk_hash="chunk-b",
                    chunk_start=256,
                    chunk_end=512,
                    actual_tokens=256,
                    total_bytes=1024,
                ),
            ),
            total_tokens=512,
            total_bytes=2048,
        ))

    launched_batch = placer.start_batch(
        results=[
            _FakeDirectPlacementResult(
                descriptor=_FakeDirectPlacementDescriptor(actual_tokens=256)),
            _FakeDirectPlacementResult(
                descriptor=_FakeDirectPlacementDescriptor(actual_tokens=256)),
        ],
        kv_caches=[torch.zeros((2, 8, 16), dtype=torch.float16) for _ in range(2)],
        slot_mapping=_FakeCudaTensor(torch.arange(512, dtype=torch.int64)),
        chunk_starts=[0, 256],
    )
    execution = placer.execution_from_launched_batch(
        launched_batch=launched_batch,
        state_tracker=state_tracker,
        frontier_table=shared_frontier_table,
    )

    execution.wait_until_contiguous_cache_ready(1, timeout_s=0.01)

    assert int(shared_frontier_table.data_ptr()) == original_storage_ptr
    assert tuple(int(v) for v in shared_frontier_table.tolist()) == (
        4,
        2,
        0,
        1,
        1,
        2,
        77,
    )


def test_direct_placement_execution_wait_predicate_prefers_shared_table(
    monkeypatch: pytest.MonkeyPatch,
):
    """execution 的 wait 判断应优先参考 shared frontier table。

    这条测试覆盖的不是“返回值长什么样”，而是更底层的等待判定口径：

    - 即使 tracker / event 本身还没推进到 ready
    - 只要 shared frontier table 已经体现出目标 frontier
    - execution 的 wait 也应该按这张表来判断是否满足条件
    """
    monkeypatch.setenv("VLLM_BAM_DIRECT_PLACEMENT_IMPL", "fused")

    layout = type(
        "Layout", (),
        {
            "page_bytes": 128 * 1024,
            "pages_per_chunk": 112,
            "num_layers": 2,
            "hidden_dim": 16,
            "slot_num_tokens": 256,
            "page_token_capacity": 128,
            "pages_per_kv_layer": 1,
            "dtype": torch.float16,
        })()
    placer = BaMDirectKVPlacer(layout=layout)

    never_ready_event = _FakeQueryEvent(ready_after_queries=10_000)
    monkeypatch.setattr(
        placer,
        "_ensure_lmcache_connector_state",
        lambda kv_caches: None,
    )
    monkeypatch.setattr(
        "vllm.bam.lmcache_bam_direct_placement.BaMDirectKVPlacer._new_cuda_event_pair",
        staticmethod(lambda: (_FakeQueryEvent(), _FakeQueryEvent())),
    )
    monkeypatch.setattr(
        placer,
        "_fused_plan_entries_to_vllm_cache",
        lambda plan, kv_caches: [
            (entry.chunk_start, entry.actual_tokens, _FakeQueryEvent(),
             never_ready_event) for entry in plan.entries
        ],
    )

    shared_frontier_table = torch.tensor(
        [1, 2, 0, 0, 0, 2, 0],
        dtype=torch.int64,
    )
    state_tracker = BaMDirectPlacementStateTracker(
        BaMDirectPlacementBatchDescriptor(
            chunks=(
                BaMDirectPlacementChunkDescriptor(
                    chunk_hash="chunk-a",
                    chunk_start=0,
                    chunk_end=256,
                    actual_tokens=256,
                    total_bytes=1024,
                ),
                BaMDirectPlacementChunkDescriptor(
                    chunk_hash="chunk-b",
                    chunk_start=256,
                    chunk_end=512,
                    actual_tokens=256,
                    total_bytes=1024,
                ),
            ),
            total_tokens=512,
            total_bytes=2048,
        ))

    launched_batch = placer.start_batch(
        results=[
            _FakeDirectPlacementResult(
                descriptor=_FakeDirectPlacementDescriptor(actual_tokens=256)),
            _FakeDirectPlacementResult(
                descriptor=_FakeDirectPlacementDescriptor(actual_tokens=256)),
        ],
        kv_caches=[torch.zeros((2, 8, 16), dtype=torch.float16) for _ in range(2)],
        slot_mapping=_FakeCudaTensor(torch.arange(512, dtype=torch.int64)),
        chunk_starts=[0, 256],
    )
    execution = placer.execution_from_launched_batch(
        launched_batch=launched_batch,
        state_tracker=state_tracker,
        frontier_table=shared_frontier_table,
    )

    # 模拟更底层 runtime 已经先把 table 推到了“全部 cache-ready”。
    shared_frontier_table.copy_(torch.tensor([3, 2, 0, 2, 0, 2, 0], dtype=torch.int64))

    stats, snapshot = execution.wait_until_launched_range_cache_ready(
        timeout_s=0.01)

    assert stats.impl == "fused"
    assert snapshot is not None
    # 等待判断是按 shared table 过的，最终同步收口后 tracker 会被补齐。
    assert snapshot.cache_ready_chunks == 2
    assert snapshot.consumable_chunks == 2
    assert never_ready_event.query() is False


def test_direct_placement_request_frontier_exposes_unified_request_level_abi(
    monkeypatch: pytest.MonkeyPatch,
):
    """request handle 应能在 blocking fallback 与 finalize 之间稳定导出统一 frontier ABI。

    这条测试保护的是第三步当前最核心的接口目标：

    - 上层未来只拿 request handle
    - 不需要理解 tracker / wave / execution 的内部细节
    - 就能观察“现在 launch 到哪、read-ready 到哪、consumable 到哪”
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

    def _fake_read(keys: list[Any]) -> list[_FakeReadResult]:
        del keys
        return fake_results

    def _make_frontier_snapshot(
        *,
        launch_chunks: int,
        read_ready_chunks: int,
        staged_ready_chunks: int,
        cache_ready_chunks: int,
        consumable_chunks: int,
        total_chunks: int,
    ):
        token_unit = 256
        return type(
            "FrontierSnapshot", (),
            {
                "frontier_row": (
                    4 if consumable_chunks > 0 else
                    3 if cache_ready_chunks > 0 else
                    2 if read_ready_chunks > 0 else 1,
                    launch_chunks,
                    read_ready_chunks,
                    cache_ready_chunks,
                    consumable_chunks,
                    total_chunks,
                    0,
                ),
                "launch_frontier_chunks": launch_chunks,
                "read_ready_frontier_chunks": read_ready_chunks,
                "staged_ready_frontier_chunks": staged_ready_chunks,
                "cache_ready_frontier_chunks": cache_ready_chunks,
                "consumable_frontier_chunks": consumable_chunks,
                "total_chunks": total_chunks,
                "read_ready_frontier_tokens": read_ready_chunks * token_unit,
                "staged_ready_frontier_tokens": staged_ready_chunks * token_unit,
                "cache_ready_frontier_tokens": cache_ready_chunks * token_unit,
                "consumable_frontier_tokens": consumable_chunks * token_unit,
                "error_code": 0,
            },
        )()

    class _FakeExecution:

        def __init__(self, state_tracker):
            self._state_tracker = state_tracker

        def advance_ready(self):
            return self._state_tracker.snapshot()

        def frontier_snapshot(self):
            return _make_frontier_snapshot(
                launch_chunks=2,
                read_ready_chunks=2,
                staged_ready_chunks=0,
                cache_ready_chunks=0,
                consumable_chunks=0,
                total_chunks=2,
            )

        def poll_frontier(self):
            self._state_tracker.mark_chunks_read_ready_upto(2)
            self._state_tracker.mark_chunks_staged_ready_upto(1)
            self._state_tracker.mark_chunks_cache_ready_upto(1)
            return _make_frontier_snapshot(
                launch_chunks=2,
                read_ready_chunks=2,
                staged_ready_chunks=1,
                cache_ready_chunks=1,
                consumable_chunks=1,
                total_chunks=2,
            )

        def wait_until_contiguous_cache_ready_frontier(self, target_chunks):
            assert target_chunks == 2
            self._state_tracker.mark_chunks_read_ready_upto(2)
            self._state_tracker.mark_chunks_staged_ready_upto(2)
            self._state_tracker.mark_chunks_cache_ready_upto(2)
            return _make_frontier_snapshot(
                launch_chunks=2,
                read_ready_chunks=2,
                staged_ready_chunks=2,
                cache_ready_chunks=2,
                consumable_chunks=2,
                total_chunks=2,
            )

        def get_stats(self):
            return type(
                "PlaceStats", (),
                {
                    "impl": "fused",
                    "refill_ms": 0.0,
                    "transfer_ms": 0.0,
                    "fused_ms": 1.0,
                    "place_ms": 1.0,
                })()

    class _FakePlacer:

        def prepare_for_batch(self, **kwargs):
            del kwargs
            return None

        def start_batch(self, **kwargs):
            del kwargs
            return object()

        def execution_from_launched_batch(self, *, launched_batch, state_tracker):
            del launched_batch
            return _FakeExecution(state_tracker)

        def log_launched_batch_step_timings(self, launched_batch):
            del launched_batch
            return None

    monkeypatch.setattr(
        store,
        "read_chunk_pages_kv_fast_path_batch",
        _fake_read,
    )
    monkeypatch.setattr(
        store,
        "_ensure_direct_kv_placer",
        lambda kv_cache_dtype: _FakePlacer(),
    )

    token_db = _FakeTokenDatabase([
        (0, 256, key_a),
        (256, 512, key_b),
    ])
    in_flight_request = store.start_direct_placement_request(
        token_database=token_db,
        tokens=torch.arange(512, dtype=torch.int64),
        mask=None,
        kv_caches=[torch.zeros((2, 8, 16), dtype=torch.float16)],
        slot_mapping=_FakeCudaTensor(torch.arange(512, dtype=torch.int64)),
        kv_cache_dtype="auto",
    )

    assert in_flight_request is not None

    frontier_before_poll = store.get_direct_placement_request_frontier(
        in_flight_request=in_flight_request,
    )
    frontier_table_before_poll = store.get_direct_placement_request_frontier_table(
        in_flight_request=in_flight_request,
    )
    assert frontier_before_poll.frontier_row == (2, 2, 2, 0, 0, 2, 0)
    assert frontier_table_before_poll is not None
    assert tuple(int(v)
                 for v in frontier_table_before_poll.tolist()) == (2, 2, 2, 0, 0,
                                                                   2, 0)

    store.poll_direct_placement_request(
        in_flight_request=in_flight_request,
    )
    frontier_after_poll = store.get_direct_placement_request_frontier(
        in_flight_request=in_flight_request,
    )
    frontier_table_after_poll = store.get_direct_placement_request_frontier_table(
        in_flight_request=in_flight_request,
    )
    assert frontier_after_poll.frontier_row == (2, 2, 2, 0, 0, 2, 0)
    assert frontier_table_after_poll is not None
    assert tuple(int(v)
                 for v in frontier_table_after_poll.tolist()) == (2, 2, 2, 0,
                                                                  0, 2, 0)

    ret = store.finalize_direct_placement_request(
        in_flight_request=in_flight_request,
    )
    assert int(ret.sum().item()) == 512
    frontier_after_finalize = store.get_direct_placement_request_frontier(
        in_flight_request=in_flight_request,
    )
    frontier_table_after_finalize = store.get_direct_placement_request_frontier_table(
        in_flight_request=in_flight_request,
    )
    assert frontier_after_finalize.frontier_row == (4, 2, 2, 2, 2, 2, 0)
    assert frontier_table_after_finalize is not None
    assert tuple(int(v)
                 for v in frontier_table_after_finalize.tolist()) == (4, 2, 2,
                                                                      2, 2, 2,
                                                                      0)


def test_direct_placement_request_frontier_table_reuses_native_kv_frontier_during_read_stage(
    monkeypatch: pytest.MonkeyPatch,
):
    """异步 native read 阶段应优先复用底层 gpu_frontier_table。

    这条测试保护两个关键语义：

    1. request handle 暴露的 frontier table 应该直接复用 native KV runtime
       自带的那张 GPU-visible frontier table
    2. 在 placement wave 尚未 launch 的 read-ready 阶段，host 侧 getter/poll
       不应反向把 tracker 行写回去，覆盖 native runtime 已经维护的 launch/read
       frontier
    """
    monkeypatch.setenv("VLLM_BAM_KV_FAST_PATH", "1")

    store = _make_store()
    key_a = _register_chunk(store, "chunk-a", slot_id=0)
    key_b = _register_chunk(store, "chunk-b", slot_id=1)

    native_frontier_table = torch.tensor(
        [1, 2, 0, 0, 0, 2, 0],
        dtype=torch.int64,
    )

    class _FakeNativeRequestTable:
        gpu_frontier_table = native_frontier_table

    class _FakeNativeHandle:
        request_table = _FakeNativeRequestTable()

    class _FakeReadHandle:
        native_handle = _FakeNativeHandle()

    def _fake_submit(keys: list[Any]):
        del keys
        return _FakeReadHandle()

    def _fake_poll(request_handle: Any):
        del request_handle
        # 模拟底层 native runtime 自己推进 frontier table。
        native_frontier_table.copy_(
            torch.tensor([2, 2, 1, 0, 0, 2, 0], dtype=torch.int64))
        return type(
            "PollSnapshot", (),
            {
                "ready": False,
                "poll_iters": 1,
                "host_status": 1,
                "launch_frontier_chunks": 2,
                "read_ready_frontier_chunks": 1,
                "cache_ready_frontier_chunks": 0,
                "consumable_frontier_chunks": 0,
                "total_chunks": 2,
                "error_code": 0,
            },
        )()

    monkeypatch.setattr(
        store,
        "submit_chunk_pages_kv_fast_path_batch_request",
        _fake_submit,
    )
    monkeypatch.setattr(
        store,
        "poll_chunk_pages_kv_fast_path_batch_request",
        _fake_poll,
    )

    token_db = _FakeTokenDatabase([
        (0, 256, key_a),
        (256, 512, key_b),
    ])
    in_flight_request = store.start_direct_placement_request(
        token_database=token_db,
        tokens=torch.arange(512, dtype=torch.int64),
        mask=None,
        kv_caches=[torch.zeros((2, 8, 16), dtype=torch.float16)],
        slot_mapping=_FakeCudaTensor(torch.arange(512, dtype=torch.int64)),
        kv_cache_dtype="auto",
    )

    assert in_flight_request is not None
    assert store.get_direct_placement_request_frontier_table(
        in_flight_request=in_flight_request) is native_frontier_table

    # getter 现在应优先直接反映共享 frontier table，而不是再绕回 host tracker
    # 重建一份平行语义。
    frontier_snapshot = store.get_direct_placement_request_frontier(
        in_flight_request=in_flight_request,
    )
    assert frontier_snapshot.frontier_row == (1, 2, 0, 0, 0, 2, 0)
    assert tuple(int(v) for v in native_frontier_table.tolist()) == (1, 2, 0, 0,
                                                                     0, 2, 0)

    poll_snapshot = store.poll_direct_placement_request(
        in_flight_request=in_flight_request,
    )
    frontier_snapshot_after_poll = store.get_direct_placement_request_frontier(
        in_flight_request=in_flight_request,
    )
    assert poll_snapshot.read_ready_chunks == 1
    assert frontier_snapshot_after_poll.frontier_row == (2, 2, 1, 0, 0, 2, 0)
    assert tuple(int(v) for v in native_frontier_table.tolist()) == (2, 2, 1, 0,
                                                                     0, 2, 0)


def test_direct_placement_request_runtime_snapshot_forwards_native_handle(
    monkeypatch: pytest.MonkeyPatch,
):
    """request-level runtime snapshot 应能透传到底层 native read handle。"""
    monkeypatch.setenv("VLLM_BAM_KV_FAST_PATH", "1")

    store = _make_store()
    key_a = _register_chunk(store, "chunk-a", slot_id=0)
    key_b = _register_chunk(store, "chunk-b", slot_id=1)

    class _FakeReadHandle:
        pass

    fake_handle = _FakeReadHandle()
    observed: dict[str, Any] = {}

    def _fake_submit(keys: list[Any]):
        del keys
        return fake_handle

    def _fake_runtime_snapshot(request_handle: Any):
        observed["request_handle"] = request_handle
        assert request_handle is fake_handle
        return type(
            "RuntimeSnapshot", (),
            {
                "service_running": True,
                "active_count": 2,
                "request_id": 17,
                "worker_backend": "kv_persistent_service_v0",
                "request_table_ptr": 0x111,
                "frontier_table_ptr": 0x222,
                "completion_table_ptr": 0x333,
                "matched_runtime_row": (0, 1, 1, 17, 2, 224, 224, 4, 112, 1,
                                        0x111, 0x222, 0x333, 0x444),
                "runtime_rows": ((0, 1, 1, 17, 2, 224, 224, 4, 112, 1, 0x111,
                                  0x222, 0x333, 0x444), ),
            },
        )()

    monkeypatch.setattr(
        store,
        "submit_chunk_pages_kv_fast_path_batch_request",
        _fake_submit,
    )
    monkeypatch.setattr(
        store,
        "get_chunk_pages_kv_fast_path_batch_request_runtime_snapshot",
        _fake_runtime_snapshot,
    )

    token_db = _FakeTokenDatabase([
        (0, 256, key_a),
        (256, 512, key_b),
    ])
    in_flight_request = store.start_direct_placement_request(
        token_database=token_db,
        tokens=torch.arange(512, dtype=torch.int64),
        mask=None,
        kv_caches=[torch.zeros((2, 8, 16), dtype=torch.float16)],
        slot_mapping=_FakeCudaTensor(torch.arange(512, dtype=torch.int64)),
        kv_cache_dtype="auto",
    )

    assert in_flight_request is not None

    runtime_snapshot = store.get_direct_placement_request_runtime_snapshot(
        in_flight_request=in_flight_request,
    )

    assert observed["request_handle"] is fake_handle
    assert runtime_snapshot is not None
    assert runtime_snapshot.service_running is True
    assert runtime_snapshot.active_count == 2
    assert runtime_snapshot.request_id == 17
    assert runtime_snapshot.worker_backend == "kv_persistent_service_v0"
    assert runtime_snapshot.matched_runtime_row is not None
    assert runtime_snapshot.matched_runtime_row[3] == 17


def test_direct_placement_request_frontier_table_does_not_regress_to_older_tracker(
    monkeypatch: pytest.MonkeyPatch,
):
    """native read 持有 frontier table 时，getter 不应被较旧 tracker 回退。

    这条用例保护的是 shared frontier table 当前最关键的单调语义：

    1. native read frontier 可能已经比 host tracker 更靠前；
    2. request-level getter 会同时看到共享 frontier table 与 tracker；
    3. 此时共享表必须保留更靠前的 frontier，而不能又被 tracker 旧视图覆盖回去。

    这是后续继续把 frontier 更新权往 GPU runtime 下放时必须稳定的边界。
    """
    monkeypatch.setenv("VLLM_BAM_KV_FAST_PATH", "1")

    store = _make_store()
    key_a = _register_chunk(store, "chunk-a", slot_id=0)
    key_b = _register_chunk(store, "chunk-b", slot_id=1)

    native_frontier_table = torch.tensor([4, 2, 2, 1, 1, 2, 0],
                                         dtype=torch.int32)

    class _FakeHandle:

        def __init__(self, frontier_table):
            self.native_handle = type(
                "NativeHandle", (),
                {
                    "request_table": type(
                        "RequestTable", (),
                        {"gpu_frontier_table": frontier_table},
                    )(),
                },
            )()

    fake_handle = _FakeHandle(native_frontier_table)

    def _fake_submit(keys: list[Any]):
        del keys
        return fake_handle

    monkeypatch.setattr(
        store,
        "submit_chunk_pages_kv_fast_path_batch_request",
        _fake_submit,
    )

    token_db = _FakeTokenDatabase([
        (0, 256, key_a),
        (256, 512, key_b),
    ])
    in_flight_request = store.start_direct_placement_request(
        token_database=token_db,
        tokens=torch.arange(512, dtype=torch.int64),
        mask=None,
        kv_caches=[torch.zeros((2, 8, 16), dtype=torch.float16)],
        slot_mapping=_FakeCudaTensor(torch.arange(512, dtype=torch.int64)),
        kv_cache_dtype="auto",
    )

    assert in_flight_request is not None

    frontier_snapshot = store.get_direct_placement_request_frontier(
        in_flight_request=in_flight_request,
    )
    frontier_table = store.get_direct_placement_request_frontier_table(
        in_flight_request=in_flight_request,
    )

    assert frontier_snapshot.frontier_row == (4, 2, 2, 1, 1, 2, 0)
    assert frontier_table is not None
    assert tuple(int(v)
                 for v in frontier_table.tolist()) == (4, 2, 2, 1, 1, 2, 0)


def test_direct_kv_placer_is_reused_per_store():
    """同一个 BaM store 内应复用 direct KV placer。

    这条约束对 warmup 很关键：如果每次 direct retrieve 都重新 new 一个
    placer，那么 Triton/JIT 的 warmup 状态无法跨请求保留，请求间就会反复
    把一次性成本记回热路径。
    """
    store = _make_store()

    placer_a = store._ensure_direct_kv_placer(kv_cache_dtype="auto")
    placer_b = store._ensure_direct_kv_placer(kv_cache_dtype="auto")

    assert placer_a is placer_b
    assert isinstance(placer_a, BaMDirectKVPlacer)


class _FakeEvent:

    def record(self) -> None:
        return None

    def elapsed_time(self, other: object) -> float:
        del other
        return 0.0


class _FakeQueryEvent(_FakeEvent):

    def __init__(self, ready_after_queries: int = 0) -> None:
        self._remaining_false_queries = max(int(ready_after_queries), 0)
        self.query_calls = 0

    def query(self) -> bool:
        self.query_calls += 1
        if self._remaining_false_queries > 0:
            self._remaining_false_queries -= 1
            return False
        return True


class _FakeCudaTensor:

    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor
        self.is_cuda = True
        self.device = torch.device("cuda:0")

    def __getitem__(self, item: Any) -> torch.Tensor:
        return self._tensor.__getitem__(item)

    def numel(self) -> int:
        return int(self._tensor.numel())
