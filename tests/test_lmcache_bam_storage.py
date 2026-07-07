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


def test_direct_place_chunks_only_returns_contiguous_consumable_prefix(
    monkeypatch: pytest.MonkeyPatch,
):
    """显式把返回目标限制为 1 个 chunk 时，只返回连续可消费前缀。

    当前正常主线已经要求：

    - 命中了多少连续 prefix
    - 就要等这多少 prefix 真正 consumable 再返回

    因此如果要验证“只返回第一个 chunk”这类更接近实验控制面的行为，就必须
    显式把当前返回目标收缩到 1 个 chunk。否则这条测试会和主线语义冲突。
    """
    monkeypatch.setenv("VLLM_BAM_KV_FAST_PATH", "1")
    monkeypatch.setenv("VLLM_BAM_DIRECT_PLACEMENT_FRONTIER_CHUNKS", "1")

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
    assert bool(ret[:256].all().item()) is True
    assert bool(ret[256:].any().item()) is False

    snapshot = store.get_last_direct_placement_state_snapshot()
    assert snapshot is not None
    assert snapshot.cache_ready_chunks == 1
    assert snapshot.cache_ready_tokens == 256
    assert snapshot.consumable_chunks == 1
    assert snapshot.consumable_tokens == 256
    assert int(ret.sum().item()) == snapshot.consumable_tokens


def test_direct_place_chunks_passes_frontier_launch_limit_to_placer(
    monkeypatch: pytest.MonkeyPatch,
):
    """显式配置 frontier chunk 限额时，应只把前 N 个 chunk 交给本轮 launch。

    这个测试验证两层语义同时成立：

    1. store 会正确解析 `VLLM_BAM_DIRECT_PLACEMENT_FRONTIER_CHUNKS`
       并透传给 prepare/start；
    2. 当本轮只 launch 第一个 chunk 时，返回给 LMCache 的 ret_mask 也只能
       覆盖这个 chunk，而不能把后面的 chunk 提前暴露成已恢复前缀。
    """
    monkeypatch.setenv("VLLM_BAM_KV_FAST_PATH", "1")
    monkeypatch.setenv("VLLM_BAM_DIRECT_PLACEMENT_FRONTIER_CHUNKS", "1")

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

    assert observed["prepare_max_chunks_to_launch"] == 1
    assert observed["start_max_chunks_to_launch"] == 1
    assert ret is not None
    assert bool(ret[:256].all().item()) is True
    assert bool(ret[256:].any().item()) is False

    snapshot = store.get_last_direct_placement_state_snapshot()
    assert snapshot is not None
    assert snapshot.cache_ready_chunks == 1
    assert snapshot.consumable_chunks == 1
    assert snapshot.consumable_tokens == 256


def test_direct_place_chunks_runs_followup_wave_without_expanding_return_mask(
    monkeypatch: pytest.MonkeyPatch,
):
    """真实 store 路径启用 followup wave 后，应补后续 chunk 但不扩张返回 mask。

    这是当前“两波策略”接入真实 store 的核心语义：

    - 第一波决定返回给 LMCache / vLLM 的可消费前缀
    - 第二波只是在同一次 direct retrieve 中继续补 resident cache

    因此：
    - `ret_mask` 仍然只能覆盖第一波
    - 但最终 tracker 可以反映 followup 后更大的 resident/cache-ready 范围
    """
    monkeypatch.setenv("VLLM_BAM_KV_FAST_PATH", "1")
    monkeypatch.setenv("VLLM_BAM_DIRECT_PLACEMENT_FRONTIER_CHUNKS", "1")
    monkeypatch.setenv("VLLM_BAM_DIRECT_PLACEMENT_FOLLOWUP_CHUNKS", "2")

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

    assert observed["prepare_calls"] == [(0, 1), (1, 2)]
    assert observed["start_calls"] == [(0, 1), (1, 2)]
    assert ret is not None
    assert bool(ret[:256].all().item()) is True
    assert bool(ret[256:].any().item()) is False

    snapshot = store.get_last_direct_placement_state_snapshot()
    assert snapshot is not None
    assert snapshot.cache_ready_chunks == 3
    assert snapshot.cache_ready_tokens == 768
    assert snapshot.consumable_chunks == 3
    assert snapshot.consumable_tokens == 768
    assert int(ret.sum().item()) == 256


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

    assert snapshot is not None
    assert snapshot.cache_ready_chunks == 1
    assert snapshot.consumable_chunks == 1
    assert snapshot.consumable_tokens == 256
    assert delayed_chunk_event.query_calls >= 1
    assert delayed_chunk_event.query() is False


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
