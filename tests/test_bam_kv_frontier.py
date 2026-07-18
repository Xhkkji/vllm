# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


GIDS_MODULE_DIR = (
    Path(__file__).resolve().parents[2] / "BaM_IOStack" / "gids_module")
if str(GIDS_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(GIDS_MODULE_DIR))

from bam_kv_store import (  # noqa: E402
    BaMGPUWorkerKVExecutor,
    BaMKVNativeBatchHandle,
    BaMKVRequestTable,
    BaMKVRequest,
    BaMKVStore,
    BaMRowCtxKVExecutor,
    KV_BATCH_CONSUMED,
    KV_BATCH_IO_DONE,
    KV_BATCH_SUBMITTED,
)
from bam_row_store import (  # noqa: E402
    BaMRowAsyncRequest,
    BaMRowStore,
)


def test_prepare_request_table_initializes_gpu_frontier_table():
    """submit 前的 GPU-visible 状态表都应初始化成 submitted 态。"""
    executor = BaMRowCtxKVExecutor(
        row_store=SimpleNamespace(),
        page_bytes=128 * 1024,
        device=torch.device("cpu"),
    )
    requests = (
        BaMKVRequest(
            request_id=1,
            chunk_id=10,
            page_offset=100,
            page_count=112,
            page_bytes=128 * 1024,
            actual_tokens=256,
        ),
        BaMKVRequest(
            request_id=2,
            chunk_id=11,
            page_offset=212,
            page_count=112,
            page_bytes=128 * 1024,
            actual_tokens=256,
        ),
    )

    table = executor._prepare_request_table(requests)

    assert tuple(table.gpu_status.tolist()) == (KV_BATCH_SUBMITTED,)
    assert tuple(table.gpu_chunk_status.tolist()) == (
        KV_BATCH_SUBMITTED,
        KV_BATCH_SUBMITTED,
    )
    assert tuple(tuple(row) for row in table.gpu_completion_table.tolist()) == (
        (10, KV_BATCH_SUBMITTED, 0, 0),
        (11, KV_BATCH_SUBMITTED, 0, 0),
    )
    assert tuple(table.gpu_frontier_table.tolist()) == (
        KV_BATCH_SUBMITTED,
        2,
        0,
        0,
        0,
        2,
        0,
    )


def test_snapshot_status_returns_host_frontier_without_debug_d2h():
    """默认热路径下，snapshot 也应直接返回 host frontier mirror。"""

    class _FakeRowStore:

        def kv_get_batch_status(self, request):
            return KV_BATCH_SUBMITTED

        def kv_get_batch_frontier_state(self, request):
            assert request.request_id == 7
            return [KV_BATCH_SUBMITTED, 4, 0, 0, 0, 4, 0]

    executor = BaMRowCtxKVExecutor(
        row_store=_FakeRowStore(),
        page_bytes=128 * 1024,
        device=torch.device("cpu"),
    )

    snapshot = executor.snapshot_status(
        SimpleNamespace(row_request=SimpleNamespace(request_id=7)))

    assert snapshot.host_status == KV_BATCH_SUBMITTED
    assert snapshot.frontier_row == (KV_BATCH_SUBMITTED, 4, 0, 0, 0, 4, 0)
    assert snapshot.gpu_frontier_row == ()
    assert snapshot.completion_rows == ()


def test_snapshot_status_debug_mode_reads_and_validates_gpu_frontier(
    monkeypatch: pytest.MonkeyPatch,
):
    """调试模式下，host frontier 与 GPU frontier 应保持一致。"""

    class _FakeRowStore:

        def kv_get_batch_status(self, request):
            return KV_BATCH_IO_DONE

        def kv_get_batch_gpu_status(self, request):
            return KV_BATCH_IO_DONE

        def kv_get_chunk_gpu_statuses(self, request):
            return [KV_BATCH_IO_DONE, KV_BATCH_IO_DONE]

        def kv_get_completion_table(self, request):
            return [
                (0, KV_BATCH_IO_DONE, 128 * 1024, 0),
                (1, KV_BATCH_IO_DONE, 128 * 1024, 0),
            ]

        def kv_get_batch_frontier_state(self, request):
            return [KV_BATCH_IO_DONE, 2, 2, 0, 0, 2, 0]

        def kv_get_batch_gpu_frontier_state(self, request):
            return (KV_BATCH_IO_DONE, 2, 2, 0, 0, 2, 0)

    monkeypatch.setenv("VLLM_BAM_KV_DEBUG_STATUS", "1")
    executor = BaMRowCtxKVExecutor(
        row_store=_FakeRowStore(),
        page_bytes=128 * 1024,
        device=torch.device("cpu"),
    )

    snapshot = executor.snapshot_status(
        SimpleNamespace(row_request=SimpleNamespace(request_id=9)))

    executor._validate_status(
        snapshot,
        expected=KV_BATCH_IO_DONE,
        stage="ready",
        request_id=9,
    )
    assert snapshot.frontier_row == (KV_BATCH_IO_DONE, 2, 2, 0, 0, 2, 0)
    assert snapshot.gpu_frontier_row == (KV_BATCH_IO_DONE, 2, 2, 0, 0, 2, 0)


def test_runtime_state_for_native_batch_matches_runtime_slot_by_request_id():
    """runtime 观察接口应能把 native batch 找回对应的 runtime slot。"""

    class _FakeRowStore:

        def kv_worker_runtime_service_running(self):
            return True

        def kv_worker_runtime_active_count(self):
            return 3

        def kv_worker_get_runtime_snapshot(self):
            return [
                (
                    0,
                    1,
                    KV_BATCH_SUBMITTED,
                    100,
                    2,
                    224,
                    224,
                    4,
                    112,
                    1,
                    0x111,
                    0x222,
                    0x333,
                    0x444,
                ),
                (
                    1,
                    2,
                    KV_BATCH_IO_DONE,
                    123,
                    2,
                    224,
                    224,
                    4,
                    112,
                    1,
                    0x555,
                    0x666,
                    0x777,
                    0x888,
                ),
            ]

    store = BaMKVStore(
        row_store=_FakeRowStore(),
        page_bytes=128 * 1024,
        device=torch.device("cpu"),
    )
    request = BaMKVRequest(
        request_id=1,
        chunk_id=10,
        page_offset=100,
        page_count=112,
        page_bytes=128 * 1024,
        actual_tokens=256,
    )
    request_table = BaMKVRequestTable(
        requests=(request, ),
        request_table=torch.zeros((1, 4), dtype=torch.int64),
        pages=torch.zeros((112, 128 * 1024), dtype=torch.uint8),
        gpu_status=torch.zeros((1, ), dtype=torch.int32),
        gpu_chunk_status=torch.zeros((1, ), dtype=torch.int32),
        gpu_completion_table=torch.zeros((1, 4), dtype=torch.int64),
        gpu_frontier_table=torch.zeros((7, ), dtype=torch.int64),
        page_count=112,
        page_bytes=128 * 1024,
    )
    handle = BaMKVNativeBatchHandle(
        request_table=request_table,
        row_request=SimpleNamespace(request_id=123),
        request_table_mode="gpu",
        total_start_s=0.0,
        submit_ms=0.0,
        poll_start_s=0.0,
        worker_backend="kv_persistent_service_v0",
    )

    snapshot = store.runtime_state_for_native_batch(handle)

    assert snapshot.service_running is True
    assert snapshot.active_count == 3
    assert snapshot.request_id == 123
    assert snapshot.worker_backend == "kv_persistent_service_v0"
    assert snapshot.matched_runtime_row is not None
    assert snapshot.matched_runtime_row[0] == 1
    assert snapshot.matched_runtime_row[3] == 123


def test_kv_worker_poll_prefers_request_status_facade_over_batch_poll():
    """wrapper 层应优先走 request-level status façade。"""

    class _FakeStore:

        def __init__(self):
            self.request_poll_calls = 0
            self.batch_poll_calls = 0

        def kv_worker_backend_name(self):
            return "kv_persistent_service_v0"

        def kv_worker_poll_request(self, request_id):
            self.request_poll_calls += 1
            assert request_id == 77
            return KV_BATCH_IO_DONE

        def kv_worker_poll_batch(self):
            self.batch_poll_calls += 1
            raise AssertionError("request-level façade should be preferred")

    row_store = BaMRowStore.__new__(BaMRowStore)
    row_store.store = _FakeStore()

    ready = row_store.kv_worker_poll(
        BaMRowAsyncRequest(
            request_id=77,
            row_count=224,
            batch_size=2,
            pages_per_chunk=112,
            is_kv_batch=True,
        ))

    assert ready is True
    assert row_store.store.request_poll_calls == 1
    assert row_store.store.batch_poll_calls == 0


def test_gpu_worker_consume_skips_host_consume_when_service_already_consumed():
    """persistent service 已经把 pages 填好时，Python consume 不应再回调 C++。

    这条用例覆盖当前最新主线：

    - GPU 后台 service 已经把 request 推到 `CONSUMED`
    - `request_table.pages` 已由设备侧直接填充完成
    - Python `consume()` 只组织结果，不再重复调用
      `row_store.kv_worker_consume()`
    """

    class _FakeRowStore:

        def __init__(self):
            self.consume_calls = 0
            self.cleanup_calls = 0
            self.poll_calls = 0

        def kv_worker_backend_name(self):
            return "kv_persistent_service_v0"

        def kv_worker_poll(self, request):
            self.poll_calls += 1
            assert request.request_id == 55
            return True

        def kv_worker_consume(self, request, out_rows):
            del request, out_rows
            self.consume_calls += 1
            raise AssertionError("host consume should be skipped")

        def kv_worker_cleanup(self, request):
            assert request.request_id == 55
            self.cleanup_calls += 1

    class _FakeRowCtxExecutor:

        def __init__(self, row_store):
            self.row_store = row_store
            self.last_validate_expected = None

        def can_run(self, requests):
            return True

        def snapshot_status(self, handle):
            del handle
            return SimpleNamespace(
                host_status=KV_BATCH_CONSUMED,
                gpu_status=None,
                chunk_statuses=(),
                completion_rows=(),
                frontier_row=(),
                gpu_frontier_row=(),
            )

        def _validate_status(self, snapshot, *, expected, stage, request_id):
            del snapshot, stage, request_id
            self.last_validate_expected = expected
            assert expected == KV_BATCH_CONSUMED

        def _build_results(self, handle, **kwargs):
            del kwargs
            return [
                SimpleNamespace(
                    descriptor=SimpleNamespace(chunk_id=7),
                    pages=handle.request_table.pages,
                    stats=SimpleNamespace(
                        submit_ms=0.0,
                        poll_ms=0.0,
                        get_ms=0.0,
                        total_ms=0.0,
                        poll_iters=0,
                        submit_status=KV_BATCH_SUBMITTED,
                        ready_status=KV_BATCH_CONSUMED,
                        consumed_status=KV_BATCH_CONSUMED,
                        submit_gpu_status=KV_BATCH_SUBMITTED,
                        ready_gpu_status=KV_BATCH_CONSUMED,
                        consumed_gpu_status=KV_BATCH_CONSUMED,
                        request_table_mode="gpu",
                        executor_name="rowctx",
                        worker_backend="rowctx",
                        submit_chunk_statuses=(),
                        ready_chunk_statuses=(),
                        consumed_chunk_statuses=(),
                        submit_completion_statuses=(),
                        ready_completion_statuses=(),
                        consumed_completion_statuses=(),
                        ready_completion_bytes=(),
                        consumed_completion_bytes=(),
                        ready_completion_errors=(),
                        consumed_completion_errors=(),
                    ),
                )
            ]

    row_store = _FakeRowStore()
    rowctx_executor = _FakeRowCtxExecutor(row_store)
    executor = BaMGPUWorkerKVExecutor(rowctx_executor)

    request = BaMKVRequest(
        request_id=1,
        chunk_id=7,
        page_offset=10,
        page_count=112,
        page_bytes=128 * 1024,
        actual_tokens=256,
    )
    request_table = BaMKVRequestTable(
        requests=(request,),
        request_table=torch.zeros((1, 4), dtype=torch.int64),
        pages=torch.zeros((112, 128 * 1024), dtype=torch.uint8),
        gpu_status=torch.zeros((1,), dtype=torch.int32),
        gpu_chunk_status=torch.zeros((1,), dtype=torch.int32),
        gpu_completion_table=torch.zeros((1, 4), dtype=torch.int64),
        gpu_frontier_table=torch.zeros((7,), dtype=torch.int64),
        page_count=112,
        page_bytes=128 * 1024,
    )
    handle = BaMKVNativeBatchHandle(
        request_table=request_table,
        row_request=SimpleNamespace(request_id=55),
        request_table_mode="gpu",
        total_start_s=0.0,
        submit_ms=0.0,
        poll_start_s=0.0,
    )

    result = executor.consume(handle)

    assert len(result.results) == 1
    assert int(result.results[0].descriptor.chunk_id) == 7
    assert row_store.poll_calls == 1
    assert row_store.consume_calls == 0
    assert row_store.cleanup_calls == 1
    assert handle.get_ms is not None
    assert handle.get_ms >= 0.0
    assert handle.consumed_snapshot is not None
