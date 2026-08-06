# SPDX-License-Identifier: Apache-2.0

"""异步 KV 调度状态机的单元测试。"""

import pytest

from vllm.core.scheduler_policy import (AsyncKVLoadEvent,
                                        AsyncKVLoadState,
                                        AsyncKVSchedulePolicy)


def test_async_kv_policy_lifecycle():
    policy = AsyncKVSchedulePolicy(max_in_flight=2)
    request = policy.submit("seq-group-0", [(1, 7), (2, 8)])

    assert request.request_id == "async-kv-0"
    assert request.block_mapping == ((1, 7), (2, 8))
    assert policy.loading_request_ids == (request.request_id,)
    assert policy.ready_request_ids == ()

    policy.apply_event(
        AsyncKVLoadEvent(request.request_id, AsyncKVLoadState.READY))
    assert policy.loading_request_ids == ()
    assert policy.ready_request_ids == (request.request_id,)
    assert policy.pop_ready() == (request,)
    assert policy.ready_request_ids == ()


def test_async_kv_policy_errors_and_capacity():
    policy = AsyncKVSchedulePolicy(max_in_flight=1)
    request = policy.submit("seq-group-0", [])

    with pytest.raises(RuntimeError):
        policy.submit("seq-group-1", [])

    policy.apply_event(
        AsyncKVLoadEvent(request.request_id, AsyncKVLoadState.ERROR,
                         error="restore failed"))
    failed = policy.pop_errors()
    assert len(failed) == 1
    assert failed[0].request == request
    assert failed[0].error == "restore failed"
    assert policy.in_flight_count == 0


def test_async_kv_policy_rejects_duplicate_completion():
    policy = AsyncKVSchedulePolicy()
    request = policy.submit("seq-group-0", [])
    event = AsyncKVLoadEvent(request.request_id, AsyncKVLoadState.READY)
    policy.apply_event(event)

    with pytest.raises(RuntimeError):
        policy.apply_event(event)
