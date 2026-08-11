# SPDX-License-Identifier: Apache-2.0

"""异步 KV 调度状态机的单元测试。"""

import pytest

from vllm.core.scheduler_policy import (AsyncKVSchedulePolicy,
                                        AsyncKVTransferEvent,
                                        AsyncKVTransferOperation,
                                        AsyncKVTransferState)


def test_async_kv_policy_lifecycle():
    policy = AsyncKVSchedulePolicy(max_in_flight=2)
    request = policy.enqueue("seq-group-0", "reservation-0",
                             AsyncKVTransferOperation.READ,
                             [(1, 7), (2, 8)])

    assert request.request_id == "async-kv-0"
    assert request.block_mapping == ((1, 7), (2, 8))
    assert policy.queued_request_ids == (request.request_id,)
    assert policy.activate_next() == (request,)
    assert policy.pending_request_ids == (request.request_id,)
    assert policy.ready_request_ids == ()

    policy.apply_event(
        AsyncKVTransferEvent(request.request_id,
                             AsyncKVTransferState.READY))
    assert policy.pending_request_ids == ()
    assert policy.ready_request_ids == (request.request_id,)
    assert policy.pop_ready() == (request,)
    assert policy.ready_request_ids == ()


def test_async_kv_policy_errors_and_capacity():
    policy = AsyncKVSchedulePolicy(max_in_flight=1)
    queued_write = policy.enqueue("seq-group-0", "reservation-0",
                                  AsyncKVTransferOperation.WRITE, [])
    critical_read = policy.enqueue("seq-group-1", "reservation-1",
                                   AsyncKVTransferOperation.READ, [])

    assert policy.activate_next() == (critical_read,)
    assert policy.activate_next() == ()
    assert policy.queued_request_ids == (queued_write.request_id,)

    policy.apply_event(
        AsyncKVTransferEvent(critical_read.request_id,
                             AsyncKVTransferState.ERROR,
                             error="restore failed"))
    failed = policy.pop_errors()
    assert len(failed) == 1
    assert failed[0].request == critical_read
    assert failed[0].error == "restore failed"
    assert policy.in_flight_count == 0
    assert policy.activate_next() == (queued_write,)


def test_async_kv_policy_rejects_duplicate_completion():
    policy = AsyncKVSchedulePolicy()
    request = policy.enqueue("seq-group-0", "reservation-0",
                             AsyncKVTransferOperation.READ, [])
    policy.activate_next()
    event = AsyncKVTransferEvent(request.request_id,
                                 AsyncKVTransferState.READY)
    policy.apply_event(event)

    with pytest.raises(RuntimeError):
        policy.apply_event(event)


def test_async_kv_policy_accepts_out_of_order_completions():
    """多槽完成顺序不应与激活顺序绑定。"""
    policy = AsyncKVSchedulePolicy(max_in_flight=2)
    first = policy.enqueue("seq-group-0", "reservation-0",
                           AsyncKVTransferOperation.READ, [(1, 2)])
    second = policy.enqueue("seq-group-1", "reservation-1",
                            AsyncKVTransferOperation.READ, [(3, 4)])
    assert policy.activate_next() == (first, second)

    policy.apply_event(
        AsyncKVTransferEvent(second.request_id,
                             AsyncKVTransferState.READY))
    assert policy.pop_ready() == (second,)
    assert policy.pending_request_ids == (first.request_id,)


def test_worker_event_can_activate_only_a_staged_prefetch_unit():
    policy = AsyncKVSchedulePolicy(max_in_flight=1)
    request = policy.enqueue(
        "seq",
        "plan",
        AsyncKVTransferOperation.READ,
        ((1, 2), ),
        prefetch_plan_id="plan",
        prefetch_unit_index=0,
    )
    assert policy.stage_plan("plan")[0].activate_on_submit is False
    policy.apply_event(
        AsyncKVTransferEvent(request.request_id,
                             AsyncKVTransferState.PENDING))
    assert policy.pending_request_ids == (request.request_id, )
