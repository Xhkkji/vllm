# SPDX-License-Identifier: Apache-2.0

"""层级 I/O plan 和父事务屏障的纯控制面测试。"""

import pytest

from vllm.core.block_reservation import LogicalBlockKey
from vllm.core.custom_schedulers.async_kv_transfer import (
    AsyncKVTransferEvent, AsyncKVTransferOperation, AsyncKVTransferPriority,
    AsyncKVTransferRequest, AsyncKVTransferState)
from vllm.core.custom_schedulers.hierarchical_io import (
    HierarchicalIOConfig, HierarchicalLayerBarrierConfig,
    HierarchicalRestoreController, PrefetchUnit, RollingPrefetchConfig,
    RollingPrefetchRuntime, activate_layer_barrier, build_layer_restore_plan,
    select_prefetch_unit_blocks, wait_for_local_layer)


def test_layer_restore_plan_is_contiguous_and_default_disabled():
    assert not HierarchicalIOConfig.from_env({}).enabled

    plan = build_layer_restore_plan(plan_id="restore-1",
                                    num_layers=10,
                                    window_layers=4,
                                    created_monotonic_ns=100)
    assert [unit.layer_range for unit in plan.units] == [
        (0, 4), (4, 8), (8, 10)
    ]
    assert [unit.index for unit in plan.units] == [0, 1, 2]
    assert all(unit.block_indices is None for unit in plan.units)


def test_prefetch_unit_projects_dense_and_sparse_block_sets():
    """同一个接口应同时表达 layer 全量读取和未来 sparse 子集读取。"""
    mapping = ((10, 20), (11, 21), (12, 22), (13, 23))
    logical_blocks = tuple(LogicalBlockKey(7, index) for index in range(4))

    dense = PrefetchUnit(index=0, start_layer=0, end_layer=2)
    assert select_prefetch_unit_blocks(dense, mapping,
                                       logical_blocks) == (mapping,
                                                           logical_blocks)

    sparse = PrefetchUnit(index=1,
                          start_layer=2,
                          end_layer=4,
                          block_indices=(0, 2, 3))
    selected_mapping, selected_keys = select_prefetch_unit_blocks(
        sparse, mapping, logical_blocks)
    assert selected_mapping == ((10, 20), (12, 22), (13, 23))
    assert [key.logical_index for key in selected_keys] == [0, 2, 3]


def test_prefetch_unit_rejects_ambiguous_or_invalid_block_selection():
    with pytest.raises(ValueError, match="strictly increasing"):
        PrefetchUnit(index=0,
                     start_layer=0,
                     end_layer=2,
                     block_indices=(1, 1))
    unit = PrefetchUnit(index=0,
                        start_layer=0,
                        end_layer=2,
                        block_indices=(2, ))
    with pytest.raises(ValueError, match="outside"):
        select_prefetch_unit_blocks(unit, ((10, 20), ),
                                    (LogicalBlockKey(7, 0), ))


def test_rolling_config_is_explicit_and_validated():
    config = RollingPrefetchConfig.from_env({
        "VLLM_BAM_MDS_HIERARCHICAL_ROLLING_ENABLE": "1",
        "VLLM_BAM_MDS_HIERARCHICAL_LEAD_WINDOWS": "2",
        "VLLM_BAM_MDS_HIERARCHICAL_MAX_LEAD_WINDOWS": "3",
        "VLLM_BAM_MDS_HIERARCHICAL_TARGET_SLACK_MS": "0.5",
    })
    assert config.enabled
    assert config.initial_units == 2
    assert config.max_lead_units == 3


def _prefetch_request(index: int, *, activate: bool) -> AsyncKVTransferRequest:
    return AsyncKVTransferRequest(
        request_id=f"unit-{index}",
        seq_group_id="seq-0",
        reservation_id="plan-0",
        operation=AsyncKVTransferOperation.READ,
        block_mapping=((10, 20), ),
        logical_blocks=(),
        priority=AsyncKVTransferPriority.CRITICAL_READ,
        layer_range=(index * 2, (index + 1) * 2),
        prefetch_plan_id="plan-0",
        prefetch_unit_index=index,
        activate_on_submit=activate,
    )


def test_rolling_runtime_activates_future_unit_from_model_progress():
    runtime = RollingPrefetchRuntime(
        RollingPrefetchConfig(enabled=True,
                              lead_units=1,
                              max_lead_units=1))
    activated = []
    ready = set()

    def submit(request, mapping):
        activated.append((request.request_id, mapping))
        return AsyncKVTransferEvent(request.request_id,
                                    AsyncKVTransferState.PENDING)

    def poll(request, _mapping):
        state = (AsyncKVTransferState.READY
                 if request.request_id in ready else
                 AsyncKVTransferState.PENDING)
        return AsyncKVTransferEvent(request.request_id, state)

    assert runtime.submit_or_stage(0, _prefetch_request(0, activate=True),
                                   "mapping-0", submit)[0].state == (
                                       AsyncKVTransferState.PENDING)
    assert not runtime.submit_or_stage(0,
                                       _prefetch_request(1, activate=False),
                                       "mapping-1", submit)
    assert activated == [("unit-0", "mapping-0")]

    ready.update(("unit-0", "unit-1"))
    runtime.wait_ready(0, ("seq-0", ), 0, submit, poll, max_active=2)
    assert activated == [("unit-0", "mapping-0"),
                         ("unit-1", "mapping-1")]
    events = runtime.poll_units(0, poll)
    assert [(event.request_id, event.state) for event in events] == [
        ("unit-1", AsyncKVTransferState.PENDING),
        ("unit-0", AsyncKVTransferState.READY),
        ("unit-1", AsyncKVTransferState.READY),
    ]
    traces = runtime.pop_traces()
    assert {(trace.phase, trace.unit_index) for trace in traces} >= {
        ("activated", 0),
        ("activated", 1),
        ("physical_ready", 0),
        ("physical_ready", 1),
        ("barrier_ready", 0),
    }


def test_first_window_admission_is_separate_from_full_restore():
    plan = build_layer_restore_plan(plan_id="restore-2",
                                    num_layers=8,
                                    window_layers=2,
                                    created_monotonic_ns=100)
    controller = HierarchicalRestoreController()
    controller.register(plan, ["w0", "w1", "w2", "w3"])

    # 后续 window 可以乱序完成，但不能冒充 first-window-ready admission。
    progress = controller.mark_ready("w2", now_monotonic_ns=150)
    assert not progress.first_unit_ready
    assert not progress.all_terminal

    progress = controller.mark_ready("w0", now_monotonic_ns=200)
    assert progress.first_unit_became_ready
    assert progress.first_unit_ready
    assert progress.first_unit_ready_monotonic_ns == 200
    assert not progress.all_terminal

    controller.mark_ready("w1", now_monotonic_ns=250)
    progress = controller.mark_ready("w3", now_monotonic_ns=300)
    assert progress.all_terminal
    assert progress.all_ready
    assert controller.release(plan.plan_id) == plan


def test_window_error_prevents_parent_publish_until_all_terminal():
    plan = build_layer_restore_plan(plan_id="restore-3",
                                    num_layers=4,
                                    window_layers=2)
    controller = HierarchicalRestoreController()
    controller.register(plan, ["w0", "w1"])

    failed = controller.mark_error("w1")
    assert failed.failed
    assert not failed.all_terminal
    ready = controller.mark_ready("w0")
    assert ready.all_terminal
    assert not ready.all_ready
    assert ready.failed
    controller.release(plan.plan_id)


def test_layer_barrier_is_forward_scoped_and_default_disabled():
    assert not HierarchicalLayerBarrierConfig.from_env({}).enabled
    assert HierarchicalLayerBarrierConfig.from_env({
        "VLLM_BAM_MDS_HIERARCHICAL_LAYER_BARRIER": "1"
    }).enabled

    observed = []
    with activate_layer_barrier(
            lambda virtual_engine, request_ids, layer: observed.append(
                (virtual_engine, tuple(request_ids), layer)),
            virtual_engine=3,
            request_ids=("request-a", "request-b")):
        wait_for_local_layer(5)

    # context 退出后模型层调用必须自然退化为 no-op，避免状态泄漏到下一 batch。
    wait_for_local_layer(6)
    assert observed == [(3, ("request-a", "request-b"), 5)]
