# SPDX-License-Identifier: Apache-2.0

"""层级 I/O plan 和父事务屏障的纯控制面测试。"""

from vllm.core.custom_schedulers.hierarchical_io import (
    HierarchicalIOConfig, HierarchicalLayerBarrierConfig,
    HierarchicalRestoreController, activate_layer_barrier,
    build_layer_restore_plan, wait_for_local_layer)


def test_layer_restore_plan_is_contiguous_and_default_disabled():
    assert not HierarchicalIOConfig.from_env({}).enabled

    plan = build_layer_restore_plan(plan_id="restore-1",
                                    num_layers=10,
                                    window_layers=4,
                                    created_monotonic_ns=100)
    assert [window.layer_range for window in plan.windows] == [
        (0, 4), (4, 8), (8, 10)
    ]
    assert [window.index for window in plan.windows] == [0, 1, 2]


def test_first_window_admission_is_separate_from_full_restore():
    plan = build_layer_restore_plan(plan_id="restore-2",
                                    num_layers=8,
                                    window_layers=2,
                                    created_monotonic_ns=100)
    controller = HierarchicalRestoreController()
    controller.register(plan, ["w0", "w1", "w2", "w3"])

    # 后续 window 可以乱序完成，但不能冒充 first-window-ready admission。
    progress = controller.mark_ready("w2", now_monotonic_ns=150)
    assert not progress.first_window_ready
    assert not progress.all_terminal

    progress = controller.mark_ready("w0", now_monotonic_ns=200)
    assert progress.first_window_became_ready
    assert progress.first_window_ready
    assert progress.first_window_ready_monotonic_ns == 200
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
