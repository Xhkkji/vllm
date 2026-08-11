# SPDX-License-Identifier: Apache-2.0

"""层级 I/O plan 和父事务屏障的纯控制面测试。"""

import threading

from vllm.core.custom_schedulers.async_kv_transfer import (
    AsyncKVTransferEvent, AsyncKVTransferOperation, AsyncKVTransferPriority,
    AsyncKVTransferRequest, AsyncKVTransferState)
from vllm.core.custom_schedulers.hierarchical_io import (
    HierarchicalIOConfig, HierarchicalLayerBarrierConfig,
    HierarchicalRestoreController, RollingPrefetchConfig,
    RollingPrefetchRuntime, activate_layer_barrier, build_layer_restore_plan,
    wait_for_local_layer)


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


def test_rolling_config_is_explicit_and_validated():
    config = RollingPrefetchConfig.from_env({
        "VLLM_BAM_MDS_HIERARCHICAL_ROLLING_ENABLE": "1",
        "VLLM_BAM_MDS_HIERARCHICAL_LEAD_WINDOWS": "2",
        "VLLM_BAM_MDS_HIERARCHICAL_MAX_LEAD_WINDOWS": "3",
        "VLLM_BAM_MDS_HIERARCHICAL_TARGET_SLACK_MS": "0.5",
        "VLLM_BAM_MDS_HIERARCHICAL_ACTIVATION_BACKEND": "gpu_visible",
    })
    assert config.enabled
    assert config.initial_windows == 2
    assert config.max_lead_windows == 3
    assert config.activation_backend == "gpu_visible"

    resident = RollingPrefetchConfig.from_env({
        "VLLM_BAM_MDS_HIERARCHICAL_ROLLING_ENABLE": "1",
        "VLLM_BAM_MDS_HIERARCHICAL_ACTIVATION_BACKEND": "resident_event",
    })
    assert resident.activation_backend == "resident_event"

    native = RollingPrefetchConfig.from_env({
        "VLLM_BAM_MDS_HIERARCHICAL_ROLLING_ENABLE": "1",
        "VLLM_BAM_MDS_HIERARCHICAL_ACTIVATION_BACKEND": "gpu_native",
    })
    assert native.activation_backend == "gpu_native"


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
                              lead_windows=1,
                              max_lead_windows=1))
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


def test_gpu_visible_activation_consumes_progress_without_cuda_sync():
    """GPU signal 只决定未来 unit 的提前激活，当前 unit 始终可兜底启动。"""

    class FakeSignal:

        def __init__(self):
            self.ready = False

        def query(self):
            return self.ready

    signals = []

    def new_signal():
        signal = FakeSignal()
        signals.append(signal)
        return signal

    runtime = RollingPrefetchRuntime(
        RollingPrefetchConfig(enabled=True,
                              lead_windows=1,
                              max_lead_windows=1,
                              activation_backend="gpu_visible"),
        progress_signal_factory=new_signal)
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

    runtime.submit_or_stage(0, _prefetch_request(0, activate=True),
                            "mapping-0", submit)
    runtime.submit_or_stage(0, _prefetch_request(1, activate=False),
                            "mapping-1", submit)
    runtime.submit_or_stage(0, _prefetch_request(2, activate=False),
                            "mapping-2", submit)

    ready.add("unit-0")
    runtime.wait_ready(0, ("seq-0", ), 0, submit, poll, max_active=2)
    assert activated == [("unit-0", "mapping-0")]

    # 不做 event.synchronize；只有 query 观察到 GPU progress 后，未来 unit
    # 才进入原有 MDS submit 路径。
    signals[0].ready = True
    runtime.wait_ready(0, ("seq-0", ), 1, submit, poll, max_active=2)
    assert activated[-1] == ("unit-1", "mapping-1")

    ready.add("unit-1")
    runtime.wait_ready(0, ("seq-0", ), 2, submit, poll, max_active=2)
    signals[1].ready = True
    runtime.wait_ready(0, ("seq-0", ), 3, submit, poll, max_active=2)
    assert activated[-1] == ("unit-2", "mapping-2")

    ready.add("unit-2")
    runtime.wait_ready(0, ("seq-0", ), 4, submit, poll, max_active=2)
    runtime.wait_ready(0, ("seq-0", ), 5, submit, poll, max_active=2)
    assert len(signals) == 2

    phases = [trace.phase for trace in runtime.pop_traces()]
    assert phases.count("activation_signal_armed") == 2
    assert phases.count("activation_signal_consumed") == 2


def test_resident_event_activates_without_another_layer_hook():
    """GPU event ready 后，后台线程应直接激活 unit，不依赖下一次 wait_ready。"""

    class FakeSignal:

        def __init__(self):
            self.ready = threading.Event()

        def synchronize(self):
            assert self.ready.wait(timeout=1.0)

    signals = []

    def new_signal():
        signal = FakeSignal()
        signals.append(signal)
        return signal

    runtime = RollingPrefetchRuntime(
        RollingPrefetchConfig(enabled=True,
                              lead_windows=1,
                              max_lead_windows=1,
                              activation_backend="resident_event"),
        progress_signal_factory=new_signal)
    activated = []
    activated_future = threading.Event()
    ready = {"unit-0"}

    def submit(request, mapping):
        activated.append((request.request_id, mapping))
        if request.request_id == "unit-1":
            activated_future.set()
        return AsyncKVTransferEvent(request.request_id,
                                    AsyncKVTransferState.PENDING)

    def poll(request, _mapping):
        state = (AsyncKVTransferState.READY
                 if request.request_id in ready else
                 AsyncKVTransferState.PENDING)
        return AsyncKVTransferEvent(request.request_id, state)

    runtime.submit_or_stage(0, _prefetch_request(0, activate=True),
                            "mapping-0", submit)
    runtime.submit_or_stage(0, _prefetch_request(1, activate=False),
                            "mapping-1", submit)
    runtime.wait_ready(0, ("seq-0", ), 0, submit, poll, max_active=2)
    assert activated == [("unit-0", "mapping-0")]

    # 不再调用第二次 layer hook。resident thread 在 GPU signal ready 后直接
    # 复用已登记模板完成 unit-1 submit。
    signals[0].ready.set()
    assert activated_future.wait(timeout=1.0)
    assert activated[-1] == ("unit-1", "mapping-1")

    phases = [trace.phase for trace in runtime.pop_traces()]
    assert "activation_signal_armed" in phases
    assert "activation_signal_consumed" in phases


def test_gpu_native_advances_staged_plan_without_request_slot_limit():
    """native unit claim 是轻量 handle；真正授权由 GPU frontier 单次推进。"""
    runtime = RollingPrefetchRuntime(
        RollingPrefetchConfig(enabled=True,
                              lead_windows=2,
                              max_lead_windows=2,
                              activation_backend="gpu_native"))
    activated = []
    advanced = []
    ready = {"unit-0"}

    def submit(request, mapping):
        activated.append((request.request_id, mapping))
        return AsyncKVTransferEvent(request.request_id,
                                    AsyncKVTransferState.PENDING)

    def poll(request, _mapping):
        state = (AsyncKVTransferState.READY
                 if request.request_id in ready else
                 AsyncKVTransferState.PENDING)
        return AsyncKVTransferEvent(request.request_id, state)

    for index in range(4):
        runtime.submit_or_stage(
            0,
            _prefetch_request(index, activate=index < 2),
            f"mapping-{index}", submit)
    assert [item[0] for item in activated] == ["unit-0", "unit-1"]

    runtime.wait_ready(
        0, ("seq-0", ), 0, submit, poll,
        # native plan claim 不占 request slot，因此 max_active=1 也不应阻止
        # unit-2 被登记；daemon 自己按实际 slot 数做 backpressure。
        max_active=1,
        advance=lambda plan_id, frontier: advanced.append(
            (plan_id, frontier)))
    assert [item[0] for item in activated] == [
        "unit-0", "unit-1", "unit-2"
    ]
    assert advanced == [("plan-0", 2)]
    phases = [trace.phase for trace in runtime.pop_traces()]
    assert phases.count("gpu_frontier_armed") == 1


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
