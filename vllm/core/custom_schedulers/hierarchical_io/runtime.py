# SPDX-License-Identifier: Apache-2.0

"""Worker-local prefetch plan runtime。

Scheduler 预留 block 并一次下发完整 plan；本模块只决定已经预授权的 unit
何时占用 MDS slot。它不能分配/释放 block，也不能发布 prefix hash。这样即使
激活发生在 model forward 内，事务所有权仍然完整留在 Scheduler。
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple

from vllm.core.custom_schedulers.async_kv_transfer import (
    AsyncKVTransferEvent, AsyncKVTransferRequest, AsyncKVTransferState)

from .plan import RollingPrefetchConfig


SubmitCallback = Callable[[AsyncKVTransferRequest, Any], AsyncKVTransferEvent]
PollCallback = Callable[[AsyncKVTransferRequest, Any], AsyncKVTransferEvent]
ProgressSignalFactory = Callable[[], Any]
AdvanceCallback = Callable[[str, int], None]


@dataclass(frozen=True)
class PrefetchRuntimeTrace:
    """一条 worker 物理时间线事件，不使用 scheduler publish 时间。"""

    phase: str
    plan_id: str
    request_id: str
    unit_index: int
    monotonic_ns: int
    layer_index: Optional[int] = None
    wait_ns: int = 0
    lead_units: int = 0


@dataclass
class _UnitState:
    request: AsyncKVTransferRequest
    mapping: Any
    active: bool = False
    terminal: Optional[AsyncKVTransferEvent] = None
    activated_monotonic_ns: Optional[int] = None
    ready_monotonic_ns: Optional[int] = None
    terminal_reported: bool = False


@dataclass(frozen=True)
class _ProgressSignalState:
    signal: Any
    source_unit: int
    target_unit: int
    armed_monotonic_ns: int


@dataclass(frozen=True)
class _ResidentActivationWork:
    """resident_event 后端的一条已授权激活任务。

    work item 只引用 Scheduler 已经 stage 的 plan/unit，不携带新的 block
    mapping，也不能创建 reservation。后台线程因此只能改变 I/O 的启动时机，
    不能越过 Scheduler 的事务所有权边界。
    """

    plan_key: Tuple[int, str]
    request_ids: frozenset[str]
    progress: _ProgressSignalState
    consumer_index: int
    submit: SubmitCallback
    poll: PollCallback
    max_active: int
    sleep_seconds: float


@dataclass
class _PlanState:
    plan_id: str
    units: Dict[int, _UnitState]
    lead_units: int
    # 每个 signal 都表示“GPU 已执行到某个 consumer 边界后，可以激活到哪个
    # unit”。signal 本身不携带 block/mapping，因此未来 sparse attention 只需
    # 产生不同的 target unit，不需要改 MDS 或 Scheduler 事务接口。
    progress_signals: list[_ProgressSignalState]
    last_signalled_unit: int = -1
    last_signalled_target_unit: int = -1
    admitted_unit: int = -1


class RollingPrefetchRuntime:
    """实现 stage -> activate -> poll -> wait_ready 的最小通用协议。

    后续 unit 首次由 Worker 激活时，会先向 Scheduler 回传一个 PENDING 事件，
    再回传 READY/ERROR。Scheduler 因而能把自己的 QUEUED 状态安全推进到
    PENDING，但不参与 forward 热路径中的激活时机。
    """

    def __init__(
        self,
        config: RollingPrefetchConfig,
        progress_signal_factory: Optional[ProgressSignalFactory] = None,
    ) -> None:
        self.config = config
        self._progress_signal_factory = progress_signal_factory
        self._plans: Dict[Tuple[int, str], _PlanState] = {}
        self._events: list[AsyncKVTransferEvent] = []
        self._traces: list[PrefetchRuntimeTrace] = []
        # resident_event 会与 model thread 并发修改 unit 状态。锁只保护很小的
        # worker-local 控制面；CUDA event 的阻塞等待发生在锁外，不会卡住
        # barrier 对当前 window 的 correctness 检查。
        self._state_lock = threading.RLock()
        self._resident_queue: queue.Queue[
            Optional[_ResidentActivationWork]] = queue.Queue()
        self._resident_thread: Optional[threading.Thread] = None
        self._resident_error: Optional[BaseException] = None

    def submit_or_stage(
        self,
        virtual_engine: int,
        request: AsyncKVTransferRequest,
        mapping: Any,
        submit: SubmitCallback,
    ) -> Tuple[AsyncKVTransferEvent, ...]:
        """登记 unit；首批 unit 立即 submit，其余只保存 descriptor template。"""
        with self._state_lock:
            self._raise_resident_error()
            plan_id, unit_index = self._identity(request)
            plan_key = (virtual_engine, plan_id)
            plan = self._plans.setdefault(
                plan_key,
                _PlanState(plan_id=plan_id,
                           units={},
                           lead_units=self.config.lead_windows,
                           progress_signals=[]))
            if unit_index in plan.units:
                raise RuntimeError(
                    f"duplicate prefetch unit: {plan_id}/{unit_index}")
            unit = _UnitState(request=request, mapping=mapping)
            plan.units[unit_index] = unit
            if not request.activate_on_submit:
                return ()

            event = self._activate(unit, submit, report_activation=False,
                                   layer_index=None,
                                   lead_units=plan.lead_units)
            # 初始 unit 已由 Scheduler 在发送 RPC 前标记为 PENDING，因此这里直接
            # 返回底层 submit 事件，沿用现有 Engine apply_event 路径。
            if event.state != AsyncKVTransferState.PENDING:
                unit.terminal_reported = True
            return (event, )

    def poll_units(
        self,
        virtual_engine: int,
        poll: PollCallback,
    ) -> Tuple[AsyncKVTransferEvent, ...]:
        """非阻塞推进当前 engine 的 active units，并回收待上报事件。"""
        with self._state_lock:
            self._raise_resident_error()
            self._progress(virtual_engine, poll)
            events = tuple(self._events)
            self._events.clear()
            for plan_key, plan in tuple(self._plans.items()):
                if plan_key[0] != virtual_engine:
                    continue
                request_ids = {event.request_id for event in events}
                for unit in plan.units.values():
                    if (unit.terminal is not None
                            and unit.request.request_id in request_ids):
                        unit.terminal_reported = True
                # 外部 poll 只发生在 forward 边界。全部 terminal 且已经上报后即可
                # 删除 worker-local 模板；Scheduler 仍独立完成 commit/abort。
                if (plan.units and all(unit.terminal is not None
                                       and unit.terminal_reported
                                       for unit in plan.units.values())):
                    del self._plans[plan_key]
            return events

    def wait_ready(
        self,
        virtual_engine: int,
        request_ids: Sequence[str],
        consumer_index: int,
        submit: SubmitCallback,
        poll: PollCallback,
        *,
        max_active: int,
        advance: Optional[AdvanceCallback] = None,
        sleep_seconds: float = 0.0001,
    ) -> None:
        """滚动激活未来 unit，并等待当前 consumer 对应 unit READY。

        ``consumer_index`` 当前是 layer index。激活只查看 Scheduler 已经 stage
        的模板；若 MDS slot 暂满，则先 poll 已激活 unit，槽位释放后再继续。
        """
        request_id_set = frozenset(request_ids)
        with self._state_lock:
            self._raise_resident_error()
            matching = tuple(
                (plan_key, plan)
                for plan_key, plan in self._plans.items()
                if plan_key[0] == virtual_engine and any(
                    unit.request.seq_group_id in request_id_set
                    and unit.request.layer_range is not None
                    and unit.request.layer_range[0] <= consumer_index <
                    unit.request.layer_range[1]
                    for unit in plan.units.values()))
        if not matching:
            return

        for plan_key, plan in matching:
            with self._state_lock:
                self._raise_resident_error()
                current = next(
                    unit for unit in plan.units.values()
                    if unit.request.seq_group_id in request_id_set
                    and unit.request.layer_range is not None
                    and unit.request.layer_range[0] <= consumer_index <
                    unit.request.layer_range[1])
                current_index = self._identity(current.request)[1]
                backend = self.config.activation_backend
                if backend == "gpu_native":
                    if advance is None:
                        raise RuntimeError(
                            "gpu_native prefetch requires an advance callback")
                    self._advance_gpu_native(
                        virtual_engine, plan, request_id_set, current_index,
                        consumer_index, submit, poll, advance, sleep_seconds)
                elif backend in ("gpu_visible", "resident_event"):
                    progress = self._record_progress_signal(
                        plan, current_index, consumer_index)
                    if progress is not None:
                        if backend == "gpu_visible":
                            plan.progress_signals.append(progress)
                        else:
                            self._enqueue_resident_activation(
                                _ResidentActivationWork(
                                    plan_key=plan_key,
                                    request_ids=request_id_set,
                                    progress=progress,
                                    consumer_index=consumer_index,
                                    submit=submit,
                                    poll=poll,
                                    max_active=max_active,
                                    sleep_seconds=sleep_seconds))
                    if backend == "gpu_visible":
                        self._consume_progress_signals(
                            virtual_engine, plan, request_id_set,
                            consumer_index, submit, poll, max_active,
                            sleep_seconds)
                else:
                    self._activate_until(
                        virtual_engine, plan, request_id_set,
                        current_index + plan.lead_units, consumer_index,
                        submit, poll, max_active, sleep_seconds)

            # 同一个 window 内的后续 layer 仍会经过 correctness barrier，但
            # READY 已经确认后不重复生成 barrier trace。resident_event 的 signal
            # 由后台线程独立消费，因此这里不再依赖后续 Python layer hook。
            with self._state_lock:
                if current_index <= plan.admitted_unit:
                    continue

            wait_started_ns = time.monotonic_ns()
            while True:
                with self._state_lock:
                    self._raise_resident_error()
                    if current.terminal is not None:
                        break
                    if self.config.activation_backend == "gpu_visible":
                        self._consume_progress_signals(
                            virtual_engine, plan, request_id_set,
                            consumer_index, submit, poll, max_active,
                            sleep_seconds)
                    if not current.active:
                        # GPU progress 只负责“提前多久激活”，不能成为 correctness
                        # 前提。consumer 已到当前 window 时必须直接兜底激活，
                        # 即使 resident signal 尚未消费也不会形成死锁。
                        self._activate_until(
                            virtual_engine, plan, request_id_set,
                            current_index, consumer_index, submit, poll,
                            (len(plan.units) if self.config.activation_backend
                             == "gpu_native" else max_active), sleep_seconds)
                        if self.config.activation_backend == "gpu_native":
                            assert advance is not None
                            advance(plan.plan_id, current_index)
                    if current.terminal is None:
                        self._progress(virtual_engine, poll)
                    ready = current.terminal is not None
                if not ready:
                    time.sleep(sleep_seconds)

            wait_ns = time.monotonic_ns() - wait_started_ns
            with self._state_lock:
                assert current.terminal is not None
                if current.terminal.state == AsyncKVTransferState.ERROR:
                    raise RuntimeError(
                        "prefetch unit failed before consumer "
                        f"{consumer_index}: {current.terminal.error}")
                plan.admitted_unit = max(plan.admitted_unit, current_index)

                # 最小反馈策略：当前 unit 仍产生超过目标的等待，说明固定 lead
                # 不够；只扩大本 plan 的未来窗口，不修改全局 scheduler 优先级。
                target_ns = int(self.config.target_slack_ms * 1.0e6)
                if (target_ns > 0 and wait_ns > target_ns
                        and plan.lead_units < self.config.max_lead_windows):
                    plan.lead_units += 1
                self._traces.append(
                    PrefetchRuntimeTrace(
                        phase="barrier_ready",
                        plan_id=plan.plan_id,
                        request_id=current.request.request_id,
                        unit_index=current_index,
                        monotonic_ns=time.monotonic_ns(),
                        layer_index=consumer_index,
                        wait_ns=wait_ns,
                        lead_units=plan.lead_units,
                    ))
                if self.config.activation_backend == "gpu_visible":
                    self._consume_progress_signals(
                        virtual_engine, plan, request_id_set,
                        consumer_index, submit, poll, max_active,
                        sleep_seconds)
                elif self.config.activation_backend == "host":
                    self._activate_until(
                        virtual_engine, plan, request_id_set,
                        current_index + plan.lead_units, consumer_index,
                        submit, poll, max_active, sleep_seconds)
                elif self.config.activation_backend == "gpu_native":
                    assert advance is not None
                    self._advance_gpu_native(
                        virtual_engine, plan, request_id_set, current_index,
                        consumer_index, submit, poll, advance, sleep_seconds)

    def _advance_gpu_native(
        self,
        virtual_engine: int,
        plan: _PlanState,
        request_ids: frozenset[str],
        current_index: int,
        consumer_index: int,
        submit: SubmitCallback,
        poll: PollCallback,
        advance: AdvanceCallback,
        sleep_seconds: float,
    ) -> None:
        """声明未来 unit 为 PENDING，并由 model stream 推进 daemon frontier。

        native plan 已经持有全部 descriptor，因此这里的 ``submit`` 只是为 unit
        建立可轮询 handle，不执行文件写入或占用 request slot。随后 advance
        在当前 CUDA stream 上排队一个 64-bit store，才是真正的 I/O 授权点。
        """
        target_index = min(max(plan.units), current_index + plan.lead_units)
        if target_index <= plan.last_signalled_target_unit:
            plan.last_signalled_unit = max(plan.last_signalled_unit,
                                           current_index)
            return
        self._activate_until(virtual_engine, plan, request_ids, target_index,
                             consumer_index, submit, poll, len(plan.units),
                             sleep_seconds)
        advance(plan.plan_id, target_index)
        now_ns = time.monotonic_ns()
        plan.last_signalled_unit = max(plan.last_signalled_unit, current_index)
        plan.last_signalled_target_unit = target_index
        target = plan.units[target_index]
        self._traces.append(
            PrefetchRuntimeTrace(
                phase="gpu_frontier_armed",
                plan_id=plan.plan_id,
                request_id=target.request.request_id,
                unit_index=target_index,
                monotonic_ns=now_ns,
                layer_index=consumer_index,
                lead_units=plan.lead_units,
            ))

    def discard_units(self, virtual_engine: int,
                      request_ids: Iterable[str]) -> None:
        """丢弃从未激活的模板；active DMA 仍必须自然到达终态。"""
        with self._state_lock:
            self._raise_resident_error()
            discarded = frozenset(request_ids)
            for plan_key, plan in tuple(self._plans.items()):
                if plan_key[0] != virtual_engine:
                    continue
                for unit_index, unit in tuple(plan.units.items()):
                    if unit.request.request_id not in discarded:
                        continue
                    if unit.active or unit.terminal is not None:
                        raise RuntimeError(
                            "cannot discard an active prefetch unit")
                    del plan.units[unit_index]
                if not plan.units:
                    del self._plans[plan_key]

    def pop_traces(self) -> Tuple[PrefetchRuntimeTrace, ...]:
        with self._state_lock:
            self._raise_resident_error()
            traces = tuple(self._traces)
            self._traces.clear()
            return traces

    def _record_progress_signal(
        self,
        plan: _PlanState,
        current_index: int,
        consumer_index: int,
    ) -> Optional[_ProgressSignalState]:
        """每个 unit 只记录一次真实 GPU progress signal。

        CUDA event 被记录到当前 model stream，只有此前已经入队的 layer kernel
        真正执行完后才会 ready。这里不做 ``synchronize`` 或 D2H ``item``，
        避免为了观察进度反而把计算流水线同步停住。
        """
        if current_index <= plan.last_signalled_unit:
            return None
        target_index = min(max(plan.units),
                           current_index + plan.lead_units)
        if target_index <= plan.last_signalled_target_unit:
            # 多个末尾 source 可能被 clamp 到同一个最后 unit。target frontier
            # 没有前进时不重复发 doorbell；source 仍记为已处理，避免同一
            # window 的后续 layer 再次进入判断。
            plan.last_signalled_unit = current_index
            return None
        if not any(index <= target_index and not unit.active
                   and unit.terminal is None
                   for index, unit in plan.units.items()):
            # 末尾 window 已经没有未来 unit 可激活时，不再创建 CUDA event。
            # 这既减少 event record，也避免 resident 线程在 forward 结束后处理
            # 一个没有任何数据面效果的 doorbell。
            plan.last_signalled_unit = current_index
            plan.last_signalled_target_unit = target_index
            return None
        factory = (self._progress_signal_factory
                   or self._new_cuda_progress_signal)
        signal = factory()
        armed_ns = time.monotonic_ns()
        progress = _ProgressSignalState(signal=signal,
                                        source_unit=current_index,
                                        target_unit=target_index,
                                        armed_monotonic_ns=armed_ns)
        plan.last_signalled_unit = current_index
        plan.last_signalled_target_unit = target_index
        current = plan.units[current_index]
        self._traces.append(
            PrefetchRuntimeTrace(
                phase="activation_signal_armed",
                plan_id=plan.plan_id,
                request_id=current.request.request_id,
                unit_index=current_index,
                monotonic_ns=armed_ns,
                layer_index=consumer_index,
                lead_units=plan.lead_units,
            ))
        return progress

    def _enqueue_resident_activation(self,
                                     work: _ResidentActivationWork) -> None:
        """把 GPU doorbell 交给唯一常驻线程，避免每个 window 新建线程。"""
        if self._resident_thread is None:
            self._resident_thread = threading.Thread(
                target=self._resident_activation_loop,
                name="bam-mds-prefetch-activation",
                daemon=True)
            self._resident_thread.start()
        self._resident_queue.put(work)

    def _resident_activation_loop(self) -> None:
        """等待真实 GPU progress，并立即激活已经 stage 的未来 unit。

        ``signal.synchronize()`` 只阻塞这个控制线程，不同步 model thread，且
        不做高频 CUDA event query。GPU 到达 event 后，线程复用原有
        ``_activate_until`` 和 MDS submit 路径；这一步仍是 host 发起 pybind/
        MDS 控制面提交，不等同于 GPU kernel 直接写 NVMe SQE。
        """
        while True:
            work = self._resident_queue.get()
            if work is None:
                self._resident_queue.task_done()
                return
            try:
                self._wait_progress_signal(work.progress.signal,
                                           work.sleep_seconds)
                consumed_ns = time.monotonic_ns()
                with self._state_lock:
                    plan = self._plans.get(work.plan_key)
                    if plan is None:
                        continue
                    source = plan.units.get(work.progress.source_unit)
                    if source is None:
                        continue
                    self._traces.append(
                        PrefetchRuntimeTrace(
                            phase="activation_signal_consumed",
                            plan_id=plan.plan_id,
                            request_id=source.request.request_id,
                            unit_index=work.progress.source_unit,
                            monotonic_ns=consumed_ns,
                            layer_index=work.consumer_index,
                            wait_ns=(consumed_ns -
                                     work.progress.armed_monotonic_ns),
                            lead_units=plan.lead_units,
                        ))
                    self._activate_until(
                        work.plan_key[0], plan, work.request_ids,
                        work.progress.target_unit, work.consumer_index,
                        work.submit, work.poll, work.max_active,
                        work.sleep_seconds)
            except BaseException as error:
                # 后台线程不能直接推进 Scheduler abort。保存首个错误，并在
                # 下一次 barrier/poll 时由 Worker 主线程抛出，沿用现有 ERROR
                # 事务清理路径。
                with self._state_lock:
                    if self._resident_error is None:
                        self._resident_error = error
            finally:
                self._resident_queue.task_done()

    @staticmethod
    def _wait_progress_signal(signal: Any, sleep_seconds: float) -> None:
        synchronize = getattr(signal, "synchronize", None)
        if synchronize is not None:
            synchronize()
            return
        # 纯 Python fake signal 只用于单元测试。真实 resident_event 必须提供
        # CUDA event.synchronize()，不会走这个低频兼容分支。
        while not signal.query():
            time.sleep(sleep_seconds)

    def _raise_resident_error(self) -> None:
        if self._resident_error is not None:
            raise RuntimeError(
                "resident prefetch activation failed") from self._resident_error

    def _consume_progress_signals(
        self,
        virtual_engine: int,
        plan: _PlanState,
        request_ids: frozenset[str],
        consumer_index: int,
        submit: SubmitCallback,
        poll: PollCallback,
        max_active: int,
        sleep_seconds: float,
    ) -> None:
        """按顺序消费 GPU signal，并复用现有 MDS unit 激活路径。"""
        while plan.progress_signals:
            progress = plan.progress_signals[0]
            if not progress.signal.query():
                return
            plan.progress_signals.pop(0)
            consumed_ns = time.monotonic_ns()
            source = plan.units[progress.source_unit]
            self._traces.append(
                PrefetchRuntimeTrace(
                    phase="activation_signal_consumed",
                    plan_id=plan.plan_id,
                    request_id=source.request.request_id,
                    unit_index=progress.source_unit,
                    monotonic_ns=consumed_ns,
                    layer_index=consumer_index,
                    wait_ns=(consumed_ns - progress.armed_monotonic_ns),
                    lead_units=plan.lead_units,
                ))
            self._activate_until(virtual_engine, plan, request_ids,
                                 progress.target_unit, consumer_index, submit,
                                 poll, max_active, sleep_seconds)

    @staticmethod
    def _new_cuda_progress_signal() -> Any:
        """创建轻量 CUDA event；仅 gpu_visible 实验模式会走到这里。"""
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "gpu_visible prefetch activation requires CUDA")
        event = torch.cuda.Event(enable_timing=False, blocking=False)
        event.record(torch.cuda.current_stream())
        return event

    def _activate_until(
        self,
        virtual_engine: int,
        plan: _PlanState,
        request_ids: frozenset[str],
        last_unit_index: int,
        consumer_index: int,
        submit: SubmitCallback,
        poll: PollCallback,
        max_active: int,
        sleep_seconds: float,
    ) -> None:
        due = tuple(
            unit for index, unit in sorted(plan.units.items())
            if index <= last_unit_index
            and unit.request.seq_group_id in request_ids
            and not unit.active and unit.terminal is None)
        for unit in due:
            while self._active_count(virtual_engine) >= max_active:
                self._progress(virtual_engine, poll)
                if self._active_count(virtual_engine) >= max_active:
                    time.sleep(sleep_seconds)
            self._activate(unit, submit, report_activation=True,
                           layer_index=consumer_index,
                           lead_units=plan.lead_units)

    def _activate(
        self,
        unit: _UnitState,
        submit: SubmitCallback,
        *,
        report_activation: bool,
        layer_index: Optional[int],
        lead_units: int,
    ) -> AsyncKVTransferEvent:
        unit.active = True
        unit.activated_monotonic_ns = time.monotonic_ns()
        if report_activation:
            self._events.append(
                AsyncKVTransferEvent(unit.request.request_id,
                                     AsyncKVTransferState.PENDING))
        event = submit(unit.request, unit.mapping)
        plan_id, unit_index = self._identity(unit.request)
        self._traces.append(
            PrefetchRuntimeTrace(
                phase="activated",
                plan_id=plan_id,
                request_id=unit.request.request_id,
                unit_index=unit_index,
                monotonic_ns=unit.activated_monotonic_ns,
                layer_index=layer_index,
                lead_units=lead_units,
            ))
        if event.state != AsyncKVTransferState.PENDING:
            self._finish(unit, event)
            if report_activation:
                self._events.append(event)
        return event

    def _progress(self, virtual_engine: int, poll: PollCallback) -> None:
        for (engine, _), plan in tuple(self._plans.items()):
            if engine != virtual_engine:
                continue
            for unit in plan.units.values():
                if not unit.active or unit.terminal is not None:
                    continue
                event = poll(unit.request, unit.mapping)
                if event.state == AsyncKVTransferState.PENDING:
                    continue
                self._finish(unit, event)
                self._events.append(event)

    def _finish(self, unit: _UnitState,
                event: AsyncKVTransferEvent) -> None:
        unit.active = False
        unit.terminal = event
        unit.ready_monotonic_ns = time.monotonic_ns()
        plan_id, unit_index = self._identity(unit.request)
        self._traces.append(
            PrefetchRuntimeTrace(
                phase=("physical_ready" if event.state ==
                       AsyncKVTransferState.READY else "physical_error"),
                plan_id=plan_id,
                request_id=unit.request.request_id,
                unit_index=unit_index,
                monotonic_ns=unit.ready_monotonic_ns,
            ))

    def _active_count(self, virtual_engine: int) -> int:
        return sum(
            1 for (engine, _), plan in self._plans.items()
            if engine == virtual_engine for unit in plan.units.values()
            if unit.active and unit.terminal is None)

    @staticmethod
    def _identity(request: AsyncKVTransferRequest) -> Tuple[str, int]:
        if (request.prefetch_plan_id is None
                or request.prefetch_unit_index is None):
            raise ValueError("prefetch request has no plan/unit identity")
        return request.prefetch_plan_id, request.prefetch_unit_index
