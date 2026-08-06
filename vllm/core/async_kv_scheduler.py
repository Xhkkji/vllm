# SPDX-License-Identifier: Apache-2.0

"""可通过参数选择的 V0 异步 KV 调度器外壳。

在 Engine/Worker 异步事件协议接通之前，本类刻意保持与原生 V0 Scheduler
完全兼容。选择本类不会暗中改变请求顺序、抢占策略或原有的计算调度；
它只提供异步 KV 调度所需的独立状态和接口。
"""

from __future__ import annotations

import time
from typing import Callable, Iterable, List, Optional, Sequence, Set, Tuple

import vllm.envs as envs
from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.core.interfaces import AllocStatus
from vllm.core.scheduler import (Scheduler, SchedulerRunningOutputs,
                                 SchedulerOutputs, SchedulerSwappedInOutputs,
                                 SchedulingBudget)
from vllm.core.scheduler_policy import (AsyncKVExecutionMarker,
                                        AsyncKVLoadEvent,
                                        AsyncKVLoadRequest,
                                        AsyncKVSchedulePolicy)
from vllm.logger import init_logger
from vllm.sequence import SequenceGroup, SequenceStatus

logger = init_logger(__name__)


class AsyncKVScheduler(Scheduler):
    """带有独立异步 KV 策略扩展的原生 V0 Scheduler。

    当前仍由父类的调度路径负责真正的 prefill、decode、抢占和 block
    管理。本类新增的方法只建立调度器侧的异步 KV 合同，后续由 Engine/
    Worker 桥接层调用。这样可以通过 ``--scheduler-cls`` 选择实验路径，
    同时保持默认的 ``vllm.core.scheduler.Scheduler`` 不变。
    """

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        lora_config: Optional[LoRAConfig],
        pipeline_parallel_size: int = 1,
        output_proc_callback: Optional[Callable] = None,
    ) -> None:
        if not scheduler_config.chunked_prefill_enabled:
            raise ValueError(
                "AsyncKVScheduler requires chunked prefill so that "
                "asynchronous KV completion is consumed at bounded "
                "model-execution intervals")
        super().__init__(scheduler_config, cache_config, lora_config,
                         pipeline_parallel_size, output_proc_callback)
        # loading 只记录已经登记异步恢复、但尚未被调度器提升的请求。
        # 当前阶段不把它们放入原生 running/swapped 队列，避免在 I/O
        # 完成事件尚未接通前改变原生调度行为。
        self.async_kv_policy = AsyncKVSchedulePolicy()
        self.loading: dict[str, SequenceGroup] = {}
        self._async_kv_loads_to_submit: List[AsyncKVLoadRequest] = []
        # READY 请求先进入 running，真正被 Engine dispatch 前保留一个观测
        # 标记。这样可以区分“状态已经可运行”和“已经进入模型执行 batch”。
        self._async_kv_execution_markers: dict[
            str, AsyncKVExecutionMarker] = {}
        # abort 不能立刻释放 loading 请求占用的 GPU block：MDS 可能仍在
        # 向该地址执行 DMA。这里先记录取消意图，等 READY/ERROR 事件到达
        # 后再释放 block，避免异步写入已经被其他请求复用的地址。
        self._cancelled_async_kv_requests: Set[str] = set()

    @property
    def scheduler_strategy(self) -> str:
        """返回稳定的策略名称，供日志和实验结果标记使用。"""
        return "async_kv"

    def submit_async_kv_load(
        self,
        seq_group: SequenceGroup,
        block_mapping: Sequence[Tuple[int, int]],
    ) -> AsyncKVLoadRequest:
        """登记异步恢复请求，并保存 request 到 sequence group 的归属关系。

        本方法当前不会直接提交设备 I/O。返回的描述对象将作为后续
        Engine/Worker 桥接层的输入，再由 Worker 转发给 MDS connector。
        保存归属关系是为了完成事件返回后，调度器能够找到对应的请求并
        在后续轮次执行 ready 提升。
        """
        request = self.async_kv_policy.submit(seq_group.request_id,
                                              block_mapping)
        self.loading[request.request_id] = seq_group
        return request

    def has_pending_async_kv_loads(self) -> bool:
        """判断是否仍有请求需要 Worker 继续 poll。

        loading 表在请求收到 READY 事件并完成 running 提升之前一直保留
        记录，因此 Engine 即使在某一轮没有可执行 token，也不会误认为
        当前 virtual engine 已经空闲。
        """
        return bool(self.loading)

    def has_unfinished_seqs(self) -> bool:
        """把 loading 请求计入 Engine 的未完成请求判断。"""
        return bool(self.loading) or super().has_unfinished_seqs()

    def get_num_unfinished_seq_groups(self) -> int:
        """返回 waiting/running/swapped/loading 的请求总数。"""
        return (len(self.loading) + super().get_num_unfinished_seq_groups())

    def apply_async_kv_event(self, event: AsyncKVLoadEvent) -> None:
        """应用 Worker 返回的后端无关异步恢复事件。

        这里只更新逻辑状态，不立即修改 running 队列。队列迁移统一在
        ``promote_ready_async_kv_loads`` 中进行，确保一轮调度开始时所有
        READY 请求以同一批次被提升。
        """
        self.async_kv_policy.apply_event(event)

    def promote_ready_async_kv_loads(self) -> None:
        """把 READY 请求提升到原生 running 队列。

        只有 MDS 返回 READY 后，才把 Sequence 从 SWAPPED 改为 RUNNING。
        block table 在提交阶段已经切换到目标 GPU block，因此此处只完成
        状态和队列迁移，不再重复调用 block_manager.swap_in。

        READY 请求按提交顺序进入 running 队列头部，使用逆序插入保持
        FIFO。下一次父类调度会正常为这些请求 append_slots 并执行 decode。
        """
        ready = self.async_kv_policy.pop_ready()
        ready_groups: List[SequenceGroup] = []
        for request in ready:
            seq_group = self.loading.pop(request.request_id, None)
            if seq_group is None:
                raise RuntimeError(
                    f"missing loading sequence group: {request.request_id}")
            if request.request_id in self._cancelled_async_kv_requests:
                self._cancelled_async_kv_requests.remove(request.request_id)
                # 请求在 I/O 完成前已经被 abort；现在才可以安全释放目标
                # GPU block，并将其从调度器的生命周期中彻底移除。
                self._free_finished_seq_group(seq_group)
                continue

            for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
                seq.status = SequenceStatus.RUNNING
            ready_groups.append(seq_group)
            self._async_kv_execution_markers[seq_group.request_id] = (
                AsyncKVExecutionMarker(
                    request_id=request.request_id,
                    seq_group_id=seq_group.request_id,
                    promoted_monotonic_ns=time.monotonic_ns()))
            if envs.VLLM_V0_SWAP_TRACE:
                marker = self._async_kv_execution_markers[
                    seq_group.request_id]
                logger.info(
                    "[V0_SWAP_TRACE][AsyncKV][Scheduler] phase=promote "
                    "request_id=%s seq_group_id=%s "
                    "promoted_monotonic_ns=%d",
                    marker.request_id,
                    marker.seq_group_id,
                    marker.promoted_monotonic_ns,
                )

        for seq_group in reversed(ready_groups):
            self.running.appendleft(seq_group)

        # MDS poll 失败时，异步请求已经不能再被 attention 使用。将它们
        # 标记为 ignored，并释放已经预留的 GPU block，避免 block 泄漏。
        for load in self.async_kv_policy.pop_errors():
            seq_group = self.loading.pop(load.request.request_id, None)
            if seq_group is None:
                raise RuntimeError(
                    f"missing failed loading sequence group: "
                    f"{load.request.request_id}")
            for seq in seq_group.get_seqs():
                if not seq.is_finished():
                    seq.status = SequenceStatus.FINISHED_IGNORED
            self._free_finished_seq_group(seq_group)

    def consume_async_kv_execution_markers(
            self, seq_group_ids: Iterable[str]
    ) -> Tuple[AsyncKVExecutionMarker, ...]:
        """取出本轮即将 dispatch 的异步恢复请求观测标记。

        Scheduler 仍然是 sequence 状态的唯一修改者；Engine 这里只消费已经
        READY/RUNNING 的只读观测信息。若 READY 请求暂时没有被当前 budget
        选中，标记会保留到它第一次真正进入执行 batch 的轮次。
        """
        markers = []
        for seq_group_id in seq_group_ids:
            marker = self._async_kv_execution_markers.pop(seq_group_id, None)
            if marker is not None:
                markers.append(marker)
        return tuple(markers)

    def abort_seq_group(
        self,
        request_id: str | Iterable[str],
        seq_id_to_seq_group=None,
    ) -> None:
        """处理 loading 请求的 abort，并延迟 block 释放到 I/O 结束。

        原生 Scheduler 只扫描 waiting/running/swapped 三个队列；loading
        请求不在这些队列中，因此需要在调用父类逻辑后单独检查。这里不
        尝试伪造 MDS cancel 协议，而是让 resident MDS 完成当前单槽 read，
        再由 ``promote_ready_async_kv_loads`` 执行最终清理。
        """
        if isinstance(request_id, str):
            request_ids = {request_id}
        else:
            request_ids = set(request_id)
        seq_id_to_seq_group = seq_id_to_seq_group or {}

        super().abort_seq_group(request_ids, seq_id_to_seq_group)
        # READY 但尚未进入执行 batch 的请求可能在这里被取消。删除纯观测
        # marker，避免一次永远不会发生的 first_execute 长期占用记录。
        for seq_group_id in tuple(self._async_kv_execution_markers):
            real_request_id = seq_group_id
            if seq_group_id in seq_id_to_seq_group:
                real_request_id = seq_id_to_seq_group[seq_group_id].group_id
            if real_request_id in request_ids:
                del self._async_kv_execution_markers[seq_group_id]

        for async_request_id, seq_group in self.loading.items():
            real_request_id = seq_group.request_id
            if seq_group.request_id in seq_id_to_seq_group:
                real_request_id = seq_id_to_seq_group[
                    seq_group.request_id].group_id
            if real_request_id not in request_ids:
                continue

            self._cancelled_async_kv_requests.add(async_request_id)
            for seq in seq_group.get_seqs():
                if not seq.is_finished():
                    seq.status = SequenceStatus.FINISHED_ABORTED

    def drain_async_kv_loads_to_submit(
            self) -> Tuple[AsyncKVLoadRequest, ...]:
        """取出本轮新生成的 submit 请求，并清空一次性发送缓冲。"""
        requests = tuple(self._async_kv_loads_to_submit)
        self._async_kv_loads_to_submit.clear()
        return requests

    def _schedule(self) -> SchedulerOutputs:
        """清空本轮 submit 缓冲后复用父类的普通调度入口。"""
        self._async_kv_loads_to_submit.clear()
        return super()._schedule()

    def _schedule_running(
        self,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
        partial_prefill_metadata=None,
    ) -> SchedulerRunningOutputs:
        """在 loading 期间禁止新的 swap-out。

        当前 block manager 的 swap_in 会立即释放源 CPU block，而 MDS
        读取可能还没有完成。若此时允许另一个请求 swap-out，就可能复用
        正在被 SSD->GPU 读取的 storage block。第一阶段使用 recompute
        作为保守保护：其他请求仍可继续 decode，但不会产生新的
        GPU->CPU/SSD swap-out。后续事务式 block reservation 完成后再解除
        这个限制。
        """
        if not self.loading:
            return super()._schedule_running(
                budget, curr_loras, enable_chunking,
                partial_prefill_metadata)

        original_mode = self.user_specified_preemption_mode
        self.user_specified_preemption_mode = "recompute"
        try:
            return super()._schedule_running(
                budget, curr_loras, enable_chunking,
                partial_prefill_metadata)
        finally:
            self.user_specified_preemption_mode = original_mode

    def _schedule_swapped(
        self,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
    ) -> SchedulerSwappedInOutputs:
        """提交一个异步 swap-in，但不把它加入当前计算 batch。

        原生实现会在 ``block_manager.swap_in`` 后立即把请求标记为
        RUNNING，并在当前 SchedulerOutputs 中执行。这里拆成：

        1. 检查 GPU block 是否足够；
        2. 移动 block table 到 GPU，并保留 mapping；
        3. 将逻辑请求登记为 loading；
        4. 等待 Worker/MDS 返回 READY；
        5. 下一轮才进入 running。

        第一阶段只允许一个 loading 请求，对应 BaM MDS 当前的单槽协议。
        """
        empty = SchedulerSwappedInOutputs.create_empty()
        if self.loading or not self.swapped:
            return empty

        seq_group = self.swapped[0]
        is_prefill = seq_group.is_prefill()
        alloc_status = self.block_manager.can_swap_in(
            seq_group,
            self._get_num_lookahead_slots(is_prefill, enable_chunking),
        )
        if alloc_status == AllocStatus.LATER:
            return empty
        if alloc_status == AllocStatus.NEVER:
            logger.warning(
                "Failing the request %s because there is not enough KV "
                "cache space for async swap-in.",
                seq_group.request_id,
            )
            self.swapped.popleft()
            for seq in seq_group.get_seqs():
                seq.status = SequenceStatus.FINISHED_IGNORED
            empty.infeasible_seq_groups.append(seq_group)
            return empty

        # 异步恢复本身不消耗当前 forward 的 token budget；真正进入 running
        # 后，父类会在下一轮按普通 decode/prefill 规则更新 budget。
        self.swapped.popleft()
        mapping = self.block_manager.swap_in(seq_group)
        if not mapping:
            # 没有物理 block 时无需经过 MDS，直接恢复为可调度请求。
            for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
                seq.status = SequenceStatus.RUNNING
            self.running.appendleft(seq_group)
            return empty

        request = self.submit_async_kv_load(seq_group, mapping)
        self._async_kv_loads_to_submit.append(request)
        logger.debug(
            "[ASYNC_KV_SCHEDULER] submitted request_id=%s seq_group=%s "
            "blocks=%d",
            request.request_id,
            seq_group.request_id,
            len(mapping),
        )
        return empty
