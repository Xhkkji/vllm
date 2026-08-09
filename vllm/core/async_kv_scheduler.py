# SPDX-License-Identifier: Apache-2.0

"""可通过参数选择的 V0 block 事务式异步 KV 调度器。"""

from __future__ import annotations

import time
import os
from typing import Callable, Iterable, List, Optional, Set, Tuple

import vllm.envs as envs
from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.core.block_reservation import BlockSwapReservation
from vllm.core.interfaces import AllocStatus
from vllm.core.scheduler import (Scheduler, SchedulerSwappedInOutputs,
                                 SchedulingBudget, PreemptionMode)
from vllm.core.custom_schedulers.async_kv_transfer import (
    AsyncKVExecutionMarker, AsyncKVSchedulePolicy, AsyncKVTransferEvent,
    AsyncKVTransferOperation, AsyncKVTransferRequest)
from vllm.logger import init_logger
from vllm.sequence import SequenceGroup, SequenceStatus

logger = init_logger(__name__)

ASYNC_KV_STRATEGY_NATIVE = "native"
ASYNC_KV_STRATEGY_CHUNKED_PRIORITY_PREEMPT = "chunked_priority_preempt"
ASYNC_KV_STRATEGIES = {
    ASYNC_KV_STRATEGY_NATIVE,
    ASYNC_KV_STRATEGY_CHUNKED_PRIORITY_PREEMPT,
}


class AsyncKVScheduler(Scheduler):
    """带有独立异步 KV 策略扩展的原生 V0 Scheduler。

    父类继续负责 prefill、decode、请求顺序和 victim 选择；本类只把同步
    swap 动作替换为 reserve -> MDS transfer -> commit/abort。实验路径通过
    ``--scheduler-cls`` 显式选择，默认原生 Scheduler 完全不受影响。
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
        # read 请求从 swapped 队列取出后暂存在 loading；write 请求仍保留
        # 在 swapped 队列，但在 SSD 副本提交前记录于 saving，禁止被重新
        # swap-in。两个方向共用同一个多槽状态机，容量必须与 MDS request
        # table 一致；小于 1 时直接失败，避免运行中出现隐式单槽回退。
        self.async_kv_policy = AsyncKVSchedulePolicy(
            max_in_flight=envs.VLLM_BAM_MDS_MAX_IN_FLIGHT)
        self.async_kv_scheduler_strategy = os.getenv(
            "VLLM_BAM_ASYNC_SCHEDULER_STRATEGY",
            ASYNC_KV_STRATEGY_NATIVE)
        if self.async_kv_scheduler_strategy not in ASYNC_KV_STRATEGIES:
            raise ValueError(
                "unsupported VLLM_BAM_ASYNC_SCHEDULER_STRATEGY="
                f"{self.async_kv_scheduler_strategy}; expected one of "
                f"{sorted(ASYNC_KV_STRATEGIES)}")
        self.loading: dict[str, SequenceGroup] = {}
        self.saving: dict[str, SequenceGroup] = {}
        self._chunked_priority_waiting_id: Optional[str] = None
        self._chunked_priority_preempted_victims: Set[str] = set()
        # READY 请求先进入 running，真正被 Engine dispatch 前保留一个观测
        # 标记。这样可以区分“状态已经可运行”和“已经进入模型执行 batch”。
        self._async_kv_execution_markers: dict[
            str, AsyncKVExecutionMarker] = {}
        # active I/O abort 不能立刻释放源或目标 block：MDS 仍可能对这些
        # 地址执行 DMA。先记录取消意图，等 READY/ERROR 后再统一回收。
        self._cancelled_async_kv_requests: Set[str] = set()

    @property
    def scheduler_strategy(self) -> str:
        """返回稳定的策略名称，供日志和实验结果标记使用。"""
        return f"async_kv:{self.async_kv_scheduler_strategy}"

    def _schedule_chunked_prefill(self):
        """在原生 chunked path 前注入可选的等待感知抢占策略。

        父类仍负责 continuous batching、running/decode 优先、chunked prefill
        token budget 和最终 SchedulerOutputs。本钩子只在 priority policy 下，
        当 waiting 高优先级请求因 KV block 不足无法进入 running 时，提前把
        一个低优先级 running victim 送入现有 async swap-out 路径。
        """
        if (self.async_kv_scheduler_strategy
                == ASYNC_KV_STRATEGY_CHUNKED_PRIORITY_PREEMPT):
            self._preempt_low_priority_running_for_waiting()
        return super()._schedule_chunked_prefill()

    def _preempt_low_priority_running_for_waiting(self) -> int:
        if self.scheduler_config.policy != "priority":
            return 0
        if not self.waiting or not self.running:
            return 0

        # Keep waiting in the same priority order used by vLLM's native
        # priority policy: lower priority value wins, then earlier arrival.
        self.waiting = type(self.waiting)(
            sorted(self.waiting, key=self._get_priority))
        waiting_head = self.waiting[0]
        if waiting_head.request_id != self._chunked_priority_waiting_id:
            self._chunked_priority_waiting_id = waiting_head.request_id
            self._chunked_priority_preempted_victims.clear()

        candidates = [
            seq_group for seq_group in self.running
            if seq_group.request_id
            not in self._chunked_priority_preempted_victims
        ]
        if not candidates:
            return 0
        victim = max(candidates, key=self._get_priority)
        if self._get_priority(victim) <= self._get_priority(waiting_head):
            return 0

        # If the high-priority waiting request can already be admitted, let
        # the native chunked scheduler handle it without extra churn.
        if self.block_manager.can_allocate(waiting_head) == AllocStatus.OK:
            return 0

        self.running.remove(victim)
        blocks_to_swap_out: List[Tuple[int, int]] = []
        preempted_mode = self._preempt(victim, blocks_to_swap_out)
        self._chunked_priority_preempted_victims.add(victim.request_id)
        if preempted_mode == PreemptionMode.SWAP:
            self.swapped.append(victim)
        else:
            self.waiting.appendleft(victim)
        if envs.VLLM_V0_SWAP_TRACE:
            logger.info(
                "[V0_SWAP_TRACE][AsyncKV][Scheduler] "
                "phase=chunked_priority_preempt victim_seq_group_id=%s "
                "waiting_seq_group_id=%s victim_priority=%s "
                "waiting_priority=%s mode=%s",
                victim.request_id,
                waiting_head.request_id,
                self._get_priority(victim),
                self._get_priority(waiting_head),
                preempted_mode.name,
            )
        return 1

    def _enqueue_async_kv_transfer(
        self,
        seq_group: SequenceGroup,
        reservation: BlockSwapReservation,
        operation: AsyncKVTransferOperation,
    ) -> AsyncKVTransferRequest:
        """把 block reservation 登记为等待 MDS transfer slot。"""
        request = self.async_kv_policy.enqueue(
            seq_group.request_id,
            reservation.reservation_id,
            operation,
            reservation.block_mapping,
            reservation.logical_blocks,
        )
        lifecycle = (self.loading if operation == AsyncKVTransferOperation.READ
                     else self.saving)
        lifecycle[request.request_id] = seq_group
        return request

    def has_active_async_kv_transfer(self) -> bool:
        """只要至少一笔 transfer 已提交，Engine 就需要轮询 Worker。"""
        return self.async_kv_policy.in_flight_count > 0

    def has_unfinished_seqs(self) -> bool:
        """把 reservation 中的请求计入 Engine 未完成判断。"""
        return (bool(self.loading) or bool(self.saving)
                or super().has_unfinished_seqs())

    def get_num_unfinished_seq_groups(self) -> int:
        """返回原生三队列和独立 loading 请求的总数。"""
        # saving 请求通常已经在原生 swapped 队列中，只为不在原生三队列
        # 中的 loading 请求额外计数，避免统计翻倍。
        return len(self.loading) + super().get_num_unfinished_seq_groups()

    def apply_async_kv_event(self, event: AsyncKVTransferEvent) -> None:
        """应用 Worker 返回的后端无关异步 transfer 事件。

        这里只更新控制面状态；block table 的 commit/abort 统一在下一次
        Engine 调度边界完成。
        """
        self.async_kv_policy.apply_event(event)

    def complete_ready_async_kv_transfers(self) -> None:
        """在调度边界提交已完成 reservation，并处理失败或取消。"""
        ready = self.async_kv_policy.pop_ready()
        ready_groups: List[SequenceGroup] = []
        for request in ready:
            lifecycle = (self.loading
                         if request.operation == AsyncKVTransferOperation.READ
                         else self.saving)
            seq_group = lifecycle.pop(request.request_id, None)
            if seq_group is None:
                raise RuntimeError(
                    f"missing async KV sequence group: {request.request_id}")
            if request.request_id in self._cancelled_async_kv_requests:
                self._cancelled_async_kv_requests.remove(request.request_id)
                # DMA 已经完成，此时 abort reservation 可以安全释放目标
                # block；随后释放仍由正式 block table 持有的源 block。
                self.block_manager.abort_block_swap(request.reservation_id)
                self._remove_from_swapped(seq_group)
                self._free_finished_seq_group(seq_group)
                continue

            self.block_manager.commit_block_swap(request.reservation_id)
            committed_ns = time.monotonic_ns()
            if envs.VLLM_V0_SWAP_TRACE:
                logger.info(
                    "[V0_SWAP_TRACE][AsyncKV][Scheduler] phase=commit "
                    "operation=%s request_id=%s seq_group_id=%s "
                    "commit_monotonic_ns=%d",
                    request.operation.value,
                    request.request_id,
                    seq_group.request_id,
                    committed_ns,
                )
            if request.operation == AsyncKVTransferOperation.READ:
                for seq in seq_group.get_seqs(
                        status=SequenceStatus.SWAPPED):
                    seq.status = SequenceStatus.RUNNING
                ready_groups.append(seq_group)
                self._async_kv_execution_markers[seq_group.request_id] = (
                    AsyncKVExecutionMarker(
                        request_id=request.request_id,
                        seq_group_id=seq_group.request_id,
                        promoted_monotonic_ns=committed_ns))
                if envs.VLLM_V0_SWAP_TRACE:
                    marker = self._async_kv_execution_markers[
                        seq_group.request_id]
                    logger.info(
                        "[V0_SWAP_TRACE][AsyncKV][Scheduler] phase=promote "
                        "operation=read request_id=%s seq_group_id=%s "
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
            request = load.request
            lifecycle = (self.loading
                         if request.operation == AsyncKVTransferOperation.READ
                         else self.saving)
            seq_group = lifecycle.pop(request.request_id, None)
            if seq_group is None:
                raise RuntimeError(
                    f"missing failed async KV sequence group: "
                    f"{request.request_id}")
            self.block_manager.abort_block_swap(request.reservation_id)
            self._remove_from_swapped(seq_group)
            for seq in seq_group.get_seqs():
                if not seq.is_finished():
                    seq.status = SequenceStatus.FINISHED_IGNORED
            self._free_finished_seq_group(seq_group)

    def _remove_from_swapped(self, seq_group: SequenceGroup) -> None:
        """按对象身份移除可能仍在原生 swapped 队列中的 write 请求。"""
        try:
            self.swapped.remove(seq_group)
        except ValueError:
            pass

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
        """处理 transfer 中请求的 abort，并延迟 block 释放到 I/O 结束。

        原生 Scheduler 只扫描 waiting/running/swapped 三个队列；loading
        请求不在这些队列中，因此需要在调用父类逻辑后单独检查。这里不
        尝试伪造 MDS cancel 协议，而是让 resident MDS 完成对应 slot 的 I/O，
        再由 ``complete_ready_async_kv_transfers`` 执行最终清理。
        """
        if isinstance(request_id, str):
            request_ids = {request_id}
        else:
            request_ids = set(request_id)
        seq_id_to_seq_group = seq_id_to_seq_group or {}

        # saving 请求虽然状态为 SWAPPED，但正式 block table 仍指向 MDS
        # 正在读取的 GPU source。先从原生 swapped 队列摘出目标请求，避免
        # 父类 abort 提前 free_seq，造成 DMA 写盘期间地址被复用。
        for seq_group in tuple(self.saving.values()):
            real_request_id = seq_group.request_id
            if seq_group.request_id in seq_id_to_seq_group:
                real_request_id = seq_id_to_seq_group[
                    seq_group.request_id].group_id
            if real_request_id in request_ids:
                self._remove_from_swapped(seq_group)

        super().abort_seq_group(request_ids, seq_id_to_seq_group)
        # READY 但尚未进入执行 batch 的请求可能在这里被取消。删除纯观测
        # marker，避免一次永远不会发生的 first_execute 长期占用记录。
        for seq_group_id in tuple(self._async_kv_execution_markers):
            real_request_id = seq_group_id
            if seq_group_id in seq_id_to_seq_group:
                real_request_id = seq_id_to_seq_group[seq_group_id].group_id
            if real_request_id in request_ids:
                del self._async_kv_execution_markers[seq_group_id]

        for async_request_id, seq_group in tuple(self.loading.items()) + tuple(
                self.saving.items()):
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

    def drain_async_kv_transfers_to_submit(
            self) -> Tuple[AsyncKVTransferRequest, ...]:
        """按后端容量批量激活 queued transfer。"""
        return self.async_kv_policy.activate_next()

    def _swap_out(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: List[Tuple[int, int]],
    ) -> None:
        """把原生同步 swap-out 改为 reservation + 后台 MDS write。

        ``blocks_to_swap_out`` 故意保持为空，防止 Worker.execute_worker
        再执行一次同步写。源 GPU block 在 write READY 前仍由 reservation
        持有，因此本轮调度不会错误地把相同地址分给其他 sequence。
        """
        if not self.block_manager.can_reserve_swap_out(seq_group):
            raise RuntimeError(
                "Aborted due to the lack of storage swap space. Please "
                "increase the swap space to avoid this error.")
        reservation = self.block_manager.reserve_swap_out(seq_group)
        request = self._enqueue_async_kv_transfer(
            seq_group, reservation, AsyncKVTransferOperation.WRITE)
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            seq.status = SequenceStatus.SWAPPED
        logger.debug(
            "[ASYNC_KV_SCHEDULER] queued operation=write request_id=%s "
            "seq_group=%s dirty_blocks=%d reused_clean_blocks=%d",
            request.request_id,
            seq_group.request_id,
            len(reservation.block_mapping),
            reservation.num_reused_blocks,
        )
        if envs.VLLM_V0_SWAP_TRACE and reservation.num_reused_blocks:
            logger.info(
                "[V0_SWAP_TRACE][AsyncKV][Scheduler] phase=avoid_write "
                "request_id=%s seq_group_id=%s clean_blocks=%d "
                "dirty_blocks=%d",
                request.request_id,
                seq_group.request_id,
                reservation.num_reused_blocks,
                len(reservation.block_mapping),
            )

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
        2. 预留目标 GPU block，正式 block table 仍指向 storage；
        3. 将逻辑请求登记为 loading；
        4. 等待 Worker/MDS 返回 READY；
        5. 下一轮才进入 running。

        read/write 可以提前排队，并由统一 policy 按 MDS slot 容量激活。
        """
        empty = SchedulerSwappedInOutputs.create_empty()
        # read reservation 会占用真实 GPU target，因此最多保留与后端 slot
        # 数相同的 loading 请求。write 继续留在 saving/swapped；从其后
        # 找到真正位于 storage 的 seq，避免对尚未写完的 GPU source 读回。
        if (len(self.loading) >= self.async_kv_policy.max_in_flight
                or not self.swapped):
            return empty

        saving_group_ids = {id(group) for group in self.saving.values()}
        seq_group = next((group for group in self.swapped
                          if id(group) not in saving_group_ids), None)
        if seq_group is None:
            return empty
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
            self.swapped.remove(seq_group)
            for seq in seq_group.get_seqs():
                seq.status = SequenceStatus.FINISHED_IGNORED
            empty.infeasible_seq_groups.append(seq_group)
            return empty

        # 异步恢复本身不消耗当前 forward 的 token budget；真正进入 running
        # 后，父类会在下一轮按普通 decode/prefill 规则更新 budget。
        self.swapped.remove(seq_group)
        reservation = self.block_manager.reserve_swap_in(seq_group)
        request = self._enqueue_async_kv_transfer(
            seq_group, reservation, AsyncKVTransferOperation.READ)
        logger.debug(
            "[ASYNC_KV_SCHEDULER] queued operation=read request_id=%s "
            "seq_group=%s blocks=%d",
            request.request_id,
            seq_group.request_id,
            len(reservation.block_mapping),
        )
        return empty
