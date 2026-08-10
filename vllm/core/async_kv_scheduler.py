# SPDX-License-Identifier: Apache-2.0

"""可通过参数选择的 V0 block 事务式异步 KV 调度器。"""

from __future__ import annotations

import os
import time
from collections import deque
from typing import Callable, Iterable, List, Optional, Set, Tuple

import vllm.envs as envs
from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.core.block.interfaces import BlockAllocator
from vllm.core.block_reservation import (BlockPrefixRestoreReservation,
                                         BlockSwapReservation)
from vllm.core.interfaces import AllocStatus
from vllm.core.scheduler import (Scheduler, SchedulerSwappedInOutputs,
                                 SchedulingBudget, PreemptionMode)
from vllm.core.custom_schedulers.async_kv_transfer import (
    AsyncKVExecutionMarker, AsyncKVSchedulePolicy, AsyncKVTransferEvent,
    AsyncKVTransferOperation, AsyncKVTransferRequest)
from vllm.logger import init_logger
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus

logger = init_logger(__name__)

ASYNC_KV_STRATEGY_NATIVE = "native"
ASYNC_KV_STRATEGY_CHUNKED_PRIORITY_PREEMPT = "chunked_priority_preempt"
ASYNC_KV_STRATEGY_LONG_CONTEXT_STRESS = "long_context_stress"
ASYNC_KV_STRATEGIES = {
    ASYNC_KV_STRATEGY_NATIVE,
    ASYNC_KV_STRATEGY_CHUNKED_PRIORITY_PREEMPT,
    ASYNC_KV_STRATEGY_LONG_CONTEXT_STRESS,
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
        # prefix populate/restore 与请求 preemption 的 swap 生命周期分开保存。
        # 两者只复用底层 MDS transfer queue，READY 后的队列迁移语义不同，
        # 因此不能通过伪造 SWAPPED 状态混在 loading/saving 中。
        self.bam_mds_prefix_enabled = envs.VLLM_BAM_MDS_PREFIX_ENABLE
        if self.bam_mds_prefix_enabled and not cache_config.enable_prefix_caching:
            raise ValueError(
                "VLLM_BAM_MDS_PREFIX_ENABLE requires --enable-prefix-caching")
        self.prefix_loading: dict[str, SequenceGroup] = {}
        self.prefix_saving: dict[str, SequenceGroup] = {}
        self._prefix_restore_admission_blocked = False
        logger.info(
            "[BAM_MDS_PREFIX] phase=init enabled=%s block_size=%d",
            self.bam_mds_prefix_enabled,
            cache_config.block_size,
        )
        self._chunked_priority_waiting_id: Optional[str] = None
        self._chunked_priority_preempted_victims: Set[str] = set()
        self._long_context_stress_min_free_blocks = int(
            os.getenv("VLLM_BAM_LONG_CONTEXT_STRESS_MIN_FREE_BLOCKS", "128"))
        self._long_context_stress_min_victim_blocks = int(
            os.getenv("VLLM_BAM_LONG_CONTEXT_STRESS_MIN_VICTIM_BLOCKS", "64"))
        self._long_context_stress_max_preempts_per_waiting = max(
            1,
            int(
                os.getenv(
                    "VLLM_BAM_LONG_CONTEXT_STRESS_MAX_PREEMPTS_PER_WAITING",
                    "1")))
        self._long_context_stress_allow_prefill = bool(
            int(os.getenv("VLLM_BAM_LONG_CONTEXT_STRESS_ALLOW_PREFILL", "1")))
        self._long_context_stress_require_priority = bool(
            int(os.getenv("VLLM_BAM_LONG_CONTEXT_STRESS_REQUIRE_PRIORITY",
                          "1")))
        self._long_context_stress_proactive = bool(
            int(os.getenv("VLLM_BAM_LONG_CONTEXT_STRESS_PROACTIVE", "0")))
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
        self._sort_waiting_for_prefix_restore()
        self._maybe_start_bam_mds_prefix_restore()

        if (self.async_kv_scheduler_strategy
                == ASYNC_KV_STRATEGY_CHUNKED_PRIORITY_PREEMPT):
            self._preempt_low_priority_running_for_waiting()
        elif (self.async_kv_scheduler_strategy
              == ASYNC_KV_STRATEGY_LONG_CONTEXT_STRESS):
            self._preempt_long_context_victims_for_waiting()
        self._sort_waiting_for_prefix_restore()
        if self._prefix_restore_admission_blocked:
            # preempt 可能刚刚释放了 HBM；在进入父类 admission 前立即重试，
            # 避免本来可 restore 的请求在同一轮退化为完整 recompute。
            self._maybe_start_bam_mds_prefix_restore()
        return self._schedule_chunked_prefill_with_reserved_slots()

    def _sort_waiting_for_prefix_restore(self) -> None:
        """让 prefix admission 与 vLLM priority policy 使用同一顺序。"""
        if (self.bam_mds_prefix_enabled
                and self.scheduler_config.policy == "priority"):
            self.waiting = type(self.waiting)(
                sorted(self.waiting, key=self._get_priority))

    def _num_async_read_sequence_slots(self) -> int:
        """返回 read reservation 已占用、但父类 budget 看不到的 seq 数。"""
        return sum(
            group.get_max_num_running_seqs()
            for group in (*self.loading.values(),
                          *self.prefix_loading.values()))

    def _schedule_chunked_prefill_with_reserved_slots(self):
        """在不复制父类调度器的前提下，为异步 read 保留 sequence slot。

        loading 请求已经持有 GPU target，READY 后会回到 running。这里只从
        本轮可见 waiting 中扣除等量 admission 容量；已有 running 和仍可
        容纳的新请求继续交给原生 chunked scheduler，因此 prefix I/O 不会
        冻结整个 waiting 队列。
        """
        reserved_slots = self._num_async_read_sequence_slots()
        if reserved_slots == 0 and not self._prefix_restore_admission_blocked:
            return super()._schedule_chunked_prefill()

        running_slots = sum(group.get_max_num_running_seqs()
                            for group in self.running)
        available_slots = max(
            0,
            self.scheduler_config.max_num_seqs - running_slots -
            reserved_slots,
        )
        if self._prefix_restore_admission_blocked:
            # 当前队首确认存在 SSD prefix，但本轮 HBM 尚不足。让 running
            # 继续推进并释放空间；不能绕过它，也不能交给父类重新计算。
            available_slots = 0

        visible_waiting = deque()
        hidden_waiting = deque()
        for seq_group in self.waiting:
            required_slots = seq_group.get_max_num_running_seqs()
            if required_slots <= available_slots and not hidden_waiting:
                visible_waiting.append(seq_group)
                available_slots -= required_slots
            else:
                # 不越过第一个放不下的请求，保持 FCFS/priority 的公平性。
                hidden_waiting.append(seq_group)

        self.waiting = type(self.waiting)(visible_waiting)
        try:
            outputs = super()._schedule_chunked_prefill()
        finally:
            # 父类可能把 preempted 请求放回 waiting；它们应位于本轮未暴露
            # 的请求之前。下一轮 prefix 前置逻辑会按 priority 再统一排序。
            self.waiting.extend(hidden_waiting)
        return outputs

    def _maybe_start_bam_mds_prefix_restore(self) -> int:
        """在父类 admission 前尝试恢复一个 SSD prefix。

        每次最多预留当前可用的 sequence slot 和 MDS prefix loading slot。
        pending target 已由 allocator 隔离，所以 running/swapped/其他 write
        不再阻止 restore；READY 后才把请求放入 running。
        """
        if not self.bam_mds_prefix_enabled:
            return 0
        self._prefix_restore_admission_blocked = False

        max_seq_slots = self.scheduler_config.max_num_seqs - (
            sum(group.get_max_num_running_seqs() for group in self.running) +
            self._num_async_read_sequence_slots())
        max_prefix_loads = (self.async_kv_policy.max_in_flight -
                            len(self.loading) - len(self.prefix_loading))
        if max_seq_slots <= 0 or max_prefix_loads <= 0:
            return 0

        started = 0
        for seq_group in tuple(self.waiting):
            if started >= min(max_seq_slots, max_prefix_loads):
                break
            waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
            if len(waiting_seqs) != 1 or seq_group.is_encoder_decoder():
                break
            seq = waiting_seqs[0]
            storage_prefix_blocks, gpu_prefix_blocks = (
                self.block_manager.get_mds_cached_prefix_block_counts(seq))
            if storage_prefix_blocks <= gpu_prefix_blocks:
                # 当前最高优先级请求应先走原生 HBM-hit/recompute admission，
                # 不能让后面的 SSD restore 提前占用它需要的 sequence/HBM slot。
                break
            alloc_status = self.block_manager.can_reserve_mds_prefix_restore(
                storage_prefix_blocks, gpu_prefix_blocks)
            if alloc_status == AllocStatus.LATER:
                self._prefix_restore_admission_blocked = True
                break
            if alloc_status == AllocStatus.NEVER:
                break

            try:
                reservation = self.block_manager.reserve_mds_prefix_restore(
                    seq_group,
                    num_prefix_blocks=storage_prefix_blocks,
                    num_gpu_cached_blocks=gpu_prefix_blocks,
                )
            except BlockAllocator.NoFreeBlocksError:
                # allocator 的可用块数量包含可驱逐 cache；逐块预留仍可能在
                # 压力边界失败。将它视为本轮 LATER，但不能越过当前请求，也
                # 不在每个 decode step 输出 INFO 干扰性能。
                logger.debug(
                    "[BAM_MDS_PREFIX] phase=restore_deferred "
                    "seq_group_id=%s storage_prefix_blocks=%d "
                    "gpu_prefix_blocks=%d reason=no_free_blocks",
                    seq_group.request_id,
                    storage_prefix_blocks,
                    gpu_prefix_blocks,
                )
                self._prefix_restore_admission_blocked = True
                break
            self.waiting.remove(seq_group)
            request = self._enqueue_async_kv_transfer(
                seq_group,
                reservation,
                AsyncKVTransferOperation.READ,
                prefix=True,
            )
            logger.info(
                "[BAM_MDS_PREFIX] phase=restore_queued request_id=%s "
                "seq_group_id=%s storage_prefix_blocks=%d "
                "gpu_prefix_blocks=%d read_blocks=%d",
                request.request_id,
                seq_group.request_id,
                storage_prefix_blocks,
                gpu_prefix_blocks,
                len(reservation.block_mapping),
            )
            started += 1
        return started

    def _preempt_long_context_victims_for_waiting(self) -> int:
        """主动制造长上下文 HBM/SSD 迁移压力的实验策略。

        这个策略只挂在 AsyncKVScheduler 的显式 env 分支下。它不改变原生
        admission / batching 代码，而是在进入父类 chunked scheduler 前，
        当高优先级 waiting 请求与长 running 请求竞争 HBM 时，主动把少量
        低优先级长 running 请求送入现有 async swap-out 路径。
        """
        if not self.waiting or not self.running:
            return 0
        if self.saving:
            return 0
        if (self.async_kv_policy.in_flight_count >=
                self.async_kv_policy.max_in_flight):
            return 0

        if self.scheduler_config.policy == "priority":
            self.waiting = type(self.waiting)(
                sorted(self.waiting, key=self._get_priority))
        waiting_head = self.waiting[0]
        if waiting_head.request_id != self._chunked_priority_waiting_id:
            self._chunked_priority_waiting_id = waiting_head.request_id
            self._chunked_priority_preempted_victims.clear()

        waiting_alloc_status = self.block_manager.can_allocate(waiting_head)
        free_blocks = self.block_manager.get_num_free_gpu_blocks()
        if waiting_alloc_status == AllocStatus.OK:
            if not self._long_context_stress_proactive:
                return 0
            if free_blocks >= self._long_context_stress_min_free_blocks:
                return 0
            if len(self.running) >= self.scheduler_config.max_num_seqs:
                return 0
        elif waiting_alloc_status == AllocStatus.NEVER:
            return 0

        preempted = 0
        selected_victims: Set[str] = set()
        while preempted < self._long_context_stress_max_preempts_per_waiting:
            victim = self._select_long_context_stress_victim(
                waiting_head, selected_victims)
            if victim is None:
                break
            selected_victims.add(victim.request_id)
            self.running.remove(victim)
            blocks_to_swap_out: List[Tuple[int, int]] = []
            preempted_mode = self._preempt(victim, blocks_to_swap_out)
            if preempted_mode == PreemptionMode.SWAP:
                self.swapped.append(victim)
            else:
                self.waiting.appendleft(victim)
            preempted += 1
            if envs.VLLM_V0_SWAP_TRACE:
                logger.info(
                    "[V0_SWAP_TRACE][AsyncKV][Scheduler] "
                    "phase=long_context_stress_preempt "
                    "victim_seq_group_id=%s waiting_seq_group_id=%s "
                    "victim_priority=%s waiting_priority=%s "
                    "victim_blocks=%d free_gpu_blocks=%d "
                    "waiting_alloc_status=%s mode=%s",
                    victim.request_id,
                    waiting_head.request_id,
                    self._get_priority(victim),
                    self._get_priority(waiting_head),
                    self._estimate_seq_group_blocks(victim),
                    free_blocks,
                    waiting_alloc_status.name,
                    preempted_mode.name,
                )
        return preempted

    def _select_long_context_stress_victim(
        self,
        waiting_head: SequenceGroup,
        selected_victims: Set[str],
    ) -> Optional[SequenceGroup]:
        candidates = []
        for seq_group in self.running:
            if seq_group.request_id in selected_victims:
                continue
            if (not self._long_context_stress_allow_prefill
                    and seq_group.is_prefill()):
                continue
            victim_blocks = self._estimate_seq_group_blocks(seq_group)
            if victim_blocks < self._long_context_stress_min_victim_blocks:
                continue
            if (self.scheduler_config.policy == "priority"
                    and self._long_context_stress_require_priority
                    and self._get_priority(seq_group) <=
                    self._get_priority(waiting_head)):
                continue
            candidates.append(seq_group)
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda group:
            (self._get_priority(group), self._estimate_seq_group_blocks(group)),
        )

    def _estimate_seq_group_blocks(self, seq_group: SequenceGroup) -> int:
        return sum(seq.n_blocks for seq in seq_group.get_seqs())

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
        reservation: BlockSwapReservation | BlockPrefixRestoreReservation,
        operation: AsyncKVTransferOperation,
        prefix: bool = False,
    ) -> AsyncKVTransferRequest:
        """把 block reservation 登记为等待 MDS transfer slot。"""
        request = self.async_kv_policy.enqueue(
            seq_group.request_id,
            reservation.reservation_id,
            operation,
            reservation.block_mapping,
            reservation.logical_blocks,
        )
        if prefix:
            lifecycle = (self.prefix_loading
                         if operation == AsyncKVTransferOperation.READ else
                         self.prefix_saving)
        else:
            lifecycle = (self.loading
                         if operation == AsyncKVTransferOperation.READ else
                         self.saving)
        lifecycle[request.request_id] = seq_group
        return request

    def has_active_async_kv_transfer(self) -> bool:
        """只要至少一笔 transfer 已提交，Engine 就需要轮询 Worker。"""
        return self.async_kv_policy.in_flight_count > 0

    def has_unfinished_seqs(self) -> bool:
        """把 reservation 中的请求计入 Engine 未完成判断。"""
        return (bool(self.loading) or bool(self.saving)
                or bool(self.prefix_loading) or bool(self.prefix_saving)
                or super().has_unfinished_seqs())

    def get_num_unfinished_seq_groups(self) -> int:
        """返回原生三队列和独立 loading 请求的总数。"""
        # saving 请求通常已经在原生 swapped 队列中，只为不在原生三队列
        # 中的 loading 请求额外计数，避免统计翻倍。
        return (len(self.loading) + len(self.prefix_loading)
                + len(self.prefix_saving)
                + super().get_num_unfinished_seq_groups())

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
            is_prefix_restore = request.request_id in self.prefix_loading
            is_prefix_store = request.request_id in self.prefix_saving
            lifecycle = (self.loading
                         if request.operation == AsyncKVTransferOperation.READ
                         else self.saving)
            if is_prefix_restore:
                lifecycle = self.prefix_loading
            elif is_prefix_store:
                lifecycle = self.prefix_saving
            seq_group = lifecycle.pop(request.request_id, None)
            if seq_group is None:
                raise RuntimeError(
                    f"missing async KV sequence group: {request.request_id}")
            if request.request_id in self._cancelled_async_kv_requests:
                self._cancelled_async_kv_requests.remove(request.request_id)
                # DMA 已经完成，此时 abort reservation 可以安全释放目标
                # block；随后释放仍由正式 block table 持有的源 block。
                if is_prefix_restore:
                    self.block_manager.abort_mds_prefix_restore(
                        request.reservation_id)
                else:
                    self.block_manager.abort_block_swap(
                        request.reservation_id)
                self._remove_from_swapped(seq_group)
                super()._free_finished_seq_group(seq_group)
                continue

            if is_prefix_restore:
                self.block_manager.commit_mds_prefix_restore(
                    request.reservation_id)
            else:
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
            if is_prefix_restore:
                for seq in seq_group.get_seqs(
                        status=SequenceStatus.WAITING):
                    seq.status = SequenceStatus.RUNNING
                ready_groups.append(seq_group)
                logger.info(
                    "[BAM_MDS_PREFIX] phase=restore_ready request_id=%s "
                    "seq_group_id=%s",
                    request.request_id,
                    seq_group.request_id,
                )
            elif is_prefix_store:
                # prefix populate 已完成，正式 table 此时位于 storage；调用
                # storage 专用释放后，完整 CPU block 留在 prefix allocator
                # 的 LRU 中，后续相同 token hash 可以获取稳定的 MDS block id。
                self._free_completed_mds_prefix_store(seq_group)
                logger.info(
                    "[BAM_MDS_PREFIX] phase=store_ready request_id=%s "
                    "seq_group_id=%s",
                    request.request_id,
                    seq_group.request_id,
                )
            elif request.operation == AsyncKVTransferOperation.READ:
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
            # abort 只登记“完成后丢弃”，不会中止正在进行的 DMA。后端既可能
            # 回 READY，也可能回 ERROR；两种终态都必须消费取消标记，否则
            # 控制面会留下一个永远无法再次完成的 request id。
            self._cancelled_async_kv_requests.discard(request.request_id)
            is_prefix_restore = request.request_id in self.prefix_loading
            is_prefix_store = request.request_id in self.prefix_saving
            lifecycle = (self.loading
                         if request.operation == AsyncKVTransferOperation.READ
                         else self.saving)
            if is_prefix_restore:
                lifecycle = self.prefix_loading
            elif is_prefix_store:
                lifecycle = self.prefix_saving
            seq_group = lifecycle.pop(request.request_id, None)
            if seq_group is None:
                raise RuntimeError(
                    f"missing failed async KV sequence group: "
                    f"{request.request_id}")
            if is_prefix_restore:
                self.block_manager.abort_mds_prefix_restore(
                    request.reservation_id)
            else:
                self.block_manager.abort_block_swap(request.reservation_id)
            self._remove_from_swapped(seq_group)
            if is_prefix_store:
                # populate 是可选缓存写；失败不能把已经正常完成的用户请求
                # 改成 ignored，只丢弃这次缓存并正常释放。
                super()._free_finished_seq_group(seq_group)
            else:
                for seq in seq_group.get_seqs():
                    if not seq.is_finished():
                        seq.status = SequenceStatus.FINISHED_IGNORED
                super()._free_finished_seq_group(seq_group)

    def _free_completed_mds_prefix_store(
            self, seq_group: SequenceGroup) -> None:
        """完成 finished bookkeeping，并释放 storage-resident table。"""
        self._free_seq_group_cross_attn_blocks(seq_group)
        self._finished_requests_ids.append(seq_group.request_id)
        for seq in seq_group.get_seqs():
            if seq.is_finished():
                self.block_manager.free_mds_prefix_store(seq)

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

    def free_seq(self, seq: Sequence) -> None:
        """拦截 single-step output processor 的即时 finished free。

        V0 会在 stop checker 返回 finished 后立刻调用 ``free_seq``，早于
        ``free_finished_seq_groups``。BaM prefix 必须在这里保住 GPU table，
        否则稍后的 group hook 已经没有可写入 MDS 的 KV source。
        """
        if (self.bam_mds_prefix_enabled and seq.is_finished()
                and seq.seq_id in self.block_manager.block_tables):
            seq_group = next(
                (group for group in self.running if any(
                    candidate is seq for candidate in group.get_seqs())),
                None,
            )
            if (seq_group is not None
                    and self._try_start_mds_prefix_store(seq_group)):
                return
        super().free_seq(seq)

    def _free_finished_seq_group(self, seq_group: SequenceGroup) -> None:
        """完成 group 清理时避免重复提交已经开始的 prefix store。"""
        if any(group is seq_group for group in self.prefix_saving.values()):
            return
        if self._try_start_mds_prefix_store(seq_group):
            return
        super()._free_finished_seq_group(seq_group)

    def _try_start_mds_prefix_store(self,
                                    seq_group: SequenceGroup) -> bool:
        """正常释放前，用 MDS 异步保存可复用的完整 prefix block。

        这是 BaM prefix populate 唯一挂点，并且只存在于显式选择的
        ``AsyncKVScheduler``。原生 Scheduler 的 finished/free 行为没有改动。
        请求对客户端已经完成，但 block 资源会保留到 MDS DONE；Engine 通过
        ``prefix_saving`` 继续推进空调度轮次，不会提前复用 DMA source。
        """
        if not self.bam_mds_prefix_enabled or not seq_group.is_finished():
            return False
        seqs = seq_group.get_seqs()
        normally_finished = all(
            seq.status in (SequenceStatus.FINISHED_STOPPED,
                           SequenceStatus.FINISHED_LENGTH_CAPPED)
            for seq in seqs)
        has_full_block = any(seq.get_len() >= self.cache_config.block_size
                             for seq in seqs)
        has_block_table = all(seq.seq_id in self.block_manager.block_tables
                              for seq in seqs)
        if (not normally_finished or not has_full_block or not has_block_table
                or not self.block_manager.can_reserve_mds_prefix_store(
                    seq_group)):
            logger.debug(
                "[BAM_MDS_PREFIX] phase=store_skipped request_id=%s "
                "normally_finished=%s has_full_block=%s has_block_table=%s",
                seq_group.request_id,
                normally_finished,
                has_full_block,
                has_block_table,
            )
            return False

        try:
            reservation = self.block_manager.reserve_mds_prefix_store(seq_group)
        except Exception as exc:
            # Prefix 是可选的缓存优化；storage 紧张时保留正常完成语义，
            # 不把用户请求失败扩大成服务失败。
            logger.warning(
                "[BAM_MDS_PREFIX] skip store request_id=%s error=%s",
                seq_group.request_id,
                exc,
            )
            return False
        request = self._enqueue_async_kv_transfer(
            seq_group,
            reservation,
            AsyncKVTransferOperation.WRITE,
            prefix=True,
        )
        logger.info(
            "[BAM_MDS_PREFIX] phase=store_queued request_id=%s "
            "seq_group_id=%s write_blocks=%d reused_blocks=%d",
            request.request_id,
            seq_group.request_id,
            len(reservation.block_mapping),
            reservation.num_reused_blocks,
        )
        return True

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

        prefix_transfers = (tuple(self.prefix_loading.items())
                            + tuple(self.prefix_saving.items()))
        for async_request_id, seq_group in (tuple(self.loading.items()) + tuple(
                self.saving.items()) + prefix_transfers):
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
