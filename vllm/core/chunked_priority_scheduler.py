# SPDX-License-Identifier: Apache-2.0

"""Minimal chunked-prefill priority scheduler for baseline experiments.

This scheduler keeps vLLM's native chunked-prefill / continuous-batching path
and only adds one small admission hook: when a high-priority waiting request
cannot allocate KV blocks, preempt one lower-priority running request. Native
vLLM recompute preemption is used, so this scheduler is suitable for vLLM /
LMCache baselines and does not depend on GranuleKV.
"""

from __future__ import annotations

from typing import Callable, Optional, Set

import vllm.envs as envs
from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.core.interfaces import AllocStatus
from vllm.core.scheduler import Scheduler
from vllm.logger import init_logger

logger = init_logger(__name__)


class ChunkedPriorityScheduler(Scheduler):
    """Native vLLM scheduler with minimal priority preemption in chunked path."""

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
                "ChunkedPriorityScheduler requires chunked prefill")
        super().__init__(scheduler_config, cache_config, lora_config,
                         pipeline_parallel_size, output_proc_callback)
        self._chunked_priority_waiting_id: Optional[str] = None
        self._chunked_priority_preempted_victims: Set[str] = set()

    @property
    def scheduler_strategy(self) -> str:
        return "chunked_priority_preempt"

    def _schedule_chunked_prefill(self):
        self._preempt_low_priority_running_for_waiting()
        return super()._schedule_chunked_prefill()

    def _preempt_low_priority_running_for_waiting(self) -> int:
        if self.scheduler_config.policy != "priority":
            return 0
        if not self.waiting or not self.running:
            return 0

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
        if self.block_manager.can_allocate(waiting_head) == AllocStatus.OK:
            return 0
        if self.user_specified_preemption_mode == "swap":
            raise RuntimeError(
                "ChunkedPriorityScheduler supports recompute preemption only; "
                "use AsyncKVScheduler for async swap experiments")

        self.running.remove(victim)
        preempted_mode = self._preempt(victim, [])
        self._chunked_priority_preempted_victims.add(victim.request_id)
        self.waiting.appendleft(victim)
        if envs.VLLM_V0_SWAP_TRACE:
            logger.info(
                "[V0_SWAP_TRACE][ChunkedPriorityScheduler] "
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
