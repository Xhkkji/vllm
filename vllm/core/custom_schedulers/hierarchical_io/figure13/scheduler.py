# SPDX-License-Identifier: Apache-2.0

"""Tutti Figure 13 隔离实验专用 scheduler。"""

from __future__ import annotations

from typing import Callable, Optional

from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.core.async_kv_scheduler import (ASYNC_KV_STRATEGY_NATIVE,
                                          AsyncKVScheduler)


class TuttiFigure13Scheduler(AsyncKVScheduler):
    """为单请求 prefix/suffix sweep 提供显式且封闭的调度入口。

    所有实际 admission、prefix reservation 和 MDS window transfer 都继续由
    ``AsyncKVScheduler`` 负责。本类只固定 Figure 13 的实验不变量，不复制
    父类调度实现：

    * FCFS，排除 priority starvation；
    * max_num_seqs=1，排除 continuous batching 队列时间；
    * native async strategy，排除主动 preemption；
    * hierarchical I/O 必须显式开启。

    因此选择这个 scheduler 不会改变普通 async-prefix/swap baseline；反过来，
    通用 scheduler 也不需要知道 Figure 13 sweep 的存在。
    """

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        lora_config: Optional[LoRAConfig],
        pipeline_parallel_size: int = 1,
        output_proc_callback: Optional[Callable] = None,
    ) -> None:
        if scheduler_config.policy != "fcfs":
            raise ValueError("TuttiFigure13Scheduler requires FCFS policy")
        if scheduler_config.max_num_seqs != 1:
            raise ValueError(
                "TuttiFigure13Scheduler requires max_num_seqs=1")
        super().__init__(scheduler_config, cache_config, lora_config,
                         pipeline_parallel_size, output_proc_callback)
        if not self.hierarchical_io_config.enabled:
            raise ValueError(
                "TuttiFigure13Scheduler requires hierarchical MDS I/O")
        if self.async_kv_scheduler_strategy != ASYNC_KV_STRATEGY_NATIVE:
            raise ValueError(
                "TuttiFigure13Scheduler requires native async strategy")

    @property
    def scheduler_strategy(self) -> str:
        return (
            "tutti_figure13:fcfs_single_request:"
            f"window_{self.hierarchical_io_config.window_layers}_layers")
