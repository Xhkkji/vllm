# SPDX-License-Identifier: Apache-2.0

"""自定义 async KV 调度策略的轻量门面。

当前阶段不重写 vLLM 的 admission/batching，只先把后续会变化的 block 级
决策从 Scheduler 主类中抽出来。等 block-aware scheduler 真正实现时，
这里会继续承载 read/write 优先级、clean block 免写、GPU 驻留保留和
预取计划。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

from vllm.core.custom_schedulers.async_kv_state import AsyncKVBlockState


@dataclass(frozen=True)
class AsyncKVEvictionDecision:
    """释放 GPU block 前的最小策略结果。"""

    skip_write_blocks: Tuple[AsyncKVBlockState, ...] = ()
    write_back_blocks: Tuple[AsyncKVBlockState, ...] = ()


class AsyncKVBlockPolicy:
    """面向 block-aware scheduler 的初始策略容器。

    现在只提供最小的 clean/dirty 分类接口，不主动改变调度顺序。Scheduler
    可以先调用这些纯函数做日志和决策替换，后续再把更多 admission 逻辑
    迁入这里。
    """

    def classify_eviction(
        self,
        candidates: Iterable[AsyncKVBlockState],
    ) -> AsyncKVEvictionDecision:
        """把待释放 block 分成 clean 免写和 dirty 写回两类。"""
        skip_write = []
        write_back = []
        for block in candidates:
            if block.can_skip_write:
                skip_write.append(block)
            else:
                write_back.append(block)
        return AsyncKVEvictionDecision(
            skip_write_blocks=tuple(skip_write),
            write_back_blocks=tuple(write_back),
        )

