# SPDX-License-Identifier: Apache-2.0

"""把一次完整 KV restore 拆成连续 layer window。"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple


@dataclass(frozen=True)
class LayerWindow:
    """一个左闭右开的本地 layer 范围。"""

    index: int
    start_layer: int
    end_layer: int

    @property
    def num_layers(self) -> int:
        return self.end_layer - self.start_layer

    @property
    def layer_range(self) -> Tuple[int, int]:
        return self.start_layer, self.end_layer


@dataclass(frozen=True)
class LayerRestorePlan:
    """一次 prefix restore 的不可变分层计划。

    plan 只描述层范围，不持有 block、SequenceGroup 或 MDS handle。这样
    scheduler 和后续 model runner layer barrier 可以共享它，而不会把
    allocator 生命周期耦合进通用策略模块。
    """

    plan_id: str
    windows: Tuple[LayerWindow, ...]
    created_monotonic_ns: int

    @property
    def first_window(self) -> LayerWindow:
        return self.windows[0]


@dataclass(frozen=True)
class HierarchicalIOConfig:
    """默认关闭的层级 restore 实验配置。"""

    enabled: bool = False
    num_layers: int = 0
    window_layers: int = 0

    @classmethod
    def from_env(
        cls,
        environ: Optional[Mapping[str, str]] = None,
    ) -> "HierarchicalIOConfig":
        values = os.environ if environ is None else environ
        enabled = bool(
            int(values.get("VLLM_BAM_MDS_HIERARCHICAL_IO_ENABLE", "0")))
        if not enabled:
            return cls()

        # SchedulerConfig 不携带模型层数；这里显式使用 PP rank 的本地层数，
        # 避免从模型名称或全局层数猜测。单卡模型通常就是总 hidden layers。
        num_layers = int(
            values.get("VLLM_BAM_MDS_HIERARCHICAL_NUM_LAYERS", "0"))
        window_layers = int(
            values.get("VLLM_BAM_MDS_HIERARCHICAL_WINDOW_LAYERS", "0"))
        if num_layers <= 0:
            raise ValueError(
                "VLLM_BAM_MDS_HIERARCHICAL_NUM_LAYERS must be positive")
        if window_layers <= 0:
            raise ValueError(
                "VLLM_BAM_MDS_HIERARCHICAL_WINDOW_LAYERS must be positive")
        return cls(enabled=True,
                   num_layers=num_layers,
                   window_layers=window_layers)

    def build_plan(self, plan_id: str) -> LayerRestorePlan:
        if not self.enabled:
            raise RuntimeError("hierarchical I/O is disabled")
        return build_layer_restore_plan(plan_id=plan_id,
                                        num_layers=self.num_layers,
                                        window_layers=self.window_layers)


def build_layer_restore_plan(
    *,
    plan_id: str,
    num_layers: int,
    window_layers: int,
    created_monotonic_ns: Optional[int] = None,
) -> LayerRestorePlan:
    """按模型执行顺序生成连续、无重叠、无空洞的 layer windows。"""
    if not plan_id:
        raise ValueError("plan_id must not be empty")
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if window_layers <= 0:
        raise ValueError("window_layers must be positive")

    windows = tuple(
        LayerWindow(index=index,
                    start_layer=start,
                    end_layer=min(start + window_layers, num_layers))
        for index, start in enumerate(range(0, num_layers, window_layers)))
    return LayerRestorePlan(
        plan_id=plan_id,
        windows=windows,
        created_monotonic_ns=(time.monotonic_ns()
                              if created_monotonic_ns is None else
                              created_monotonic_ns),
    )
