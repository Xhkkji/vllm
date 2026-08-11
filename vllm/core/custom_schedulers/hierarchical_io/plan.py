# SPDX-License-Identifier: Apache-2.0

"""把一次完整 KV restore 表达成可滚动激活的 prefetch plan。"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple


@dataclass(frozen=True)
class LayerWindow:
    """一个可独立激活的 prefetch unit。

    当前 consumer 是模型 layer，因此仍保留 ``start_layer/end_layer`` 命名。
    sparse attention 后续也可以把一个 token/block tile 映射到相同的线性
    consumer 区间，而不需要改变 stage/activate/poll/wait 四段生命周期。
    """

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
    """一次 prefix restore 的不可变 prefetch plan。

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

    def window_for_layer(self, layer_index: int) -> LayerWindow:
        """返回消费 ``layer_index`` 的 unit，并尽早拒绝错误模型配置。"""
        for window in self.windows:
            if window.start_layer <= layer_index < window.end_layer:
                return window
        raise ValueError(f"layer is outside prefetch plan: {layer_index}")


# 通用接口直接复用 layer-window 数据结构，不再额外套一层只有转发作用的
# wrapper。未来其他策略只需提供相同的 index/consumer range 语义。
PrefetchUnit = LayerWindow
PrefetchPlan = LayerRestorePlan


@dataclass(frozen=True)
class RollingPrefetchConfig:
    """worker-local rolling activation 配置，默认关闭。

    ``lead_windows`` 表示模型消费当前 window 时，至少再激活多少个未来
    window。``target_slack_ms`` 为 0 时只使用固定 lead；大于 0 时，如果某次
    barrier 仍发生等待，runtime 会临时把 lead 增大 1，最多到
    ``max_lead_windows``。这是最小的反馈策略，不在 scheduler 中预测 I/O。
    """

    enabled: bool = False
    lead_windows: int = 1
    max_lead_windows: int = 1
    target_slack_ms: float = 0.0

    @classmethod
    def from_env(
        cls,
        environ: Optional[Mapping[str, str]] = None,
    ) -> "RollingPrefetchConfig":
        values = os.environ if environ is None else environ
        enabled = bool(
            int(values.get(
                "VLLM_BAM_MDS_HIERARCHICAL_ROLLING_ENABLE", "0")))
        if not enabled:
            return cls()
        lead_windows = int(values.get(
            "VLLM_BAM_MDS_HIERARCHICAL_LEAD_WINDOWS", "1"))
        max_lead_windows = int(values.get(
            "VLLM_BAM_MDS_HIERARCHICAL_MAX_LEAD_WINDOWS",
            str(lead_windows)))
        target_slack_ms = float(values.get(
            "VLLM_BAM_MDS_HIERARCHICAL_TARGET_SLACK_MS", "0"))
        if lead_windows < 0:
            raise ValueError(
                "VLLM_BAM_MDS_HIERARCHICAL_LEAD_WINDOWS must be non-negative")
        if max_lead_windows < lead_windows:
            raise ValueError(
                "VLLM_BAM_MDS_HIERARCHICAL_MAX_LEAD_WINDOWS must be >= "
                "VLLM_BAM_MDS_HIERARCHICAL_LEAD_WINDOWS")
        if target_slack_ms < 0:
            raise ValueError(
                "VLLM_BAM_MDS_HIERARCHICAL_TARGET_SLACK_MS must be non-negative")
        return cls(enabled=True,
                   lead_windows=lead_windows,
                   max_lead_windows=max_lead_windows,
                   target_slack_ms=target_slack_ms)

    @property
    def initial_windows(self) -> int:
        """首批激活窗口数；之后由模型进度滚动激活。"""
        return max(1, self.lead_windows)


@dataclass(frozen=True)
class HierarchicalIOConfig:
    """默认关闭的层级 restore 实验配置。"""

    enabled: bool = False
    num_layers: int = 0
    window_layers: int = 0
    rolling: RollingPrefetchConfig = RollingPrefetchConfig()

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
                   window_layers=window_layers,
                   rolling=RollingPrefetchConfig.from_env(values))

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
