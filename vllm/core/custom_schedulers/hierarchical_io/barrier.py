# SPDX-License-Identifier: Apache-2.0

"""层级 MDS restore 的 model-forward 层屏障。

这个模块故意不依赖 Scheduler、Worker 或具体模型。它只在一次 model
forward 的动态作用域中保存一个很小的回调：模型进入第 N 层前调用
``wait_for_local_layer(N)``，回调负责确认属于当前请求的 N 所在 window
已经完成 SSD -> HBM DMA。

这样模型代码不知道 MDS request id，Worker 也不需要依赖 Qwen2 的实现。
默认没有激活 session，入口立即返回，因此普通 vLLM 路径不改变数据语义。
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Callable, Iterator, Mapping, Optional, Sequence


LayerWaitCallback = Callable[[int, Sequence[str], int], None]


@dataclass(frozen=True)
class HierarchicalLayerBarrierConfig:
    """Step 4 的显式开关，默认保持 Step 1/2 的 full-restore 行为。"""

    enabled: bool = False

    @classmethod
    def from_env(
        cls,
        environ: Optional[Mapping[str, str]] = None,
    ) -> "HierarchicalLayerBarrierConfig":
        values = environ if environ is not None else os.environ
        return cls(enabled=bool(
            int(values.get("VLLM_BAM_MDS_HIERARCHICAL_LAYER_BARRIER", "0"))))


@dataclass(frozen=True)
class _LayerBarrierSession:
    """一次 forward 对应的 worker-local barrier 上下文。"""

    callback: LayerWaitCallback
    virtual_engine: int
    request_ids: tuple[str, ...]


_ACTIVE_LAYER_BARRIER: ContextVar[Optional[_LayerBarrierSession]] = ContextVar(
    "bam_hierarchical_layer_barrier", default=None)


@contextmanager
def activate_layer_barrier(
    callback: LayerWaitCallback,
    *,
    virtual_engine: int,
    request_ids: Sequence[str],
) -> Iterator[None]:
    """在 model forward 的动态范围内安装 worker-local 回调。

    request id 来自该 batch 的 ``ModelInput``。连续 batching 时一个 forward
    可能包含普通请求和层级 restore 请求；callback 必须仅处理这组 id 中自己
    管理的 restore window，其他请求自然是 no-op。
    """
    token = _ACTIVE_LAYER_BARRIER.set(
        _LayerBarrierSession(callback=callback,
                             virtual_engine=virtual_engine,
                             request_ids=tuple(request_ids)))
    try:
        yield
    finally:
        _ACTIVE_LAYER_BARRIER.reset(token)


def wait_for_local_layer(layer_index: int) -> None:
    """在 attention 使用该层 KV 前确认对应 window READY。

    没有活动 session 时直接返回。该情况包括开关关闭、decode batch 以及不属于
    MDS hierarchical restore 的普通请求，因此模型代码不需要区分这些路径。
    """
    session = _ACTIVE_LAYER_BARRIER.get()
    if session is None:
        return
    session.callback(session.virtual_engine, session.request_ids, layer_index)
