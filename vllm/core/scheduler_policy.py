# SPDX-License-Identifier: Apache-2.0

"""调度器管理异步 KV 恢复时使用的后端无关数据结构和状态机。

这个模块有意不依赖任何具体 KV 后端。调度器只需要知道恢复请求是否已经
提交、是否完成以及是否失败；具体的异步 handle、poll 和 finish 操作由
Worker、CacheEngine 以及底层 connector 负责。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Optional, Sequence, Tuple


BlockMapping = Tuple[Tuple[int, int], ...]


class AsyncKVLoadState(Enum):
    """异步 KV 传输路径返回给调度器的状态。"""

    PENDING = auto()
    READY = auto()
    ERROR = auto()


@dataclass(frozen=True)
class AsyncKVLoadRequest:
    """调度器发送给 Worker 的异步 KV 恢复请求描述。

    ``block_mapping`` 描述逻辑 KV block 到目标 GPU block 的映射。这个
    描述只表达调度意图，不携带 MDS 或 LMCache 的具体实现对象，因此可以
    在 Engine 和 Worker 之间安全传递。
    """

    request_id: str
    seq_group_id: str
    block_mapping: BlockMapping


@dataclass(frozen=True)
class AsyncKVLoadEvent:
    """Worker 返回给调度器的异步 KV 恢复事件。

    Worker 可以先返回 ``PENDING``，在后续轮询中返回 ``READY`` 或
    ``ERROR``。调度器只有收到 ``READY`` 后，才应该把对应请求提升为可
    计算状态。
    """

    request_id: str
    state: AsyncKVLoadState
    error: Optional[str] = None


@dataclass(frozen=True)
class AsyncKVExecutionMarker:
    """标记一个异步恢复请求第一次进入模型执行 batch 的信息。

    ``promoted_monotonic_ns`` 由 Scheduler 在 READY 请求提升为 RUNNING 时
    记录；Engine 在真正 dispatch 当前 batch 前消费这个标记并补上
    ``first_execute_monotonic_ns``。该对象只用于观测，不参与调度决策。
    """

    request_id: str
    seq_group_id: str
    promoted_monotonic_ns: int


@dataclass
class PendingAsyncKVLoad:
    """调度器为一个逻辑 KV 恢复请求维护的状态记录。"""

    request: AsyncKVLoadRequest
    state: AsyncKVLoadState = AsyncKVLoadState.PENDING
    error: Optional[str] = None


class AsyncKVSchedulePolicy:
    """异步 KV 调度策略使用的轻量状态机。

    该策略不持有 waiting/running/swapped 队列，也不直接操作 block manager。
    它只维护逻辑恢复请求的生命周期，因此可以在没有 GPU 的环境中单独
    测试，后续也可以连接 MDS、LMCache 或其他异步 KV 后端。

    当前状态转换为：

    ``submit -> PENDING -> READY``
    ``submit -> PENDING -> ERROR``

    READY 和 ERROR 都是终态。上层调度器分别通过 ``pop_ready`` 和
    ``pop_errors`` 取走这些请求，并负责后续的队列迁移或 block 清理。
    """

    def __init__(self, max_in_flight: int = 1) -> None:
        if max_in_flight <= 0:
            raise ValueError("max_in_flight must be positive")
        self.max_in_flight = max_in_flight
        self._request_counter = itertools.count()
        self._loads: Dict[str, PendingAsyncKVLoad] = {}

    @property
    def loading_request_ids(self) -> Tuple[str, ...]:
        """返回已经提交、但尚未进入终态的请求 ID。"""
        return tuple(request_id for request_id, load in self._loads.items()
                     if load.state == AsyncKVLoadState.PENDING)

    @property
    def ready_request_ids(self) -> Tuple[str, ...]:
        """返回已经完成、等待调度器提升的请求 ID。"""
        return tuple(request_id for request_id, load in self._loads.items()
                     if load.state == AsyncKVLoadState.READY)

    @property
    def in_flight_count(self) -> int:
        """返回当前仍在进行中的异步恢复数量。"""
        return len(self.loading_request_ids)

    def submit(self, seq_group_id: str,
               block_mapping: Sequence[Tuple[int, int]]) -> AsyncKVLoadRequest:
        """登记一个逻辑恢复请求，并生成传给 Worker 的请求描述。

        这里仅登记状态，不会直接发起设备 I/O。后续 Engine/Worker 桥接层
        拿到返回的 request 后，才会调用具体 connector 的异步提交接口。
        ``max_in_flight`` 用来限制第一阶段的并发控制槽数量。
        """
        if self.in_flight_count >= self.max_in_flight:
            raise RuntimeError("async KV load capacity is exhausted")

        request_id = f"async-kv-{next(self._request_counter)}"
        request = AsyncKVLoadRequest(
            request_id=request_id,
            seq_group_id=seq_group_id,
            block_mapping=tuple(tuple(pair) for pair in block_mapping),
        )
        self._loads[request_id] = PendingAsyncKVLoad(request=request)
        return request

    def apply_event(self, event: AsyncKVLoadEvent) -> None:
        """应用一个来自 Worker 的完成事件。

        未知 request、重复终态事件都被视为协议错误，直接抛出异常，避免
        错误事件被静默地应用到其他请求。PENDING 事件只表示仍在等待，
        不改变当前状态。
        """
        load = self._loads.get(event.request_id)
        if load is None:
            raise KeyError(f"unknown async KV request: {event.request_id}")
        if load.state != AsyncKVLoadState.PENDING:
            raise RuntimeError(
                f"async KV request already completed: {event.request_id}")
        if event.state == AsyncKVLoadState.PENDING:
            return
        if event.state not in (AsyncKVLoadState.READY,
                               AsyncKVLoadState.ERROR):
            raise ValueError(f"unsupported async KV event: {event.state}")

        load.state = event.state
        load.error = event.error

    def get_request(self, request_id: str) -> AsyncKVLoadRequest:
        """返回指定请求的只读描述，供 Worker 提交后端 I/O 使用。"""
        return self._get_load(request_id).request

    def pop_ready(self) -> Tuple[AsyncKVLoadRequest, ...]:
        """按提交顺序取出已完成请求，交给调度器在后续轮次提升。

        取出后请求不再由本状态机管理；GPU block 的真正释放或重新进入
        running 队列由上层调度器完成。
        """
        ready = tuple(load.request for load in self._loads.values()
                      if load.state == AsyncKVLoadState.READY)
        for request in ready:
            del self._loads[request.request_id]
        return ready

    def pop_errors(self) -> Tuple[PendingAsyncKVLoad, ...]:
        """取出失败请求，交给调度器执行 block 清理和错误处理。"""
        errors = tuple(load for load in self._loads.values()
                       if load.state == AsyncKVLoadState.ERROR)
        for load in errors:
            del self._loads[load.request.request_id]
        return errors

    def cancel(self, request_id: str) -> AsyncKVLoadRequest:
        """在 Worker 取消底层 I/O 后移除请求并释放策略槽位。"""
        load = self._loads.pop(request_id, None)
        if load is None:
            raise KeyError(f"unknown async KV request: {request_id}")
        return load.request

    def _get_load(self, request_id: str) -> PendingAsyncKVLoad:
        """查找内部状态；不存在时统一报告协议错误。"""
        load = self._loads.get(request_id)
        if load is None:
            raise KeyError(f"unknown async KV request: {request_id}")
        return load
