# SPDX-License-Identifier: Apache-2.0

"""异步 KV read/write 共用的后端无关控制面状态机。"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Optional, Sequence, Tuple

from vllm.core.block_reservation import BlockMapping, LogicalBlockKey


class AsyncKVTransferOperation(Enum):
    """一次 block transfer 的数据方向。"""

    READ = "read"
    WRITE = "write"


class AsyncKVTransferPriority(Enum):
    """后端无关的最小 I/O 优先级。

    restore read 位于请求恢复关键路径；write 已经拥有 GPU source，可以
    留在队列中等待。应用层以后可以改变入队顺序，但不需要修改 Worker/MDS。
    """

    CRITICAL_READ = 0
    DEFERRED_WRITE = 1


class AsyncKVTransferState(Enum):
    """Scheduler 观察到的异步 transfer 生命周期。"""

    QUEUED = auto()
    PENDING = auto()
    READY = auto()
    ERROR = auto()


@dataclass(frozen=True)
class AsyncKVTransferRequest:
    """Scheduler 发送给 Worker 的后端无关 block transfer 描述。"""

    request_id: str
    seq_group_id: str
    reservation_id: str
    operation: AsyncKVTransferOperation
    block_mapping: BlockMapping
    logical_blocks: Tuple[LogicalBlockKey, ...]
    priority: AsyncKVTransferPriority


@dataclass(frozen=True)
class AsyncKVTransferEvent:
    """Worker 返回给 Scheduler 的非阻塞完成事件。"""

    request_id: str
    state: AsyncKVTransferState
    error: Optional[str] = None


@dataclass(frozen=True)
class AsyncKVExecutionMarker:
    """记录 read READY 请求第一次真正进入模型 batch 的时间。"""

    request_id: str
    seq_group_id: str
    promoted_monotonic_ns: int


@dataclass
class PendingAsyncKVTransfer:
    """状态机内部保存的一次 transfer。"""

    request: AsyncKVTransferRequest
    state: AsyncKVTransferState = AsyncKVTransferState.QUEUED
    error: Optional[str] = None


class AsyncKVSchedulePolicy:
    """管理 queued -> pending -> ready/error 的轻量多槽状态机。

    Scheduler 可以提前建立多个 block reservation；``activate_next`` 只按
    ``max_in_flight`` 激活后端能够容纳的请求。所有 slot 共用一套状态表，
    completion 可以乱序返回，request identity 不依赖提交顺序。
    """

    def __init__(self, max_in_flight: int = 1) -> None:
        if max_in_flight <= 0:
            raise ValueError("max_in_flight must be positive")
        self.max_in_flight = max_in_flight
        self._request_counter = itertools.count()
        self._transfers: Dict[str, PendingAsyncKVTransfer] = {}

    @property
    def queued_request_ids(self) -> Tuple[str, ...]:
        return self._ids_in_state(AsyncKVTransferState.QUEUED)

    @property
    def pending_request_ids(self) -> Tuple[str, ...]:
        return self._ids_in_state(AsyncKVTransferState.PENDING)

    @property
    def ready_request_ids(self) -> Tuple[str, ...]:
        return self._ids_in_state(AsyncKVTransferState.READY)

    @property
    def in_flight_count(self) -> int:
        return len(self.pending_request_ids)

    @property
    def available_capacity(self) -> int:
        """返回当前还可提交给后端的 logical transfer slot 数。"""
        return self.max_in_flight - self.in_flight_count

    @property
    def has_outstanding(self) -> bool:
        return bool(self._transfers)

    def enqueue(
        self,
        seq_group_id: str,
        reservation_id: str,
        operation: AsyncKVTransferOperation,
        block_mapping: Sequence[Tuple[int, int]],
        logical_blocks: Sequence[LogicalBlockKey] = (),
        priority: Optional[AsyncKVTransferPriority] = None,
    ) -> AsyncKVTransferRequest:
        """登记 reservation，但暂不占用 Worker/MDS request slot。"""
        if priority is None:
            priority = (AsyncKVTransferPriority.CRITICAL_READ
                        if operation == AsyncKVTransferOperation.READ else
                        AsyncKVTransferPriority.DEFERRED_WRITE)
        request_id = f"async-kv-{next(self._request_counter)}"
        request = AsyncKVTransferRequest(
            request_id=request_id,
            seq_group_id=seq_group_id,
            reservation_id=reservation_id,
            operation=operation,
            block_mapping=tuple(tuple(pair) for pair in block_mapping),
            logical_blocks=tuple(logical_blocks),
            priority=priority,
        )
        self._transfers[request_id] = PendingAsyncKVTransfer(request=request)
        return request

    def activate_next(self) -> Tuple[AsyncKVTransferRequest, ...]:
        """优先激活 critical read，再用剩余槽位处理 deferred write。

        已经 IN_FLIGHT 的 write 不会被伪取消；后到 read 只越过仍为 QUEUED
        的 write。这样不复制两套队列状态机，同时保持同优先级 FIFO。
        """
        capacity = self.available_capacity
        if capacity <= 0:
            return ()
        activated = []
        for priority in AsyncKVTransferPriority:
            for transfer in self._transfers.values():
                if (transfer.state != AsyncKVTransferState.QUEUED
                        or transfer.request.priority != priority):
                    continue
                transfer.state = AsyncKVTransferState.PENDING
                activated.append(transfer.request)
                if len(activated) == capacity:
                    return tuple(activated)
        return tuple(activated)

    def apply_event(self, event: AsyncKVTransferEvent) -> None:
        """应用 active request 的 PENDING/READY/ERROR 事件。"""
        transfer = self._get_transfer(event.request_id)
        if transfer.state != AsyncKVTransferState.PENDING:
            raise RuntimeError(
                f"async KV request is not active: {event.request_id}")
        if event.state == AsyncKVTransferState.PENDING:
            return
        if event.state not in (AsyncKVTransferState.READY,
                               AsyncKVTransferState.ERROR):
            raise ValueError(f"unsupported async KV event: {event.state}")
        transfer.state = event.state
        transfer.error = event.error

    def pop_ready(self) -> Tuple[AsyncKVTransferRequest, ...]:
        """按入队顺序消费已经成功完成的请求。"""
        ready = tuple(
            transfer.request for transfer in self._transfers.values()
            if transfer.state == AsyncKVTransferState.READY)
        for request in ready:
            del self._transfers[request.request_id]
        return ready

    def pop_errors(self) -> Tuple[PendingAsyncKVTransfer, ...]:
        """按入队顺序消费失败请求及其错误信息。"""
        errors = tuple(
            transfer for transfer in self._transfers.values()
            if transfer.state == AsyncKVTransferState.ERROR)
        for transfer in errors:
            del self._transfers[transfer.request.request_id]
        return errors

    def cancel_queued(self, request_id: str) -> AsyncKVTransferRequest:
        """取消尚未提交给 Worker 的请求；active I/O 不能在此处撤销。"""
        transfer = self._get_transfer(request_id)
        if transfer.state != AsyncKVTransferState.QUEUED:
            raise RuntimeError(
                f"cannot cancel active async KV request: {request_id}")
        del self._transfers[request_id]
        return transfer.request

    def _ids_in_state(self,
                      state: AsyncKVTransferState) -> Tuple[str, ...]:
        return tuple(
            request_id for request_id, transfer in self._transfers.items()
            if transfer.state == state)

    def _get_transfer(self, request_id: str) -> PendingAsyncKVTransfer:
        transfer = self._transfers.get(request_id)
        if transfer is None:
            raise KeyError(f"unknown async KV request: {request_id}")
        return transfer
