# SPDX-License-Identifier: Apache-2.0

"""异步 KV 换入换出使用的最小 block reservation 描述。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from vllm.utils import Device


BlockMapping = Tuple[Tuple[int, int], ...]


@dataclass(frozen=True, order=True)
class LogicalBlockKey:
    """稳定的 scheduler-visible block 身份，不包含可复用的物理地址。"""

    seq_id: int
    logical_index: int


@dataclass(frozen=True)
class BlockResidency:
    """一个逻辑 KV block 当前可供上层策略查询的副本状态。"""

    key: LogicalBlockKey
    storage_block_id: Optional[int]
    storage_replica_clean: bool
    pin_count: int


@dataclass(frozen=True)
class BlockSwapReservation:
    """一次尚未提交到正式 block table 的设备间 block 转换。

    reservation 只暴露后端执行 I/O 所需的物理 block mapping。源 block、
    目标 block 对象以及 refcount 的具体处理仍由 BlockSpaceManager 私有
    持有，避免 Scheduler 或 GranuleKV connector 直接依赖 vLLM allocator 实现。
    """

    reservation_id: str
    seq_group_id: str
    source_device: Device
    target_device: Device
    block_mapping: BlockMapping
    # write reservation 可以直接复用 read 后保留的 clean SSD block；这些
    # block 不出现在 I/O mapping 中，但会在 commit 时进入正式 storage 表。
    num_reused_blocks: int = 0
    # I/O mapping 与逻辑 block 一一对应；后续策略可以在不解析 allocator
    # 私有对象的前提下，把 completion 关联回 residency directory。
    logical_blocks: Tuple[LogicalBlockKey, ...] = ()


@dataclass(frozen=True)
class BlockPrefixRestoreReservation:
    """一次 GranuleKV prefix SSD -> GPU 恢复事务。

    与 swap-in 不同，目标请求原本没有 storage block table。Block manager
    会先为新请求建立 GPU table，再把命中的连续 prefix 映射到这些目标
    block；READY 前 Scheduler 不会让该请求进入计算队列。
    """

    reservation_id: str
    seq_group_id: str
    block_mapping: BlockMapping
    logical_blocks: Tuple[LogicalBlockKey, ...]
    num_prefix_blocks: int
