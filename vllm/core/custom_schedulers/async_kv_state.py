# SPDX-License-Identifier: Apache-2.0

"""自定义 KV 调度策略使用的 block residency 视图。

这里刻意不直接操作 vLLM allocator，也不保存物理 block 对象。Scheduler
后续只把 BlockSpaceManager 暴露的只读 residency snapshot 转成这里的轻量
状态，再交给 policy 做“保留、读取、写回、释放”的决策。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

from vllm.core.block_reservation import BlockResidency, LogicalBlockKey


@dataclass(frozen=True)
class AsyncKVBlockState:
    """一个 logical KV block 面向调度策略的最小状态。

    ``gpu_present`` 代表当前 block 仍可直接参与 attention；``ssd_present``
    代表已经存在 storage 副本；``storage_clean`` 代表 storage 副本和当前
    GPU 内容一致，释放 GPU block 时可以跳过写盘。
    """

    key: LogicalBlockKey
    gpu_present: bool
    ssd_present: bool
    storage_clean: bool
    pin_count: int = 0
    storage_block_id: Optional[int] = None

    @property
    def can_skip_write(self) -> bool:
        """返回释放 GPU block 时是否可以复用已有 clean SSD 副本。"""
        return self.ssd_present and self.storage_clean

    @property
    def needs_restore_read(self) -> bool:
        """返回参与计算前是否需要从 SSD 恢复到 GPU。"""
        return not self.gpu_present and self.ssd_present


class AsyncKVResidencyView:
    """调度层使用的只读 residency 索引。

    这个类目前只做 snapshot 查询，不负责修改状态。真正的状态提交仍由
    block reservation 的 commit/abort 路径完成，避免 policy 绕过 vLLM
    block manager 直接改 allocator。
    """

    def __init__(self, blocks: Iterable[AsyncKVBlockState] = ()) -> None:
        self._blocks: Dict[LogicalBlockKey, AsyncKVBlockState] = {
            block.key: block for block in blocks
        }

    @classmethod
    def from_block_residency(
        cls,
        snapshots: Iterable[BlockResidency],
        *,
        gpu_present: bool = True,
    ) -> "AsyncKVResidencyView":
        """从 BlockSpaceManager 的只读 snapshot 构造 policy 视图。"""
        return cls(
            AsyncKVBlockState(
                key=snapshot.key,
                gpu_present=gpu_present,
                ssd_present=snapshot.storage_block_id is not None,
                storage_clean=snapshot.storage_replica_clean,
                pin_count=snapshot.pin_count,
                storage_block_id=snapshot.storage_block_id,
            ) for snapshot in snapshots)

    def get(self, key: LogicalBlockKey) -> Optional[AsyncKVBlockState]:
        """查询单个 logical block 的状态；不存在表示当前 snapshot 不认识它。"""
        return self._blocks.get(key)

    def iter_blocks(self) -> Tuple[AsyncKVBlockState, ...]:
        """按 snapshot 构造顺序返回所有 block 状态。"""
        return tuple(self._blocks.values())

