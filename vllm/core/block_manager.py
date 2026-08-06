# SPDX-License-Identifier: Apache-2.0
"""A block manager that manages token blocks."""
import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional
from typing import Sequence as GenericSequence
from typing import Tuple

import vllm.envs as envs
from vllm.core.block.block_table import BlockTable
from vllm.core.block.cpu_gpu_block_allocator import CpuGpuBlockAllocator
from vllm.core.block.interfaces import Block
from vllm.core.block.prefix_caching_block import (ComputedBlocksTracker,
                                                  LastAccessBlocksTracker)
from vllm.core.block.utils import check_no_caching_or_swa_for_blockmgr_encdec
from vllm.core.block_reservation import BlockSwapReservation
from vllm.core.interfaces import AllocStatus, BlockSpaceManager
from vllm.logger import init_logger
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.utils import Device

SeqId = int
EncoderSeqId = str

logger = init_logger(__name__)


@dataclass(frozen=True)
class _ReservedBlockTable:
    """一个 sequence 在 reservation 前后的两份 block table 内容。"""

    seq_id: SeqId
    source_blocks: Tuple[Block, ...]
    target_blocks: Tuple[Block, ...]
    # 只有 newly_allocated_targets 归当前 reservation 临时所有；复用的
    # clean storage prefix 仍由 residency directory 持有，abort 时不能释放。
    newly_allocated_targets: Tuple[Block, ...]
    stale_storage_replicas: Tuple[Block, ...]


@dataclass(frozen=True)
class _BlockSwapReservationRecord:
    """BlockSpaceManager 私有保存的 reservation 资源所有权。"""

    public: BlockSwapReservation
    tables: Tuple[_ReservedBlockTable, ...]


class SelfAttnBlockSpaceManager(BlockSpaceManager):
    """BlockSpaceManager which manages the allocation of KV cache.

    It owns responsibility for allocation, swapping, allocating memory for
    autoregressively-generated tokens, and other advanced features such as
    prefix caching, forking/copy-on-write, and sliding-window memory allocation.

    This class implements the design described in
    https://github.com/vllm-project/vllm/pull/3492.

    Lookahead slots
        The block manager has the notion of a "lookahead slot". These are slots
        in the KV cache that are allocated for a sequence. Unlike the other
        allocated slots, the content of these slots is undefined -- the worker
        may use the memory allocations in any way.

        In practice, a worker could use these lookahead slots to run multiple
        forward passes for a single scheduler invocation. Each successive
        forward pass would write KV activations to the corresponding lookahead
        slot. This allows low inter-token latency use-cases, where the overhead
        of continuous batching scheduling is amortized over >1 generated tokens.

        Speculative decoding uses lookahead slots to store KV activations of
        proposal tokens.

        See https://github.com/vllm-project/vllm/pull/3250 for more information
        on lookahead scheduling.

    Args:
        block_size (int): The size of each memory block.
        num_gpu_blocks (int): The number of memory blocks allocated on GPU.
        num_cpu_blocks (int): The number of memory blocks allocated on CPU.
        watermark (float, optional): The threshold used for memory swapping.
            Defaults to 0.01.
        sliding_window (Optional[int], optional): The size of the sliding
            window. Defaults to None.
        enable_caching (bool, optional): Flag indicating whether caching is
            enabled. Defaults to False.
    """

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        watermark: float = 0.01,
        sliding_window: Optional[int] = None,
        enable_caching: bool = False,
    ) -> None:
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks

        self.sliding_window = sliding_window
        # max_block_sliding_window is the max number of blocks that need to be
        # allocated
        self.max_block_sliding_window = None
        if sliding_window is not None:
            # +1 here because // rounds down
            num_blocks = sliding_window // block_size + 1
            # +1 here because the last block may not be full,
            # and so the sequence stretches one more block at the beginning
            # For example, if sliding_window is 3 and block_size is 4,
            # we may need 2 blocks when the second block only holds 1 token.
            self.max_block_sliding_window = num_blocks + 1

        self.watermark = watermark
        assert watermark >= 0.0

        self.enable_caching = enable_caching

        self.watermark_blocks = int(watermark * num_gpu_blocks)

        self.block_allocator = CpuGpuBlockAllocator.create(
            allocator_type="prefix_caching" if enable_caching else "naive",
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
            block_size=block_size,
        )

        self.block_tables: Dict[SeqId, BlockTable] = {}
        self.cross_block_tables: Dict[EncoderSeqId, BlockTable] = {}

        self._computed_blocks_tracker = ComputedBlocksTracker(
            self.block_allocator, self.block_size, self.enable_caching)
        self._last_access_blocks_tracker = LastAccessBlocksTracker(
            self.block_allocator)

        # reservation 中的目标 block 已经从目标 allocator 分配，但尚未
        # 发布到正式 block table。只要记录仍在这里，源和目标两份物理
        # block 就都不能被复用；commit/abort 必须且只能消费一次记录。
        self._reservation_counter = itertools.count()
        self._block_swap_reservations: Dict[
            str, _BlockSwapReservationRecord] = {}
        # 正式 block table 指向 GPU 时，这里持有最近一次 read 后仍有效的
        # storage 副本。自回归 decode 只会修改尾部，因此后续 write 可以
        # 复用最长 clean prefix，只为 dirty suffix 分配和提交 SSD I/O。
        self._storage_replicas: Dict[SeqId, Tuple[Block, ...]] = {}

    def can_allocate(self,
                     seq_group: SequenceGroup,
                     num_lookahead_slots: int = 0) -> AllocStatus:
        # FIXME(woosuk): Here we assume that all sequences in the group share
        # the same prompt. This may not be true for preempted sequences.

        check_no_caching_or_swa_for_blockmgr_encdec(self, seq_group)

        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]
        num_required_blocks = BlockTable.get_num_required_blocks(
            seq.get_token_ids(),
            block_size=self.block_size,
            num_lookahead_slots=num_lookahead_slots,
        )

        if seq_group.is_encoder_decoder():
            encoder_seq = seq_group.get_encoder_seq()
            assert encoder_seq is not None
            num_required_blocks += BlockTable.get_num_required_blocks(
                encoder_seq.get_token_ids(),
                block_size=self.block_size,
            )

        if self.max_block_sliding_window is not None:
            num_required_blocks = min(num_required_blocks,
                                      self.max_block_sliding_window)

        num_free_gpu_blocks = self.block_allocator.get_num_free_blocks(
            device=Device.GPU)

        # Use watermark to avoid frequent cache eviction.
        if (self.num_total_gpu_blocks - num_required_blocks
                < self.watermark_blocks):
            return AllocStatus.NEVER
        if num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks:
            return AllocStatus.OK
        else:
            return AllocStatus.LATER

    def _allocate_sequence(self, seq: Sequence) -> BlockTable:
        block_table = BlockTable(
            block_size=self.block_size,
            block_allocator=self.block_allocator,
            max_block_sliding_window=self.max_block_sliding_window,
        )
        if seq.get_token_ids():
            # NOTE: If there are any factors affecting the block besides
            # token_ids, they should be added as input to extra_hash.
            extra_hash = seq.extra_hash()

            # Add blocks to the block table only if the sequence is non empty.
            block_table.allocate(token_ids=seq.get_token_ids(),
                                 extra_hash=extra_hash)

        return block_table

    def allocate(self, seq_group: SequenceGroup) -> None:

        # Allocate self-attention block tables for decoder sequences
        waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
        assert not (set(seq.seq_id for seq in waiting_seqs)
                    & self.block_tables.keys()), "block table already exists"

        # NOTE: Here we assume that all sequences in the group have the same
        # prompt.
        seq = waiting_seqs[0]
        block_table: BlockTable = self._allocate_sequence(seq)
        self.block_tables[seq.seq_id] = block_table

        # Track seq
        self._last_access_blocks_tracker.add_seq(seq.seq_id)

        # Assign the block table for each sequence.
        for seq in waiting_seqs[1:]:
            self.block_tables[seq.seq_id] = block_table.fork()

            # Track seq
            self._last_access_blocks_tracker.add_seq(seq.seq_id)

        # Allocate cross-attention block table for encoder sequence
        #
        # NOTE: Here we assume that all sequences in the group have the same
        # encoder prompt.
        request_id = seq_group.request_id

        assert (request_id
                not in self.cross_block_tables), \
            "block table already exists"

        check_no_caching_or_swa_for_blockmgr_encdec(self, seq_group)

        if seq_group.is_encoder_decoder():
            encoder_seq = seq_group.get_encoder_seq()
            assert encoder_seq is not None
            block_table = self._allocate_sequence(encoder_seq)
            self.cross_block_tables[request_id] = block_table

    def can_append_slots(self, seq_group: SequenceGroup,
                         num_lookahead_slots: int) -> bool:
        """Determine if there is enough space in the GPU KV cache to continue
        generation of the specified sequence group.

        We use a worst-case heuristic: assume each touched block will require a
        new allocation (either via CoW or new block). We can append slots if the
        number of touched blocks is less than the number of free blocks.

        "Lookahead slots" are slots that are allocated in addition to the slots
        for known tokens. The contents of the lookahead slots are not defined.
        This is used by speculative decoding when speculating future tokens.
        """

        num_touched_blocks = 0
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            block_table = self.block_tables[seq.seq_id]

            num_touched_blocks += (
                block_table.get_num_blocks_touched_by_append_slots(
                    token_ids=block_table.get_unseen_token_ids(
                        seq.get_token_ids()),
                    num_lookahead_slots=num_lookahead_slots,
                ))

        num_free_gpu_blocks = self.block_allocator.get_num_free_blocks(
            Device.GPU)
        return num_touched_blocks <= num_free_gpu_blocks

    def append_slots(
        self,
        seq: Sequence,
        num_lookahead_slots: int,
    ) -> List[Tuple[int, int]]:

        block_table = self.block_tables[seq.seq_id]

        block_table.append_token_ids(
            token_ids=block_table.get_unseen_token_ids(seq.get_token_ids()),
            num_lookahead_slots=num_lookahead_slots,
            num_computed_slots=seq.data.get_num_computed_tokens(),
            extra_hash=seq.extra_hash(),
        )
        # Return any new copy-on-writes.
        new_cows = self.block_allocator.clear_copy_on_writes()
        return new_cows

    def free(self, seq: Sequence) -> None:
        seq_id = seq.seq_id

        if seq_id not in self.block_tables:
            # Already freed or haven't been scheduled yet.
            return

        # Update seq block ids with the latest access time
        self._last_access_blocks_tracker.update_seq_blocks_last_access(
            seq_id, self.block_tables[seq.seq_id].physical_block_ids)

        # Untrack seq
        self._last_access_blocks_tracker.remove_seq(seq_id)
        self._computed_blocks_tracker.remove_seq(seq_id)

        # storage replica 不属于正式 block table，sequence 结束时必须单独
        # 回收；若 table 当前就在 storage，directory 中不会保存重复所有权。
        for replica in self._storage_replicas.pop(seq_id, ()):
            self.block_allocator.free(replica)

        # Free table/blocks
        self.block_tables[seq_id].free()
        del self.block_tables[seq_id]

    def free_cross(self, seq_group: SequenceGroup) -> None:
        request_id = seq_group.request_id
        if request_id not in self.cross_block_tables:
            # Already freed or hasn't been scheduled yet.
            return
        self.cross_block_tables[request_id].free()
        del self.cross_block_tables[request_id]

    def get_block_table(self, seq: Sequence) -> List[int]:
        block_ids = self.block_tables[seq.seq_id].physical_block_ids
        return block_ids  # type: ignore

    def get_cross_block_table(self, seq_group: SequenceGroup) -> List[int]:
        request_id = seq_group.request_id
        assert request_id in self.cross_block_tables
        block_ids = self.cross_block_tables[request_id].physical_block_ids
        assert all(b is not None for b in block_ids)
        return block_ids  # type: ignore

    def access_all_blocks_in_seq(self, seq: Sequence, now: float):
        if self.enable_caching:
            # Record the latest access time for the sequence. The actual update
            # of the block ids is deferred to the sequence free(..) call, since
            # only during freeing of block ids, the blocks are actually added to
            # the evictor (which is when the most updated time is required)
            # (This avoids expensive calls to mark_blocks_as_accessed(..))
            self._last_access_blocks_tracker.update_last_access(
                seq.seq_id, now)

    def mark_blocks_as_computed(self, seq_group: SequenceGroup,
                                token_chunk_size: int):
        # If prefix caching is enabled, mark immutable blocks as computed
        # right after they have been scheduled (for prefill). This assumes
        # the scheduler is synchronous so blocks are actually computed when
        # scheduling the next batch.
        self.block_allocator.mark_blocks_as_computed([])

    def get_common_computed_block_ids(
            self, seqs: List[Sequence]) -> GenericSequence[int]:
        """Determine which blocks for which we skip prefill.

        With prefix caching we can skip prefill for previously-generated blocks.
        Currently, the attention implementation only supports skipping cached
        blocks if they are a contiguous prefix of cached blocks.

        This method determines which blocks can be safely skipped for all
        sequences in the sequence group.
        """
        computed_seq_block_ids = []
        for seq in seqs:
            all_blocks = self.block_tables[seq.seq_id].physical_block_ids
            num_cached_tokens = (
                self._computed_blocks_tracker.get_num_cached_tokens(seq))
            assert num_cached_tokens % self.block_size == 0
            num_cached_blocks = num_cached_tokens // self.block_size
            computed_block_ids = all_blocks[:num_cached_blocks]
            computed_seq_block_ids.append(computed_block_ids)

        # NOTE(sang): This assumes seq_block_ids doesn't contain any None.
        return self.block_allocator.get_common_computed_block_ids(
            computed_seq_block_ids)  # type: ignore

    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        if parent_seq.seq_id not in self.block_tables:
            # Parent sequence has either been freed or never existed.
            return
        src_block_table = self.block_tables[parent_seq.seq_id]
        self.block_tables[child_seq.seq_id] = src_block_table.fork()

        # Track child seq
        self._last_access_blocks_tracker.add_seq(child_seq.seq_id)

    def can_swap_in(self, seq_group: SequenceGroup,
                    num_lookahead_slots: int) -> AllocStatus:
        """Returns the AllocStatus for the given sequence_group 
        with num_lookahead_slots.

        Args:
            sequence_group (SequenceGroup): The sequence group to swap in.
            num_lookahead_slots (int): Number of lookahead slots used in 
                speculative decoding, default to 0.

        Returns:
            AllocStatus: The AllocStatus for the given sequence group.
        """
        return self._can_swap(seq_group, Device.GPU, SequenceStatus.SWAPPED,
                              num_lookahead_slots)

    def swap_in(self, seq_group: SequenceGroup) -> List[Tuple[int, int]]:
        """Returns the block id mapping (from CPU to GPU) generated by
        swapping in the given seq_group with num_lookahead_slots.

        Args:
            seq_group (SequenceGroup): The sequence group to swap in.

        Returns:
            List[Tuple[int, int]]: The mapping of swapping block from CPU 
                to GPU.
        """
        physical_block_id_mapping = []
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            blocks = self.block_tables[seq.seq_id].blocks
            if len(blocks) == 0:
                continue

            seq_swap_mapping = self.block_allocator.swap(blocks=blocks,
                                                         src_device=Device.CPU,
                                                         dst_device=Device.GPU)

            # Refresh the block ids of the table (post-swap)
            self.block_tables[seq.seq_id].update(blocks)

            seq_physical_block_id_mapping = {
                self.block_allocator.get_physical_block_id(
                    Device.CPU, cpu_block_id):
                self.block_allocator.get_physical_block_id(
                    Device.GPU, gpu_block_id)
                for cpu_block_id, gpu_block_id in seq_swap_mapping.items()
            }

            physical_block_id_mapping.extend(
                list(seq_physical_block_id_mapping.items()))

        if envs.VLLM_V0_SWAP_TRACE:
            logger.info(
                "[V0_SWAP_TRACE][BlockManager] op=swap_in request_id=%s "
                "seqs=%d mappings=%d",
                seq_group.request_id,
                len(seq_group.get_seqs(status=SequenceStatus.SWAPPED)),
                len(physical_block_id_mapping),
            )

        return physical_block_id_mapping

    def reserve_swap_in(self,
                        seq_group: SequenceGroup) -> BlockSwapReservation:
        """预留 SSD/storage -> GPU 的目标 block，但不发布 block table。"""
        return self._reserve_swap(seq_group, SequenceStatus.SWAPPED,
                                  Device.CPU, Device.GPU)

    def can_swap_out(self, seq_group: SequenceGroup) -> bool:
        """Returns whether we can swap out the given sequence_group 
        with num_lookahead_slots.

        Args:
            seq_group (SequenceGroup): The sequence group to swap out.
            num_lookahead_slots (int): Number of lookahead slots used in 
                speculative decoding, default to 0.

        Returns:
            bool: Whether it's possible to swap out current sequence group.
        """
        alloc_status = self._can_swap(seq_group, Device.CPU,
                                      SequenceStatus.RUNNING)
        return alloc_status == AllocStatus.OK

    def swap_out(self, seq_group: SequenceGroup) -> List[Tuple[int, int]]:
        """Returns the block id mapping (from GPU to CPU) generated by
        swapping out the given sequence_group with num_lookahead_slots.

        Args:
            sequence_group (SequenceGroup): The sequence group to swap out.

        Returns:
            List[Tuple[int, int]]: The mapping of swapping block from 
                GPU to CPU.
        """
        physical_block_id_mapping = []
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            blocks = self.block_tables[seq.seq_id].blocks
            if len(blocks) == 0:
                continue

            seq_swap_mapping = self.block_allocator.swap(blocks=blocks,
                                                         src_device=Device.GPU,
                                                         dst_device=Device.CPU)

            # Refresh the block ids of the table (post-swap)
            self.block_tables[seq.seq_id].update(blocks)

            seq_physical_block_id_mapping = {
                self.block_allocator.get_physical_block_id(
                    Device.GPU, gpu_block_id):
                self.block_allocator.get_physical_block_id(
                    Device.CPU, cpu_block_id)
                for gpu_block_id, cpu_block_id in seq_swap_mapping.items()
            }

            physical_block_id_mapping.extend(
                list(seq_physical_block_id_mapping.items()))

        if envs.VLLM_V0_SWAP_TRACE:
            logger.info(
                "[V0_SWAP_TRACE][BlockManager] op=swap_out request_id=%s "
                "seqs=%d mappings=%d",
                seq_group.request_id,
                len(seq_group.get_seqs(status=SequenceStatus.RUNNING)),
                len(physical_block_id_mapping),
            )

        return physical_block_id_mapping

    def reserve_swap_out(self,
                         seq_group: SequenceGroup) -> BlockSwapReservation:
        """预留 GPU -> storage，只为 dirty suffix 创建写盘 mapping。"""
        return self._reserve_swap(seq_group, SequenceStatus.RUNNING,
                                  Device.GPU, Device.CPU)

    def can_reserve_swap_out(self, seq_group: SequenceGroup) -> bool:
        """判断 storage 空间能否容纳当前 dirty suffix。

        原生 ``can_swap_out`` 按整条 sequence 估算目标 block；有 clean SSD
        replica 后会明显高估空间。异步路径只需要为第一个不匹配 block
        开始的后缀分配新 storage block，因此在这里使用同一口径检查。
        """
        required = 0
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            source_blocks = tuple(self.block_tables[seq.seq_id].blocks)
            replicas = self._storage_replicas.get(seq.seq_id, ())
            required += len(source_blocks) - self._clean_replica_prefix_length(
                source_blocks, replicas)
        return required <= self.get_num_free_cpu_blocks()

    def commit_block_swap(self, reservation_id: str) -> None:
        """在后端 I/O 完成后发布目标 block，并更新双副本所有权。

        更新顺序刻意固定为“先让 block table 指向目标对象，再释放源对象”。
        read 的 storage source 会转交给 replica directory 而不是释放；write
        才释放 GPU source。Scheduler 只会在本方法返回后改变状态，因此
        attention 不会观察到未完成的目标 block。
        """
        record = self._get_block_swap_reservation(reservation_id)
        for table_record in record.tables:
            block_table = self.block_tables.get(table_record.seq_id)
            if block_table is None:
                raise RuntimeError(
                    "cannot commit block reservation after table removal: "
                    f"{reservation_id}")
            current_blocks = tuple(block_table.blocks)
            if current_blocks != table_record.source_blocks:
                raise RuntimeError(
                    "source block table changed before reservation commit: "
                    f"{reservation_id}")
        # 所有 source table 都验证通过后才消费记录；校验失败时调用方仍可
        # abort，避免目标 block 因记录提前弹出而泄漏。
        del self._block_swap_reservations[reservation_id]
        is_read = record.public.source_device == Device.CPU
        for table_record in record.tables:
            block_table = self.block_tables[table_record.seq_id]
            block_table.update(list(table_record.target_blocks))
            if is_read:
                # read 后正式表切到 GPU，但 SSD source 继续由 directory
                # 持有。只要 decode 尚未修改对应 block，它就是 clean replica。
                old_replicas = self._storage_replicas.pop(
                    table_record.seq_id, ())
                for replica in old_replicas:
                    self.block_allocator.free(replica)
                self._storage_replicas[table_record.seq_id] = (
                    table_record.source_blocks)
            else:
                # write 后 target storage blocks 转交给正式 block table。
                # clean prefix 原来由 directory 持有，删除目录项只转移所有权；
                # 被 dirty suffix 替换的旧 replica 才需要真正释放。
                self._storage_replicas.pop(table_record.seq_id, None)
                for source_block in table_record.source_blocks:
                    self.block_allocator.free(source_block)
                for replica in table_record.stale_storage_replicas:
                    self.block_allocator.free(replica)

    def abort_block_swap(self, reservation_id: str) -> None:
        """取消 reservation，只释放未发布的目标 block，源表保持不变。"""
        record = self._pop_block_swap_reservation(reservation_id)
        for table_record in record.tables:
            for target_block in table_record.newly_allocated_targets:
                self.block_allocator.free(target_block)

    def _reserve_swap(
        self,
        seq_group: SequenceGroup,
        sequence_status: SequenceStatus,
        source_device: Device,
        target_device: Device,
    ) -> BlockSwapReservation:
        """建立一笔双副本 block 事务，并返回后端可直接执行的 mapping。

        目标 block 使用 allocator 的普通公开接口分配，因此 refcount、prefix
        hash 和 block object pool 仍由原生实现维护。发生部分分配失败时，
        已分配目标会在抛出异常前全部回收，源 block table 不受影响。
        """
        table_records: List[_ReservedBlockTable] = []
        physical_mapping: List[Tuple[int, int]] = []
        reused_blocks = 0
        try:
            for seq in seq_group.get_seqs(status=sequence_status):
                source_blocks = tuple(self.block_tables[seq.seq_id].blocks)
                if source_device == Device.GPU:
                    replicas = self._storage_replicas.get(seq.seq_id, ())
                    clean_prefix = self._clean_replica_prefix_length(
                        source_blocks, replicas)
                    reused_prefix = replicas[:clean_prefix]
                    dirty_source = source_blocks[clean_prefix:]
                    new_targets = tuple(
                        self._clone_blocks_to_device(
                            dirty_source,
                            target_device,
                            previous_target=(reused_prefix[-1]
                                             if reused_prefix else None)))
                    target_blocks = reused_prefix + new_targets
                    stale_replicas = replicas[clean_prefix:]
                    reused_blocks += clean_prefix
                    mapping_sources = dirty_source
                    mapping_targets = new_targets
                else:
                    new_targets = tuple(
                        self._clone_blocks_to_device(source_blocks,
                                                     target_device))
                    target_blocks = new_targets
                    stale_replicas = ()
                    mapping_sources = source_blocks
                    mapping_targets = new_targets
                table_records.append(
                    _ReservedBlockTable(seq_id=seq.seq_id,
                                        source_blocks=source_blocks,
                                        target_blocks=target_blocks,
                                        newly_allocated_targets=new_targets,
                                        stale_storage_replicas=
                                        stale_replicas))
                for source_block, target_block in zip(mapping_sources,
                                                      mapping_targets):
                    assert source_block.block_id is not None
                    assert target_block.block_id is not None
                    physical_mapping.append((
                        self.block_allocator.get_physical_block_id(
                            source_device, source_block.block_id),
                        self.block_allocator.get_physical_block_id(
                            target_device, target_block.block_id),
                    ))
        except Exception:
            for table_record in table_records:
                for target_block in table_record.newly_allocated_targets:
                    self.block_allocator.free(target_block)
            raise

        reservation_id = f"block-swap-{next(self._reservation_counter)}"
        public = BlockSwapReservation(
            reservation_id=reservation_id,
            seq_group_id=seq_group.request_id,
            source_device=source_device,
            target_device=target_device,
            block_mapping=tuple(physical_mapping),
            num_reused_blocks=reused_blocks,
        )
        self._block_swap_reservations[reservation_id] = (
            _BlockSwapReservationRecord(public=public,
                                        tables=tuple(table_records)))
        return public

    def _clone_blocks_to_device(
        self,
        source_blocks: Tuple[Block, ...],
        target_device: Device,
        previous_target: Optional[Block] = None,
    ) -> List[Block]:
        """按原 token 链在目标设备分配一份尚未发布的 block 对象。"""
        target_blocks: List[Block] = []
        try:
            for source_block in source_blocks:
                extra_hash = getattr(source_block, "extra_hash", None)
                if source_block.is_full:
                    target_block = self.block_allocator.allocate_immutable_block(
                        previous_target,
                        list(source_block.token_ids),
                        target_device,
                        extra_hash=extra_hash,
                    )
                else:
                    target_block = self.block_allocator.allocate_mutable_block(
                        previous_target,
                        target_device,
                        extra_hash=extra_hash,
                    )
                    target_block.append_token_ids(
                        list(source_block.token_ids))
                target_blocks.append(target_block)
                previous_target = target_block
        except Exception:
            for target_block in target_blocks:
                self.block_allocator.free(target_block)
            raise
        return target_blocks

    @staticmethod
    def _clean_replica_prefix_length(
        source_blocks: Tuple[Block, ...],
        replicas: Tuple[Block, ...],
    ) -> int:
        """返回 GPU 与 SSD 内容一致的最长 block 前缀长度。

        对同一 seq_id，block 的位置和 token 内容共同构成最小逻辑身份。
        decode 只向尾部追加 token，因此一旦发现不匹配，后面的 replica
        都按 dirty suffix 处理，既保持 block 链正确，也避免逐项复杂修补。
        """
        clean_prefix = 0
        for source, replica in zip(source_blocks, replicas):
            if (source.token_ids != replica.token_ids
                    or source.extra_hash != replica.extra_hash):
                break
            clean_prefix += 1
        return clean_prefix

    def get_num_clean_storage_replicas(self, seq: Sequence) -> int:
        """返回指定 GPU sequence 当前仍有效的 clean SSD prefix 长度。"""
        replicas = self._storage_replicas.get(seq.seq_id, ())
        blocks = tuple(self.block_tables[seq.seq_id].blocks)
        return self._clean_replica_prefix_length(blocks, replicas)

    def _pop_block_swap_reservation(
            self, reservation_id: str) -> _BlockSwapReservationRecord:
        record = self._block_swap_reservations.pop(reservation_id, None)
        if record is None:
            raise KeyError(f"unknown block reservation: {reservation_id}")
        return record

    def _get_block_swap_reservation(
            self, reservation_id: str) -> _BlockSwapReservationRecord:
        record = self._block_swap_reservations.get(reservation_id)
        if record is None:
            raise KeyError(f"unknown block reservation: {reservation_id}")
        return record

    def get_num_free_gpu_blocks(self) -> int:
        return self.block_allocator.get_num_free_blocks(Device.GPU)

    def get_num_free_cpu_blocks(self) -> int:
        return self.block_allocator.get_num_free_blocks(Device.CPU)

    def get_prefix_cache_hit_rate(self, device: Device) -> float:
        return self.block_allocator.get_prefix_cache_hit_rate(device)

    def reset_prefix_cache(self, device: Optional[Device] = None) -> bool:
        return self.block_allocator.reset_prefix_cache(device)

    def _can_swap(self,
                  seq_group: SequenceGroup,
                  device: Device,
                  status: SequenceStatus,
                  num_lookahead_slots: int = 0) -> AllocStatus:
        """Returns the AllocStatus for swapping in/out the given sequence_group 
        on to the 'device'.

        Args:
            sequence_group (SequenceGroup): The sequence group to swap in/out.
            device (Device): device to swap the 'seq_group' on.
            status (SequenceStatus): The status of sequence which is needed
                for action. RUNNING for swap out and SWAPPED for swap in
            num_lookahead_slots (int): Number of lookahead slots used in 
                speculative decoding, default to 0.

        Returns:
            AllocStatus: The AllocStatus for swapping in/out the given 
                sequence_group on to the 'device'.
        """
        # First determine the number of blocks that will be touched by this
        # swap. Then verify if there are available blocks in the device
        # to perform the swap.
        num_blocks_touched = 0
        blocks: List[Block] = []
        for seq in seq_group.get_seqs(status=status):
            block_table = self.block_tables[seq.seq_id]
            if block_table.blocks is not None:
                # Compute the number blocks to touch for the tokens to be
                # appended. This does NOT include the full blocks that need
                # to be touched for the swap.
                num_blocks_touched += \
                    block_table.get_num_blocks_touched_by_append_slots(
                        block_table.get_unseen_token_ids(seq.get_token_ids()),
                        num_lookahead_slots=num_lookahead_slots)
                blocks.extend(block_table.blocks)
        # Compute the number of full blocks to touch and add it to the
        # existing count of blocks to touch.
        num_blocks_touched += self.block_allocator.get_num_full_blocks_touched(
            blocks, device=device)

        watermark_blocks = 0
        if device == Device.GPU:
            watermark_blocks = self.watermark_blocks

        if self.block_allocator.get_num_total_blocks(
                device) < num_blocks_touched:
            return AllocStatus.NEVER
        elif self.block_allocator.get_num_free_blocks(
                device) - num_blocks_touched >= watermark_blocks:
            return AllocStatus.OK
        else:
            return AllocStatus.LATER

    def get_num_cached_tokens(self, seq: Sequence) -> int:
        """Get the number of tokens in blocks that are already computed and
        cached in the block manager for the sequence.
        """
        return self._computed_blocks_tracker.get_num_cached_tokens(seq)
