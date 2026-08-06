# SPDX-License-Identifier: Apache-2.0

"""AsyncKVScheduler 队列迁移和 block mapping 的单元测试。"""

import pytest

from vllm.config import CacheConfig, SchedulerConfig
from vllm.core.async_kv_scheduler import AsyncKVScheduler
from vllm.core.scheduler import Scheduler, SchedulingBudget
from vllm.core.scheduler_policy import (AsyncKVTransferEvent,
                                        AsyncKVTransferOperation,
                                        AsyncKVTransferState)
from vllm.sequence import SequenceStatus

from .utils import (append_new_token_seq_group, create_dummy_prompt,
                    schedule_and_update_computed_tokens)


def _create_scheduler() -> AsyncKVScheduler:
    """创建一个同时具备 GPU block 和 storage block 的小型调度器。"""
    block_size = 4
    scheduler_config = SchedulerConfig(
        "generate",
        max_num_batched_tokens=32,
        max_num_seqs=8,
        max_model_len=32,
        enable_chunked_prefill=True,
    )
    cache_config = CacheConfig(block_size, 1.0, 1, "auto")
    cache_config.num_cpu_blocks = 16
    cache_config.num_gpu_blocks = 16
    return AsyncKVScheduler(scheduler_config, cache_config, None)


def _sync_swap_out(scheduler: AsyncKVScheduler, seq_group) -> None:
    """只用于构造已有 SSD 副本；被测异步路径仍由 AsyncKVScheduler 执行。"""
    Scheduler._swap_out(scheduler, seq_group, [])


def _restore_with_clean_storage_replica(scheduler: AsyncKVScheduler,
                                        request_id: str):
    """构造 GPU 正式表和 SSD clean replica 同时存在的 decode 请求。"""
    seq, seq_group = create_dummy_prompt(
        request_id, prompt_length=8, block_size=4)
    scheduler._allocate_and_set_running(seq_group)
    _sync_swap_out(scheduler, seq_group)
    scheduler._add_seq_group_to_swapped(seq_group)
    scheduler._schedule_swapped(
        SchedulingBudget(token_budget=32, max_num_seqs=8), None)
    request = scheduler.drain_async_kv_transfers_to_submit()[0]
    scheduler.apply_async_kv_event(
        AsyncKVTransferEvent(request.request_id,
                             AsyncKVTransferState.READY))
    scheduler.complete_ready_async_kv_transfers()
    return seq, seq_group


def test_async_scheduler_requires_chunked_prefill():
    """异步恢复必须依赖有界的 chunk 调度边界。"""
    scheduler_config = SchedulerConfig(
        "generate",
        max_num_batched_tokens=32,
        max_num_seqs=8,
        max_model_len=32,
        enable_chunked_prefill=False,
    )
    cache_config = CacheConfig(4, 1.0, 1, "auto")
    cache_config.num_cpu_blocks = 16
    cache_config.num_gpu_blocks = 16
    with pytest.raises(ValueError, match="requires chunked prefill"):
        AsyncKVScheduler(scheduler_config, cache_config, None)


def test_swapped_request_waits_for_async_kv_ready():
    """KV 未 READY 时不计算，READY 后才进入下一轮 decode。"""
    scheduler = _create_scheduler()
    seq, seq_group = create_dummy_prompt(
        "1", prompt_length=8, block_size=4)

    # 先构造一个已经完成 prefill、随后被 swap-out 的 decode 请求。
    scheduler._allocate_and_set_running(seq_group)
    append_new_token_seq_group(8, seq_group, 99)
    _sync_swap_out(scheduler, seq_group)
    scheduler._add_seq_group_to_swapped(seq_group)
    assert seq.status == SequenceStatus.SWAPPED

    # 新策略只预留 GPU block 并生成异步请求，不能把该请求放进当前 batch。
    storage_block_ids = list(scheduler.block_manager.get_block_table(seq))
    free_gpu_before_reserve = scheduler.block_manager.get_num_free_gpu_blocks()
    output = scheduler._schedule_swapped(
        SchedulingBudget(token_budget=32, max_num_seqs=8), None)
    requests = scheduler.drain_async_kv_transfers_to_submit()
    assert not output.decode_seq_groups
    assert not output.prefill_seq_groups
    assert not output.blocks_to_swap_in
    assert len(requests) == 1
    assert requests[0].seq_group_id == seq_group.request_id
    assert requests[0].operation == AsyncKVTransferOperation.READ
    assert requests[0].block_mapping
    # reserve 阶段正式表仍指向 storage，目标 GPU block 只是被预留。
    assert scheduler.block_manager.get_block_table(seq) == storage_block_ids
    assert (scheduler.block_manager.get_num_free_gpu_blocks()
            < free_gpu_before_reserve)
    assert seq.status == SequenceStatus.SWAPPED
    assert scheduler.has_active_async_kv_transfer()
    assert scheduler.get_num_unfinished_seq_groups() == 1

    # PENDING 事件不得改变队列或 SequenceStatus。
    scheduler.apply_async_kv_event(
        AsyncKVTransferEvent(requests[0].request_id,
                             AsyncKVTransferState.PENDING))
    scheduler.complete_ready_async_kv_transfers()
    assert seq.status == SequenceStatus.SWAPPED
    assert not scheduler.running

    # READY 后先完成 swapped -> running，再由父类普通调度执行 decode。
    scheduler.apply_async_kv_event(
        AsyncKVTransferEvent(requests[0].request_id,
                             AsyncKVTransferState.READY))
    scheduler.complete_ready_async_kv_transfers()
    assert seq.status == SequenceStatus.RUNNING
    assert scheduler.block_manager.get_block_table(seq) != storage_block_ids
    assert list(scheduler.running) == [seq_group]
    assert not scheduler.loading

    metadata, scheduled = schedule_and_update_computed_tokens(scheduler)
    assert len(metadata) == 1
    assert [item.seq_group
            for item in scheduled.scheduled_seq_groups] == [seq_group]
    assert not scheduled.blocks_to_swap_in

    # READY -> RUNNING 的观测标记只在请求第一次进入执行 batch 前消费，
    # 重复消费不会产生第二个 first_execute 事件。
    markers = scheduler.consume_async_kv_execution_markers(
        [seq_group.request_id])
    assert len(markers) == 1
    assert markers[0].request_id == requests[0].request_id
    assert markers[0].seq_group_id == seq_group.request_id
    assert not scheduler.consume_async_kv_execution_markers(
        [seq_group.request_id])


def test_abort_loading_request_defers_block_free_until_ready():
    """loading 期间 abort 不得提前释放 MDS 正在写入的 GPU block。"""
    scheduler = _create_scheduler()
    seq, seq_group = create_dummy_prompt(
        "2", prompt_length=8, block_size=4)
    scheduler._allocate_and_set_running(seq_group)
    append_new_token_seq_group(8, seq_group, 100)
    _sync_swap_out(scheduler, seq_group)
    scheduler._add_seq_group_to_swapped(seq_group)

    scheduler._schedule_swapped(
        SchedulingBudget(token_budget=32, max_num_seqs=8), None)
    request = scheduler.drain_async_kv_transfers_to_submit()[0]

    scheduler.abort_seq_group(seq_group.request_id)
    assert seq.status == SequenceStatus.FINISHED_ABORTED
    assert request.request_id in scheduler.loading
    assert scheduler.has_unfinished_seqs()

    # MDS 完成后才真正释放 block 并移除 loading 生命周期。
    scheduler.apply_async_kv_event(
        AsyncKVTransferEvent(request.request_id,
                             AsyncKVTransferState.READY))
    scheduler.complete_ready_async_kv_transfers()
    assert not scheduler.loading
    assert not scheduler.running
    assert not scheduler.has_unfinished_seqs()


def test_ready_decode_runs_with_remaining_prefill_chunk():
    """READY decode 应在下一个边界与剩余 prefill chunk 一起调度。"""
    scheduler = _create_scheduler()

    restored_seq, restored_group = create_dummy_prompt(
        "10", prompt_length=8, block_size=4)
    scheduler._allocate_and_set_running(restored_group)
    append_new_token_seq_group(8, restored_group, 101)
    _sync_swap_out(scheduler, restored_group)
    scheduler._add_seq_group_to_swapped(restored_group)

    _, prefill_group = create_dummy_prompt(
        "11", prompt_length=40, block_size=4)
    scheduler.add_seq_group(prefill_group)

    # 第一轮提交异步恢复，同时只计算长 prompt 的第一个 32-token chunk。
    first_metadata, first_output = schedule_and_update_computed_tokens(
        scheduler)
    requests = scheduler.drain_async_kv_transfers_to_submit()
    assert len(requests) == 1
    assert [item.seq_group
            for item in first_output.scheduled_seq_groups] == [prefill_group]
    assert first_metadata[0].token_chunk_size == 32
    assert restored_seq.status == SequenceStatus.SWAPPED
    assert prefill_group.is_prefill()

    scheduler.apply_async_kv_event(
        AsyncKVTransferEvent(requests[0].request_id,
                             AsyncKVTransferState.READY))
    scheduler.complete_ready_async_kv_transfers()

    # 第二轮优先恢复 decode，但 32-token budget 仍足以容纳剩余 8-token
    # prefill；SchedulerOutputs 为 attention backend 保持 prefill 在前。
    second_metadata, second_output = schedule_and_update_computed_tokens(
        scheduler)
    scheduled_groups = [
        item.seq_group for item in second_output.scheduled_seq_groups
    ]
    assert scheduled_groups == [prefill_group, restored_group]
    assert second_output.num_prefill_groups == 1
    assert [metadata.token_chunk_size
            for metadata in second_metadata] == [8, 1]
    assert restored_seq.status == SequenceStatus.RUNNING


def test_async_swap_out_keeps_gpu_blocks_until_write_ready():
    """异步 write 完成前 GPU source 不得从正式 block table 释放。"""
    scheduler = _create_scheduler()
    seq, seq_group = create_dummy_prompt(
        "20", prompt_length=8, block_size=4)
    scheduler._allocate_and_set_running(seq_group)
    append_new_token_seq_group(8, seq_group, 202)

    gpu_block_ids = list(scheduler.block_manager.get_block_table(seq))
    free_gpu_before = scheduler.block_manager.get_num_free_gpu_blocks()
    scheduler._swap_out(seq_group, [])
    scheduler._add_seq_group_to_swapped(seq_group)
    request = scheduler.drain_async_kv_transfers_to_submit()[0]

    assert request.operation == AsyncKVTransferOperation.WRITE
    assert seq.status == SequenceStatus.SWAPPED
    assert scheduler.block_manager.get_block_table(seq) == gpu_block_ids
    assert scheduler.block_manager.get_num_free_gpu_blocks() == free_gpu_before
    assert request.request_id in scheduler.saving

    scheduler.apply_async_kv_event(
        AsyncKVTransferEvent(request.request_id,
                             AsyncKVTransferState.READY))
    scheduler.complete_ready_async_kv_transfers()

    assert request.request_id not in scheduler.saving
    assert scheduler.block_manager.get_block_table(seq) != gpu_block_ids
    assert scheduler.block_manager.get_num_free_gpu_blocks() > free_gpu_before
    assert list(scheduler.swapped) == [seq_group]


def test_async_writes_share_one_mds_slot():
    """多个 write 可以先 reserve，但 Worker/MDS 每次只接收一个。"""
    scheduler = _create_scheduler()
    groups = []
    for index in range(2):
        _, seq_group = create_dummy_prompt(
            str(30 + index), prompt_length=8, block_size=4)
        scheduler._allocate_and_set_running(seq_group)
        scheduler._swap_out(seq_group, [])
        scheduler._add_seq_group_to_swapped(seq_group)
        groups.append(seq_group)

    first = scheduler.drain_async_kv_transfers_to_submit()
    assert len(first) == 1
    assert scheduler.drain_async_kv_transfers_to_submit() == ()
    assert len(scheduler.async_kv_policy.queued_request_ids) == 1

    scheduler.apply_async_kv_event(
        AsyncKVTransferEvent(first[0].request_id,
                             AsyncKVTransferState.READY))
    scheduler.complete_ready_async_kv_transfers()
    second = scheduler.drain_async_kv_transfers_to_submit()
    assert len(second) == 1
    assert second[0].seq_group_id == groups[1].request_id


def test_abort_active_write_keeps_gpu_source_until_completion():
    """取消 write 时不能让父类提前释放 MDS 正在读取的 GPU 地址。"""
    scheduler = _create_scheduler()
    seq, seq_group = create_dummy_prompt(
        "40", prompt_length=8, block_size=4)
    scheduler._allocate_and_set_running(seq_group)
    gpu_block_ids = list(scheduler.block_manager.get_block_table(seq))
    scheduler._swap_out(seq_group, [])
    scheduler._add_seq_group_to_swapped(seq_group)
    request = scheduler.drain_async_kv_transfers_to_submit()[0]

    scheduler.abort_seq_group(seq_group.request_id)
    assert seq.status == SequenceStatus.FINISHED_ABORTED
    assert scheduler.block_manager.get_block_table(seq) == gpu_block_ids
    assert request.request_id in scheduler.saving

    scheduler.apply_async_kv_event(
        AsyncKVTransferEvent(request.request_id,
                             AsyncKVTransferState.READY))
    scheduler.complete_ready_async_kv_transfers()
    assert seq.seq_id not in scheduler.block_manager.block_tables
    assert not scheduler.has_unfinished_seqs()


def test_clean_storage_replica_avoids_write_io():
    """GPU 内容未变化时，swap-out 应直接复用 SSD replica。"""
    scheduler = _create_scheduler()
    seq, seq_group = _restore_with_clean_storage_replica(scheduler, "50")
    assert scheduler.block_manager.get_num_clean_storage_replicas(seq) == 2

    scheduler._swap_out(seq_group, [])
    scheduler._add_seq_group_to_swapped(seq_group)
    request = scheduler.drain_async_kv_transfers_to_submit()[0]
    assert request.operation == AsyncKVTransferOperation.WRITE
    assert request.block_mapping == ()

    scheduler.apply_async_kv_event(
        AsyncKVTransferEvent(request.request_id,
                             AsyncKVTransferState.READY))
    scheduler.complete_ready_async_kv_transfers()
    assert seq.status == SequenceStatus.SWAPPED
    assert scheduler.block_manager.get_num_free_gpu_blocks() == 16


def test_only_dirty_suffix_is_written():
    """decode 新增尾块后，只写 dirty suffix，不重写 clean prefix。"""
    scheduler = _create_scheduler()
    seq, seq_group = _restore_with_clean_storage_replica(scheduler, "51")
    append_new_token_seq_group(8, seq_group, 303)
    scheduler.block_manager.append_slots(seq, num_lookahead_slots=0)
    assert len(scheduler.block_manager.get_block_table(seq)) == 3

    scheduler._swap_out(seq_group, [])
    request = scheduler.drain_async_kv_transfers_to_submit()[0]
    assert len(request.block_mapping) == 1
    scheduler.apply_async_kv_event(
        AsyncKVTransferEvent(request.request_id,
                             AsyncKVTransferState.READY))
    scheduler.complete_ready_async_kv_transfers()
    assert len(scheduler.block_manager.get_block_table(seq)) == 3


def test_free_gpu_sequence_releases_storage_replicas():
    """sequence 结束时，正式 GPU 表和目录中的 SSD replica 都必须释放。"""
    scheduler = _create_scheduler()
    seq, _ = _restore_with_clean_storage_replica(scheduler, "52")
    assert scheduler.block_manager.get_num_free_cpu_blocks() == 14
    scheduler.free_seq(seq)
    assert scheduler.block_manager.get_num_free_cpu_blocks() == 16
    assert scheduler.block_manager.get_num_free_gpu_blocks() == 16


def test_queued_read_precedes_deferred_write():
    """active write 完成后，应先提交已排队的 critical read。"""
    scheduler = _create_scheduler()
    _, write_group = create_dummy_prompt(
        "60", prompt_length=8, block_size=4)
    scheduler._allocate_and_set_running(write_group)
    scheduler._swap_out(write_group, [])
    scheduler._add_seq_group_to_swapped(write_group)
    active_write = scheduler.drain_async_kv_transfers_to_submit()[0]

    _, read_group = create_dummy_prompt(
        "61", prompt_length=8, block_size=4)
    scheduler._allocate_and_set_running(read_group)
    _sync_swap_out(scheduler, read_group)
    scheduler._add_seq_group_to_swapped(read_group)
    scheduler._schedule_swapped(
        SchedulingBudget(token_budget=32, max_num_seqs=8), None)

    scheduler.apply_async_kv_event(
        AsyncKVTransferEvent(active_write.request_id,
                             AsyncKVTransferState.READY))
    scheduler.complete_ready_async_kv_transfers()
    next_request = scheduler.drain_async_kv_transfers_to_submit()[0]
    assert next_request.operation == AsyncKVTransferOperation.READ
    assert next_request.seq_group_id == read_group.request_id
