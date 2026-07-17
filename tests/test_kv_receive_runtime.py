from types import SimpleNamespace

import vllm.envs as envs
from vllm.distributed.kv_transfer.kv_connector.base import (
    KVReceiveResult, KVReceiveStatus)
from vllm.distributed.kv_transfer.kv_connector.lmcache_connector import (
    LMCacheConnector, _PendingDeferredRetrieveState)


def _make_model_input(
    *,
    request_ids=("req-a", ),
    seq_lens=(1024, ),
    query_lens=(256, ),
    finished_requests_ids=(),
    request_ids_to_seq_ids=None,
):
    if request_ids_to_seq_ids is None:
        request_ids_to_seq_ids = {
            request_id: [idx]
            for idx, request_id in enumerate(request_ids)
        }
    return SimpleNamespace(
        sampling_metadata=SimpleNamespace(
            seq_groups=[
                SimpleNamespace(request_id=request_id)
                for request_id in request_ids
            ]),
        request_ids_to_seq_ids=request_ids_to_seq_ids,
        seq_lens=list(seq_lens),
        query_lens=list(query_lens),
        finished_requests_ids=list(finished_requests_ids),
    )


def test_kv_receive_result_helpers_cover_three_runtime_statuses():
    model_input = _make_model_input()

    ready_forward = KVReceiveResult.ready_forward(model_input=model_input)
    assert ready_forward.status == KVReceiveStatus.READY_FORWARD
    assert ready_forward.bypass_model_exec is False

    ready_bypass = KVReceiveResult.ready_bypass(
        model_input=model_input,
        hidden_or_intermediate_states="hidden",
    )
    assert ready_bypass.status == KVReceiveStatus.READY_BYPASS
    assert ready_bypass.bypass_model_exec is True
    assert ready_bypass.hidden_or_intermediate_states == "hidden"

    deferred = KVReceiveResult.deferred(model_input=model_input)
    assert deferred.status == KVReceiveStatus.DEFERRED
    assert deferred.bypass_model_exec is False
    assert deferred.hidden_or_intermediate_states is None


def test_lmcache_connector_build_receive_batch_key_uses_request_and_length_context():
    connector = LMCacheConnector.__new__(LMCacheConnector)

    key_a = connector._build_receive_batch_key(
        _make_model_input(
            request_ids=("req-a", "req-b"),
            seq_lens=(1024, 768),
            query_lens=(256, 128),
        ))
    key_b = connector._build_receive_batch_key(
        _make_model_input(
            request_ids=("req-a", "req-b"),
            seq_lens=(1024, 512),
            query_lens=(256, 128),
        ))

    assert key_a[0] == ("req-a", "req-b")
    assert key_a != key_b


def test_lmcache_connector_build_receive_batch_key_prefers_request_ids_to_seq_ids():
    connector = LMCacheConnector.__new__(LMCacheConnector)
    model_input = _make_model_input(
        request_ids=("display-only-a", "display-only-b"),
        request_ids_to_seq_ids={
            "real-req-a": [7],
            "real-req-b": [8],
        },
    )

    batch_key = connector._build_receive_batch_key(model_input)

    assert batch_key[0] == ("real-req-a", "real-req-b")


def test_lmcache_connector_cleanup_finished_pending_retrieves():
    connector = LMCacheConnector.__new__(LMCacheConnector)
    key_keep = (("req-a", ), (1024, ), (256, ))
    key_drop = (("req-b", "req-c"), (2048, 1024), (256, 256))
    connector._pending_deferred_retrieves = {
        key_keep:
        _PendingDeferredRetrieveState(
            batch_key=key_keep,
            deferred_batch=object(),
            created_at_s=1.0,
        ),
        key_drop:
        _PendingDeferredRetrieveState(
            batch_key=key_drop,
            deferred_batch=object(),
            created_at_s=2.0,
        ),
    }

    connector._cleanup_finished_pending_retrieves(
        _make_model_input(finished_requests_ids=("req-c", )))

    assert key_keep in connector._pending_deferred_retrieves
    assert key_drop not in connector._pending_deferred_retrieves


def test_lmcache_connector_forced_min_defer_polls():
    connector = LMCacheConnector.__new__(LMCacheConnector)
    connector.cache_config = object()
    connector._pending_deferred_retrieves = {}
    connector.engine = object()
    connector.lmcache_start_deferrable_retrieve_kv = (
        lambda *args, **kwargs: "deferred-batch")
    connector.lmcache_poll_deferrable_retrieve_kv = (
        lambda deferred_batch: True)
    connector.lmcache_finalize_deferrable_retrieve_kv = (
        lambda deferred_batch, **kwargs: (
            kwargs["model_input"], False, None))
    connector._runtime_defer_enabled = lambda: True

    old_flag = envs.VLLM_BAM_DIRECT_PLACEMENT_DEFER_MIN_POLLS
    envs.VLLM_BAM_DIRECT_PLACEMENT_DEFER_MIN_POLLS = 1
    try:
        model_input = _make_model_input(request_ids=("req-a", ))
        first_result = connector._try_drive_deferred_receive(
            model_executable=object(),
            model_input=model_input,
            kv_caches=[],
            retrieve_status=[],
        )
        assert first_result is not None
        assert first_result.status == KVReceiveStatus.DEFERRED
        assert len(connector._pending_deferred_retrieves) == 1

        second_result = connector._try_drive_deferred_receive(
            model_executable=object(),
            model_input=model_input,
            kv_caches=[],
            retrieve_status=[],
        )
        assert second_result is not None
        assert second_result.status == KVReceiveStatus.READY_FORWARD
        assert len(connector._pending_deferred_retrieves) == 0
    finally:
        envs.VLLM_BAM_DIRECT_PLACEMENT_DEFER_MIN_POLLS = old_flag


def test_lmcache_connector_ready_batch_does_not_force_defer_when_min_polls_zero():
    """验证当前主线语义：ready 就直接 finalize，不再人为多 defer 一轮。

    这条用例对应现在的默认实验口径：

    - runtime defer 能力打开；
    - request handle 三段式路径生效；
    - 但 `min_defer_polls=0`，因此只有真实 `not_ready` 才应该返回
      `KVReceiveStatus.DEFERRED`。

    如果这条用例失败，就说明“为了验证跨 iteration 而强制多 defer 一轮”的旧
    行为又漏回主线了。
    """
    connector = LMCacheConnector.__new__(LMCacheConnector)
    connector.cache_config = object()
    connector._pending_deferred_retrieves = {}
    connector.engine = object()
    connector.lmcache_start_deferrable_retrieve_kv = (
        lambda *args, **kwargs: "deferred-batch")
    connector.lmcache_poll_deferrable_retrieve_kv = (
        lambda deferred_batch: True)
    connector.lmcache_finalize_deferrable_retrieve_kv = (
        lambda deferred_batch, **kwargs: (
            kwargs["model_input"], False, None))
    connector._runtime_defer_enabled = lambda: True

    old_flag = envs.VLLM_BAM_DIRECT_PLACEMENT_DEFER_MIN_POLLS
    envs.VLLM_BAM_DIRECT_PLACEMENT_DEFER_MIN_POLLS = 0
    try:
        model_input = _make_model_input(request_ids=("req-a", ))
        result = connector._try_drive_deferred_receive(
            model_executable=object(),
            model_input=model_input,
            kv_caches=[],
            retrieve_status=[],
        )

        assert result is not None
        assert result.status == KVReceiveStatus.READY_FORWARD
        assert len(connector._pending_deferred_retrieves) == 0
    finally:
        envs.VLLM_BAM_DIRECT_PLACEMENT_DEFER_MIN_POLLS = old_flag
