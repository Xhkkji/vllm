from types import SimpleNamespace
from vllm.distributed.kv_transfer.kv_connector.base import (
    KVReceiveResult, KVReceiveStatus)


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
