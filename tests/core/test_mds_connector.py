# SPDX-License-Identifier: Apache-2.0

"""MDS connector 多 request-slot 控制面的无 GPU 单元测试。"""

from dataclasses import dataclass

import pytest
import torch

from vllm.bam.mds.connector import BaMMDSConnector


@dataclass(frozen=True)
class _FakeHandle:
    request_id: int


class _FakeMDSClient:
    def __init__(self) -> None:
        self.next_request_id = 1
        self.ready: set[int] = set()
        self.live: set[int] = set()
        self.submitted_payloads = []

    def submit_read_async(self, payload):
        self.submitted_payloads.append(payload)
        return self._submit()

    def submit_write_async(self, payload):
        self.submitted_payloads.append(payload)
        return self._submit()

    def _submit(self):
        handle = _FakeHandle(self.next_request_id)
        self.next_request_id += 1
        self.live.add(handle.request_id)
        return handle

    def poll(self, handle) -> bool:
        assert handle.request_id in self.live
        return handle.request_id in self.ready

    def finish(self, handle) -> int:
        self.live.remove(handle.request_id)
        return 123

    def discard(self, handle) -> None:
        self.live.remove(handle.request_id)


def test_connector_tracks_multiple_out_of_order_transfers():
    connector = BaMMDSConnector.__new__(BaMMDSConnector)
    connector.client = _FakeMDSClient()
    connector._pending_transfers = {}

    first_mapping = torch.tensor([[1, 11]], dtype=torch.int64)
    second_mapping = torch.tensor([[2, 12]], dtype=torch.int64)
    assert not connector.submit_transfer_async(
        "scheduler-1", first_mapping, operation="write")
    assert not connector.submit_transfer_async(
        "scheduler-2", second_mapping, operation="read")
    assert set(connector._pending_transfers) == {"scheduler-1", "scheduler-2"}

    second = connector._pending_transfers["scheduler-2"]
    connector.client.ready.add(second.handle.request_id)
    assert connector.poll_transfer_async("scheduler-2")
    assert set(connector._pending_transfers) == {"scheduler-1"}
    assert not connector.poll_transfer_async("scheduler-1")


def test_connector_forwards_and_validates_layer_range():
    connector = BaMMDSConnector.__new__(BaMMDSConnector)
    connector.client = _FakeMDSClient()
    connector._pending_transfers = {}
    connector.layout = type("Layout", (), {"num_layers": 8})()

    mapping = torch.tensor([[7, 3]], dtype=torch.int64)
    assert not connector.submit_transfer_async(
        "window-0", mapping, operation="read", layer_range=(2, 6))
    assert connector.client.submitted_payloads == [{
        "gpu_block_ids": [3],
        "storage_block_ids": [7],
        "layer_start": 2,
        "layer_end": 6,
    }]
    with pytest.raises(ValueError, match="outside local KV cache"):
        connector.submit_transfer_async(
            "window-bad", mapping, operation="read", layer_range=(6, 9))
