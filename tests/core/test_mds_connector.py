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
        self.prefetch_plans = {}

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

    def register_prefetch_plan(self, plan_id, units) -> None:
        self.prefetch_plans[plan_id] = dict(units)

    def activate_prefetch_units(self, plan_id, unit_ids):
        handles = []
        for unit_id in unit_ids:
            payload, _operation = self.prefetch_plans[plan_id][unit_id]
            self.submitted_payloads.append(payload)
            handles.append(self._submit())
        return tuple(handles)

    def finish_prefetch_unit(self, plan_id, unit_id) -> int:
        del self.prefetch_plans[plan_id][unit_id]
        handle = next(handle for handle in self.live
                      if handle in self.ready)
        self.live.remove(handle)
        return 123

    def discard_prefetch_units(self, plan_id, unit_ids) -> None:
        for unit_id in unit_ids:
            del self.prefetch_plans[plan_id][unit_id]

    def release_prefetch_plan(self, plan_id) -> None:
        assert not self.prefetch_plans[plan_id]
        del self.prefetch_plans[plan_id]


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


def test_connector_stages_and_releases_prefetch_plan():
    connector = BaMMDSConnector.__new__(BaMMDSConnector)
    connector.client = _FakeMDSClient()
    connector._pending_transfers = {}
    connector._prefetch_templates = {}
    connector.layout = type("Layout", (), {"num_layers": 8})()

    first = torch.tensor([[7, 3]], dtype=torch.int64)
    second = torch.tensor([[8, 4]], dtype=torch.int64)
    connector.stage_prefetch_plan(
        "plan-0", (("unit-0", first, "read", (0, 4)),
                   ("unit-1", second, "read", (4, 8))))
    assert not connector.client.submitted_payloads

    assert not connector.activate_prefetch_transfer_async(
        "plan-0", "unit-0", first, operation="read", layer_range=(0, 4))
    pending = connector._pending_transfers["unit-0"]
    connector.client.ready.add(pending.handle.request_id)
    assert connector.poll_transfer_async("unit-0")
    assert "unit-1" in connector._prefetch_templates

    connector.discard_staged_prefetch_units(("unit-1", ))
    assert not connector._prefetch_templates
    assert not connector.client.prefetch_plans
