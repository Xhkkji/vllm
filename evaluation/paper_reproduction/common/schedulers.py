"""Request ordering policies with no ownership of I/O lifecycle."""

from __future__ import annotations

from typing import Sequence, Tuple

from .plans import RequestScheduler, RequestWorkItem


class FCFSScheduler:
    name = "fcfs"

    def order(self, items: Sequence[RequestWorkItem]) -> Tuple[RequestWorkItem,
                                                               ...]:
        return tuple(items)


class DiskHRRNScheduler:
    """Lightweight ready/preparing ordering for a Bidaw-style baseline.

    This ranks descriptions only.  Admission, submit, query, complete, and
    cancel remain owned by the existing AsyncKVTransferQueue/Connector.
    """

    name = "disk_hrrn"

    def order(self, items: Sequence[RequestWorkItem]) -> Tuple[RequestWorkItem,
                                                               ...]:
        def score(item: RequestWorkItem) -> float:
            service = max(item.estimated_service_us, 1.0)
            return (max(item.ready_deadline_us, 0.0) + service) / service

        return tuple(sorted(items, key=score, reverse=True))
