"""Bidaw-style ready/preparing ordering over common request descriptions."""

from ...common.schedulers import DiskHRRNScheduler


class BidawScheduler(DiskHRRNScheduler):
    name = "bidaw_style_disk_hrrn"
