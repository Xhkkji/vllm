# Bidaw-Style Scheduler

This adapter only ranks request descriptions with a ready/preparing and
Disk-HRRN-style order. It deliberately excludes host eviction, ghost cache,
answer-length prediction, and intermediate tensor caching from the first
reproduction pass. The GranuleKV request lifecycle remains unchanged.
