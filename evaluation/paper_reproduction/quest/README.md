# Quest Adapter

This directory isolates the selector experiment. It emits the existing
`SparseKVAccessPlan` and uses the existing layer-window planner; it does not
copy a Quest runtime or change GranuleKV. The proxy selector is only a smoke
path. A faithful run must provide per-layer choices through
`selected_blocks_by_layer` and report selector recall and attention error.

Run the plan-only smoke check from the repository root:

```bash
python -m evaluation.paper_reproduction.common.runner.plan_smoke \
  --strategy quest --output evaluation/paper_reproduction/quest/results/smoke/plan.json
```

Use `--prefetcher on_demand` for Quest synchronous/on-demand comparison and
`--prefetcher layer_wise --window N` for asynchronous layer prefetch.
