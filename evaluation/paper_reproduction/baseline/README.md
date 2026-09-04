# Fixed Baseline

The baseline keeps the model, prefix, SSD image, HBM capacity, and request
workload constant. It compares a blocking GranuleKV restore with the same
payload restored through layer-wise asynchronous requests. Both paths use the
production GranuleKV connector and native daemon; no synthetic timing or
second I/O protocol is introduced.

## Plan smoke

Use the `pytorch-vllm` interpreter explicitly:

```bash
PYTHON_BIN=/home/xhk/miniconda3/envs/pytorch-vllm/bin/python \
  bash evaluation/paper_reproduction/baseline/scripts/run_smoke.sh
```

## Real transport comparison

The real runner creates independent control directories and daemon logs for
sync and async. It reuses the canonical root-owned MPS instance and refuses
to overwrite an existing result directory:

```bash
RUN_ID=YYYYMMDD_granulekv_transport_formal \
  bash evaluation/paper_reproduction/baseline/scripts/run_real_transport.sh
```

Before running, the matching root MPS instance must be available:

```bash
bash /home/xhk/llm-inference/GranuleKV/gids_module/start_granulekv_mps.sh
bash /home/xhk/llm-inference/GranuleKV/gids_module/check_granulekv_mps.sh
```

The runner executes `GPU -> SSD write -> GPU clear -> SSD -> GPU restore ->
CUDA compute -> payload verify` for both backends. `RESULTS.md` is generated
only when both summaries are verified and both daemon logs contain real
`GranuleKV_BATCH_DONE` write/read markers. `overlap_ms` is the intersection of
request submit-to-ready intervals and CUDA compute intervals; it demonstrates
overlap, but is not by itself a model-level speedup claim.

The first validated runs are kept under `results/` locally. The small smoke
run validates the lifecycle and marker path. The formal run validates the
larger payload and reports the measured end-to-end result, including cases
where asynchronous overlap exists without an end-to-end speedup.
