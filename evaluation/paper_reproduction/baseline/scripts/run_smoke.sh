#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PYTHON_BIN="${PYTHON_BIN:-/home/xhk/miniconda3/envs/pytorch-vllm/bin/python}"
"${PYTHON_BIN}" -m evaluation.paper_reproduction.common.runner.plan_smoke \
  --strategy dense \
  --prefetcher on_demand \
  --output evaluation/paper_reproduction/baseline/results/smoke/plan.json
