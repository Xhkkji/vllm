#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
python -m evaluation.paper_reproduction.common.runner.plan_smoke \
  --strategy quest \
  --output evaluation/paper_reproduction/granulekv_joint/results/smoke/plan.json
