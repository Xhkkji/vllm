#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
python -m evaluation.paper_reproduction.common.runner.plan_smoke \
  --strategy solidattention \
  --output evaluation/paper_reproduction/solidattention/results/smoke/plan.json
