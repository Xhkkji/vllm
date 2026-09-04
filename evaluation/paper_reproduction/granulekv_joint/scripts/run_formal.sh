#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export VLLM_PAPER_STRATEGY=granulekv_joint
export VLLM_PAPER_RESULT_DIR=evaluation/paper_reproduction/granulekv_joint/results/formal
bash evaluation/paper_reproduction/common/runner/run_formal.sh
