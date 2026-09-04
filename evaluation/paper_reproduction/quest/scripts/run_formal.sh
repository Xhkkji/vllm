#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export VLLM_PAPER_STRATEGY=quest
export VLLM_PAPER_RESULT_DIR=evaluation/paper_reproduction/quest/results/formal
bash evaluation/paper_reproduction/common/runner/run_formal.sh
