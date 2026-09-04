#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${VLLM_PAPER_REPRODUCTION_COMMAND:-}" ]]; then
  printf '%s\n' \
    'Set VLLM_PAPER_REPRODUCTION_COMMAND to the existing local vLLM launch command.' \
    'This wrapper intentionally does not invent a model, SSD image, or backend command.' >&2
  exit 2
fi

exec bash -lc "$VLLM_PAPER_REPRODUCTION_COMMAND"
