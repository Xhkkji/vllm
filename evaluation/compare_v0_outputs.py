#!/usr/bin/env python3
"""逐请求比较两个 v0_swap_trace_eval 结构化输出。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    return parser.parse_args()


def load_output(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        payload = json.load(source)
    if payload.get("schema_version") != 1:
        raise RuntimeError(f"unsupported output schema: {path}")
    return payload


def main() -> None:
    args = parse_args()
    reference = load_output(args.reference)
    candidate = load_output(args.candidate)
    config_fields = ("model", "seed", "temperature", "n", "best_of",
                     "prompt_len", "max_tokens")
    for field in config_fields:
        if reference[field] != candidate[field]:
            raise RuntimeError(
                f"configuration mismatch: {field}: "
                f"{reference[field]!r} != {candidate[field]!r}")

    reference_requests = reference["requests"]
    candidate_requests = candidate["requests"]
    if len(reference_requests) != len(candidate_requests):
        raise RuntimeError("request count mismatch")
    compared_tokens = 0
    for request_index, (expected, actual) in enumerate(
            zip(reference_requests, candidate_requests)):
        if (expected["prompt_token_ids_sha256"]
                != actual["prompt_token_ids_sha256"]):
            raise RuntimeError(f"prompt mismatch at request {request_index}")
        if len(expected["candidates"]) != len(actual["candidates"]):
            raise RuntimeError(
                f"candidate count mismatch at request {request_index}")
        for candidate_index, (expected_candidate,
                              actual_candidate) in enumerate(
                                  zip(expected["candidates"],
                                      actual["candidates"])):
            expected_ids = expected_candidate["token_ids"]
            actual_ids = actual_candidate["token_ids"]
            if expected_ids != actual_ids:
                first_difference = next(
                    (index for index, values in enumerate(
                        zip(expected_ids, actual_ids))
                     if values[0] != values[1]),
                    min(len(expected_ids), len(actual_ids)))
                raise RuntimeError(
                    "token mismatch at "
                    f"request={request_index} candidate={candidate_index} "
                    f"token={first_difference}")
            compared_tokens += len(expected_ids)

    if reference["token_ids_sha256"] != candidate["token_ids_sha256"]:
        raise RuntimeError("token digest mismatch after element-wise comparison")
    print(
        "V0_OUTPUT_CONSISTENCY_PASS "
        f"requests={len(reference_requests)} tokens={compared_tokens} "
        f"sha256={reference['token_ids_sha256']}")


if __name__ == "__main__":
    main()
