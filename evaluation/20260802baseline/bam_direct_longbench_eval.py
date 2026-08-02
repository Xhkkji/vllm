#!/usr/bin/env python3
"""Run real LongBench prompts through the vLLM V0 BaM direct KVStore path."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.v0_swap_trace_eval import close_runtime, setup_file_logging


DEFAULT_MODEL = "/home/xhk/llm-inference/models/Qwen2.5-7B-Instruct"
DEFAULT_MANIFEST = (
    "/home/xhk/llm-inference/datasets/longbench/organized/triviaqa/"
    "qwen25/full/buckets/lt4k.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real LongBench prompts through BaM direct KVStore."
    )
    parser.add_argument("model")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--num-samples", type=int, default=25)
    parser.add_argument("--prompt-len", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--best-of", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    parser.add_argument("--swap-space", type=float, default=4.0)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--num-gpu-blocks-override", type=int, default=260)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--dtype", default="half")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def load_rows(manifest: str, limit: int) -> list[dict]:
    rows: list[dict] = []
    with Path(manifest).open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("prompt"):
                raise ValueError("LongBench row has no non-empty 'prompt' field")
            rows.append(row)
            if limit > 0 and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"No rows loaded from {manifest}")
    return rows


def fixed_length_prompt(tokenizer: AutoTokenizer, text: str, length: int,
                        index: int) -> dict[str, list[int]]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"Tokenizer returned no tokens for sample {index}")

    # 保留真实文本内容；短样本只重复其自身 token，避免引入合成 stress 文本。
    repeated = (token_ids * ((length + len(token_ids) - 1) // len(token_ids)))
    token_ids = repeated[:length]
    return {"prompt_token_ids": token_ids}


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir)
    setup_file_logging(log_dir, args.model)
    os.environ.setdefault("VLLM_USE_V1", "0")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=False, use_fast=True
    )
    rows = load_rows(args.manifest, args.num_samples)
    prompts = [
        fixed_length_prompt(tokenizer, row["prompt"], args.prompt_len, index)
        for index, row in enumerate(rows)
    ]

    from vllm import LLM, SamplingParams

    sampling_params = SamplingParams(
        n=1,
        best_of=args.best_of,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    print("=" * 80)
    print("BaM direct LongBench evaluation")
    print(f"model={args.model}")
    print(f"manifest={args.manifest}")
    print(f"num_samples={len(rows)}")
    print(f"prompt_len={args.prompt_len}")
    print(f"max_tokens={args.max_tokens}")
    print(f"best_of={args.best_of}")
    print(f"max_num_seqs={args.max_num_seqs}")
    print(f"num_gpu_blocks_override={args.num_gpu_blocks_override}")
    print("=" * 80)

    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        tensor_parallel_size=1,
        dtype=args.dtype,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        swap_space=args.swap_space,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        disable_async_output_proc=True,
        preemption_mode="swap",
        max_num_seqs=args.max_num_seqs,
        num_gpu_blocks_override=args.num_gpu_blocks_override,
    )

    try:
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        elapsed = time.perf_counter() - start

        total_prompt_tokens = len(prompts) * args.prompt_len
        total_generated_tokens = sum(
            len(candidate.token_ids)
            for output in outputs
            for candidate in output.outputs
        )
        print("=" * 80)
        print("Run summary")
        print(f"elapsed={elapsed:.3f}s")
        print(f"total_prompt_tokens={total_prompt_tokens}")
        print(f"total_generated_tokens={total_generated_tokens}")
        if elapsed > 0:
            print(f"prompt_tokens_per_sec={total_prompt_tokens / elapsed:.2f}")
            print(
                "generated_tokens_per_sec="
                f"{total_generated_tokens / elapsed:.2f}"
            )
        print("=" * 80)
    finally:
        close_runtime(llm)


if __name__ == "__main__":
    main()
