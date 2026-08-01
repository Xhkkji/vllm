#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""生成稳定的长上下文并发请求，用于观察 vLLM V0 swap 路径。"""

from __future__ import annotations

import argparse
import atexit
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, TextIO

from transformers import AutoTokenizer

# 允许直接以脚本路径运行，无需预先 `pip install -e .`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class TeeStream:
    """把终端输出同步写入日志文件，便于后续回看 swap 轨迹。"""

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)()
                   for stream in self.streams)


def _sanitize_name(name: str) -> str:
    safe = []
    for ch in name:
        safe.append(ch if ch.isalnum() or ch in ("-", "_", ".") else "_")
    return "".join(safe).strip("_") or "model"


def setup_file_logging(log_dir: Path, model_name: str) -> Path:
    """将 stdout/stderr 同步写入 evaluation/logs 下的文件。"""
    log_dir.mkdir(parents=True, exist_ok=True)
    model_stub = _sanitize_name(Path(model_name).name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"v0_swap_trace_{model_stub}_{timestamp}.log"

    log_file = log_path.open("w", encoding="utf-8")
    atexit.register(log_file.close)

    sys.stdout = TeeStream(sys.stdout, log_file)
    sys.stderr = TeeStream(sys.stderr, log_file)
    print(f"Logs will be written to: {log_path}", flush=True)
    return log_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在 V100/V0 路径上制造 KV cache 压力，观察 swap 行为。")
    parser.add_argument("model", help="模型名或本地模型路径")
    parser.add_argument("--tokenizer",
                        default=None,
                        help="可选 tokenizer 路径，默认与模型相同")
    parser.add_argument("--prompt-len",
                        type=int,
                        default=2048,
                        help="每个请求的 prompt token 长度")
    parser.add_argument("--num-prompts",
                        type=int,
                        default=32,
                        help="并发请求数")
    parser.add_argument("--max-tokens",
                        type=int,
                        default=16,
                        help="每个请求继续生成的 token 数")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="采样温度。默认 0.0 为 greedy；>0 配合 best_of 更容易制造多序列压力。")
    parser.add_argument("--max-model-len",
                        type=int,
                        default=4096,
                        help="传给 vLLM 的最大上下文长度")
    parser.add_argument("--gpu-memory-utilization",
                        type=float,
                        default=0.6,
                        help="vLLM 的 gpu_memory_utilization")
    parser.add_argument("--swap-space",
                        type=float,
                        default=8.0,
                        help="vLLM 的 swap_space，单位 GiB")
    parser.add_argument("--dtype",
                        default="auto",
                        help="传给 vLLM 的 dtype，例如 auto/half")
    parser.add_argument("--tensor-parallel-size",
                        type=int,
                        default=1,
                        help="张量并行数，V100 单卡实验通常保持 1")
    parser.add_argument("--max-num-seqs",
                        type=int,
                        default=None,
                        help="可选，显式限制 scheduler 的最大并发序列数")
    parser.add_argument("--max-num-batched-tokens",
                        type=int,
                        default=None,
                        help="可选，显式限制 scheduler 的单步 token budget")
    parser.add_argument(
        "--num-gpu-blocks-override",
        type=int,
        default=None,
        help=("可选，覆盖 profiling 得到的 GPU KV block 数。用于以确定方式制造 "
              "swap/preemption 压力，不影响未设置该参数的历史 baseline。"))
    parser.add_argument("--device",
                        default="cuda",
                        help="显式指定 vLLM 设备类型，V100 实验建议设为 cuda")
    parser.add_argument("--log-dir",
                        default=str(REPO_ROOT / "evaluation" / "logs"),
                        help="日志目录，默认写到 evaluation/logs")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="随机种子")
    parser.add_argument("--trust-remote-code",
                        action="store_true",
                        help="是否信任远程自定义代码")
    parser.add_argument("--enforce-eager",
                        action="store_true",
                        help="是否强制 eager 模式")
    parser.add_argument(
        "--preemption-mode",
        choices=["auto", "swap", "recompute"],
        default="auto",
        help="V0 抢占模式。auto 使用 vLLM 默认策略；swap 可强制走换入换出。")
    parser.add_argument("--n",
                        type=int,
                        default=1,
                        help="每个请求最终返回的候选数量")
    parser.add_argument(
        "--best-of",
        type=int,
        default=1,
        help="每个请求内部保留的候选数量。>1 时会形成多序列组，更容易触发 swap。")
    parser.add_argument("--disable-async-output-proc",
                        action="store_true",
                        default=True,
                        help="关闭 async output processor，避免未识别平台报错")
    return parser.parse_args()


def build_token_prompts(tokenizer: AutoTokenizer, prompt_len: int,
                        num_prompts: int) -> List[dict[str, List[int]]]:
    """构造固定长度 token prompts，避免字符串分词长度不稳定。"""
    base_text = (
        "This is a long context stress test for vLLM swap tracing. "
        "We repeat structured content so that the tokenizer output length "
        "is predictable and the KV cache pressure is easier to control. ")
    base_ids = tokenizer.encode(base_text, add_special_tokens=False)
    if not base_ids:
        raise ValueError("Tokenizer returned empty base token ids.")

    prompts: List[dict[str, List[int]]] = []
    for request_idx in range(num_prompts):
        token_ids: List[int] = []
        while len(token_ids) < prompt_len:
            token_ids.extend(base_ids)

        # 在尾部加入少量请求编号信息，避免所有请求完全相同。
        suffix_ids = tokenizer.encode(f" request-{request_idx} ",
                                      add_special_tokens=False)
        suffix_keep = min(len(suffix_ids), prompt_len)
        token_ids = token_ids[:prompt_len]
        token_ids[-suffix_keep:] = suffix_ids[:suffix_keep]
        # 当前 vLLM 版本要求显式使用 TokensPrompt，而不是直接传裸 token id 列表。
        prompts.append({"prompt_token_ids": token_ids})
    return prompts


def close_runtime(llm: object) -> None:
    """显式关闭实验 runtime，避免 Python finalization 阶段遗留 GPU service。

    普通 vLLM 路径没有 ``bam_direct_kv_store``，此函数会直接跳过；新 direct
    backend 则必须先停止 persistent CQ worker、解除 DMA mapping，再销毁 NCCL
    process group。该顺序与生产进程的显式 shutdown 语义一致。
    """
    llm_engine = getattr(llm, "llm_engine", None)
    model_executor = getattr(llm_engine, "model_executor", None)
    driver_worker = getattr(model_executor, "driver_worker", None)
    worker = getattr(driver_worker, "worker", None)
    for cache_engine in getattr(worker, "cache_engine", ()):
        direct_store = getattr(cache_engine, "bam_direct_kv_store", None)
        if direct_store is not None:
            direct_store.close()

    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
    cleanup_dist_env_and_memory()


def main() -> None:
    args = parse_args()
    setup_file_logging(Path(args.log_dir), args.model)

    # 这套实验只针对 V0 路径；如果外部没设置，这里给出安全默认值。
    os.environ.setdefault("VLLM_USE_V1", "0")

    from vllm import LLM, SamplingParams

    tokenizer_name = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    prompts = build_token_prompts(tokenizer, args.prompt_len,
                                  args.num_prompts)

    sampling_params = SamplingParams(
        n=args.n,
        best_of=args.best_of if args.best_of > 1 else None,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    print("=" * 80)
    print("V0 swap trace evaluation")
    print(f"model={args.model}")
    print(f"tokenizer={tokenizer_name}")
    print(f"num_prompts={args.num_prompts}")
    print(f"prompt_len={args.prompt_len}")
    print(f"max_tokens={args.max_tokens}")
    print(f"temperature={args.temperature}")
    print(f"max_model_len={args.max_model_len}")
    print(f"gpu_memory_utilization={args.gpu_memory_utilization}")
    print(f"swap_space={args.swap_space} GiB")
    print(f"device={args.device}")
    print(f"preemption_mode={args.preemption_mode}")
    print(f"n={args.n}")
    print(f"best_of={args.best_of}")
    print(f"max_num_seqs={args.max_num_seqs}")
    print(f"max_num_batched_tokens={args.max_num_batched_tokens}")
    print(f"num_gpu_blocks_override={args.num_gpu_blocks_override}")
    print(
        f"disable_async_output_proc={args.disable_async_output_proc}")
    print(f"VLLM_USE_V1={os.environ.get('VLLM_USE_V1')}")
    print(f"VLLM_V0_SWAP_TRACE={os.environ.get('VLLM_V0_SWAP_TRACE', '0')}")
    print("=" * 80)

    llm_kwargs = dict(
        model=args.model,
        tokenizer=tokenizer_name,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        swap_space=args.swap_space,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        disable_async_output_proc=args.disable_async_output_proc,
    )
    # 只在显式指定时覆盖 vLLM 默认 scheduler 配置，方便做对照实验。
    if args.preemption_mode != "auto":
        llm_kwargs["preemption_mode"] = args.preemption_mode
    if args.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs
    if args.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.num_gpu_blocks_override is not None:
        llm_kwargs["num_gpu_blocks_override"] = args.num_gpu_blocks_override

    llm = LLM(**llm_kwargs)

    start_time = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed_s = time.perf_counter() - start_time

    total_prompt_tokens = args.num_prompts * args.prompt_len
    total_generated_tokens = sum(
        len(candidate.token_ids)
        for output in outputs
        for candidate in output.outputs)

    print("=" * 80)
    print("Run summary")
    print(f"elapsed={elapsed_s:.3f}s")
    print(f"total_prompt_tokens={total_prompt_tokens}")
    print(f"total_generated_tokens={total_generated_tokens}")
    if elapsed_s > 0:
        print(f"prompt_tokens_per_sec={total_prompt_tokens / elapsed_s:.2f}")
        print(
            f"generated_tokens_per_sec={total_generated_tokens / elapsed_s:.2f}"
        )
    print("=" * 80)

    # 打印极少量结果，确认请求确实成功返回。
    preview_count = min(2, len(outputs))
    for idx in range(preview_count):
        text = outputs[idx].outputs[0].text
        print(f"[preview {idx}] {text[:200]!r}")

    close_runtime(llm)


if __name__ == "__main__":
    main()
