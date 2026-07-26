#!/usr/bin/env python3
"""Run LongBench-TriviaQA through the LMCache SSD baseline path.

这个 runner 专门服务当前 SSD-backend baseline：

1. 读取已经组织好的 LongBench TriviaQA manifest；
2. 每条样本默认连续跑两次同一个 prompt；
3. request_1 用来写入/建立 LMCache SSD 数据；
4. request_2 用来触发从 SSD/GDS 读回 KV chunk；
5. 输出逐样本 JSONL 指标，便于后续和 BaM one-copy 对比。

注意：
这里不把 LongBench 直接接进 vLLM 通用 benchmark dataset 框架，是为了保持
LMCache/BaM 这条实验链路的控制语义清楚：每个样本都明确有 write/read 两
个阶段，并且可以单独观察 request_2 的 SSD-backed KV restore 开销。
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Iterable

from lmcache.integration.vllm.utils import ENGINE_NAME
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

try:
    from lmcache.experimental.cache_engine import LMCacheEngineBuilder
except ImportError:  # pragma: no cover - 兼容不同 LMCache 版本
    from lmcache.v1.cache_engine import LMCacheEngineBuilder


DEFAULT_MODEL = "/home/xhk/llm-inference/models/Qwen2.5-7B-Instruct"
DEFAULT_MANIFEST = (
    "/home/xhk/llm-inference/datasets/longbench/organized/triviaqa/"
    "qwen25/full/buckets/lt4k.jsonl"
)

_LOGGER_HANDLE_PATCHED = False


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def install_benchmark_log_filter(debug_log: bool) -> None:
    """默认压掉底层调试 INFO，只保留性能/告警/错误相关日志。

    这里故意放在 benchmark runner 层做过滤，而不是改 BaM/vLLM 的核心逻辑：
    - 不影响实际调用链路和数据搬运正确性；
    - 需要排错时设置 `LONGBENCH_DEBUG_LOG=1` 即可恢复完整日志；
    - 默认只保留每个 request/iter 的端到端耗时，以及 BaM 侧单行性能摘要。
    """
    global _LOGGER_HANDLE_PATCHED
    if debug_log or _LOGGER_HANDLE_PATCHED:
        return

    original_handle = logging.Logger.handle
    perf_markers = (
        # BaM/vLLM 侧每个 retrieve iter 的单行性能摘要。
        "[LMCACHE_BAM_ITER_PERF]",
        # GPU-initiated 分支的关键事件必须默认可见，否则无法判断新逻辑是否命中。
        "GPU_INITIATED_PREFETCH",
        "[LMCACHE_BAM_KV_FAST_PATH_PREFETCH_ENQUEUE]",
        "[LMCACHE_BAM_EARLY_PREFETCH]",
        # ref/lifecycle 只有在对应 debug 开关打开时才会产生，这里允许透出。
        "[LMCACHE_BAM_KV_REF_DEBUG_STATS]",
        "[LMCACHE_BAM_CACHE_LIFECYCLE_STATS]",
        # benchmark 结束后的临时 idle-stop 观察点，默认不改变数据链路。
        "[LMCACHE_BAM_RUNTIME_IDLE_STOP",
        # 关闭路径必须默认可见，用来确认 wrapper 没有截断 LMCache 原生 close。
        "[LMCACHE_BAM_STORAGE_MANAGER_CLOSE",
    )

    def filtered_handle(self: logging.Logger, record: logging.LogRecord) -> None:
        if record.levelno >= logging.WARNING:
            original_handle(self, record)
            return
        try:
            message = record.getMessage()
        except Exception:
            message = str(record.msg)
        lower_message = message.lower()
        if any(marker.lower() in lower_message for marker in perf_markers):
            original_handle(self, record)

    logging.Logger.handle = filtered_handle
    _LOGGER_HANDLE_PATCHED = True


def load_manifest(path: Path, limit: int) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def iter_request_plan(rows: Iterable[dict], repeat_read: int) -> Iterable[tuple[dict, str]]:
    """生成每条样本的请求序列。

    当前 baseline 的语义固定为：
    - 第一次：request_1_write，用于把 KV 写入 LMCache SSD/GDS shadow；
    - 后续：request_N_read，用于读回同一个 prompt 的 KV。

    `repeat_read` 默认是 1，也就是每条样本总共跑两次。后续要测 steady-state
    时可以调大，让同一 prompt 连续读多次。
    """
    for row in rows:
        yield row, "write"
        for _ in range(max(repeat_read, 0)):
            yield row, "read"


@contextlib.contextmanager
def build_llm(args: argparse.Namespace):
    ktc = KVTransferConfig.from_cli(
        '{"kv_connector":"LMCacheConnector","kv_role":"kv_both"}'
    )
    llm = LLM(
        model=args.model,
        kv_transfer_config=ktc,
        max_model_len=args.max_model_len,
        enable_chunked_prefill=args.enable_chunked_prefill,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        enforce_eager=args.enforce_eager,
        trust_remote_code=False,
    )
    try:
        yield llm
    finally:
        LMCacheEngineBuilder.destroy(ENGINE_NAME)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run LongBench-TriviaQA LMCache SSD baseline.")
    parser.add_argument("--manifest", type=Path, default=Path(DEFAULT_MANIFEST))
    parser.add_argument("--metrics-jsonl", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("MODEL_PATH", DEFAULT_MODEL))
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    parser.add_argument("--dtype", default="half")
    parser.add_argument("--max-tokens", type=int, default=32)
    # 默认跑完整 Qwen-tokenized lt4k bucket；传 0 表示不截断。
    parser.add_argument("--num-samples", type=int, default=25)
    parser.add_argument("--repeat-read", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--enable-chunked-prefill",
                        action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--print-output", action="store_true")
    parser.add_argument(
        "--debug-log",
        action=argparse.BooleanOptionalAction,
        default=env_flag("LONGBENCH_DEBUG_LOG", False),
        help="print full backend debug logs instead of performance-only logs",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    install_benchmark_log_filter(debug_log=args.debug_log)

    rows = load_manifest(args.manifest, limit=args.num_samples)
    if not rows:
        raise RuntimeError(f"empty manifest: {args.manifest}")

    args.metrics_jsonl.parent.mkdir(parents=True, exist_ok=True)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    print("[longbench-triviaqa] manifest=", args.manifest)
    print("[longbench-triviaqa] metrics_jsonl=", args.metrics_jsonl)
    print("[longbench-triviaqa] samples=", len(rows))
    print("[longbench-triviaqa] repeat_read=", args.repeat_read)
    print("[longbench-triviaqa] max_model_len=", args.max_model_len)
    print("[longbench-triviaqa] max_tokens=", args.max_tokens)
    print("[longbench-triviaqa] debug_log=", args.debug_log)
    print("[longbench-triviaqa] lmcache_local_disk=", os.environ.get("LMCACHE_LOCAL_DISK"))
    print("[longbench-triviaqa] gds_path=", os.environ.get("VLLM_GDS_LMCACHE_PATH"))

    with args.metrics_jsonl.open("w", encoding="utf-8") as metrics_f:
        with build_llm(args) as llm:
            request_idx = 0
            total_elapsed_s = 0.0
            phase_elapsed_s: dict[str, float] = {"write": 0.0, "read": 0.0}
            phase_counts: dict[str, int] = {"write": 0, "read": 0}
            for row, phase in iter_request_plan(rows, repeat_read=args.repeat_read):
                request_idx += 1
                sample_id = row["_id"]
                prompt = row["prompt"]
                start = time.perf_counter()
                # 使用 runner 自己的逐请求指标作为统一性能输出，避免 vLLM 每次
                # generate 的 tqdm/progress bar 淹没底层 cache 统计。
                outputs = llm.generate([prompt], sampling_params=sampling_params,
                                       use_tqdm=False)
                elapsed_s = time.perf_counter() - start
                total_elapsed_s += elapsed_s
                phase_elapsed_s[phase] = phase_elapsed_s.get(phase, 0.0) + elapsed_s
                phase_counts[phase] = phase_counts.get(phase, 0) + 1
                total_avg_s = total_elapsed_s / max(request_idx, 1)
                phase_avg_s = (
                    phase_elapsed_s[phase] / max(phase_counts[phase], 1))
                write_avg_s = (
                    phase_elapsed_s.get("write", 0.0) /
                    max(phase_counts.get("write", 0), 1))
                read_avg_s = (
                    phase_elapsed_s.get("read", 0.0) /
                    max(phase_counts.get("read", 0), 1))
                generated = outputs[0].outputs[0].text
                output_tokens = len(outputs[0].outputs[0].token_ids)
                record = {
                    "request_idx": request_idx,
                    "sample_id": sample_id,
                    "phase": phase,
                    "elapsed_s": elapsed_s,
                    "source_dataset": row.get("source_dataset"),
                    "prompt_mode": row.get("prompt_mode"),
                    "length": row.get("length"),
                    "length_bucket": row.get("length_bucket"),
                    "prompt_sha1": row.get("prompt_sha1"),
                    "output_tokens": output_tokens,
                    "answers": row.get("answers", []),
                    "generated_text": generated,
                }
                metrics_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                metrics_f.flush()

                print(
                    "[longbench-triviaqa-iter] "
                    f"iter={request_idx} sample_id={sample_id} phase={phase} "
                    f"length={row.get('length')} bucket={row.get('length_bucket')} "
                    f"elapsed_s={elapsed_s:.4f} phase_avg_s={phase_avg_s:.4f} "
                    f"write_avg_s={write_avg_s:.4f} read_avg_s={read_avg_s:.4f} "
                    f"total_avg_s={total_avg_s:.4f} output_tokens={output_tokens}",
                    flush=True,
                )
                if args.print_output:
                    print("[longbench-triviaqa-output]", generated)

            print(
                "[longbench-triviaqa-summary] "
                f"requests={request_idx} samples={len(rows)} "
                f"repeat_read={args.repeat_read} total_elapsed_s={total_elapsed_s:.4f} "
                f"avg_request_s={total_elapsed_s / max(request_idx, 1):.4f} "
                f"write_avg_s={phase_elapsed_s.get('write', 0.0) / max(phase_counts.get('write', 0), 1):.4f} "
                f"read_avg_s={phase_elapsed_s.get('read', 0.0) / max(phase_counts.get('read', 0), 1):.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
