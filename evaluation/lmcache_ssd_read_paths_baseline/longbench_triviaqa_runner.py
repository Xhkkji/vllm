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


def env_float(name: str, default: float = 0.0) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


@contextlib.contextmanager
def maybe_nvtx_range(name: str):
    """按需给 Nsight Systems 标出 runner 侧 request 阶段。

    默认关闭，避免给普通 benchmark 引入额外依赖和开销。打开
    `LONGBENCH_NVTX_TRACE=1` 后，Nsight timeline 里可以直接区分 write/read
    request，再结合 vLLM 内部的 prefill/decode range 做 decode-only 分析。
    """
    if not env_flag("LONGBENCH_NVTX_TRACE", False):
        yield
        return
    import torch
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def drop_kernel_page_cache_before_read(request_idx: int, sample_id: str) -> None:
    """在 read request 计时前清 Linux page cache。

    这个开关只服务 LMCache 原生 SSD cold-read baseline。默认关闭，避免影响
    常规 BaM/GDS/LMCache 回归。调用方必须以 root 运行，否则无法写
    `/proc/sys/vm/drop_caches`。
    """
    if os.geteuid() != 0:
        raise RuntimeError(
            "LONGBENCH_DROP_CACHES_BEFORE_READ=1 requires root because "
            "it writes /proc/sys/vm/drop_caches")

    print(
        "[longbench-triviaqa-drop-caches] "
        f"iter={request_idx} sample_id={sample_id} action=sync_drop_caches",
        flush=True,
    )
    os.sync()
    Path("/proc/sys/vm/drop_caches").write_text("3\n", encoding="ascii")
    settle_s = env_float("LONGBENCH_DROP_CACHES_SETTLE_S", 0.0)
    if settle_s > 0:
        time.sleep(settle_s)


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
        # 轻量请求/connector 阶段标记，用来定位无 debug 压测卡点。
        "[LMCACHE_BAM_RECEIVE_STAGE]",
        "[LMCACHE_BAM_DIRECT_PLACEMENT_READ_SUBMIT",
        "[LMCACHE_BAM_DIRECT_PLACEMENT_PIPELINE]",
        "[LMCACHE_BAM_DIRECT_PLACEMENT_RUNTIME_ATTACH]",
        "[LMCACHE_BAM_DIRECT_PLACEMENT_RUNTIME_POLL_STALL]",
        "[LMCACHE_BAM_DIRECT_PLACEMENT_READ_FRONTIER]",
        "[LMCACHE_BAM_DIRECT_PLACEMENT_RUNTIME_READY]",
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


def iter_batch_request_plan(
    rows: list[dict],
    repeat_read: int,
    batch_size: int,
) -> Iterable[tuple[list[dict], str]]:
    """按 batch 组织 write/read 请求。

    batch_size=1 时等价于原来的逐样本 write/read；batch_size>1 时，同一批
    prompt 会先一起 write，再一起 read，从而真正触发 vLLM 的多请求 batch
    调度，而不是只在 Python for-loop 中串行跑多个单请求。
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]
        yield batch_rows, "write"
        for _ in range(max(repeat_read, 0)):
            yield batch_rows, "read"


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
        swap_space=args.swap_space,
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
    parser.add_argument("--swap-space", type=float,
                        default=env_float("SWAP_SPACE", 4.0))
    parser.add_argument("--max-tokens", type=int, default=32)
    # 默认跑完整 Qwen-tokenized lt4k bucket；传 0 表示不截断。
    parser.add_argument("--num-samples", type=int, default=25)
    parser.add_argument("--repeat-read", type=int, default=1)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("BATCH_SIZE", "1")),
        help="number of prompts submitted in one llm.generate call",
    )
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
    print("[longbench-triviaqa] batch_size=", args.batch_size)
    print("[longbench-triviaqa] max_model_len=", args.max_model_len)
    print("[longbench-triviaqa] max_tokens=", args.max_tokens)
    print("[longbench-triviaqa] swap_space=", args.swap_space)
    print("[longbench-triviaqa] debug_log=", args.debug_log)
    print("[longbench-triviaqa] drop_caches_before_read=",
          env_flag("LONGBENCH_DROP_CACHES_BEFORE_READ", False))
    print("[longbench-triviaqa] lmcache_local_disk=", os.environ.get("LMCACHE_LOCAL_DISK"))
    print("[longbench-triviaqa] gds_path=", os.environ.get("VLLM_GDS_LMCACHE_PATH"))

    with args.metrics_jsonl.open("w", encoding="utf-8") as metrics_f:
        with build_llm(args) as llm:
            request_idx = 0
            total_elapsed_s = 0.0
            phase_elapsed_s: dict[str, float] = {"write": 0.0, "read": 0.0}
            phase_counts: dict[str, int] = {"write": 0, "read": 0}
            plan_iter = iter_batch_request_plan(
                rows,
                repeat_read=args.repeat_read,
                batch_size=args.batch_size,
            )
            for batch_rows, phase in plan_iter:
                request_idx += 1
                batch_sample_ids = [row["_id"] for row in batch_rows]
                prompts = [row["prompt"] for row in batch_rows]
                print(
                    "[longbench-triviaqa-begin] "
                    f"iter={request_idx} sample_ids={','.join(batch_sample_ids)} "
                    f"phase={phase} batch_size={len(batch_rows)} "
                    f"lengths={','.join(str(row.get('length')) for row in batch_rows)} "
                    f"bucket={batch_rows[0].get('length_bucket')}",
                    flush=True,
                )
                dropped_caches = False
                if (phase == "read"
                        and env_flag("LONGBENCH_DROP_CACHES_BEFORE_READ", False)):
                    drop_kernel_page_cache_before_read(
                        request_idx,
                        ",".join(batch_sample_ids),
                    )
                    dropped_caches = True
                start = time.perf_counter()
                # 使用 runner 自己的逐请求指标作为统一性能输出，避免 vLLM 每次
                # generate 的 tqdm/progress bar 淹没底层 cache 统计。
                nvtx_name = (
                    f"longbench_request:{phase}:iter={request_idx}:"
                    f"batch_size={len(batch_rows)}:"
                    f"samples={','.join(batch_sample_ids)}")
                with maybe_nvtx_range(nvtx_name):
                    outputs = llm.generate(
                        prompts,
                        sampling_params=sampling_params,
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
                output_tokens_list = [
                    len(output.outputs[0].token_ids) for output in outputs
                ]
                batch_record = {
                    "request_idx": request_idx,
                    "sample_id": ",".join(batch_sample_ids),
                    "sample_ids": batch_sample_ids,
                    "phase": phase,
                    "record_type": "batch",
                    "batch_size": len(batch_rows),
                    "elapsed_s": elapsed_s,
                    "per_sample_elapsed_s": elapsed_s / max(len(batch_rows), 1),
                    "source_dataset": batch_rows[0].get("source_dataset"),
                    "prompt_mode": batch_rows[0].get("prompt_mode"),
                    "length": [row.get("length") for row in batch_rows],
                    "length_bucket": batch_rows[0].get("length_bucket"),
                    "prompt_sha1": [row.get("prompt_sha1") for row in batch_rows],
                    "output_tokens": output_tokens_list,
                    "drop_caches_before_request": dropped_caches,
                    "answers": [row.get("answers", []) for row in batch_rows],
                    "generated_text": [
                        output.outputs[0].text for output in outputs
                    ],
                }
                metrics_f.write(json.dumps(batch_record, ensure_ascii=False) + "\n")
                for row, output in zip(batch_rows, outputs):
                    generated = output.outputs[0].text
                    output_tokens = len(output.outputs[0].token_ids)
                    record = {
                        "request_idx": request_idx,
                        "sample_id": row["_id"],
                        "phase": phase,
                        "record_type": "sample",
                        "batch_size": len(batch_rows),
                        "elapsed_s": elapsed_s,
                        "per_sample_elapsed_s": elapsed_s / max(len(batch_rows), 1),
                        "source_dataset": row.get("source_dataset"),
                        "prompt_mode": row.get("prompt_mode"),
                        "length": row.get("length"),
                        "length_bucket": row.get("length_bucket"),
                        "prompt_sha1": row.get("prompt_sha1"),
                        "output_tokens": output_tokens,
                        "drop_caches_before_request": dropped_caches,
                        "answers": row.get("answers", []),
                        "generated_text": generated,
                    }
                    metrics_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                metrics_f.flush()

                print(
                    "[longbench-triviaqa-iter] "
                    f"iter={request_idx} sample_ids={','.join(batch_sample_ids)} "
                    f"phase={phase} batch_size={len(batch_rows)} "
                    f"bucket={batch_rows[0].get('length_bucket')} "
                    f"elapsed_s={elapsed_s:.4f} phase_avg_s={phase_avg_s:.4f} "
                    f"write_avg_s={write_avg_s:.4f} read_avg_s={read_avg_s:.4f} "
                    f"total_avg_s={total_avg_s:.4f} "
                    f"output_tokens={','.join(str(v) for v in output_tokens_list)}",
                    flush=True,
                )
                if args.print_output:
                    for sample_id, output in zip(batch_sample_ids, outputs):
                        print("[longbench-triviaqa-output]", sample_id,
                              output.outputs[0].text)

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
