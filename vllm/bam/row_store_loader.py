# SPDX-License-Identifier: Apache-2.0
import importlib
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import vllm.envs as envs


def parse_optional_int_list(value: Optional[str]) -> Optional[List[int]]:
    if value is None or value.strip() == "":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _candidate_import_paths() -> Iterable[Path]:
    # 优先使用用户显式指定的 BaM Python 路径。
    if envs.VLLM_BAM_IMPORT_PATH:
        yield Path(envs.VLLM_BAM_IMPORT_PATH)

    llm_inference_dir = Path(__file__).resolve().parents[3]
    bam_gids_dir = llm_inference_dir / "BaM_IOStack" / "gids_module"
    yield bam_gids_dir

    if bam_gids_dir.exists():
        for build_dir in sorted(bam_gids_dir.glob("build*")):
            yield build_dir


def import_bam_row_store():
    """按本地实验环境的常见路径探测并导入 BaMRowStore。"""
    errors: List[str] = []
    for path in _candidate_import_paths():
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)
        try:
            module = importlib.import_module("bam_row_store")
            return module.BaMRowStore
        except Exception as exc:  # pragma: no cover - 这里只做环境探测
            errors.append(f"{path_str}: {type(exc).__name__}: {exc}")

    try:
        module = importlib.import_module("bam_row_store")
        return module.BaMRowStore
    except Exception as exc:
        errors.append(f"default sys.path: {type(exc).__name__}: {exc}")

    raise ImportError("Failed to import bam_row_store. Tried paths:\n" +
                      "\n".join(errors))
