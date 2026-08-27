#!/usr/bin/env bash
set -euo pipefail

NATIVE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GRANULEKV_DIR="$(cd -- "${NATIVE_DIR}/.." && pwd)"
BUILD_LIB="${GRANULEKV_DIR}/build/torch_bridge"
BUILD_TEMP="${GRANULEKV_DIR}/build/torch_bridge_temp"
PYTHON_BIN="${PYTHON_BIN:-/home/xhk/miniconda3/envs/pytorch/bin/python}"

mkdir -p "${BUILD_LIB}" "${BUILD_TEMP}"
"${PYTHON_BIN}" "${NATIVE_DIR}/setup.py" build_ext \
  --build-lib "${BUILD_LIB}" --build-temp "${BUILD_TEMP}"

echo "built_dir=${BUILD_LIB}"
