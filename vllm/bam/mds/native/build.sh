#!/usr/bin/env bash
set -euo pipefail

NATIVE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MDS_DIR="$(cd -- "${NATIVE_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xhk/miniconda3/envs/pytorch-vllm/bin/python}"
BUILD_LIB="${MDS_DIR}/build/torch_bridge"
BUILD_TEMP="${MDS_DIR}/build/torch_bridge_temp"

# 使用当前 vLLM 环境 PyTorch 的 ABI；该扩展不编译 CUDA kernel，也不加入 vLLM
# 主 CMake，避免结构整理触发整个 vLLM native stack 重建。
mkdir -p "${BUILD_LIB}" "${BUILD_TEMP}"
"${PYTHON_BIN}" "${NATIVE_DIR}/setup.py" \
  build_ext \
  --build-lib "${BUILD_LIB}" \
  --build-temp "${BUILD_TEMP}"

echo "built_dir=${BUILD_LIB}"
