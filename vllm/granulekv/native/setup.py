#!/usr/bin/env python3
"""Build the independent GranuleKV pointer-to-Tensor bridge."""

from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


ROOT = Path(__file__).resolve().parent

setup(
    name="granulekv_torch_bridge",
    ext_modules=[
        CppExtension(
            name="granulekv_torch_bridge",
            sources=[str(ROOT / "torch_ipc_tensor_bridge.cpp")],
            extra_compile_args=["-O2"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
