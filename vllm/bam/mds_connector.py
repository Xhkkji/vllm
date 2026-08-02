# SPDX-License-Identifier: Apache-2.0
"""兼容旧 import；新代码使用 ``vllm.bam.mds.connector``。"""

from vllm.bam.mds.connector import BaMMDSConnector

__all__ = ["BaMMDSConnector"]
