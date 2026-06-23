# SPDX-License-Identifier: Apache-2.0
"""兼容旧导入路径。

主实现已经迁到 vllm.bam.row_store_loader。这里保留同名导出，避免已有实验
脚本或临时调试代码失效。
"""

from vllm.bam.row_store_loader import import_bam_row_store, parse_optional_int_list

__all__ = ["import_bam_row_store", "parse_optional_int_list"]
