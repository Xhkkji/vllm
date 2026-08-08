# SPDX-License-Identifier: Apache-2.0

"""兼容旧 import 路径的异步 KV transfer 状态机入口。

新的自定义调度代码应直接从
``vllm.core.custom_schedulers.async_kv_transfer`` 导入。这个文件暂时保留，
避免旧测试、Worker 或外部脚本因为模块迁移立即失效。
"""

from vllm.core.custom_schedulers.async_kv_transfer import (  # noqa: F401
    AsyncKVExecutionMarker,
    AsyncKVSchedulePolicy,
    AsyncKVTransferEvent,
    AsyncKVTransferOperation,
    AsyncKVTransferPriority,
    AsyncKVTransferQueue,
    AsyncKVTransferRequest,
    AsyncKVTransferState,
    PendingAsyncKVTransfer,
)

