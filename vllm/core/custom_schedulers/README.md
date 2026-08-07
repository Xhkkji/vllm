# custom_schedulers

这个目录用于放和原生 `vllm.core.scheduler.Scheduler` 并列的自定义调度实现。

建议后续文件组织：

- `async_kv_block_scheduler.py`
- `async_kv_policy.py`
- `async_kv_state.py`
- `async_kv_transfer.py`

当前阶段先保留目录，等调度逻辑稳定后再逐步迁入。
