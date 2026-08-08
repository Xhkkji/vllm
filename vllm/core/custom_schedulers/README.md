# custom_schedulers

这个目录用于放和原生 `vllm.core.scheduler.Scheduler` 并列的自定义调度实现。

建议后续文件组织：

- `async_kv_transfer.py`：异步 read/write 请求、事件和多槽状态机。
- `async_kv_state.py`：调度策略使用的 KV block residency 只读视图。
- `async_kv_policy.py`：block-aware 调度决策入口，逐步承载 read/write 优先级、免写释放和预取策略。
- `async_kv_block_scheduler.py`：后续真正独立的自定义 scheduler 主类。

当前阶段先迁入 transfer/state/policy 三类可独立演进的模块；现有
`vllm.core.async_kv_scheduler.AsyncKVScheduler` 仍作为可跑的过渡入口。
