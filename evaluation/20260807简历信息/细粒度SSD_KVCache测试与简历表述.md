# 20260807 简历信息：GPU-Initiated 细粒度 SSD KV Cache

## 项目标题建议

推荐标题：

```latex
\textbf{面向 LLM 推理的 GPU-Initiated 细粒度 SSD KV Cache 系统}
```

如果希望同时强调应用场景，可以使用：

```latex
\textbf{面向长上下文与在线多轮对话的 GPU-Initiated 细粒度 SSD KV Cache 系统}
```

当前更建议使用第一版。它把重点放在系统能力上，即 GPU-Initiated、SSD KV Cache、细粒度 I/O，而不是把项目过早绑定到单一负载类型。

## 简历项目描述建议

```latex
\datedsubsection{\textbf{面向 LLM 推理的 GPU-Initiated 细粒度 SSD KV Cache 系统}}{2026.04 - 至今}
\datasection{\item \textbullet\ \textbf{SSD-backed KV Cache：} 面向 LLM 推理中的 KV Cache 显存瓶颈，设计 SSD-backed KV Cache 原型系统，将冷 KV 数据下沉至 NVMe SSD，并在推理阶段按需恢复至 GPU KV Cache，扩展单卡可服务上下文与并发请求规模。}
\datasection{\item \textbullet\ \textbf{Serving Runtime 接入：} 基于 vLLM / LMCache 打通 Prefix Cache、KV Connector 与 SSD$\rightarrow$GPU 存储后端链路，实现 KV 数据写入、索引管理、缓存命中检测与上层推理透明恢复。}
\datasection{\item \textbullet\ \textbf{细粒度 SSD KV I/O：} 设计 GPU-Initiated descriptor pool，实现 Block/Fragment 级 SSD KV 异步读取；相比 LMCache 原生 chunk 恢复路径，在 2048-token KV 恢复场景下降低 restore p95 约 27.3\%，有效带宽提升约 1.52$\times$，并支持原生 chunk 路径难以覆盖的 128-token 小粒度恢复。}
\datasection{\item \textbullet\ \textbf{GPU-Resident IO Runtime：} 设计 GPU 侧常驻 IO Runtime，将请求提交、完成轮询、状态推进与数据放置下沉至 GPU 端，减少 CPU 轮询、Host 拷贝与同步开销。}
\datasection{\item \textbullet\ \textbf{端到端验证链路：} 构建 Qwen2.5-7B-Instruct + vLLM / LMCache 评测链路，覆盖 KV write/read 对拍、cache retrieve、混合读写压力、descriptor pool 回收与异步一致性诊断，统计 KV restore latency、有效带宽、TTFT 与尾延迟。}
```

## 最新细粒度 I/O 对比结果

### 测试目的

这组测试不是完整论文级端到端评测，而是用于支撑简历中的核心优势：

> 相比 LMCache 原生 chunk 级 SSD 恢复路径，当前 GPU-Initiated descriptor pool 支持 block / fragment 级细粒度 KV 读取，在小粒度恢复场景下可以避免 chunk 粒度限制，在较大恢复规模下也表现出更高有效带宽和更低尾延迟。

### 测试配置

| 项 | 设置 |
|---|---|
| LMCache baseline | 原生默认 `LMCACHE_CHUNK_SIZE=256` |
| LMCache 路径 | SSD -> CPU -> GPU |
| BaM 路径 | GPU-Initiated descriptor pool，SSD -> GPU |
| 模型 | Qwen2.5-7B-Instruct |
| LMCache repeats | 每组 5 次 |
| BaM repeats | warmup 2 次，正式 7 次 |
| 对应规模 | 8 / 32 / 128 vLLM blocks，约 128 / 512 / 2048 tokens |

为了让 LMCache baseline 真正走 SSD read，测试中关闭了 vLLM 自身 GPU prefix cache；否则同进程第二遍 reuse 会被 vLLM GPU prefix cache 截住，无法观测 LMCache local disk retrieve。

### 对比结果

| 方法 | 目标规模 | 实际恢复 KV | SSD 读取量 | Restore median | Restore p95 | 有效带宽 |
|---|---:|---:|---:|---:|---:|---:|
| LMCache native chunk | 8 blocks / 128 tokens | 0 tokens | 0 MiB | -- | -- | -- |
| BaM descriptor pool | 8 blocks / 128 tokens | 128 tokens | 7 MiB | 2.42 ms | 3.51 ms | 2.83 GiB/s |
| LMCache native chunk | 32 blocks / 512 tokens | 256 tokens | 14 MiB | 3.68 ms | 4.40 ms | 3.71 GiB/s |
| BaM descriptor pool | 32 blocks / 512 tokens | 512 tokens | 28 MiB | 6.21 ms | 6.23 ms | 4.41 GiB/s |
| LMCache native chunk | 128 blocks / 2048 tokens | 1792 tokens | 98 MiB | 24.18 ms | 26.59 ms | 3.96 GiB/s |
| BaM descriptor pool | 128 blocks / 2048 tokens | 2048 tokens | 112 MiB | 18.21 ms | 19.33 ms | 6.01 GiB/s |

### 结果解读

1. 128-token 小粒度场景下，LMCache 默认 256-token chunk 没有触发 SSD restore，而 BaM 可以直接恢复 8 个 block。
2. 512-token 场景下，LMCache 只恢复了 1 个 256-token chunk，剩余部分仍要走 suffix prefill；BaM 可以按 block 完整恢复目标 KV。
3. 2048-token 场景下，BaM 读取的数据更多，但 restore p95 更低：19.33 ms vs 26.59 ms；有效带宽也更高：6.01 GiB/s vs 3.96 GiB/s。

### 简历可用结论

可以在简历或项目介绍中使用下面这句话：

> 实现 GPU-Initiated descriptor pool 细粒度 KV 读取路径，在 2048-token KV 恢复场景下，相比 LMCache 原生 SSD chunk 路径将 restore p95 从 26.59 ms 降至 19.33 ms，并将有效恢复带宽从 3.96 GiB/s 提升至 6.01 GiB/s；在 128-token 小粒度场景下支持原生 chunk 路径无法覆盖的 block 级恢复。

如果希望更保守，可以写成：

> 构建 GPU-Initiated Block/Fragment 级 SSD KV 读取路径，相比 LMCache 原生 chunk 级恢复，在小粒度 KV 命中场景下减少恢复粒度限制，并在 2048-token 恢复规模下观察到更低 p95 延迟和更高有效带宽。

## 测试数据来源

BaM direct read-size sweep：

```text
/home/xhk/llm-inference/BaM_IOStack/vllm_evaluation/mds_poc/result/read_size_sweep/20260807_181541/summary.json
```

LMCache native chunk baseline：

```text
/home/xhk/llm-inference/BaM_IOStack/vllm_evaluation/BaM_sync_baseline/result/lmcache_native_chunk_compare/20260807_181541/
```

## 口径注意

LMCache 结果中的“实际恢复 KV”来自 `LMCacheEngine.retrieve()` 的 `retrieved_tokens`。默认 256-token chunk 策略下，LMCache 不一定恢复完整目标 prompt：128-token 场景没有 SSD restore，512-token 场景恢复 256 tokens，2048-token 场景恢复 1792 tokens。因此这组结果重点体现的是原生 chunk 粒度与当前 BaM block/fragment 粒度之间的恢复能力差异，而不是完整 serving 负载下的最终 TTFT / TPOT 对比。
