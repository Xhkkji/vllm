# LMCache + BaM 128KB Page Result Summary

这份摘要专门记录 `2026-06-22` 这轮在 `vllm-bam` 上完成的单卡
`LMCache SSD-only` 与 `LMCache + BaM shadow` 对比结果，便于后续直接引用。

`2026-06-23` 已经进一步跑通 `LMCache + BaM prefer-load`。也就是：
`LMCache` 在第二个请求恢复共享前缀时，优先从 `BaM` 读回 KV chunk，
再交给 vLLM 继续执行。这个结果单独记录在：

- [SINGLE_GPU_LMCACHE_BAM_PREFER_LOAD_QWEN25.md](/home/xhk/llm-inference/vllm-bam/evaluation/SINGLE_GPU_LMCACHE_BAM_PREFER_LOAD_QWEN25.md)

## 对比口径

- 模型：`/home/xhk/llm-inference/models/Qwen2.5-7B-Instruct`
- GPU：`Tesla V100S-PCIE-32GB`
- 引擎：`vLLM V0`
- 统一配置：
  - `PROMPT_REPEAT=400`
  - `MAX_MODEL_LEN=8192`
  - `dtype=half`
  - `enable_prefix_caching=false`
  - `enable_chunked_prefill=false`

## Baseline

`LMCache SSD-only` 日志：

- [run.log](/home/xhk/llm-inference/vllm-bam/evaluation/logs/single_gpu_lmcache_no_prefix_reuse_qwen25/20260622_225752/run.log)

关键结果：

- `request_1_elapsed_s=3.0629`
- `request_2_elapsed_s=2.6346`

## BaM 128KB Page

`LMCache + BaM shadow` 日志：

- [run.log](/home/xhk/llm-inference/vllm-bam/evaluation/logs/single_gpu_lmcache_no_prefix_reuse_qwen25/20260622_233047/run.log)

当前页布局：

- `page_bytes=131072`
- `pages_per_chunk=112`
- `page_token_capacity=128`
- `pages_per_kv_layer=2`

关键结果：

- `warmup bam_shadow_elapsed_s=2.2156`
- `request_1_elapsed_s=2.9673`
- `request_2_elapsed_s=2.7374`

稳态写入带宽：

- 大多数满 chunk 写入约 `3.2 ~ 3.6 GiB/s`

补充 microbench：

- [lmcache_bam_write_microbench_20260622_235727.log](/home/xhk/llm-inference/vllm-bam/evaluation/logs/lmcache_bam_write_microbench_20260622_235727.log)

这个 microbench 的口径是：

- 同一个进程
- 同一个 dummy chunk 大小：`14 MiB`
- 每轮使用新的 `chunk_hash`
- 顺序写入新的 BaM 槽位
- `BaM` 初始化不计入循环内计时

关键结果：

- `first_write=67.647 ms`，约 `0.202 GiB/s`
- `second_write=6.564 ms`，约 `2.083 GiB/s`
- `steady_state mean=4.198 ms`，约 `3.257 GiB/s`
- `steady_state median=3.616 ms`
- 最快单次约 `2.507 ms`，约 `5.453 GiB/s`

这说明：

- 当前主线瓶颈已经不再是稳态顺序写吞吐
- 真实 `LMCache + BaM shadow` 日志里的 `3.x GiB/s` 稳态结果是可信的
- 剩余尾巴主要集中在首块冷启动，而不是后续顺序写

## 直接结论

和 `LMCache SSD-only baseline` 相比：

- `request_1` 从 `3.0629s` 变成 `2.9673s`
- `request_2` 从 `2.6346s` 变成 `2.7374s`

因此当前可以先认为：

- `LMCache + BaM shadow` 在 `128KB page` 版本下，端到端时延已经和
  `LMCache SSD-only baseline` 接近
- 之前旧的 `1KB page` 路径造成的明显额外开销，已经基本消失

## Prefer-Load 最新结果

最新 `LMCache + BaM prefer-load` 日志：

- [run.log](/home/xhk/llm-inference/vllm-bam/evaluation/logs/single_gpu_lmcache_no_prefix_reuse_qwen25/20260623_174138/run.log)

这轮使用：

- `PROMPT_REPEAT=100`
- `MAX_MODEL_LEN=8192`
- `VLLM_BAM_LMCACHE_SHADOW_ENABLE=1`
- `VLLM_BAM_LMCACHE_PREFER_LOAD_ENABLE=1`

关键结果：

- `warmup bam_shadow_elapsed_s=2.1491`
- `request_1_elapsed_s=1.8220`
- `request_2_elapsed_s=1.6120`
- BaM read 带宽约 `2.95 / 3.13 / 3.68 / 3.54 GiB/s`
- `LMCACHE_BAM_VERIFY exact_equal=True`
- `max_abs_diff=0.000000`
- 日志中出现 `prefer-load hit`
- 日志中出现 `LMCACHE_REBUILD`
- 日志中出现 `XFORMERS_PREFIX_FALLBACK`

这个结果说明：

- `BaM` 读取链路已经真实进入第二个请求
- 读回数据和 LMCache 原始数据完全一致
- vLLM 能正常消费 BaM 读回的 KV cache
- 第二个请求输出正常，没有乱码

需要注意：这轮 `PROMPT_REPEAT=100`，而上面的 `LMCache SSD-only baseline`
是 `PROMPT_REPEAT=400`，所以暂时不要把 `1.6120s` 和 `2.6346s`
直接当作公平性能对比。下一步需要用同一个 prompt 长度分别重跑：

- `LMCache SSD-only`
- `LMCache + BaM prefer-load`

## 和旧版本相比的提升

旧的 `1KB token-row/page` 版本大致表现为：

- `warmup ≈ 95s`
- 稳态写带宽约 `0.5 GiB/s`

新的 `128KB page` 版本大致表现为：

- `warmup ≈ 2.2s`
- 稳态写带宽约 `3.2 ~ 3.6 GiB/s`

所以这轮最重要的结论是：

- 把 `LMCache -> BaM` 的物理组织从 `1KB page` 改成固定 `128KB page`
  后，首写异常和稳态带宽都得到了数量级改善。
- 并且顺序写 microbench 已确认：当前 `BaM` 主线稳态写吞吐与真实
  `vLLM shadow write` 日志一致，约为 `3.x GiB/s`
