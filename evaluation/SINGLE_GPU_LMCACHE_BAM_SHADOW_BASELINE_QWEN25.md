# 单卡 LMCache + BaM Shadow Baseline

这份文档整理当前在 `vllm-bam` 中已经跑通的单卡 `LMCache + BaM shadow write` 基线，目标是固定一条后续可继续推进到 `Prefer BaM load` 的中间状态。

## 结论

当前这条链路已经稳定跑通：

- `LMCache` 继续负责原始 `SSD` 存储路径
- `BaM` 额外对同一份 KV chunk 做 `shadow write`
- `warmup` 单独执行，不计入正式请求耗时

当前它的定位是：

- 验证 `LMCache -> BaM` 写链路已经真实打通
- 验证 `BaM` 现在可以同时处理满块和尾块
- 作为后续 `Prefer BaM load` 的直接前置基线

当前它还不是：

- 最终的 `BaM SSD backend` 性能结论
- `BaM read path` 的性能结论
- 论文口径下的最终对比结果

## 当前环境

- GPU: `Tesla V100S-PCIE-32GB`
- Compute capability: `7.0`
- `vLLM V1` 不可用，当前统一走 `V0`
- attention backend: `XFormers`
- model: `/home/xhk/llm-inference/models/Qwen2.5-7B-Instruct`
- repo: `/home/xhk/llm-inference/vllm-bam`
- LMCache repo: `/home/xhk/llm-inference/LMCache-v0-torch26`
- BaM import path: `/home/xhk/llm-inference/BaM_IOStack/gids_module`

## 执行命令

先进入仓库：

```bash
cd /home/xhk/llm-inference/vllm-bam
```

然后运行：

```bash
VLLM_BAM_LMCACHE_SHADOW_ENABLE=1 \
PYTHON_BIN=/home/xhk/miniconda3/envs/pytorch-vllm/bin/python \
bash evaluation/run_single_gpu_lmcache_no_prefix_reuse_qwen25.sh
```

## 本次结果

归档目录：

- [single_gpu_lmcache_bam_shadow_qwen25_20260622_035605](/home/xhk/llm-inference/vllm-bam/evaluation/logs/single_gpu_lmcache_bam_shadow_qwen25_20260622_035605)

主日志：

- [vllm-bam-single-gpu-lmcache-no-prefix-reuse-qwen25.log](/home/xhk/llm-inference/vllm-bam/evaluation/logs/single_gpu_lmcache_bam_shadow_qwen25_20260622_035605/vllm-bam-single-gpu-lmcache-no-prefix-reuse-qwen25.log)

关键结果：

- `warmup bam_shadow_elapsed_s=105.7516`
- `request_1_elapsed_s=2.5348`
- `request_2_elapsed_s=1.8315`

## 最新复跑结果

为了和当前 `LMCache SSD-only` baseline 对齐，本轮又在：

- `PROMPT_REPEAT=400`
- `MAX_MODEL_LEN=8192`
- `VLLM_BAM_LMCACHE_PREFER_LOAD_ENABLE=0`

的条件下重新跑了一次 `LMCache + BaM shadow write`。

最新日志：

- [run.log](/home/xhk/llm-inference/vllm-bam/evaluation/logs/single_gpu_lmcache_no_prefix_reuse_qwen25/20260622_225155/run.log)

关键结果：

- `warmup bam_shadow_elapsed_s=94.9162`
- `request_1_elapsed_s=3.8644`
- `request_2_elapsed_s=3.2046`

和这轮对应的 `LMCache SSD-only` baseline：

- [run.log](/home/xhk/llm-inference/vllm-bam/evaluation/logs/single_gpu_lmcache_no_prefix_reuse_qwen25/20260622_225752/run.log)
- `request_1_elapsed_s=3.0629`
- `request_2_elapsed_s=2.6346`

对比可得：

- `request_1` 变慢约 `0.80s`，约 `+26%`
- `request_2` 变慢约 `0.57s`，约 `+22%`

这说明当前 `BaM shadow write` 已经能稳定接到 `LMCache put` 生命周期里，
但正式请求阶段会引入一段清晰可见的写侧开销。

## 128KB Page 版本最新结果

随后又把 `LMCache -> BaM` 的物理布局从旧的 `1KB token-row/page` 改成了：

- 固定 `slot_num_tokens=256`
- 固定 `page_bytes=128KB`
- 每页对应一个 `(K/V, layer, 128-token block)`

也就是一个满 chunk：

- `[2, 28, 256, 512]`

会被切成：

- `112` 个 `128KB` page

最新日志：

- [run.log](/home/xhk/llm-inference/vllm-bam/evaluation/logs/single_gpu_lmcache_no_prefix_reuse_qwen25/20260622_233047/run.log)

关键结果：

- `page_bytes=131072`
- `pages_per_chunk=112`
- `page_token_capacity=128`
- `pages_per_kv_layer=2`
- `warmup bam_shadow_elapsed_s=2.2156`
- `request_1_elapsed_s=2.9673`
- `request_2_elapsed_s=2.7374`

和旧的 `1KB page` 版本相比，这轮变化非常明显：

- 首写异常从 `90s+` 下降到首个满块约 `66ms`
- warmup 从 `94.9s` 下降到 `2.2s`
- 稳态写带宽从约 `0.5 GiB/s` 提升到约 `3.2 ~ 3.6 GiB/s`
- 端到端请求时延已经接近 `LMCache SSD-only baseline`

如果和当前 `LMCache SSD-only` baseline：

- `request_1_elapsed_s=3.0629`
- `request_2_elapsed_s=2.6346`

相比，这轮 `128KB page` 的 `BaM shadow` 已基本回到同一量级，不再像旧版本那样
带来明显的额外写侧拖慢。

## 这次确认了什么

这次日志里已经明确确认：

- `slot_num_tokens=256`
- 满块可以正常写入：
  - `actual_rows=14336`
- 尾块也可以正常写入：
  - `actual_rows=13272`
  - `actual_rows=7616`

并且日志中没有再出现：

- `row count overflow`
- `row count mismatch`

这说明当前 `BaM shadow` 适配已经满足：

- 固定槽位大小
- 允许变长尾块
- warmup 不会把正式请求的槽位尺寸定小
- 最新复跑中没有再出现：
  - `prefer-load hit`
  - `LLVM ERROR`

## 为什么当前写入看起来很慢

当前慢写入其实分成两层：

1. `warmup` 里的首个写入异常慢

最新日志里第一个满块写入：

- `total_bytes=14680064`
- `elapsed_ms=92164.193`

但后续同样大小的满块写入大多在：

- `24ms ~ 29ms`
- `bw_gib_s≈0.52 ~ 0.56`

这说明 `90s+` 这个异常值不是稳定态吞吐，而更像是首个 `store_rows + flush_cache`
触发的一次性成本。结合当前代码路径，计时区间里包含：

- CPU buffer 到控制 GPU 的显式搬运
- `store_tensor_rows(...)`
- `h_pc->flush_cache()`
- 两次 `cudaDeviceSynchronize()`

因此，这个首写时间不是“纯 SSD 写 14MB 的时间”，而是把首次 BaM 运行时/flush
的同步成本一起算进去了。

2. 正式请求阶段的稳定带宽也偏低

即使去掉首写异常值，当前稳定写带宽仍只有大约 `0.5 GiB/s`，这也明显偏低。
从当前实现看，主要有两个直接原因：

- 旧实现里 `row_bytes=1024`，也就是把 LMCache KV payload 按 `1KB row / 1KB page`
  去写 BaM
- 每次 chunk 写入后都会立刻 `flush_cache()`，而且前后都显式 `cudaDeviceSynchronize()`

这意味着当前路径更像：

- 把一个 `14MB` chunk 拆成 `14336` 个 `1KB` 小页
- 同步写入
- 立刻同步 flush

这种布局和调度方式本身就不适合跑高 SSD 吞吐，所以旧版本里看到的更像是
“功能已打通的保守同步基线”，而不是 `BaM` 最终应有的吞吐上限。

而在最新的 `128KB page` 版本里，这个主要瓶颈已经被明显缓解：

- 一个 chunk 从 `14336` 个 `1KB` 小页
- 变成 `112` 个 `128KB` 大页

因此当前更新后的判断应该是：

- 旧的 `1KB page` 路径解释了此前 `90s+` 首写和 `0.5 GiB/s` 稳态写入
- 新的 `128KB page` 路径已经显著改善这两个问题
- 目前剩下的主要冷启动成本，只体现在 warmup 里第一个满块仍然比后续稳态块慢

## 如何理解这组数字

当前更准确的表述是：

- `LMCache SSD-only` baseline 已稳定
- `LMCache + BaM shadow write` baseline 已稳定
- `BaM` 的写链路已经接到 `LMCache` 生命周期里
- 旧版 `1KB page` 首写 `90s+` 是一次性异常值，不能代表稳定态写带宽
- 旧版 `1KB page` 稳定态满块写入大约在 `0.52 ~ 0.56 GiB/s`
- 新版 `128KB page` 稳定态满块写入已提升到大约 `3.2 ~ 3.6 GiB/s`
- 新版 `128KB page` 端到端请求时延已接近 `LMCache SSD-only baseline`

## 当前性能尾巴

虽然 `128KB page` 版本已经把主要问题解决了，但日志里仍然有一个比较明确的
冷启动尾巴：

- warmup 第一块满 chunk：`66.157 ms`
- 后续稳态满 chunk：大多约 `3.8 ~ 4.1 ms`

目前从代码和日志看，这个差值更像“首块冷启动成本”，而不是稳态 SSD 吞吐问题。

主要依据：

1. `BaMRowStore` / `page_cache` / queue pair 的初始化都发生在首块之前，
   但第一块仍然明显比第二块慢，说明不仅是 Python 层对象构造。
2. `store_tensor_rows(...)` 里每次都会：
   - 启动写 kernel
   - `cudaDeviceSynchronize()`
   - `flush_cache()`
   - 再 `cudaDeviceSynchronize()`
3. 第一块写入时最可能叠加了：
   - 首次 H2D 拷贝
   - 首次 kernel launch
   - 首次 page cache flush
   - 首次 NVMe 提交/完成路径热身

因此当前更合理的判断是：

- 主要性能问题已经从“页粒度过小导致全程都慢”
- 收敛成“首块冷启动还有一小段额外成本”

这类尾巴后续如果还要继续抠，优先级应该低于主路径正确性和稳态带宽。

## 顺序写 microbench 复核

为了确认“当前慢点到底在首块，还是稳态顺序写本身”，又补跑了一次
独立的 BaM 写入 microbench。这个 microbench 的口径是：

- 同一个进程
- 固定 dummy chunk 形状：`[2, 28, 256, 512]`
- chunk 大小：`14 MiB`
- 每轮使用新的 `chunk_hash`
- 顺序写入新的 BaM 槽位
- `LMCacheBaMAdapter` 初始化放在计时循环外

日志：

- [lmcache_bam_write_microbench_20260622_235727.log](/home/xhk/llm-inference/vllm-bam/evaluation/logs/lmcache_bam_write_microbench_20260622_235727.log)

关键结果：

- `first_write=67.647 ms`，约 `0.202 GiB/s`
- `second_write=6.564 ms`，约 `2.083 GiB/s`
- `steady_state mean=4.198 ms`，约 `3.257 GiB/s`
- `steady_state median=3.616 ms`
- 最快单次约 `2.507 ms`，约 `5.453 GiB/s`

这和真实 `LMCache + BaM shadow` 日志里的稳态结果是对得上的：

- 真实链路稳态大多约 `3.2 ~ 3.6 GiB/s`
- microbench 稳态平均约 `3.26 GiB/s`

所以现在可以明确判断：

- 当前 `128KB page` 版本的主线问题已经不是稳态顺序写吞吐
- `BaM` 顺序写新槽位这条主链路已经基本正常
- 剩余尾巴主要就是首块冷启动

但这组数字暂时不要直接解读成：

- `BaM backend` 的最终性能
- `BaM` 读取链路已经优于 `LMCache SSD`

因为当前正式请求仍然是：

- `LMCache` 原始路径负责读写和返回
- `BaM` 只是在 `put` 时额外 shadow 一份

所以它本质上回答的是：

- 在现有 `LMCache SSD` 基线上，额外加上 `BaM shadow write` 后，整条链路能否稳定跑通

## 与后续工作的关系

这条 baseline 现在已经可以作为下一阶段的直接起点：

1. 保留当前 `LMCache + BaM shadow write`
2. 在 `LMCache load` 生命周期里优先从 `BaM` 读取
3. 如果 `BaM load` 失败或校验不通过，再 fallback 到原始 `LMCache SSD`

也就是说，下一步最合理的主线是：

- `LMCache SSD-only`
- `LMCache + BaM shadow write`
- `LMCache + Prefer BaM load`

## 当前一句话定位

截至现在，可以把这条结果写成：

- `vllm-bam` 上的单卡 `LMCache + BaM shadow write` 基线已经跑通，且已支持固定 256-token 槽位下的变长 chunk 写入。
