# vLLM KV Cache 与 BaM SSD 数据组织总结

本文档总结当前阶段已经确认的两部分信息：

1. `vLLM V0` 在 `V100` 上的 KV cache 数据组织方式与块大小
2. 本地 `BaM_IOStack` 在 SSD 侧的数据组织方式，以及它对后续 KV `swap_out` / `swap_in` 设计的约束

本文档只基于当前本地代码与已经完成的实验，不引入外部实现假设。

---

## 1. vLLM 侧：KV cache 是怎样组织的

### 1.1 整体结构

当前 `vLLM V0` 的 `CacheEngine` 会分别维护：

- `gpu_cache`
- `cpu_cache`

二者都是：

- 一个按 `attention layer` 分层的列表
- 每一层对应一个 tensor

参考：
- [cache_engine.py](/home/xhk/llm-inference/vllm/vllm/worker/cache_engine.py:65)
- [cache_engine.py](/home/xhk/llm-inference/vllm/vllm/worker/cache_engine.py:95)

也就是说，`vLLM` 的 KV cache 不是“一个大 tensor 装下所有层”，而是：

- `gpu_cache[layer_id]`
- `cpu_cache[layer_id]`

每个 layer 都有自己的块数组。

### 1.2 swap 的基本单位

当前 `swap_in / swap_out` 的基本单位是 **block**。

`CacheEngine.swap_out()` / `swap_in()` 的逻辑都是：

- 遍历所有 attention layers
- 对每一层调用 `attn_backend.swap_blocks(...)`
- 使用同一份 `src_to_dst` block 映射

参考：
- [cache_engine.py](/home/xhk/llm-inference/vllm/vllm/worker/cache_engine.py:133)
- [cache_engine.py](/home/xhk/llm-inference/vllm/vllm/worker/cache_engine.py:140)

因此，一个逻辑上的 `KV block` 实际上是：

- 同一个 block id
- 在全部 attention layers 上对应的一组 layer-local slice

可以理解为：

`KV block = {layer_0_slice, layer_1_slice, ..., layer_(L-1)_slice}`

---

## 2. 当前实验下 vLLM KV block 的实际大小

### 2.1 已测得的基本参数

在当前实验环境下，已经确认：

- `block_size_tokens = 16`
- `block_bytes = 1835008`
- `num_attention_layers = 28`

参考：
- [V100_V0_CPU_SWAP_BASELINE.md](/home/xhk/llm-inference/vllm/evaluation/V100_V0_CPU_SWAP_BASELINE.md:47)

### 2.2 block 大小公式

`vLLM` 中 block 大小的计算方式来自：

```python
total = num_attention_layers * cache_config.block_size * \
    (key_cache_entry + value_cache_entry)
return dtype_size * total
```

参考：
- [cache_engine.py](/home/xhk/llm-inference/vllm/vllm/worker/cache_engine.py:151)

这里的含义是：

- 一个 block 里有 `block_size_tokens` 个 token slot
- 每个 token slot 在每层都要存 KV
- 每层的 KV 大小由：
  - `num_kv_heads`
  - `head_size`
  - `dtype_size`
 共同决定

### 2.3 每层 slice 的实际大小

根据当前测量值：

```text
per_layer_slice_bytes = block_bytes / num_attention_layers
                      = 1835008 / 28
                      = 65536 B
                      = 64 KiB
```

这说明当前这版 `Qwen3-0.6B + float16 + block_size=16` 下：

- **一个完整 KV block 大小约为 `1.75 MiB`**
- **其中每个 attention layer 对应一个 `64 KiB` 的 layer slice**

这 64KiB 已经是该层完整的 `key + value` 之和，不需要再继续拆成“key 64KiB + value 64KiB”。

---

## 3. vLLM 当前 swap 的观测结果

已经完成的 CPU swap 基线表明：

- 大批量单向 `swap_out ≈ 0.42 ms/block`
- 大批量单向 `swap_in ≈ 0.424 ms/block`
- 往返 `round_trip ≈ 0.84 ms/block`
- 有效带宽约 `4.0 GiB/s`

参考：
- [V100_V0_CPU_SWAP_BASELINE.md](/home/xhk/llm-inference/vllm/evaluation/V100_V0_CPU_SWAP_BASELINE.md:113)

因此后续若引入 `BaM`，需要记住当前主比较口径是：

- 每 block 的单向搬运成本
- 每 block 的往返成本
- 大批量时的有效带宽

---

## 4. BaM 侧：SSD 数据是怎样组织的

### 4.1 page / range / array 的基本抽象

本地 `BaM_IOStack` 的核心 SSD 数据组织抽象不是“对象存储”，而是：

- `page_cache_t`
- `range_t<T>`
- `array_t<T>`

其中：

- `page_cache_t` 管理 GPU 侧缓存页与底层 I/O
- `range_t<T>` 定义一个逻辑连续的数据区间
- `array_t<T>` 提供按逻辑索引访问数据的抽象

参考：
- [page_cache.h](/home/xhk/llm-inference/BaM_IOStack/bam/include/page_cache.h:1760)
- [page_cache.h](/home/xhk/llm-inference/BaM_IOStack/bam/include/page_cache.h:1821)
- [page_cache.h](/home/xhk/llm-inference/BaM_IOStack/bam/include/page_cache.h:1852)

其核心地址映射方式是：

- 逻辑元素索引 `i`
- 经 `sizeof(T)` 和 `page_size` 换算
- 找到所属 page 与 page 内 offset

例如：

```cpp
page = ((i - index_start) * sizeof(T) + page_start_offset) >> page_size_log
subindex = ((i - index_start) * sizeof(T) + page_start_offset) & page_size_minus_1
```

参考：
- [page_cache.h](/home/xhk/llm-inference/BaM_IOStack/bam/include/page_cache.h:1852)
- [page_cache.h](/home/xhk/llm-inference/BaM_IOStack/bam/include/page_cache.h:1861)

### 4.2 底层写 I/O 的能力

`BaM` 底层实际支持的写语义是：

- 给定 `starting_lba`
- 给定 `n_blocks`
- 给定一个已准备好的 GPU-side DMA buffer / cache entry
- 发起 `NVM_IO_WRITE`

参考：
- [page_cache.h](/home/xhk/llm-inference/BaM_IOStack/bam/include/page_cache.h:5115)

因此从底层能力上说，`BaM` 支持：

- 把一段 GPU 上的数据
- 写到 SSD 上某个明确的 LBA 区间

这与 `vLLM` 的 `swap_out` 需求在能力层面是匹配的。

---

## 5. BaM 上层的 row 语义与当前约束

### 5.1 本地代码里 row 的痕迹

本地 `BaM_IOStack` 的高层异步/registered 路径里，存在明显的 row 语义：

- `s_ctx.row_index`
- `dim`
- `cache_dim`
- `key_off`

参考：
- [bam_iostack.cuh](/home/xhk/llm-inference/BaM_IOStack/bam/include/bam_iostack.cuh:37)
- [bam_iostack.cuh](/home/xhk/llm-inference/BaM_IOStack/bam/include/bam_iostack.cuh:98)
- [bam_iostack.cuh](/home/xhk/llm-inference/BaM_IOStack/bam/include/bam_iostack.cuh:144)

这说明在你当前要复用的那套 `BaM` 上层抽象里，更自然的数据视角是：

- 数据按“行”组织
- 行内再由 `dim / cache_dim` 去解释

### 5.2 已知约束：单 row 最大 128KB

在你当前对系统的理解中，`BaM` 可自然支持的单 row 大小上限约为：

- `128KB`

这会直接限制 KV 在 SSD 上的组织方式：

- 不能把一个完整 `vLLM KV block` 当成一行
- 因为一个完整 block 约 `1.75 MiB`

同时，这个上限也提供了一个非常有用的判断：

- 当前每层 slice 是 `64KB`
- `64KB < 128KB`

因此：

- **一层一个 row 可行**
- **一个完整 block 一个 row 不可行**

---

## 6. 结合 vLLM 与 BaM，最合适的 SSD 组织方式

### 6.1 不合适的方式

#### 方式 A：一个完整 KV block 对应一个 row

不合适，原因是：

- `block_bytes ≈ 1.75 MiB`
- 超出 `BaM` 当前 row 上限 `128KB`

#### 方式 B：多个 layers 拼成一个 row

例如：

- 2 层合并为 1 个 row
- `2 * 64KB = 128KB`

从大小上刚好卡住上限，但当前不建议第一版采用，原因是：

- 容易卡边界，不够稳
- 不够通用，未来换模型后可能越界
- 与 `vLLM` 当前按 layer 持有 cache 的方式不够自然

### 6.2 推荐方式：一层一个 row

当前最合适的方式是：

- **一个 row 对应一个 block 在一个 attention layer 上的完整 KV slice**

即：

```text
row(slot_id, layer_id) = kv_block(slot_id).layer_slice(layer_id)
row_bytes = 64KB
```

这样：

- 一个完整 KV block 对应 `28` 个 rows
- 每个 row 是 `64KB`
- 落在 `BaM` 当前 `128KB` row 上限之内

---

## 7. 推荐的逻辑映射公式

### 7.1 逻辑 ID 映射

建议定义：

- `slot_id`: 某个被换出的 KV block 在 SSD 上占用的槽号
- `layer_id`: 当前层编号，范围为 `[0, num_layers - 1]`

然后定义：

```text
row_id = slot_id * num_layers + layer_id
```

这表示：

- 同一个 `slot_id` 的 28 个连续 rows
- 共同构成一个完整 KV block

### 7.2 row 大小

建议固定：

```text
row_bytes = 64KB
```

即：

- 一行正好存一个 layer slice

### 7.3 完整 block 的 SSD 占用

一个 block 占用：

```text
num_layers * row_bytes = 28 * 64KB = 1.75MB
```

这与当前 `vLLM` 实测 block 大小一致。

---

## 8. 这个组织方式为什么适合当前第一版 BaM 接入

### 8.1 与 vLLM 当前 cache 结构匹配

`vLLM` 当前就是：

- `gpu_cache[layer_id]`
- `cpu_cache[layer_id]`

因此在 `swap_out` 时，很自然就能拿到“每层一片”的数据视图。

### 8.2 与 BaM row 上限匹配

当前每层 `64KB`：

- 小于 `128KB`
- 无需把一个 block 再拆得更细

### 8.3 不需要先改 BaM 的 page cache 核心状态机

第一版真正需要新增的是：

- KV 专用 row layout
- `slot_id / layer_id -> row_id`
- `row_id -> SSD offset / LBA`

而不是先改 `page_cache.h` 的核心状态管理与替换逻辑。

---

## 9. 第一版设计建议

### 9.1 SSD 上的数据组织

第一版推荐：

- 一个 KV block 分配一个 `slot_id`
- 一个 block 在 SSD 上拆成 `num_layers` 行
- 每行存该 block 某一层的 `64KB` slice

即：

- `slot_id = 0` 对应 rows `[0, 27]`
- `slot_id = 1` 对应 rows `[28, 55]`
- 以此类推

### 9.2 swap_out 的写入方式

当一次 `swap_out` 有 `N` 个 blocks 时：

1. 为每个 block 分配 `slot_id`
2. 对每个 `layer_id`
3. 取出这一层中本轮要换出的 `N` 个 `64KB` slices
4. 将它们分别写到对应的：
   - `row(slot_id, layer_id)`

从调试和实现角度上说，这相当于：

- 逻辑上按 row 组织
- 实际上仍可以按 layer 批量提交

### 9.3 swap_in 的恢复方式

后续若做 `swap_in`，则：

1. 根据 block 的 `slot_id`
2. 读取这 28 个 rows
3. 将它们分别放回 `gpu_cache[layer_id]` 的对应 block 位置

因此当前推荐的 SSD 布局，同时兼容未来的 `swap_in`。

---

## 10. 当前结论

### vLLM 侧

- 当前 `vLLM V0` 的 KV cache 是按 layer 分层存放的
- 一个逻辑 block 覆盖全部 `28` 个 attention layers
- 当前实验配置下：
  - `block_size_tokens = 16`
  - `block_bytes = 1835008 ≈ 1.75 MiB`
  - `per_layer_slice = 64 KiB`

### BaM 侧

- 当前本地 `BaM_IOStack` 的底层写能力支持：
  - 将给定 GPU-side buffer 写到指定 SSD LBA 范围
- 当前上层更自然的数据抽象是：
  - page / range / array
  - 以及带 `row_index / dim / cache_dim` 的 row 语义
- 在已知“单 row 最大约 `128KB`”的前提下：
  - 一个完整 `vLLM block` 不能作为一行
  - 一个 `64KB` 的 layer slice 可以作为一行

### 推荐的联合组织方式

- **一个 KV block 对应一个 `slot_id`**
- **一个 block 在 SSD 上拆成 `28` 个 rows**
- **每个 row 对应一个 layer 的 `64KB` KV slice**
- **推荐公式：**

```text
row_id = slot_id * num_layers + layer_id
```

这就是当前阶段最贴合：

- `vLLM` 现有 KV 结构
- `BaM` 现有 row / array 语义
- `128KB row` 上限

的 SSD 组织方式。
