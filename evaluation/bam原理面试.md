# BaM 原理面试整理

## 1. 一句话说明 BaM

BaM 的核心是 **GPU-Initiated SSD I/O**：

```text
传统路径：CPU 决定 I/O -> CPU 调用存储 API -> SSD DMA 到 GPU -> GPU 消费
BaM 路径：GPU 线程直接发起 NVMe 请求 -> SSD DMA 到 GPU -> GPU 侧推进完成状态
```

它想解决的问题不是“能不能从 SSD 读到 GPU”，而是：

```text
当访问模式很细、很随机、强依赖 GPU 计算结果时，
不要让 CPU 成为每个小 I/O 的提交者和同步点。
```

因此 BaM 更适合 GNN、图算法、稀疏数据访问、长上下文 sparse KV restore 这类 workload。

## 2. BaM 论文里的核心问题

BaM 论文讨论的是 GPU 访问外部存储时的三个矛盾：

1. GPU 算得很快，但 GPU HBM 容量有限；
2. 数据在 SSD 上足够大，但传统 I/O 路径通常由 CPU 发起；
3. 很多 GPU workload 的访问地址要在 GPU 计算过程中才知道。

传统 CPU-initiated I/O 会导致：

- CPU/GPU 频繁同步；
- 小粒度 I/O 的 API 和驱动开销占比高；
- 很难维持足够高的 NVMe queue depth；
- GPU 需要等待 CPU 把下一批 I/O 请求整理好；
- 数据依赖越强，CPU 调度越容易成为瓶颈。

BaM 的思路是把 NVMe queue 暴露给 GPU，让 GPU 线程直接提交存储请求，并通过 GPU 侧软件 cache 和 completion 机制管理这些请求。

## 3. BaM 和传统 GDS 的区别

### 3.1 传统 GDS 解决了什么

传统 GPUDirect Storage 主要解决：

```text
SSD -> CPU memory -> GPU memory
```

这条路径里的 CPU 内存中转问题。使用 GDS 后，数据可以直接 DMA 到 GPU buffer：

```text
SSD -> GPU memory
```

这对大块顺序读非常有效。

### 3.2 传统 GDS 仍然保留的问题

多数 GDS 使用方式下，I/O 仍然由 CPU 发起：

```text
CPU calls cuFileRead / cuFileWrite
  -> GDS driver
  -> NVMe
  -> DMA to GPU buffer
  -> CPU 或 CUDA stream 观察完成
  -> GPU 消费数据
```

所以它虽然绕过了 CPU 内存拷贝，但没有完全绕过 CPU 控制面。

小粒度随机 I/O 下，这会暴露几个问题：

- 每个小 I/O 都有 API、driver、文件系统或 block layer 开销；
- 每笔 I/O 的 payload 很小，固定开销占比很高；
- GPU 生成访问地址后，还要通知 CPU 发起 I/O；
- CPU 组织 batch 的速度可能跟不上 GPU 消费节奏；
- queue depth 不够时，SSD 很难跑满带宽；
- chunk 粒度偏粗时，会读入 attention 实际不需要的数据。

### 3.3 BaM 的不同点

BaM 不只是 “direct DMA to GPU”，而是进一步做：

```text
GPU 生成或消费 I/O descriptor
GPU 写 NVMe SQ
GPU ring doorbell
GPU poll CQ
GPU 更新 completion/status
GPU 侧或上层 runtime 消费数据
```

也就是说，BaM 把高频 I/O 数据面从 CPU 下沉到了 GPU。

## 4. 当前代码里的 BaM 底层机制

### 4.1 GPU 侧 NVMe SQ submit

底层核心在：

```text
BaM_IOStack/bam/include/nvm_parallel_queue.h
```

其中 `sq_enqueue()` 是 device 函数，运行在 GPU 上。它做的事情是：

```text
1. GPU 线程通过 atomic ticket 获取 SQ slot
2. 把 64B NVMe command 写入 SQ memory
3. 推进 SQ tail
4. 通过 MMIO 写 SQ doorbell
```

这说明 BaM 的提交路径不是 CPU 调系统调用完成的，而是 GPU kernel 内部直接写 NVMe submission queue。

### 4.2 GPU 侧 CQ service

completion 侧对应：

```text
cq_try_peek_head()
cq_dequeue()
put_cid()
finalize completion
```

当前代码里 CQ service kernel 会读取 NVMe CQ entry，根据 cid 找到对应 request，然后把状态推进为 DONE。

这一步很关键：

```text
如果 submit 在 GPU，但 completion 还要 CPU 高频轮询，
细粒度 I/O 的收益会被 CPU poll 和同步吃掉。
```

BaM 的设计是让 GPU 也能推进 completion。

### 4.3 多 queue 和 queue depth

当前初始化里常见配置是：

```text
queue_depth = 4096
num_queues = 128
```

这说明 BaM 不是靠单个小请求跑带宽，而是靠大量 outstanding requests 把 SSD 喂满。

细粒度随机 I/O 想要高带宽，需要满足：

```text
请求足够多
queue depth 足够高
completion 回收足够快
上层能持续 refill 新请求
```

BaM 的 SQ/CQ 设计正是围绕这个目标组织的。

## 5. 当前 KV direct 路径里的 descriptor pool

当前 `BaM_IOStack/gids_module/bam_direct_kv_io.py` 是 KV direct I/O 的 Python 包装。

它的定位是：

```text
只承载 SSD <-> CUDA direct I/O，
不承载完整 KV cache 语义。
```

上层传入一批 fragment descriptor：

```text
operation
ssd_byte_offset
region_id
region_offset
length
```

底层维护一个 native descriptor pool。每次提交不是马上同步等待一个 I/O，而是：

```text
1. 从 descriptor pool 申请槽位
2. 把请求写入 GPU-visible request table
3. 发布 SUBMITTED 状态
4. GPU service 看到请求后发 NVMe command
5. completion 回来后根据 cid 找到 descriptor
6. 标记 DONE
7. progress() 统一回收完成的 slots
```

这套机制的价值是：

- 上层可以提交很多小 fragment；
- descriptor slot 可以乱序复用；
- completion 可以乱序回来；
- poll/progress 不需要逐个 handle 做昂贵同步；
- 多个小 I/O 可以形成持续的 in-flight 流。

## 6. direct KV 路径为何适合细粒度 KV restore

KV restore 的需求天然是 block/fragment 级的：

```text
一个 logical KV block
  -> 多层 K/V 数据
  -> 对应 SSD 上一段或多段 byte range
  -> 恢复到 GPU KV cache 的某个 region offset
```

传统 chunk 路径通常是：

```text
按照较大的 chunk 恢复
即使 attention 只需要其中一部分 block，也可能读整个 chunk
```

BaM direct KV 路径可以表达：

```text
只读 selected blocks
只读当前 layer/window 需要的 fragments
读到指定 GPU region
完成后只发布这些 blocks READY
```

因此它适合后续 sparse attention：

```text
SparseKVAccessPlan
  -> fragment descriptors
  -> BaM descriptor pool
  -> GPU-initiated SSD restore
  -> per-layer sparse block table
```

## 7. BaM page cache / rowctx 路径的作用

除了 direct KV 路径，BaM 原生还有 page cache / rowctx 机制。

page cache 里有：

```text
INVALID
BUSY
VALID
DIRTY
ref_count
polling_flag
```

它的作用是：

```text
多个 GPU thread 访问同一个 page 时，
不要每个线程都重复发 SSD I/O。
```

典型流程：

```text
cache hit:
  直接返回 GPU page cache 地址

cache miss:
  一个线程抢到 page slot
  把 page 标成 BUSY
  发起 NVMe read
  completion 后标成 VALID
  其他线程等待或复用这个 page
```

这就是 BaM 论文里强调的 software cache 思路：它保留细粒度访问接口，但底层会把重复访问和并发访问组织起来。

## 8. 为什么传统 GDS 小粒度随机 I/O 差

### 8.1 固定开销占比高

假设每次只读 4KB、16KB 或 64KB，那么真正的数据传输很快，慢的是：

```text
CPU API 调用
driver 路径
参数检查
文件 offset / GPU pointer 处理
completion 管理
CPU/GPU 同步
```

payload 越小，这些固定开销占比越高。

### 8.2 动态访问地址要从 GPU 回到 CPU

sparse attention / GNN / graph traversal 里，下一批要读什么经常由 GPU 计算结果决定。

传统 GDS 需要：

```text
GPU 算出 block ids
  -> 同步或传给 CPU
  -> CPU 调 cuFileRead
  -> SSD DMA 到 GPU
  -> GPU 继续算
```

这个闭环太长，不适合高频小请求。

### 8.3 queue depth 很难持续拉满

SSD 需要足够多 outstanding requests 才能跑出带宽。传统路径如果是一笔笔小 I/O：

```text
submit
wait
submit
wait
```

设备很容易空转。即使用 CPU batch，也会受 CPU 组织频率、同步点和上层调度节奏影响。

BaM 则更自然：

```text
GPU 上大量 request descriptors
  -> 多 queue 并发 submit
  -> CQ service 持续回收
  -> descriptor pool 持续 refill
```

### 8.4 chunk 粒度导致 I/O amplification

很多 GDS-based KV 系统为了管理简单，会以 chunk/object 为恢复单位。  
如果 chunk 是 256 tokens，但 sparse attention 只需要其中几个 block，仍可能恢复整个 chunk。

BaM 的 block/fragment descriptor 可以更贴近实际消费粒度：

```text
需要哪个 block，就恢复哪个 block；
需要哪个 layer/window，就恢复哪个 layer/window；
不需要的 prefix KV 不恢复。
```

## 9. 面试里可以这样讲

较短版本：

```text
BaM 和传统 GDS 的区别在于，GDS 主要解决 SSD 到 GPU 的数据通路，
但 I/O 发起和 completion 管理通常仍由 CPU 控制；BaM 则把 NVMe SQ/CQ
暴露给 GPU，让 GPU 线程直接提交 I/O、服务 completion，并配合软件 cache
和 descriptor pool 管理大量细粒度请求。因此在 GNN 或 sparse KV restore
这种 GPU 计算动态决定访问地址的场景下，BaM 可以减少 CPU-GPU 同步，
维持更高 queue depth，降低小随机 I/O 的固定开销和 I/O 放大。
```

更贴近当前项目的版本：

```text
在长上下文 LLM 推理里，sparse attention 每层可能只需要部分历史 KV block。
如果走传统 chunk/GDS 路径，容易按较粗 chunk 恢复，读入很多不用的 KV；
而 BaM 可以把这些 block 访问转成 fragment descriptor，由 GPU-side SQ/CQ
和 MDS runtime 直接推进 SSD 到 GPU 的恢复。这样既保留细粒度访问，又通过
descriptor pool、多队列和高 in-flight 保持 SSD 带宽，是后续做 sparse KV
restore 和 layerwise overlap 的基础。
```

## 10. 当前实现边界

面试时要注意不要说过头。

当前可以说：

- 已经打通 BaM/MDS SSD 到 GPU KV restore 链路；
- 已经支持 block/fragment 级 descriptor 组织；
- 已经能做 partial-block restore profiling；
- 已经观察到部分 block 访问能降低 physical restore 时间；
- 当前代码已经把 sparse access plan 和 layer-window restore 初步统一。

当前不要说：

- 已经完整实现 end-to-end sparse attention；
- 已经完全由 GPU 自主生成所有 I/O 请求；
- 已经完成真正物理 KV eviction；
- 已经完全复现 Tutti 的 GPU-side rolling activation。

更准确的表述是：

```text
底层 BaM 已具备 GPU-side NVMe submit/CQ primitive；
当前 KV 路线正在把控制面从 CPU batch 推进逐步下沉到 GPU-visible
descriptor 和 completion table；
短期先允许 CPU 做粗粒度调度，但让高频 I/O 数据面由 BaM/MDS 执行。
```
