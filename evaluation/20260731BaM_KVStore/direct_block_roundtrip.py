#!/usr/bin/env python3
"""BaM KVStore：data-plane 与真实 CacheEngine direct SSD roundtrip。

测试包含两层：

1. vLLM-layout CUDA allocation -> direct block data plane -> exact_equal；
2. vLLM CacheEngine -> scheduler swap mapping -> direct KVStore -> exact_equal。

不导入 LMCache，不创建 BaM page cache，不经过 output-pages/refill/scatter。
"""

from __future__ import annotations

import os
import time

import torch

from bam_direct_kv_io import BaMDirectKVIO
from vllm.bam.direct_block_store import BaMDirectBlockStore


GPU_DMA_ALIGNMENT = 64 * 1024
SSD_RESERVE_BYTES = 64 * 1024 * 1024
POLL_TIMEOUT_S = 30.0


def allocate_aligned_kv_layer(
    *, num_blocks: int, block_elems: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """【BaM KVStore 直通调用链】分配测试用 64KB 对齐 KV tensor。

    owner 比最终 tensor 多保留一个 GPU page；aligned view 始终落在 owner
    allocation 内。这样 direct IO 只映射明确属于本测试的地址范围，不依赖
    PyTorch caching allocator 恰好返回 64KB 对齐的小 allocation。
    """
    shape = (2, num_blocks, block_elems)
    tensor_bytes = 2 * num_blocks * block_elems * torch.empty(
        (), dtype=dtype
    ).element_size()
    if tensor_bytes % GPU_DMA_ALIGNMENT != 0:
        raise ValueError(
            f"layer bytes must be 64KB aligned, got {tensor_bytes}"
        )

    owner = torch.empty(
        tensor_bytes + GPU_DMA_ALIGNMENT, dtype=torch.uint8, device="cuda"
    )
    byte_offset = (-owner.data_ptr()) % GPU_DMA_ALIGNMENT
    aligned_bytes = owner.narrow(0, byte_offset, tensor_bytes)
    layer = aligned_bytes.view(dtype).view(shape)
    if layer.data_ptr() % GPU_DMA_ALIGNMENT != 0:
        raise AssertionError("failed to create a 64KB-aligned CUDA view")
    return owner, layer


def wait_until_ready(direct_io: BaMDirectKVIO, handle: object) -> int:
    """【BaM KVStore 直通调用链】等待 GPU worker 发布 request ready。

    CPU 只读取 request 状态，不读取 NVMe CQ，也不推进数据传输。
    """
    deadline = time.monotonic() + POLL_TIMEOUT_S
    poll_count = 0
    while True:
        poll_count += 1
        if direct_io.poll(handle):
            return poll_count
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"direct KV batch did not become ready in {POLL_TIMEOUT_S}s"
            )
        time.sleep(0.001)


def main() -> None:
    """【BaM KVStore 直通调用链】验证独立 block data plane。"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    # Qwen2.5-7B / TP=1 / FP16 的典型 V0 paged-KV block：
    # block_size=16, num_kv_heads=4, head_size=128，因此每个 K/V fragment
    # 含 8192 个 FP16 元素，即 16KB。
    num_layers = 28
    num_gpu_blocks = 8
    block_elems = 16 * 4 * 128
    test_blocks = 4
    dtype = torch.float16
    source_block_ids = list(range(test_blocks))
    destination_block_ids = list(
        range(test_blocks, test_blocks * 2)
    )
    storage_block_ids = list(range(test_blocks))

    torch.cuda.set_device(0)
    torch.manual_seed(20260731)
    owners: list[torch.Tensor] = []
    gpu_cache: list[torch.Tensor] = []
    for _ in range(num_layers):
        owner, layer = allocate_aligned_kv_layer(
            num_blocks=num_gpu_blocks,
            block_elems=block_elems,
            dtype=dtype,
        )
        owners.append(owner)
        gpu_cache.append(layer)

    source_index = torch.tensor(
        source_block_ids, device="cuda", dtype=torch.long
    )
    expected_host: list[torch.Tensor] = []
    for layer_id, layer in enumerate(gpu_cache):
        layer.zero_()
        # 每层使用不同 seed 语义，避免“所有层内容相同”掩盖 SSD offset 错位。
        values = torch.randn(
            (2, test_blocks, block_elems),
            device="cuda",
            dtype=dtype,
        ) + float(layer_id)
        layer.index_copy_(1, source_index, values)
        expected_host.append(values.cpu())

    # persistent CQ service 启动后不允许再触发 cudaMalloc/cudaFree。旧 BaM
    # one-copy 也是先准备完 output/placement workspace，再启动常驻 worker。
    # marker 只用于验证常驻 1 CTA 时前台 CUDA kernel 仍能获得调度。
    marker = torch.zeros(1, device="cuda")

    direct_io = BaMDirectKVIO(
        ssd_list=(0,),
        request_capacity=512,
        region_capacity=32,
        queue_depth=4096,
        num_queues=128,
    )
    try:
        namespace_bytes = direct_io.namespace_size_bytes
        if namespace_bytes <= SSD_RESERVE_BYTES:
            raise RuntimeError(
                f"SSD namespace is too small: {namespace_bytes} bytes"
            )
        # 旧 row/page 路径从 namespace 低地址开始布局。新 direct roundtrip 只使用
        # 末尾 64MB 的独立测试窗口，并在下面再次检查实际 block extent 不越界。
        ssd_base_offset = (
            (namespace_bytes - SSD_RESERVE_BYTES) // 4096 * 4096
        )
        store = BaMDirectBlockStore(
            direct_io=direct_io,
            gpu_cache=gpu_cache,
            ssd_base_offset=ssd_base_offset,
        )
        test_end = (
            ssd_base_offset
            + test_blocks * store.layout.logical_block_bytes
        )
        if test_end > namespace_bytes:
            raise RuntimeError(
                f"test SSD range exceeds namespace: end={test_end}, "
                f"namespace={namespace_bytes}"
            )

        producer_stream = torch.cuda.current_stream()
        stream = torch.cuda.Stream()
        stream.wait_stream(producer_stream)
        print("[DIRECT_KV_ROUNDTRIP] begin")
        print(f"device={torch.cuda.get_device_name(0)}")
        print(f"namespace_bytes={namespace_bytes}")
        print(f"ssd_test_range=[{ssd_base_offset}, {test_end})")
        print(
            f"layers={num_layers} gpu_blocks={num_gpu_blocks} "
            f"test_blocks={test_blocks} "
            f"fragment_bytes={store.layout.fragment_bytes} "
            f"logical_block_bytes={store.layout.logical_block_bytes}"
        )
        print(
            "data_path=SSD<->vLLM_KV_direct "
            "bam_page_cache=0 staging=0 refill=0 "
            "gpu_cq_poll=1 cpu_cq_poll=0 cpu_data_move=0 cpu_ready_check=1"
        )

        write_start = time.perf_counter()
        write_handle = store.write_blocks(
            gpu_block_ids=source_block_ids,
            storage_block_ids=storage_block_ids,
            stream=stream,
        )
        stream.synchronize()
        write_polls = wait_until_ready(direct_io, write_handle.native_handle)
        store.finish(write_handle)
        write_ms = (time.perf_counter() - write_start) * 1000.0

        destination_begin = destination_block_ids[0]
        destination_end = destination_block_ids[-1] + 1
        read_start = time.perf_counter()
        read_handle = store.read_blocks(
            storage_block_ids=storage_block_ids,
            gpu_block_ids=destination_block_ids,
            stream=stream,
        )
        stream.synchronize()
        read_polls = wait_until_ready(direct_io, read_handle.native_handle)
        with torch.cuda.stream(stream):
            marker.fill_(1)
        stream.synchronize()
        compute_launch_ok = int(marker.item())
        store.finish(read_handle)
        read_ms = (time.perf_counter() - read_start) * 1000.0

        # marker 已验证 CPU 能在 ready 后发起前台 GPU 计算。exact-equal 是测试
        # 收尾阶段的逐层 D2H 校验，不属于推理链路；先正常停止常驻 service，
        # 避免 28 次同步 D2H 干扰对数据正确性的判断。
        direct_io.close()
        # persistent service 启动后不再创建 GPU reduction workspace。这里只把
        # 最终 vLLM block fragment 做一次 D2H 快照，并在 CPU 上执行测试校验。
        for layer_id, layer in enumerate(gpu_cache):
            restored_host = layer[:, destination_begin:destination_end, :].cpu()
            if not torch.equal(restored_host, expected_host[layer_id]):
                raise AssertionError(
                    f"direct KV block mismatch at layer {layer_id}"
                )

        total_bytes = test_blocks * store.layout.logical_block_bytes
        write_gib_s = total_bytes / (write_ms / 1000.0) / (1024**3)
        read_gib_s = total_bytes / (read_ms / 1000.0) / (1024**3)
        print(
            f"write blocks={test_blocks} fragments={write_handle.fragment_count} "
            f"bytes={total_bytes} elapsed_ms={write_ms:.3f} "
            f"bw_gib_s={write_gib_s:.3f} polls={write_polls}"
        )
        print(
            f"read blocks={test_blocks} fragments={read_handle.fragment_count} "
            f"bytes={total_bytes} elapsed_ms={read_ms:.3f} "
            f"bw_gib_s={read_gib_s:.3f} polls={read_polls}"
        )
        print(f"compute_launch_while_service_running={compute_launch_ok}")
        print("exact_equal=1")
        print("[DIRECT_KV_ROUNDTRIP] PASS")
    finally:
        direct_io.close()
        # owners 必须活到 DMA unmap 完成后；这一行也明确记录了生命周期约束。
        del owners


def run_cache_engine_roundtrip() -> None:
    """【BaM KVStore 直通调用链】验证真实 CacheEngine swap 接入口。

    这里不加载模型权重，但使用 Qwen2.5-7B 的真实 ``ModelConfig``、XFormers
    paged-KV shape 和 ``CacheEngine`` allocation。测试中的两张 mapping 与 V0
    scheduler 交给 Worker 的语义完全相同：write 为 GPU->storage，read 为
    storage->GPU。
    """
    os.environ["VLLM_BAM_DIRECT_KVSTORE_ENABLE"] = "1"
    os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
    os.environ.setdefault("VLLM_BAM_SSD_LIST", "0")

    from vllm.config import (CacheConfig, DeviceConfig, ModelConfig,
                             ParallelConfig)
    from vllm.worker.cache_engine import CacheEngine

    model_path = "/home/xhk/llm-inference/models/Qwen2.5-7B-Instruct"
    model_config = ModelConfig(
        model_path,
        task="auto",
        tokenizer=model_path,
        tokenizer_mode="auto",
        trust_remote_code=False,
        seed=0,
        dtype="float16",
        enforce_eager=True,
        max_model_len=4096,
    )
    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.9,
        swap_space=1,
        cache_dtype="auto",
    )
    cache_config.num_gpu_blocks = 8
    cache_config.num_cpu_blocks = 4

    cache_engine = CacheEngine(
        cache_config,
        model_config,
        ParallelConfig(),
        DeviceConfig("cuda"),
    )

    source_block_ids = list(range(4))
    destination_block_ids = list(range(4, 8))
    source_index = torch.tensor(source_block_ids,
                                device="cuda",
                                dtype=torch.long)
    destination_index = torch.tensor(destination_block_ids,
                                     device="cuda",
                                     dtype=torch.long)
    expected_host: list[torch.Tensor] = []
    torch.manual_seed(20260731)
    for layer_id, layer in enumerate(cache_engine.gpu_cache):
        values = torch.randn(
            (2, 4, layer.shape[2]), device="cuda", dtype=layer.dtype
        ) + float(layer_id)
        layer.index_copy_(1, source_index, values)
        expected_host.append(values.cpu())

    # persistent service 到首次 swap_out 才启动。所有测试 tensor、CPU 快照和
    # marker 均在此之前准备，复用真实服务“先完成 allocation，再启动 worker”
    # 的生命周期约束。
    marker = torch.zeros(1, device="cuda")
    write_mapping = torch.tensor(
        [[gpu_id, storage_id]
         for storage_id, gpu_id in enumerate(source_block_ids)],
        dtype=torch.int64,
        device="cpu",
    )
    read_mapping = torch.tensor(
        [[storage_id, gpu_id]
         for storage_id, gpu_id in enumerate(destination_block_ids)],
        dtype=torch.int64,
        device="cpu",
    )

    # 真实 Worker 会在 _warm_up_model() 返回后执行同一个显式初始化。测试没有
    # 加载模型，因此以上 KV 内容、CPU snapshot 和 marker 就是全部 CUDA
    # allocation；到这里才允许注册 DMA region。
    cache_engine.initialize_bam_direct_kv_store()
    direct_store = cache_engine.bam_direct_kv_store
    if direct_store is None:
        raise AssertionError("CacheEngine did not bind direct KVStore")

    try:
        print("[DIRECT_KV_CACHE_ENGINE] begin", flush=True)
        print("[DIRECT_KV_CACHE_ENGINE] swap_out", flush=True)
        cache_engine.swap_out(write_mapping)
        print("[DIRECT_KV_CACHE_ENGINE] swap_in", flush=True)
        cache_engine.swap_in(read_mapping)
        print("[DIRECT_KV_CACHE_ENGINE] attention_marker", flush=True)
        marker.fill_(1)
        torch.cuda.current_stream().synchronize()
        compute_launch_ok = int(marker.item())

        # 与推理链路相同，attention marker 只在 request ready 后发起。随后先
        # 正常停止 persistent service，再做测试专用的逐层 D2H exact 校验。
        direct_store.close()
        for layer_id, layer in enumerate(cache_engine.gpu_cache):
            restored_host = layer.index_select(1, destination_index).cpu()
            if not torch.equal(restored_host, expected_host[layer_id]):
                raise AssertionError(
                    f"CacheEngine direct KV mismatch at layer {layer_id}"
                )
        print(
            "cache_engine_layout="
            f"{tuple(cache_engine.gpu_cache[0].shape)} "
            f"stride={tuple(cache_engine.gpu_cache[0].stride())}"
        )
        print(f"compute_launch_after_ready={compute_launch_ok}")
        print("cpu_kv_payload_cache_layers=0")
        print("bam_page_cache=0 staging=0 refill=0 exact_equal=1")
        print("[DIRECT_KV_CACHE_ENGINE] PASS")
    finally:
        direct_store.close()


if __name__ == "__main__":
    main()
    run_cache_engine_roundtrip()
