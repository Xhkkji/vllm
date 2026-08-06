# SPDX-License-Identifier: Apache-2.0
"""CacheEngine class for managing the KV cache."""
import time
from typing import List

import torch

import vllm.envs as envs
from vllm.attention import get_attn_backend
from vllm.config import CacheConfig, DeviceConfig, ModelConfig, ParallelConfig
from vllm.core.scheduler_policy import (AsyncKVTransferEvent,
                                        AsyncKVTransferOperation,
                                        AsyncKVTransferState)
from vllm.logger import init_logger
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, LayerBlockType,
                        get_dtype_size, is_pin_memory_available)

logger = init_logger(__name__)


class CacheEngine:
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        device_config: DeviceConfig,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.device_config = device_config
        self.bam_direct_kvstore_enabled = envs.VLLM_BAM_DIRECT_KVSTORE_ENABLE
        self.bam_mds_enabled = envs.VLLM_BAM_MDS_ENABLE
        if self.bam_direct_kvstore_enabled and self.bam_mds_enabled:
            raise ValueError(
                "VLLM_BAM_MDS_ENABLE and VLLM_BAM_DIRECT_KVSTORE_ENABLE are "
                "mutually exclusive")
        if (self.bam_direct_kvstore_enabled or self.bam_mds_enabled) and (
                envs.VLLM_BAM_SHADOW_ENABLE
                or envs.VLLM_BAM_SWAPIN_ENABLE):
            raise ValueError(
                "MDS/direct KVStore cannot be combined with the legacy BaM "
                "cache-backed V0 swap path")
        if self.bam_mds_enabled and (
                parallel_config.tensor_parallel_size != 1
                or parallel_config.pipeline_parallel_size != 1):
            raise ValueError(
                "Phase 4A MDS single-slot connector requires TP=1 and PP=1")

        # 【BaM KVStore 直通调用链】owner / DMA region 只为真实 GPU KV cache
        # 保活。普通 vLLM、LMCache 和 GDS 路径不会创建或读取这两张表。
        self._bam_direct_gpu_cache_owners: List[torch.Tensor] = []
        self._bam_direct_gpu_cache_regions: List[torch.Tensor] = []
        self.bam_mds_connector = None
        self._bam_mds_transfer_started_at: float | None = None
        # 以下字段只用于异步 KV 延迟分解，不参与 block 或 I/O 状态管理。
        self._bam_mds_async_kv_submitted_at_ns: int | None = None
        self._bam_mds_async_kv_first_poll_at_ns: int | None = None
        self._bam_mds_async_kv_operation: AsyncKVTransferOperation | None = None

        self.head_size = model_config.get_head_size()
        # Models like Jamba, have mixed typed layers, E.g Mamba
        self.num_attention_layers = model_config.get_num_layers_by_block_type(
            parallel_config, LayerBlockType.attention)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        if self.num_gpu_blocks:
            self.num_gpu_blocks //= parallel_config.pipeline_parallel_size
        self.num_cpu_blocks = cache_config.num_cpu_blocks
        if self.num_cpu_blocks:
            self.num_cpu_blocks //= parallel_config.pipeline_parallel_size

        if cache_config.cache_dtype == "auto":
            self.dtype = model_config.dtype
        else:
            self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        # Get attention backend.
        self.attn_backend = get_attn_backend(self.head_size,
                                             model_config.dtype,
                                             cache_config.cache_dtype,
                                             self.block_size,
                                             model_config.is_attention_free,
                                             use_mla=model_config.use_mla)

        # Initialize the cache.
        self.gpu_cache = self._allocate_kv_cache(
            self.num_gpu_blocks, self.device_config.device_type)
        # 【BaM KVStore 直通调用链】scheduler 仍使用 CPU block id 作为稳定的
        # storage block id，但 payload 已经落到 SSD，因此不再分配等大的 CPU
        # KV tensor。关闭新开关时仍完整保留 vLLM 原生 CPU swap baseline。
        self.cpu_cache = (
            [] if (self.bam_direct_kvstore_enabled or self.bam_mds_enabled) else
            self._allocate_kv_cache(self.num_cpu_blocks, "cpu")
        )
        self.swap_trace_enabled = envs.VLLM_V0_SWAP_TRACE
        # 【BaM KVStore 直通调用链】此处只完成真实 KV allocation，不立刻执行
        # NVMe controller 初始化或 DMA registration。Worker 必须等模型 warmup、
        # CUDA Graph capture 和全部 workspace allocation 完成后，再显式调用
        # initialize_bam_direct_kv_store()。这样不会让 BaM runtime 介入后续
        # cudaMalloc/capture 生命周期。
        self.bam_direct_kv_store = None
        self.bam_block_store = self._init_bam_block_store()
        self.bam_shadow_writer = self._init_bam_shadow_writer()
        self.bam_swap_reader = self._init_bam_swap_reader()

    def initialize_bam_direct_kv_store(self) -> None:
        """在所有 CUDA warmup 后启用当前选中的 BaM KVStore transport。"""
        if self.bam_mds_enabled:
            assert self.bam_mds_connector is not None
            self.bam_mds_connector.start()
            return
        if not self.bam_direct_kvstore_enabled:
            return
        if self.bam_direct_kv_store is not None:
            return
        from vllm.bam.direct_block_store import BaMVLLMDirectKVStore
        self.bam_direct_kv_store = BaMVLLMDirectKVStore(
            gpu_cache=self.gpu_cache,
            dma_regions=self._bam_direct_gpu_cache_regions,
            num_storage_blocks=self.num_cpu_blocks,
        )

    def _init_bam_block_store(self):
        if not (envs.VLLM_BAM_SHADOW_ENABLE or envs.VLLM_BAM_SWAPIN_ENABLE):
            return None

        from vllm.worker.bam_block_store import BaMBlockStore
        return BaMBlockStore(self.gpu_cache, self.num_cpu_blocks)

    def _init_bam_shadow_writer(self):
        if not envs.VLLM_BAM_SHADOW_ENABLE:
            return None

        from vllm.worker.bam_shadow_writer import BaMShadowWriter
        assert self.bam_block_store is not None
        return BaMShadowWriter(self.bam_block_store, self.dtype)

    def _init_bam_swap_reader(self):
        if not envs.VLLM_BAM_SWAPIN_ENABLE:
            return None

        from vllm.worker.bam_swap_reader import BaMSwapReader
        assert self.bam_block_store is not None
        return BaMSwapReader(self.bam_block_store, self.dtype)

    def _log_swap_event(self, op_name: str, src_to_dst: torch.Tensor,
                        elapsed_s: float) -> None:
        """记录一次 swap 的核心信息，便于观察 block 粒度和耗时。"""
        if not self.swap_trace_enabled:
            return

        num_mappings = src_to_dst.shape[0]
        block_bytes = self.get_cache_block_size(self.cache_config,
                                                self.model_config,
                                                self.parallel_config)
        total_bytes = num_mappings * block_bytes
        logger.info(
            "[V0_SWAP_TRACE][CacheEngine] op=%s mappings=%d "
            "block_size=%d block_bytes=%d total_bytes=%d "
            "num_attention_layers=%d elapsed_ms=%.3f",
            op_name,
            num_mappings,
            self.block_size,
            block_bytes,
            total_bytes,
            self.num_attention_layers,
            elapsed_s * 1000,
        )

    def _allocate_kv_cache(
        self,
        num_blocks: int,
        device: str,
    ) -> List[torch.Tensor]:
        """Allocates KV cache on the specified device.

        【BaM KVStore 直通调用链】新后端开启时，GPU 分配改用 64KB 对齐 owner，
        但返回给 attention 的 tensor shape/stride 与原生分配完全一致。
        """
        kv_cache_generic_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_kv_heads, self.head_size)
        pin_memory = is_pin_memory_available() if device == "cpu" else False
        kv_cache: List[torch.Tensor] = []
        try:
            kv_cache_stride_order = self.attn_backend.get_kv_cache_stride_order(
            )
        except (AttributeError, NotImplementedError):
            kv_cache_stride_order = tuple(range(len(kv_cache_generic_shape)))

        # The allocation respects the backend-defined stride order to ensure
        # the semantic remains consistent for each backend. We first obtain the
        # generic kv cache shape and then permute it according to the stride
        # order which could result in a non-contiguous tensor.
        kv_cache_allocation_shape = tuple(kv_cache_generic_shape[i]
                                          for i in kv_cache_stride_order)

        if self.bam_mds_enabled and device == "cuda":
            from vllm.bam.mds.connector import BaMMDSConnector
            self.bam_mds_connector = BaMMDSConnector(
                allocation_shape=kv_cache_allocation_shape,
                stride_order=kv_cache_stride_order,
                dtype=self.dtype,
                device_index=self.device_config.device.index or 0,
                num_layers=self.num_attention_layers,
                num_gpu_blocks=num_blocks,
                num_storage_blocks=self.num_cpu_blocks,
            )
            return self.bam_mds_connector.gpu_cache

        for _ in range(self.num_attention_layers):
            # null block in CpuGpuBlockAllocator requires at least that
            # block to be zeroed-out.
            # We zero-out everything for simplicity.
            if self.bam_direct_kvstore_enabled and device == "cuda":
                from vllm.bam.direct_block_store import (
                    allocate_aligned_kv_region)
                owner, dma_region, layer_kv_cache = (
                    allocate_aligned_kv_region(
                        allocation_shape=kv_cache_allocation_shape,
                        stride_order=kv_cache_stride_order,
                        dtype=self.dtype,
                        device=device,
                    ))
                self._bam_direct_gpu_cache_owners.append(owner)
                self._bam_direct_gpu_cache_regions.append(dma_region)
            else:
                layer_kv_cache = torch.zeros(
                    kv_cache_allocation_shape,
                    dtype=self.dtype,
                    pin_memory=pin_memory,
                    device=device).permute(*kv_cache_stride_order)

            # view back to (TOTAL_PAGES, PAGE_SIZE, entry_shape...) for cases
            # when entry_shape is higher than 1D
            kv_cache.append(layer_kv_cache)
        return kv_cache

    def swap_in(self, src_to_dst: torch.Tensor) -> None:
        """把 scheduler 的 storage->GPU block mapping 恢复到 KV cache。

        【BaM KVStore 直通调用链】新后端下，本函数等待的只是 GPU worker 发布
        request ready；CPU 不读取 NVMe CQ，也不参与 payload 搬运。返回后 Worker
        才会继续发起当前 engine step 的 attention。
        """
        start = time.perf_counter()
        if self.bam_mds_connector is not None:
            self.bam_mds_connector.swap_in(src_to_dst)
        elif self.bam_direct_kv_store is not None:
            self.bam_direct_kv_store.swap_in(src_to_dst)
        elif self.bam_swap_reader is not None:
            self.bam_swap_reader.swap_in(self.gpu_cache, self.cpu_cache,
                                         src_to_dst)
        else:
            for i in range(self.num_attention_layers):
                self.attn_backend.swap_blocks(self.cpu_cache[i],
                                              self.gpu_cache[i], src_to_dst)
        self._log_swap_event("swap_in", src_to_dst, time.perf_counter() - start)

    def swap_in_async(self, src_to_dst: torch.Tensor) -> bool:
        """推进 resident MDS swap-in，未完成时由 engine defer 当前 batch。

        该接口只服务 MDS direct 路径。目标 GPU block 在返回 True 前不能被
        attention 消费；SSD completion 和数据可见性由 daemon 内的常驻 GPU CQ
        service 保证。
        """
        if self.bam_mds_connector is None:
            raise RuntimeError("async swap-in is only supported by resident MDS")
        if self._bam_mds_transfer_started_at is None:
            self._bam_mds_transfer_started_at = time.perf_counter()
        ready = self.bam_mds_connector.swap_in_async(src_to_dst)
        if not ready:
            return False
        elapsed_s = time.perf_counter() - self._bam_mds_transfer_started_at
        self._bam_mds_transfer_started_at = None
        self._log_swap_event("swap_in", src_to_dst, elapsed_s)
        return True

    def submit_async_kv_transfer(
        self,
        request_id: str,
        operation: AsyncKVTransferOperation,
        src_to_dst: torch.Tensor,
    ) -> AsyncKVTransferEvent:
        """把 Scheduler 的异步 read/write 提交给 resident MDS。

        该方法只负责控制面提交，不等待 SSD 数据完成。目标 GPU block 已经
        由 AsyncKVScheduler 预留，因此从提交开始到 READY 之前都禁止
        attention 使用这些 block。
        """
        if self.bam_mds_connector is None:
            raise RuntimeError(
                "async KV scheduling requires the resident MDS connector")
        if self._bam_mds_transfer_started_at is not None:
            raise RuntimeError(
                "CacheEngine already has a pending async KV transfer")

        self._bam_mds_transfer_started_at = time.perf_counter()
        self._bam_mds_async_kv_submitted_at_ns = time.monotonic_ns()
        self._bam_mds_async_kv_first_poll_at_ns = None
        self._bam_mds_async_kv_operation = operation
        if self.swap_trace_enabled:
            logger.info(
                "[V0_SWAP_TRACE][AsyncKV][CacheEngine] phase=submit "
                "operation=%s request_id=%s submit_monotonic_ns=%d "
                "mappings=%d",
                operation.value,
                request_id,
                self._bam_mds_async_kv_submitted_at_ns,
                src_to_dst.shape[0],
            )
        try:
            ready = self.bam_mds_connector.submit_transfer_async(
                request_id, src_to_dst, operation=operation.value)
        except Exception:
            # submit 失败时没有一个可继续 poll 的逻辑请求，必须复位计时
            # 状态，否则后续合法请求会被误判为“已有 pending transfer”。
            self._bam_mds_transfer_started_at = None
            self._bam_mds_async_kv_submitted_at_ns = None
            self._bam_mds_async_kv_first_poll_at_ns = None
            self._bam_mds_async_kv_operation = None
            raise
        if ready:
            self._finish_async_kv_transfer_trace(request_id, src_to_dst)
            return AsyncKVTransferEvent(request_id,
                                        AsyncKVTransferState.READY)
        return AsyncKVTransferEvent(request_id, AsyncKVTransferState.PENDING)

    def poll_async_kv_transfer(
            self, request_id: str,
            src_to_dst: torch.Tensor) -> AsyncKVTransferEvent:
        """非阻塞查询一个已经提交的 MDS read/write。"""
        if self.bam_mds_connector is None:
            raise RuntimeError(
                "async KV scheduling requires the resident MDS connector")
        if self._bam_mds_transfer_started_at is None:
            raise RuntimeError("CacheEngine has no pending async KV transfer")
        if self._bam_mds_async_kv_operation is None:
            raise RuntimeError("CacheEngine lost async KV operation state")

        if self._bam_mds_async_kv_first_poll_at_ns is None:
            self._bam_mds_async_kv_first_poll_at_ns = time.monotonic_ns()
            if self.swap_trace_enabled:
                logger.info(
                    "[V0_SWAP_TRACE][AsyncKV][CacheEngine] "
                    "phase=first_poll operation=%s request_id=%s "
                    "first_poll_monotonic_ns=%d",
                    self._bam_mds_async_kv_operation.value,
                    request_id,
                    self._bam_mds_async_kv_first_poll_at_ns,
                )

        ready = self.bam_mds_connector.poll_transfer_async(request_id)
        if not ready:
            return AsyncKVTransferEvent(request_id,
                                        AsyncKVTransferState.PENDING)
        self._finish_async_kv_transfer_trace(request_id, src_to_dst)
        return AsyncKVTransferEvent(request_id, AsyncKVTransferState.READY)

    def _finish_async_kv_transfer_trace(self, request_id: str,
                                        src_to_dst: torch.Tensor) -> None:
        """在异步 read/write 完成后记录耗时并复位单槽状态。"""
        assert self._bam_mds_transfer_started_at is not None
        assert self._bam_mds_async_kv_submitted_at_ns is not None
        assert self._bam_mds_async_kv_operation is not None
        ready_ns = time.monotonic_ns()
        elapsed_s = time.perf_counter() - self._bam_mds_transfer_started_at
        if self.swap_trace_enabled:
            logger.info(
                "[V0_SWAP_TRACE][AsyncKV][CacheEngine] phase=ready "
                "operation=%s request_id=%s submitted_monotonic_ns=%d "
                "first_poll_monotonic_ns=%s ready_monotonic_ns=%d "
                "submit_to_ready_ms=%.3f",
                self._bam_mds_async_kv_operation.value,
                request_id,
                self._bam_mds_async_kv_submitted_at_ns,
                self._bam_mds_async_kv_first_poll_at_ns,
                ready_ns,
                (ready_ns - self._bam_mds_async_kv_submitted_at_ns) / 1.0e6,
            )
        operation = self._bam_mds_async_kv_operation
        self._bam_mds_transfer_started_at = None
        self._log_swap_event(
            "swap_in" if operation == AsyncKVTransferOperation.READ else
            "swap_out", src_to_dst, elapsed_s)
        self._bam_mds_async_kv_submitted_at_ns = None
        self._bam_mds_async_kv_first_poll_at_ns = None
        self._bam_mds_async_kv_operation = None

    def swap_out(self, src_to_dst: torch.Tensor) -> None:
        """把 scheduler 的 GPU->storage block mapping 写出。

        【BaM KVStore 直通调用链】新后端直接从 vLLM physical block 发起 SSD
        write，不再先复制到 CPU cache，也不经过 BaM payload cache。
        """
        start = time.perf_counter()
        if self.bam_mds_connector is not None:
            self.bam_mds_connector.swap_out(src_to_dst)
        elif self.bam_direct_kv_store is not None:
            self.bam_direct_kv_store.swap_out(src_to_dst)
        else:
            for i in range(self.num_attention_layers):
                self.attn_backend.swap_blocks(self.gpu_cache[i],
                                              self.cpu_cache[i], src_to_dst)
            if self.bam_shadow_writer is not None:
                self.bam_shadow_writer.on_swap_out(self.gpu_cache, src_to_dst)
        self._log_swap_event("swap_out", src_to_dst,
                             time.perf_counter() - start)

    def copy(self, src_to_dsts: torch.Tensor) -> None:
        self.attn_backend.copy_blocks(self.gpu_cache, src_to_dsts)

    @staticmethod
    def get_cache_block_size(
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_attention_layers = model_config.get_num_layers_by_block_type(
            parallel_config, LayerBlockType.attention)

        if cache_config.cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        key_cache_entry = num_heads * head_size

        # For MLA there is no value cache, since the latent vector
        # is joint keys and values.
        value_cache_entry = key_cache_entry if not model_config.use_mla else 0
        total = num_attention_layers * cache_config.block_size * \
            (key_cache_entry + value_cache_entry)

        dtype_size = get_dtype_size(dtype)
        return dtype_size * total
