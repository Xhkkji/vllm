# SPDX-License-Identifier: Apache-2.0
"""CacheEngine class for managing the KV cache."""
import time
from typing import List

import torch

import vllm.envs as envs
from vllm.attention import get_attn_backend
from vllm.config import CacheConfig, DeviceConfig, ModelConfig, ParallelConfig
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
        self.cpu_cache = self._allocate_kv_cache(self.num_cpu_blocks, "cpu")
        self.swap_trace_enabled = envs.VLLM_V0_SWAP_TRACE
        self.bam_block_store = self._init_bam_block_store()
        self.bam_shadow_writer = self._init_bam_shadow_writer()
        self.bam_swap_reader = self._init_bam_swap_reader()

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
        """Allocates KV cache on the specified device."""
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

        for _ in range(self.num_attention_layers):
            # null block in CpuGpuBlockAllocator requires at least that
            # block to be zeroed-out.
            # We zero-out everything for simplicity.
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
        start = time.perf_counter()
        if self.bam_swap_reader is not None:
            self.bam_swap_reader.swap_in(self.gpu_cache, self.cpu_cache,
                                         src_to_dst)
        else:
            for i in range(self.num_attention_layers):
                self.attn_backend.swap_blocks(self.cpu_cache[i],
                                              self.gpu_cache[i], src_to_dst)
        self._log_swap_event("swap_in", src_to_dst, time.perf_counter() - start)

    def swap_out(self, src_to_dst: torch.Tensor) -> None:
        start = time.perf_counter()
        for i in range(self.num_attention_layers):
            self.attn_backend.swap_blocks(self.gpu_cache[i], self.cpu_cache[i],
                                          src_to_dst)
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
