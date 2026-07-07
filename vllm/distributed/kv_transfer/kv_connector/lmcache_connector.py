# SPDX-License-Identifier: Apache-2.0
"""
LMCache KV Cache Connector for Distributed Machine Learning Inference

The LMCacheConnector can (1) transfer KV caches between prefill vLLM worker
(KV cache producer) and decode vLLM worker (KV cache consumer) using LMCache;
(2) offload and share KV caches.
"""

from typing import TYPE_CHECKING, List, Tuple, Union

import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

logger = init_logger(__name__)


def _get_lmcache_retrieve_token_limit(
    total_seq_len: int,
    vllm_num_computed_tokens: int,
    min_query_len: int,
) -> int:
    """复制 LMCache V0 retrieve 前缀截断规则。

    这里不能直接取完整 prompt 去 prefetch，否则 BaM 会预取 LMCache retrieve
    最后不会真正消费的尾部 chunk。规则要和 LMCache V0 adapter 里的
    `_get_retrieve_token_limit()` 保持一致：

    - 至少给 vLLM 留 `min_query_len` 个 token 自己重算
    - LMCache/BaM 只预取真正可能被 `engine.retrieve()` 消费的前缀 chunk
    """
    min_query_len = max(min_query_len, 1)
    min_query_len = min(min_query_len, total_seq_len)
    max_computed_tokens = max(total_seq_len - min_query_len, 0)
    max_lmc_num_computed_tokens = max(
        max_computed_tokens - vllm_num_computed_tokens, 0)
    return vllm_num_computed_tokens + max_lmc_num_computed_tokens


def _maybe_prefetch_lmcache_bam_chunks(
    engine: object,
    model_input: "ModelInputForGPUWithSamplingMetadata",
    retrieve_status: List[object],
) -> None:
    """在 LMCache retrieve 前提前提交 BaM chunk 读取。

    当前 V0 LMCache adapter 的真实 retrieve 仍是逐 chunk blocking get：

    ```text
    engine.retrieve(...)
      -> storage_manager.get(key)
      -> BaM load / LMCache fallback
    ```

    这里复用 LMCache 已有的 `engine.prefetch(tokens, mask)` 入口，在进入
    `engine.retrieve()` 前先把同一批 chunk key 交给 storage manager。
    对 BaM wrapper 来说，这会触发：

    ```text
    storage_manager.prefetch(key)
      -> BaM prepare_request()
      -> BaM submit_request()
    ```

    之后真正的 `storage_manager.get(key)` 会消费已经提交的 request，
    做 poll/complete/refill。CPU 仍负责“哪些 chunk 要读”的粗粒度决策；
    BaM/GPU 负责 page id 请求表和数据读取链路。
    """
    if not (envs.VLLM_BAM_LMCACHE_PREFER_LOAD_ENABLE
            and envs.VLLM_BAM_LMCACHE_READ_MODE == "prefetch"):
        return

    if getattr(engine, "config", None) is None:
        return
    if getattr(engine.config, "enable_blending", False):
        return

    attn_metadata = model_input.attn_metadata
    sampling_metadata = model_input.sampling_metadata
    if attn_metadata is None or sampling_metadata is None:
        return

    query_start_loc = getattr(attn_metadata, "query_start_loc", None)
    seq_lens = getattr(attn_metadata, "seq_lens", None)
    seq_group_list = getattr(sampling_metadata, "seq_groups", None)
    if query_start_loc is None or seq_lens is None or seq_group_list is None:
        return

    chunk_size = int(engine.config.chunk_size)
    idx = 0
    for seq_group in seq_group_list:
        for seq_id in seq_group.seq_ids:
            seq_data = seq_group.seq_data[seq_id]
            status = retrieve_status[idx]
            status_name = getattr(status, "name", str(status))

            # 这几个判断要和 LMCache V0 retrieve 保持一致，避免预取不会被
            # retrieve 消费的 chunk。
            if status_name == "NONE":
                idx += 1
                continue

            total_seq_len = (seq_lens[idx] if status_name == "CHUNK_PREFILL"
                             else seq_data.get_len())
            full_token_tensor = torch.tensor(
                seq_data.get_token_ids()[:total_seq_len], device="cpu")

            vllm_num_required_tokens = (query_start_loc[idx + 1] -
                                        query_start_loc[idx]).item()
            vllm_num_required_tokens = int(vllm_num_required_tokens)
            if vllm_num_required_tokens < chunk_size:
                idx += 1
                continue

            vllm_num_computed_tokens = total_seq_len - vllm_num_required_tokens
            vllm_num_computed_tokens_align = (
                vllm_num_computed_tokens // chunk_size * chunk_size)

            token_mask = torch.ones_like(full_token_tensor, dtype=torch.bool)
            token_mask[:vllm_num_computed_tokens_align] = False
            retrieve_token_limit = _get_lmcache_retrieve_token_limit(
                total_seq_len=total_seq_len,
                vllm_num_computed_tokens=vllm_num_computed_tokens,
                min_query_len=chunk_size,
            )

            prefetch_tokens = full_token_tensor[:retrieve_token_limit]
            prefetch_mask = token_mask[:retrieve_token_limit]
            if torch.sum(prefetch_mask).item() <= 0:
                idx += 1
                continue

            logger.info(
                "[LMCACHE_BAM_EARLY_PREFETCH] engine.prefetch tokens=%d "
                "masked_tokens=%d chunk_size=%d",
                len(prefetch_tokens),
                int(torch.sum(prefetch_mask).item()),
                chunk_size,
            )
            engine.prefetch(prefetch_tokens, prefetch_mask)
            idx += 1


class LMCacheConnector(KVConnectorBase):

    def __init__(
        self,
        rank: int,
        local_rank: int,
        config: VllmConfig,
    ):

        self.transfer_config = config.kv_transfer_config
        self.vllm_config = config

        from lmcache.experimental.cache_engine import LMCacheEngineBuilder
        from lmcache.integration.vllm.utils import ENGINE_NAME
        from lmcache.integration.vllm.vllm_adapter import (
            RetrieveStatus, StoreStatus, init_lmcache_engine,
            lmcache_retrieve_kv, lmcache_should_retrieve, lmcache_should_store,
            lmcache_store_kv)
        logger.info("Initializing LMCacheConfig under kv_transfer_config %s",
                    self.transfer_config)

        # TODO (Jiayi): Find model_config, parallel_config, and cache_config
        self.engine = init_lmcache_engine(config.model_config,
                                          config.parallel_config,
                                          config.cache_config)
        self.lmcache_engine_name = ENGINE_NAME
        self.lmcache_engine_builder = LMCacheEngineBuilder

        self.model_config = config.model_config
        self.parallel_config = config.parallel_config
        self.cache_config = config.cache_config
        self.lmcache_retrieve_kv = lmcache_retrieve_kv
        self.lmcache_store_kv = lmcache_store_kv
        self.lmcache_should_retrieve = lmcache_should_retrieve
        self.lmcache_should_store = lmcache_should_store
        self.store_status = StoreStatus
        self.retrieve_status = RetrieveStatus

        if (envs.VLLM_GDS_LMCACHE_SHADOW_ENABLE
                or envs.VLLM_GDS_LMCACHE_PREFER_LOAD_ENABLE) and \
                self.engine is not None:
            # 可选 LMCache-style GDS wrapper：
            # - 默认关闭，不影响原始 LMCache SSD
            # - 可单独 shadow / prefer-load
            # - 若同时开启 BaM，BaM wrapper 会在外层，优先级更高
            from vllm.bam.lmcache_gds_storage import LMCacheGDSStorageManager
            self.engine.storage_manager = LMCacheGDSStorageManager(
                self.engine.storage_manager)
            logger.info(
                "Enabled LMCache-style GDS storage wrapper for V0 connector. "
                "shadow_enable=%s prefer_load_enable=%s path=%s use_gds=%s",
                envs.VLLM_GDS_LMCACHE_SHADOW_ENABLE,
                envs.VLLM_GDS_LMCACHE_PREFER_LOAD_ENABLE,
                envs.VLLM_GDS_LMCACHE_PATH,
                envs.VLLM_GDS_LMCACHE_USE_GDS,
            )

        if (envs.VLLM_BAM_LMCACHE_SHADOW_ENABLE
                or envs.VLLM_BAM_LMCACHE_PREFER_LOAD_ENABLE) and \
                self.engine is not None:
            # 统一包装 LMCache storage manager：
            # - put: 可做 BaM shadow write
            # - get: 可做 BaM prefer-load，失败再回退原始 LMCache
            from vllm.bam.lmcache_bam_storage import LMCacheBaMStorageManager
            self.engine.storage_manager = LMCacheBaMStorageManager(
                self.engine.storage_manager)
            logger.info(
                "Enabled LMCache <-> BaM storage wrapper for V0 connector. "
                "shadow_enable=%s prefer_load_enable=%s",
                envs.VLLM_BAM_LMCACHE_SHADOW_ENABLE,
                envs.VLLM_BAM_LMCACHE_PREFER_LOAD_ENABLE,
            )

    def recv_kv_caches_and_hidden_states(
        self, model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor]
    ) -> Tuple[Union[torch.Tensor, IntermediateTensors], bool,
               "ModelInputForGPUWithSamplingMetadata"]:
        """按当前 V0 connector 契约同步收口一次 retrieve。

        这里需要明确当前 runtime 边界：

        - `recv_kv_caches_and_hidden_states()` 是 blocking 调用
        - 返回值只有两种稳定语义：
          1. `bypass_model_exec=True`：完全跳过当前前向
          2. `bypass_model_exec=False`：立刻用返回的 `model_input` 继续当前前向

        因此即使下层 BaM direct placement 已经具备 `start/poll/finalize`
        request-handle 结构，当前 V0 主线里它也只能在这一次 blocking `recv`
        调用内部完成同步收口。

        如果未来要把 live in-flight request 跨 `recv` 调用保留下来，就必须先
        扩展 connector / model_runner 的 runtime 契约，引入“延后当前 request、
        下轮再继续执行”的安全中间态；否则后台 direct placement 可能会和当前
        正常 model forward 对同一片 paged KV cache 发生写入竞争。
        """

        retrieve_status = self.lmcache_should_retrieve(model_input)
        self._maybe_prefetch_bam_chunks(model_input, retrieve_status)
        model_input, bypass_model_exec, hidden_or_intermediate_states =\
            self.lmcache_retrieve_kv(
                model_executable, model_input, self.cache_config, kv_caches,
                retrieve_status)
        return hidden_or_intermediate_states, bypass_model_exec, model_input

    def _maybe_prefetch_bam_chunks(
        self,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        retrieve_status: List[object],
    ) -> None:
        """真实 vLLM retrieve 前的 BaM early-prefetch hook。

        hook 放在 connector 层而不是 LMCache 源码里，是为了保持 vllm-bam
        主线和 LMCache V0 repo 解耦。失败只记日志，不影响原始 retrieve。
        """
        if self.engine is None:
            return
        try:
            _maybe_prefetch_lmcache_bam_chunks(self.engine, model_input,
                                               retrieve_status)
        except Exception:
            logger.exception(
                "[LMCACHE_BAM_EARLY_PREFETCH] failed before retrieve; "
                "continue with normal LMCache retrieve")

    def send_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor,
                                             IntermediateTensors],
    ) -> None:

        store_status = self.lmcache_should_store(model_input)
        self.lmcache_store_kv(
            self.model_config,
            self.parallel_config,
            self.cache_config,
            model_executable,
            model_input,
            kv_caches,
            store_status,
        )

    def close(self):
        self.lmcache_engine_builder.destroy(self.lmcache_engine_name)
