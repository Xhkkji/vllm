# SPDX-License-Identifier: Apache-2.0
"""
LMCache KV Cache Connector for Distributed Machine Learning Inference

The LMCacheConnector can (1) transfer KV caches between prefill vLLM worker
(KV cache producer) and decode vLLM worker (KV cache consumer) using LMCache;
(2) offload and share KV caches.
"""

import dataclasses
import os
import time
from typing import TYPE_CHECKING, Dict, List, Union

import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.base import (KVConnectorBase,
                                                            KVReceiveResult)
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

logger = init_logger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    """解析本 connector 只读调试开关。

    这里不能复用 LMCache adapter 里的同名 helper，因为两个模块分属不同仓库、
    导入方向也不同。connector 只需要用它判断是否打印热路径 poll 日志，不改变
    任何 retrieve / placement 数据语义。
    """
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _should_log_deferred_poll(attempts: int) -> bool:
    """限制 deferred poll 热路径日志频率。

    `GIDS_KV_DEBUG=1` 用来观察状态机，但 deferred retrieve 每个 engine
    iteration 都会 poll 一次。如果每轮都打印，会把真正有用的状态变化淹没在
    大量重复 WAIT 里，也会明显污染性能。因此这里采用固定节流：

    - 前 5 次完整打印，方便看启动阶段；
    - 之后每 1000 次打印一次，方便确认是否长期卡住。
    """
    attempts = int(attempts)
    return attempts <= 5 or attempts % 1000 == 0


@dataclasses.dataclass
class _PendingDeferredRetrieveState:
    """记录一批已经 start、等待后续 poll/finalize 的 direct retrieve。

    这里保存的是 connector/runtime 这一层需要的 request 级状态，而不是底层
    page/chunk 的 ready 状态。

    额外记录 `poll_attempts` 的原因是：
    - 默认情况下，如果同一轮 poll 已经 ready，就可以立刻 finalize
    - 但为了验证“跨 engine iteration 保留 live handle”这条 runtime 主线，
      我们还需要一个可控的“至少再 defer N 轮”的实验能力
    """

    batch_key: tuple[tuple[str, ...], tuple[int, ...], tuple[int, ...]]
    deferred_batch: object
    created_at_s: float
    poll_attempts: int = 0


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

    if (envs.VLLM_BAM_GPU_INITIATED_PREFETCH
            and envs.VLLM_BAM_DIRECT_PLACEMENT
            and envs.VLLM_BAM_KV_FAST_PATH):
        # GPU-initiated descriptor 分支不再复用 LMCache engine.prefetch()：
        # connector 会直接登记本轮 retrieve plan，direct placement start 再消费
        # 同源 descriptor。这里如果继续走 engine.prefetch()，会额外生成一批
        # pending key，既不能覆盖 deferrable one-copy 主线，也容易在 write / miss
        # 阶段产生不可消费的 prefetch 日志。
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
    prefetch_calls = 0
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
            prefetch_calls += 1
            idx += 1

    if prefetch_calls <= 0:
        return

    # `engine.prefetch()` 内部会按 chunk key 调用 storage_manager.prefetch(key)。
    # 对兼容的 descriptor prefetch 分支来说，单个 key 进入 storage_manager 时
    # 还不能立即组成 one-copy 所需的 batch；因此这里在本轮所有 prefetch 调用
    # 结束后，通过 wrapper 的薄接口显式 flush：
    #
    #   pending keys -> prepared descriptor -> direct placement start 消费 descriptor
    #
    # 如果当前 storage manager 不是 BaM wrapper，或者新分支未启用，这个调用
    # 自然是 no-op，不影响 LMCache 原生路径。
    storage_manager = getattr(engine, "storage_manager", None)
    flush_prefetch = getattr(storage_manager,
                             "flush_bam_gpu_initiated_prefetch", None)
    if callable(flush_prefetch):
        flushed_chunks = int(flush_prefetch())
        if flushed_chunks > 0:
            logger.info(
                "[LMCACHE_BAM_GPU_INITIATED_PREFETCH_FLUSHED] "
                "prefetch_calls=%d chunks=%d",
                prefetch_calls,
                flushed_chunks,
            )


def _maybe_stage_lmcache_bam_gpu_initiated_prefetch_plan(
    engine: object,
    model_input: "ModelInputForGPUWithSamplingMetadata",
    retrieve_status: List[object],
) -> int:
    """在 deferred direct retrieve start 前登记上层 KV descriptor 计划。

    这是当前 GPU-initiated 演进线的上层计划入口。它仍然复用 LMCache V0
    retrieve 的 tokens/mask 规则，但不再调用 `engine.prefetch()`，也不在
    connector 阶段发起 native submit：

    ```text
    connector 已知当前 batch 的 retrieve 输入
      -> storage_manager.stage_bam_gpu_initiated_prefetch_plan()
      -> BaM store 用 direct-placement 同源 key 规则收集 prefix hit chunks
      -> 准备 compact KV descriptor
      -> 后续 direct placement start 消费 descriptor，再进入统一 read 生命周期
    ```

    这里属于控制面预热路径；开关未启用、缺少 wrapper 或当前 batch 无 prefix
    hit 时都返回 0，不改变原始 retrieve 正确性。真正 SSD read 仍在
    direct-placement request start 边界提交，避免把旧的 CPU early-submit 继续
    伪装成 GPU-initiated。
    """
    if not (envs.VLLM_BAM_LMCACHE_PREFER_LOAD_ENABLE
            and envs.VLLM_BAM_GPU_INITIATED_PREFETCH
            and envs.VLLM_BAM_DIRECT_PLACEMENT
            and envs.VLLM_BAM_KV_FAST_PATH):
        return 0

    if getattr(engine, "config", None) is None:
        return 0
    if getattr(engine.config, "enable_blending", False):
        return 0

    storage_manager = getattr(engine, "storage_manager", None)
    stage_prefetch_plan = getattr(
        storage_manager,
        "stage_bam_gpu_initiated_prefetch_plan",
        None,
    )
    token_database = getattr(engine, "token_database", None)
    if not callable(stage_prefetch_plan) or token_database is None:
        return 0

    attn_metadata = model_input.attn_metadata
    sampling_metadata = model_input.sampling_metadata
    if attn_metadata is None or sampling_metadata is None:
        return 0

    query_start_loc = getattr(attn_metadata, "query_start_loc", None)
    seq_lens = getattr(attn_metadata, "seq_lens", None)
    seq_group_list = getattr(sampling_metadata, "seq_groups", None)
    if query_start_loc is None or seq_lens is None or seq_group_list is None:
        return 0

    chunk_size = int(engine.config.chunk_size)
    staged_chunks = 0
    plan_calls = 0
    idx = 0
    for seq_group in seq_group_list:
        for seq_id in seq_group.seq_ids:
            seq_data = seq_group.seq_data[seq_id]
            status = retrieve_status[idx]
            status_name = getattr(status, "name", str(status))

            if status_name == "NONE":
                idx += 1
                continue

            total_seq_len = (int(seq_lens[idx])
                             if status_name == "CHUNK_PREFILL" else
                             int(seq_data.get_len()))
            full_token_tensor = torch.tensor(
                seq_data.get_token_ids()[:total_seq_len], device="cpu")

            vllm_num_required_tokens = int(
                (query_start_loc[idx + 1] - query_start_loc[idx]).item())
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
            masked_tokens = int(prefetch_mask.sum().item())
            if masked_tokens <= 0:
                idx += 1
                continue

            chunks = int(
                stage_prefetch_plan(
                    token_database=token_database,
                    tokens=prefetch_tokens,
                    mask=prefetch_mask,
                    source="connector_deferred_receive",
                ))
            staged_chunks += chunks
            plan_calls += 1
            if chunks > 0:
                logger.info(
                    "[LMCACHE_BAM_GPU_INITIATED_PREFETCH_PLAN] "
                    "seq_idx=%d tokens=%d masked_tokens=%d chunks=%d "
                    "chunk_size=%d",
                    idx,
                    len(prefetch_tokens),
                    masked_tokens,
                    chunks,
                    chunk_size,
                )
            idx += 1

    if plan_calls > 0:
        logger.info(
            "[LMCACHE_BAM_GPU_INITIATED_PREFETCH_PLAN_SUMMARY] "
            "plan_calls=%d staged_chunks=%d",
            plan_calls,
            staged_chunks,
        )
    return staged_chunks


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
            lmcache_finalize_deferrable_retrieve_kv,
            lmcache_poll_deferrable_retrieve_kv,
            lmcache_retrieve_kv, lmcache_should_retrieve,
            lmcache_should_store, lmcache_start_deferrable_retrieve_kv,
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
        self.lmcache_finalize_deferrable_retrieve_kv = (
            lmcache_finalize_deferrable_retrieve_kv)
        self.lmcache_poll_deferrable_retrieve_kv = (
            lmcache_poll_deferrable_retrieve_kv)
        self.lmcache_start_deferrable_retrieve_kv = (
            lmcache_start_deferrable_retrieve_kv)
        self.lmcache_store_kv = lmcache_store_kv
        self.lmcache_should_retrieve = lmcache_should_retrieve
        self.lmcache_should_store = lmcache_should_store
        self.store_status = StoreStatus
        self.retrieve_status = RetrieveStatus
        self._pending_deferred_retrieves: Dict[
            tuple[tuple[str, ...], tuple[int, ...], tuple[int, ...]],
            _PendingDeferredRetrieveState,
        ] = {}

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
    ) -> KVReceiveResult:
        """按当前 V0 connector 契约推进一次 retrieve。

        当前存在两条主线：

        1. 默认 blocking 主线
           仍然在一次 `recv()` 调用内部同步完成 retrieve/finalize。
        2. 可选 deferred runtime 主线
           当 direct placement request-handle 与上层 runtime 开关都开启时，
           connector 可以把当前 batch 显式返回成 `DEFERRED`，让 engine 在
           下一轮继续对同一批输入做 poll/finalize。

        新增的 `DEFERRED` 语义就是为了让 live in-flight direct retrieve
        不必再被硬塞进一次 blocking `recv()` 调用里同步收口。
        """

        self._cleanup_finished_pending_retrieves(model_input)
        retrieve_status = self.lmcache_should_retrieve(model_input)
        if self._runtime_defer_enabled():
            batch_key = self._build_receive_batch_key(model_input)
            if batch_key not in self._pending_deferred_retrieves:
                self._maybe_stage_gpu_initiated_prefetch_plan(
                    model_input,
                    retrieve_status,
                )
            deferred_result = self._try_drive_deferred_receive(
                model_executable=model_executable,
                model_input=model_input,
                kv_caches=kv_caches,
                retrieve_status=retrieve_status,
            )
            if deferred_result is not None:
                return deferred_result
        self._maybe_prefetch_bam_chunks(model_input, retrieve_status)
        model_input, bypass_model_exec, hidden_or_intermediate_states =\
            self.lmcache_retrieve_kv(
                model_executable, model_input, self.cache_config, kv_caches,
                retrieve_status)
        if bypass_model_exec:
            return KVReceiveResult.ready_bypass(
                model_input=model_input,
                hidden_or_intermediate_states=hidden_or_intermediate_states,
            )
        return KVReceiveResult.ready_forward(
            model_input=model_input,
            hidden_or_intermediate_states=hidden_or_intermediate_states,
        )

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

    def _maybe_stage_gpu_initiated_prefetch_plan(
        self,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        retrieve_status: List[object],
    ) -> int:
        """在 deferred direct retrieve start 前尝试登记上层 descriptor plan。"""
        if self.engine is None:
            return 0
        try:
            return _maybe_stage_lmcache_bam_gpu_initiated_prefetch_plan(
                self.engine,
                model_input,
                retrieve_status,
            )
        except Exception:
            logger.exception(
                "[LMCACHE_BAM_GPU_INITIATED_PREFETCH_PLAN] stage failed "
                "before deferred retrieve start")
            return 0

    def _runtime_defer_enabled(self) -> bool:
        """是否启用 direct placement 的 runtime-level defer 主线。"""
        return (bool(envs.VLLM_BAM_DIRECT_PLACEMENT_DEFER_RUNTIME)
                and bool(envs.VLLM_BAM_DIRECT_PLACEMENT)
                and bool(envs.VLLM_BAM_KV_FAST_PATH)
                and self.engine is not None
                and self.lmcache_start_deferrable_retrieve_kv is not None
                and self.lmcache_poll_deferrable_retrieve_kv is not None
                and self.lmcache_finalize_deferrable_retrieve_kv is not None)

    def _get_min_runtime_defer_polls(self) -> int:
        """返回当前实验要求的最少 defer 轮数。"""
        return max(int(envs.VLLM_BAM_DIRECT_PLACEMENT_DEFER_MIN_POLLS), 0)

    def _build_receive_batch_key(
        self,
        model_input: "ModelInputForGPUWithSamplingMetadata",
    ) -> tuple[tuple[str, ...], tuple[int, ...], tuple[int, ...]]:
        """为当前 receive batch 构造一个稳定键。

        这里使用：
        - request id 顺序
        - 当前 seq_lens
        - 当前 query_lens

        这样可以区分“同一个 request id 在不同 prefill/chunk 阶段”的上下文，
        避免把上一个调度轮次的 live handle 误绑定到新的输入上。
        """
        # 这里优先使用 `request_ids_to_seq_ids`，因为它是 model runner 在构造
        # batch 时显式整理出来的 request 级映射，语义最稳定：
        #
        # - key: 真实 request_id
        # - value: 该 request 当前 batch 中对应的 seq_ids
        #
        # 反过来，`sampling_metadata.seq_groups` 在 v0 路径里是
        # `SequenceGroupToSample`，它并不保证携带 `request_id` 字段，
        # 直接依赖它会把 batch key 绑定到一个并不稳定的中间表示上。
        request_ids_to_seq_ids = model_input.request_ids_to_seq_ids
        request_ids: tuple[str, ...] = tuple()
        if request_ids_to_seq_ids is not None:
            request_ids = tuple(request_ids_to_seq_ids.keys())

        seq_lens = tuple(int(x) for x in (model_input.seq_lens or []))
        query_lens = tuple(int(x) for x in (model_input.query_lens or []))
        return request_ids, seq_lens, query_lens

    def _cleanup_finished_pending_retrieves(
        self,
        model_input: "ModelInputForGPUWithSamplingMetadata",
    ) -> None:
        """清理已经结束 request 残留的 pending retrieve 句柄。"""
        finished_request_ids = set(model_input.finished_requests_ids or [])
        if not finished_request_ids:
            return

        stale_keys = [
            batch_key for batch_key in self._pending_deferred_retrieves
            if finished_request_ids.intersection(batch_key[0])
        ]
        for batch_key in stale_keys:
            self._pending_deferred_retrieves.pop(batch_key, None)

    def _try_drive_deferred_receive(
        self,
        *,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor],
        retrieve_status: List[object],
    ) -> KVReceiveResult | None:
        """尝试推进当前 batch 的 deferred direct retrieve 主线。

        返回：
        - `KVReceiveResult`：说明当前 batch 已进入 deferred runtime 主线，
          这轮要么直接 `DEFERRED`，要么已经成功 finalize。
        - `None`：说明这轮不适合走 deferred 主线，调用方应继续 blocking 路径。
        """
        batch_key = self._build_receive_batch_key(model_input)
        pending_state = self._pending_deferred_retrieves.get(batch_key)

        if pending_state is None:
            deferred_batch = self.lmcache_start_deferrable_retrieve_kv(
                model_executable,
                model_input,
                self.cache_config,
                kv_caches,
                retrieve_status,
            )
            if deferred_batch is None:
                return None
            pending_state = _PendingDeferredRetrieveState(
                batch_key=batch_key,
                deferred_batch=deferred_batch,
                created_at_s=time.perf_counter(),
            )
            self._pending_deferred_retrieves[batch_key] = pending_state
            logger.info(
                "[LMCACHE_BAM_DEFERRED_RETRIEVE_START] request_ids=%s",
                ",".join(batch_key[0]),
            )

        pending_state.poll_attempts += 1
        ready = self.lmcache_poll_deferrable_retrieve_kv(
            pending_state.deferred_batch)
        min_defer_polls = self._get_min_runtime_defer_polls()
        force_extra_defer = (ready and min_defer_polls > 0
                             and pending_state.poll_attempts <= min_defer_polls)
        if not ready or force_extra_defer:
            wait_reason = ("forced_min_defer"
                           if force_extra_defer else "not_ready")
            # WAIT 发生在每个 engine iteration 的非阻塞 poll 热路径里。
            # 性能跑默认只保留最终 DONE 统计；需要逐轮观察调度等待时再打开
            # GIDS_KV_DEBUG=1，避免几百次 WAIT 日志污染延迟口径。
            if (_env_flag("GIDS_KV_DEBUG")
                    and _should_log_deferred_poll(
                        pending_state.poll_attempts)):
                logger.info(
                    "[LMCACHE_BAM_DEFERRED_RETRIEVE_WAIT] request_ids=%s "
                    "wait_ms=%.3f poll_attempts=%d min_defer_polls=%d "
                    "reason=%s",
                    ",".join(batch_key[0]),
                    ((time.perf_counter() - pending_state.created_at_s) *
                     1000.0),
                    pending_state.poll_attempts,
                    min_defer_polls,
                    wait_reason,
                )
            return KVReceiveResult.deferred(model_input=model_input)

        self._pending_deferred_retrieves.pop(batch_key, None)
        model_input, bypass_model_exec, hidden_or_intermediate_states = (
            self.lmcache_finalize_deferrable_retrieve_kv(
                pending_state.deferred_batch,
                model_executable=model_executable,
                model_input=model_input,
                cache_config=self.cache_config,
                kv_caches=kv_caches,
            ))
        logger.info(
            "[LMCACHE_BAM_DEFERRED_RETRIEVE_DONE] request_ids=%s "
            "wait_ms=%.3f bypass=%s poll_attempts=%d min_defer_polls=%d",
            ",".join(batch_key[0]),
            (time.perf_counter() - pending_state.created_at_s) * 1000.0,
            bypass_model_exec,
            pending_state.poll_attempts,
            min_defer_polls,
        )
        if bypass_model_exec:
            return KVReceiveResult.ready_bypass(
                model_input=model_input,
                hidden_or_intermediate_states=hidden_or_intermediate_states,
            )
        return KVReceiveResult.ready_forward(
            model_input=model_input,
            hidden_or_intermediate_states=hidden_or_intermediate_states,
        )

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
        self._pending_deferred_retrieves.clear()
        self.lmcache_engine_builder.destroy(self.lmcache_engine_name)
