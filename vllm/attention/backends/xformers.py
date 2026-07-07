# SPDX-License-Identifier: Apache-2.0
"""Attention layer with xFormers and PagedAttention."""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - Triton 可能在某些环境下不可用
    triton = None
    tl = None
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import (AttentionBias,
                                         BlockDiagonalCausalMask,
                                         BlockDiagonalMask,
                                         LowerTriangularMaskWithTensorBias)

from vllm import _custom_ops as ops
from vllm import envs
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer,
                                              AttentionMetadata, AttentionType)
from vllm.attention.backends.utils import (
    CommonAttentionState, CommonMetadataBuilder,
    get_num_prefill_decode_query_kv_tokens, get_seq_len_block_table_args,
    is_all_cross_attn_metadata_set, is_all_encoder_attn_metadata_set)
from vllm.attention.ops.paged_attn import (PagedAttention,
                                           PagedAttentionMetadata)
from vllm.logger import init_logger

logger = init_logger(__name__)


if triton is not None:

    @triton.jit
    def _gather_packed_key_cache_kernel(
        key_cache_ptr,
        block_ids_ptr,
        block_offsets_ptr,
        dst_token_positions_ptr,
        out_ptr,
        total_elements,
        num_kv_heads: tl.constexpr,
        head_size: tl.constexpr,
        pack_size: tl.constexpr,
        stride_cache_block,
        stride_cache_head,
        stride_cache_d_outer,
        stride_cache_token,
        stride_cache_pack,
        stride_out_token,
        stride_out_head,
        stride_out_dim,
        BLOCK_SIZE: tl.constexpr,
    ):
        """从 vLLM packed key cache 直接 gather 到连续 prefix key。

        当前 fallback 最重的一段并不是 `gather_cache` 本身，而是前面的：

        ```text
        key_cache.permute(...).contiguous().view(...)
        ```

        这一步会把 packed key cache 重新排成“token-major 连续布局”，
        代价约 0.6ms / layer。这里直接按 packed layout 寻址，把 prefix token
        scatter/gather 成连续输出，从而绕开这次整块重排。
        """
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_elements

        hidden_per_token = num_kv_heads * head_size
        token_idx = offsets // hidden_per_token
        hidden_idx = offsets % hidden_per_token
        kv_head = hidden_idx // head_size
        head_offset = hidden_idx % head_size

        block_id = tl.load(block_ids_ptr + token_idx, mask=mask, other=0)
        block_offset = tl.load(block_offsets_ptr + token_idx, mask=mask, other=0)
        dst_token_idx = tl.load(dst_token_positions_ptr + token_idx,
                                mask=mask,
                                other=0)
        d_outer = head_offset // pack_size
        d_inner = head_offset % pack_size

        src_offsets = (
            block_id * stride_cache_block +
            kv_head * stride_cache_head +
            d_outer * stride_cache_d_outer +
            block_offset * stride_cache_token +
            d_inner * stride_cache_pack
        )
        dst_offsets = (
            dst_token_idx * stride_out_token +
            kv_head * stride_out_head +
            head_offset * stride_out_dim
        )
        values = tl.load(key_cache_ptr + src_offsets, mask=mask)
        tl.store(out_ptr + dst_offsets, values, mask=mask)


    @triton.jit
    def _gather_packed_value_cache_kernel(
        value_cache_ptr,
        block_ids_ptr,
        block_offsets_ptr,
        dst_token_positions_ptr,
        out_ptr,
        total_elements,
        num_kv_heads: tl.constexpr,
        head_size: tl.constexpr,
        stride_cache_block,
        stride_cache_head,
        stride_cache_dim,
        stride_cache_token,
        stride_out_token,
        stride_out_head,
        stride_out_dim,
        BLOCK_SIZE: tl.constexpr,
    ):
        """从 vLLM packed value cache 直接 gather 到连续 prefix value。"""
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_elements

        hidden_per_token = num_kv_heads * head_size
        token_idx = offsets // hidden_per_token
        hidden_idx = offsets % hidden_per_token
        kv_head = hidden_idx // head_size
        head_offset = hidden_idx % head_size

        block_id = tl.load(block_ids_ptr + token_idx, mask=mask, other=0)
        block_offset = tl.load(block_offsets_ptr + token_idx, mask=mask, other=0)
        dst_token_idx = tl.load(dst_token_positions_ptr + token_idx,
                                mask=mask,
                                other=0)

        src_offsets = (
            block_id * stride_cache_block +
            kv_head * stride_cache_head +
            head_offset * stride_cache_dim +
            block_offset * stride_cache_token
        )
        dst_offsets = (
            dst_token_idx * stride_out_token +
            kv_head * stride_out_head +
            head_offset * stride_out_dim
        )
        values = tl.load(value_cache_ptr + src_offsets, mask=mask)
        tl.store(out_ptr + dst_offsets, values, mask=mask)


@dataclass(frozen=True)
class _XFormersPrefixFallbackSegment:
    """描述一条请求在 fallback 中的 prefix/query 拼接区间。

    `xformers` fallback 当前仍然需要把 prefix KV 从 paged cache gather 成连续
    tensor，再和本轮 query 对应的新 K/V 拼成完整连续序列。

    这一步真正随层变化的数据只有：

    - 当前层的 `prefix_key/prefix_value`
    - 当前层的 `key/value`

    但“每个请求应该从 prefix 连续缓冲区取哪一段、再从 query 连续缓冲区取哪一段，
    以及它们在最终 full KV 序列里应该落到哪里”这一层控制面其实只和 metadata
    有关，和 layer 无关。因此把它提前收成 plan，后续每层只复用这份区间描述即可，
    避免重复做 Python 切片规划。
    """

    prefix_start: int
    prefix_end: int
    query_start: int
    query_end: int
    full_prefix_start: int
    full_prefix_end: int
    full_query_start: int
    full_query_end: int


@dataclass(frozen=True)
class _XFormersPrefixFallbackPlan:
    """prefix fallback 的跨层可复用控制面计划。

    这里刻意只缓存“稳定的控制面”，不缓存任何层相关的 KV 数据：

    - `query_lens/context_lens/kv_lens`
    - `cu_context_lens`
    - `attn_bias`
    - prefix/query 的拼接区间

    这样做的原因是：

    1. prefix KV 数据本身是逐层不同的，不能跨层直接复用；
    2. 但这些长度、区间和 bias 在同一个 prefill batch 内是稳定不变的，
       完全没必要每层都重新构造一遍。
    """

    query_lens: tuple[int, ...]
    context_lens: tuple[int, ...]
    kv_lens: tuple[int, ...]
    segments: tuple[_XFormersPrefixFallbackSegment, ...]
    cu_context_lens: torch.Tensor
    attn_bias: AttentionBias
    total_context_tokens: int
    total_kv_tokens: int
    context_block_ids: torch.Tensor
    context_block_offsets: torch.Tensor
    context_compact_positions: torch.Tensor
    context_full_positions: torch.Tensor


@dataclass(frozen=True)
class _XFormersPrefixFallbackProfile:
    """一次 xformers prefix fallback 的细粒度阶段统计。"""

    cache_view_ms: float
    gather_key_ms: float
    gather_value_ms: float
    concat_ms: float
    xformers_forward_ms: float
    total_ms: float


@dataclass(frozen=True)
class _XFormersPrefixGatherProfile:
    """一次 prefix KV gather 的细粒度阶段统计。"""

    mode: str
    cache_view_ms: float
    gather_key_ms: float
    gather_value_ms: float


@dataclass
class _XFormersPrefixFallbackWorkspace:
    """prefix fallback 的跨层可复用 scratch buffer。

    当前 fallback 仍然需要把 prefix/query 组织成一段完整连续的 KV 序列后再交给
    xFormers。旧实现每层都会：

    1. 生成很多小 slice
    2. 用 Python list 收集
    3. 再 `torch.cat()` 出新的 `full_key/full_value`

    这会引入额外分配和拼接开销。这里把“最终 full KV 缓冲区”也缓存下来，
    每层只做原地填充，从而把控制面和数据面都收得更紧一些。
    """

    full_key: torch.Tensor
    full_value: torch.Tensor


class XFormersBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "XFORMERS"

    @staticmethod
    def get_impl_cls() -> Type["XFormersImpl"]:
        return XFormersImpl

    @staticmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        return XFormersMetadata

    @staticmethod
    def get_builder_cls() -> Type["XFormersMetadataBuilder"]:
        return XFormersMetadataBuilder

    @staticmethod
    def get_state_cls() -> Type["CommonAttentionState"]:
        return CommonAttentionState

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return PagedAttention.get_kv_cache_shape(num_blocks, block_size,
                                                 num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: Dict[int, int],
    ) -> None:
        PagedAttention.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        PagedAttention.copy_blocks(kv_caches, src_to_dists)


@dataclass
class XFormersMetadata(AttentionMetadata, PagedAttentionMetadata):
    """Metadata for XFormersbackend.

    NOTE: Any python object stored here is not updated when it is
    cuda-graph replayed. If you have values that need to be changed
    dynamically, it should be stored in tensor. The tensor has to be
    updated from `CUDAGraphRunner.forward` API.
    """

    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ----------------------|
    #                                   |-- query_len ---|

    # seq_lens stored as a tensor.
    seq_lens_tensor: Optional[torch.Tensor]

    # FIXME: It is for flash attn.
    # Maximum sequence length among prefill batch. 0 if there are decoding
    # requests only.
    max_prefill_seq_len: int
    # Maximum sequence length among decode batch. 0 if there are prefill
    # requests only.
    max_decode_seq_len: int

    # Whether or not if cuda graph is enabled.
    # Cuda-graph is currently enabled for decoding only.
    # TODO(woosuk): Move `use_cuda_graph` out since it's unrelated to attention.
    use_cuda_graph: bool

    # (batch_size,). The sequence length per sequence. Sequence length means
    # the computed tokens + new tokens None if it is a decoding.
    seq_lens: Optional[List[int]] = None

    # FIXME: It is for flash attn.
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    seq_start_loc: Optional[torch.Tensor] = None

    # (batch_size,) A tensor of context lengths (tokens that are computed
    # so far).
    context_lens_tensor: Optional[torch.Tensor] = None

    # Maximum query length in the batch. None for decoding.
    max_query_len: Optional[int] = None

    # Max number of query tokens among request in the batch.
    max_decode_query_len: Optional[int] = None

    # (batch_size + 1,). The cumulative subquery lengths of the sequences in
    # the batch, used to index into subquery. E.g., if the subquery length
    # is [4, 6], it is [0, 4, 10].
    query_start_loc: Optional[torch.Tensor] = None

    # Self-attention prefill/decode metadata cache
    _cached_prefill_metadata: Optional["XFormersMetadata"] = None
    _cached_decode_metadata: Optional["XFormersMetadata"] = None

    # Begin encoder attn & enc/dec cross-attn fields...

    # Encoder sequence lengths representation
    encoder_seq_lens: Optional[List[int]] = None
    encoder_seq_lens_tensor: Optional[torch.Tensor] = None
    # FIXME: It is for flash attn.
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    encoder_seq_start_loc: Optional[torch.Tensor] = None

    # Maximum sequence length among encoder sequences
    max_encoder_seq_len: Optional[int] = None

    # Number of tokens input to encoder
    num_encoder_tokens: Optional[int] = None

    # Cross-attention memory-mapping data structures: slot mapping
    # and block tables
    cross_slot_mapping: Optional[torch.Tensor] = None
    cross_block_tables: Optional[torch.Tensor] = None

    def __post_init__(self):
        # Set during the execution of the first attention op.
        # It is a list because it is needed to set per prompt
        # when alibi slopes is used. It is because of the limitation
        # from xformer API.
        # will not appear in the __repr__ and __init__
        self.attn_bias: Optional[List[AttentionBias]] = None
        self.encoder_attn_bias: Optional[List[AttentionBias]] = None
        self.cross_attn_bias: Optional[List[AttentionBias]] = None
        # prefix fallback 只在 V100/Turing 等当前无法稳定走 paged prefix
        # Triton kernel 的环境里使用。这里把“与层无关的控制面计划”缓存到
        # metadata 上，允许同一个 prefill batch 的所有 layer 复用。
        self._cached_prefix_fallback_plan: Optional[
            _XFormersPrefixFallbackPlan] = None
        self._cached_prefix_fallback_plan_key: Optional[Tuple[Any, ...]] = None
        # prefix fallback 的 scratch buffer 也按 prefill batch 缓存在 metadata 上。
        # 原因和 plan 一样：它只和本轮 batch 的稳定形状有关，和 layer 无关，
        # 因此同一批次所有层都可以复用，避免每层重新分配 full KV 缓冲区。
        self._cached_prefix_fallback_workspace: Optional[
            _XFormersPrefixFallbackWorkspace] = None
        self._cached_prefix_fallback_workspace_key: Optional[
            Tuple[Any, ...]] = None
        # 仅服务于 “sm<80 + no-alibi 时尝试借道 alibi kernel” 这个实验开关。
        # 这组全 0 slope 只和 head 数 / device / dtype 有关，因此可以和其它
        # prefix 控制面缓存一样，按当前 prefill batch 复用。
        self._cached_zero_alibi_slopes: Optional[torch.Tensor] = None
        self._cached_zero_alibi_slopes_key: Optional[Tuple[Any, ...]] = None

    @property
    def is_all_encoder_attn_metadata_set(self):
        '''
        All attention metadata required for encoder attention is set.
        '''
        return is_all_encoder_attn_metadata_set(self)

    @property
    def is_all_cross_attn_metadata_set(self):
        '''
        All attention metadata required for enc/dec cross-attention is set.

        Superset of encoder attention required metadata.
        '''
        return is_all_cross_attn_metadata_set(self)

    @property
    def prefill_metadata(self) -> Optional["XFormersMetadata"]:
        if self.num_prefills == 0:
            return None

        if self._cached_prefill_metadata is not None:
            # Recover cached prefill-phase attention
            # metadata structure
            return self._cached_prefill_metadata

        assert ((self.seq_lens is not None)
                or (self.encoder_seq_lens is not None))
        assert ((self.seq_lens_tensor is not None)
                or (self.encoder_seq_lens_tensor is not None))

        # Compute some attn_metadata fields which default to None
        query_start_loc = (None if self.query_start_loc is None else
                           self.query_start_loc[:self.num_prefills + 1])
        seq_start_loc = (None if self.seq_start_loc is None else
                         self.seq_start_loc[:self.num_prefills + 1])
        slot_mapping = (None if self.slot_mapping is None else
                        self.slot_mapping[:self.num_prefill_tokens])
        seq_lens = (None if self.seq_lens is None else
                    self.seq_lens[:self.num_prefills])
        seq_lens_tensor = (None if self.seq_lens_tensor is None else
                           self.seq_lens_tensor[:self.num_prefills])
        context_lens_tensor = (None if self.context_lens_tensor is None else
                               self.context_lens_tensor[:self.num_prefills])
        block_tables = (None if self.block_tables is None else
                        self.block_tables[:self.num_prefills])

        # Construct & cache prefill-phase attention metadata structure
        self._cached_prefill_metadata = XFormersMetadata(
            num_prefills=self.num_prefills,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=0,
            slot_mapping=slot_mapping,
            multi_modal_placeholder_index_maps=self.
            multi_modal_placeholder_index_maps,
            enable_kv_scales_calculation=self.enable_kv_scales_calculation,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            max_query_len=self.max_query_len,
            max_prefill_seq_len=self.max_prefill_seq_len,
            max_decode_seq_len=0,
            query_start_loc=query_start_loc,
            seq_start_loc=seq_start_loc,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables,
            use_cuda_graph=False,
            # Begin encoder & cross attn fields below...
            encoder_seq_lens=self.encoder_seq_lens,
            encoder_seq_lens_tensor=self.encoder_seq_lens_tensor,
            max_encoder_seq_len=self.max_encoder_seq_len,
            cross_slot_mapping=self.cross_slot_mapping,
            cross_block_tables=self.cross_block_tables)
        return self._cached_prefill_metadata

    @property
    def decode_metadata(self) -> Optional["XFormersMetadata"]:
        if self.num_decode_tokens == 0:
            return None

        if self._cached_decode_metadata is not None:
            # Recover cached decode-phase attention
            # metadata structure
            return self._cached_decode_metadata
        assert ((self.seq_lens_tensor is not None)
                or (self.encoder_seq_lens_tensor is not None))

        # Compute some attn_metadata fields which default to None
        slot_mapping = (None if self.slot_mapping is None else
                        self.slot_mapping[self.num_prefill_tokens:])
        seq_lens_tensor = (None if self.seq_lens_tensor is None else
                           self.seq_lens_tensor[self.num_prefills:])
        block_tables = (None if self.block_tables is None else
                        self.block_tables[self.num_prefills:])

        # Construct & cache decode-phase attention metadata structure
        self._cached_decode_metadata = XFormersMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=self.num_decode_tokens,
            slot_mapping=slot_mapping,
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=True,
            seq_lens_tensor=seq_lens_tensor,
            max_prefill_seq_len=0,
            max_decode_seq_len=self.max_decode_seq_len,
            block_tables=block_tables,
            use_cuda_graph=self.use_cuda_graph,
            # Begin encoder & cross attn fields below...
            encoder_seq_lens=self.encoder_seq_lens,
            encoder_seq_lens_tensor=self.encoder_seq_lens_tensor,
            max_encoder_seq_len=self.max_encoder_seq_len,
            cross_slot_mapping=self.cross_slot_mapping,
            cross_block_tables=self.cross_block_tables)

        # Batch may be composed of prefill|decodes, adjust query start indices
        # to refer to the start of decodes when the two are split apart.
        # E.g. in tokens:[3 prefills|6 decodes], query_start_loc=[3,9] => [0,6].
        if self._cached_decode_metadata.query_start_loc is not None:
            qs = self._cached_decode_metadata.query_start_loc
            self._cached_decode_metadata.query_start_loc = qs - qs[0]
        return self._cached_decode_metadata


def _get_attn_bias(
    attn_metadata: XFormersMetadata,
    attn_type: str,
) -> Optional[AttentionBias]:
    '''
    Extract appropriate attention bias from attention metadata
    according to attention type.

    Arguments:

    * attn_metadata: Attention metadata structure associated with attention
    * attn_type: encoder attention, decoder self-attention,
                 encoder/decoder cross-attention

    Returns:
    * Appropriate attention bias value given the attention type
    '''

    if (attn_type == AttentionType.DECODER
            or attn_type == AttentionType.ENCODER_ONLY):
        return attn_metadata.attn_bias
    elif attn_type == AttentionType.ENCODER:
        return attn_metadata.encoder_attn_bias
    elif attn_type == AttentionType.ENCODER_DECODER:
        return attn_metadata.cross_attn_bias
    else:
        raise AttributeError(f"Invalid attention type {str(attn_type)}")


def _set_attn_bias(
    attn_metadata: XFormersMetadata,
    attn_bias: List[Optional[AttentionBias]],
    attn_type: str,
) -> None:
    '''
    Update appropriate attention bias field of attention metadata,
    according to attention type.

    Arguments:

    * attn_metadata: Attention metadata structure associated with attention
    * attn_bias: The desired attention bias value
    * attn_type: encoder attention, decoder self-attention,
                 encoder/decoder cross-attention
    '''

    if (attn_type == AttentionType.DECODER
            or attn_type == AttentionType.ENCODER_ONLY):
        attn_metadata.attn_bias = attn_bias
    elif attn_type == AttentionType.ENCODER:
        attn_metadata.encoder_attn_bias = attn_bias
    elif attn_type == AttentionType.ENCODER_DECODER:
        attn_metadata.cross_attn_bias = attn_bias
    else:
        raise AttributeError(f"Invalid attention type {str(attn_type)}")


class XFormersMetadataBuilder(CommonMetadataBuilder[XFormersMetadata]):

    _metadata_cls = XFormersMetadata


class XFormersImpl(AttentionImpl[XFormersMetadata]):
    """
    If the input tensors contain prompt tokens, the layout is as follows:
    |<--------------- num_prefill_tokens ----------------->|	
    |<--prefill_0-->|<--prefill_1-->|...|<--prefill_N-1--->|

    Otherwise, the layout is as follows:	
    |<----------------- num_decode_tokens ------------------>|	
    |<--decode_0-->|..........|<--decode_M-1-->|<--padding-->|

    Generation tokens can contain padding when cuda-graph is used.
    Currently, prompt tokens don't contain any padding.

    The prompts might have different lengths, while the generation tokens
    always have length 1.

    If chunked prefill is enabled, prefill tokens and decode tokens can be
    batched together in a flattened 1D query.

    |<----- num_prefill_tokens ---->|<------- num_decode_tokens --------->|
    |<-prefill_0->|...|<-prefill_N-1->|<--decode_0-->|...|<--decode_M-1-->|

    Currently, cuda graph is disabled for chunked prefill, meaning there's no
    padding between prefill and decode tokens.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
        use_irope: bool = False,
    ) -> None:
        if blocksparse_params is not None:
            raise ValueError(
                "XFormers does not support block-sparse attention.")
        if logits_soft_cap is not None:
            logger.warning_once("XFormers does not support logits soft cap. "
                                "Outputs may be slightly off.")
        if use_irope:
            logger.warning_once(
                "Using irope in XFormers is not supported yet, it will fall"
                " back to global attention for long context.")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        supported_head_sizes = PagedAttention.get_supported_head_sizes()
        if head_size not in supported_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by PagedAttention. "
                f"Supported head sizes are: {supported_head_sizes}.")

        self.attn_type = attn_type

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        kv_cache: torch.Tensor,
        attn_metadata: "XFormersMetadata",
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with xFormers and PagedAttention.

        For decoder-only models: query, key and value must be non-None.

        For encoder/decoder models:
        * XFormersImpl.forward() may be invoked for both self- and cross-
          attention layers.
        * For self-attention: query, key and value must be non-None.
        * For cross-attention:
            * Query must be non-None
            * During prefill, key and value must be non-None; key and value
              get cached for use during decode.
            * During decode, key and value may be None, since:
              (1) key and value tensors were cached during prefill, and
              (2) cross-attention key and value tensors do not grow during
                  decode
        
        A note on how the attn_type (attention type enum) argument impacts
        attention forward() behavior:
    
            * DECODER: normal decoder-only behavior;
                use decoder self-attention block table
            * ENCODER: no KV caching; pass encoder sequence
                attributes (encoder_seq_lens/encoder_seq_lens_tensor/
                max_encoder_seq_len) to kernel, in lieu of decoder
                sequence attributes (seq_lens/seq_lens_tensor/max_seq_len).
                Used for encoder branch of encoder-decoder models.
            * ENCODER_ONLY: no kv_caching, uses the normal attention 
                attributes (seq_lens/seq_lens_tensor/max_seq_len).
            * ENCODER_DECODER: cross-attention behavior;
                use cross-attention block table for caching KVs derived
                from encoder hidden states; since KV sequence lengths
                will match encoder sequence lengths, pass encoder sequence
                attributes to kernel (encoder_seq_lens/encoder_seq_lens_tensor/
                max_encoder_seq_len)
    
        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size * num_kv_heads * head_size]
                NOTE: kv_cache will be an empty tensor with shape [0]
                for profiling run.
            attn_metadata: Metadata for attention.
            attn_type: Select attention type, between encoder attention,
                       decoder self-attention, or encoder/decoder cross-
                       attention. Defaults to decoder self-attention,
                       which is the vLLM default generally
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        attn_type = self.attn_type
        # Check that appropriate attention metadata attributes are
        # selected for the desired attention type
        if (attn_type == AttentionType.ENCODER
                and (not attn_metadata.is_all_encoder_attn_metadata_set)):
            raise AttributeError("Encoder attention requires setting "
                                 "encoder metadata attributes.")

        elif (attn_type == AttentionType.ENCODER_DECODER
              and (not attn_metadata.is_all_cross_attn_metadata_set)):
            raise AttributeError("Encoder/decoder cross-attention "
                                 "requires setting cross-attention "
                                 "metadata attributes.")

        query = query.view(-1, self.num_heads, self.head_size)
        if key is not None:
            assert value is not None
            key = key.view(-1, self.num_kv_heads, self.head_size)
            value = value.view(-1, self.num_kv_heads, self.head_size)
        else:
            assert value is None

        # Self-attention vs. cross-attention will impact
        # which KV cache memory-mapping & which
        # seqlen datastructures we utilize

        if (attn_type != AttentionType.ENCODER and kv_cache.numel() > 0):
            # KV-cache during decoder-self- or
            # encoder-decoder-cross-attention, but not
            # during encoder attention.
            #
            # Even if there are no new key/value pairs to cache,
            # we still need to break out key_cache and value_cache
            # i.e. for later use by paged attention
            key_cache, value_cache = PagedAttention.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_size)

            if (key is not None) and (value is not None):

                if attn_type == AttentionType.ENCODER_DECODER:
                    # Update cross-attention KV cache (prefill-only)
                    # During cross-attention decode, key & value will be None,
                    # preventing this IF-statement branch from running
                    updated_slot_mapping = attn_metadata.cross_slot_mapping
                else:
                    # Update self-attention KV cache (prefill/decode)
                    updated_slot_mapping = attn_metadata.slot_mapping

                # Reshape the input keys and values and store them in the cache.
                # If kv_cache is not provided, the new key and value tensors are
                # not cached. This happens during the initial memory
                # profiling run.
                PagedAttention.write_to_paged_cache(
                    key, value, key_cache, value_cache, updated_slot_mapping,
                    self.kv_cache_dtype, layer._k_scale, layer._v_scale)
        (num_prefill_query_tokens, num_prefill_kv_tokens,
        num_decode_query_tokens) = \
            get_num_prefill_decode_query_kv_tokens(attn_metadata, attn_type)

        output = torch.empty_like(query)
        # Query for decode. KV is not needed because it is already cached.
        decode_query = query[num_prefill_query_tokens:]
        # QKV for prefill.
        query = query[:num_prefill_query_tokens]
        if key is not None and value is not None:
            key = key[:num_prefill_kv_tokens]
            value = value[:num_prefill_kv_tokens]

        assert query.shape[0] == num_prefill_query_tokens
        assert decode_query.shape[0] == num_decode_query_tokens

        if prefill_meta := attn_metadata.prefill_metadata:
            # Prompt run.
            if kv_cache.numel() == 0 or prefill_meta.block_tables.numel() == 0:
                # normal attention.
                # block tables are empty if the prompt does not have a cached
                # prefix.
                out = self._run_memory_efficient_xformers_forward(
                    query, key, value, prefill_meta, attn_type=attn_type)
                assert out.shape == output[:num_prefill_query_tokens].shape
                output[:num_prefill_query_tokens] = out
            else:
                assert attn_type != AttentionType.ENCODER_ONLY, (
                    "Encoder-only models should not have prefix attention.")

                assert prefill_meta.query_start_loc is not None
                assert prefill_meta.max_query_len is not None

                # 这条日志在每一层 prefix attention 都会打印一次，长期打开会对
                # 主线性能和日志可读性都造成干扰。这里收敛成：
                #
                # - 平时默认不打
                # - 只有在显式打开 fallback profile 时，才一起输出这份上下文
                #
                # 这样既保留联调时需要的输入形状信息，又避免稳定跑分时每层都刷
                # 一条大日志。
                if envs.VLLM_BAM_XFORMERS_PREFIX_FALLBACK_PROFILE:
                    logger.info(
                        "[XFORMERS_PREFIX] query_shape=%s key_shape=%s "
                        "value_shape=%s block_tables_shape=%s "
                        "query_start_loc=%s seq_lens=%s seq_lens_tensor=%s "
                        "num_prefills=%d num_prefill_tokens=%d "
                        "max_query_len=%s max_prefill_seq_len=%s",
                        tuple(query.shape),
                        None if key is None else tuple(key.shape),
                        None if value is None else tuple(value.shape),
                        tuple(prefill_meta.block_tables.shape),
                        prefill_meta.query_start_loc.tolist(),
                        prefill_meta.seq_lens,
                        None if prefill_meta.seq_lens_tensor is None else
                        prefill_meta.seq_lens_tensor.tolist(),
                        prefill_meta.num_prefills,
                        prefill_meta.num_prefill_tokens,
                        prefill_meta.max_query_len,
                        prefill_meta.max_prefill_seq_len,
                    )

                # V100 + V0 + xFormers 的 prefix kernel 在当前环境会触发
                # LLVM layout 崩溃。这里保留 prefix hit 语义和 paged KV /
                # SSD 读负载，只把最后一步 attention 计算切到稳定的
                # xFormers 普通 varlen 路径。
                device_capability = torch.cuda.get_device_capability(
                    query.device)
                if device_capability < (8, 0) and self.alibi_slopes is None:
                    if envs.VLLM_BAM_TRY_PAGED_PREFIX_ZERO_ALIBI:
                        out = self._try_run_paged_prefix_with_zero_alibi(
                            query=query,
                            key=key,
                            value=value,
                            key_cache=key_cache,
                            value_cache=value_cache,
                            prefill_meta=prefill_meta,
                            layer=layer,
                        )
                    else:
                        out = self._run_prefix_attention_fallback(
                            query=query,
                            key=key,
                            value=value,
                            key_cache=key_cache,
                            value_cache=value_cache,
                            prefill_meta=prefill_meta,
                        )
                else:
                    # prefix-enabled attention
                    # TODO(Hai) this triton kernel has regression issue (broke)
                    # to deal with different data types between KV and FP8 KV
                    # cache, to be addressed separately.
                    out = PagedAttention.forward_prefix(
                        query,
                        key,
                        value,
                        self.kv_cache_dtype,
                        key_cache,
                        value_cache,
                        prefill_meta.block_tables,
                        prefill_meta.query_start_loc,
                        prefill_meta.seq_lens_tensor,
                        prefill_meta.max_query_len,
                        self.alibi_slopes,
                        self.sliding_window,
                        layer._k_scale,
                        layer._v_scale,
                    )
                assert output[:num_prefill_query_tokens].shape == out.shape
                output[:num_prefill_query_tokens] = out

        if decode_meta := attn_metadata.decode_metadata:
            assert attn_type != AttentionType.ENCODER_ONLY, (
                "Encoder-only models should not have decode metadata.")

            (
                seq_lens_arg,
                max_seq_len_arg,
                block_tables_arg,
            ) = get_seq_len_block_table_args(decode_meta, False, attn_type)

            output[num_prefill_query_tokens:] = PagedAttention.forward_decode(
                decode_query,
                key_cache,
                value_cache,
                block_tables_arg,
                seq_lens_arg,
                max_seq_len_arg,
                self.kv_cache_dtype,
                self.num_kv_heads,
                self.scale,
                self.alibi_slopes,
                layer._k_scale,
                layer._v_scale,
            )

        # Reshape the output tensor.
        return output.view(-1, self.num_heads * self.head_size)

    def _run_prefix_attention_fallback(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        prefill_meta: XFormersMetadata,
    ) -> torch.Tensor:
        """绕开不稳定的 prefix kernel，但保留 prefix hit 与 paged KV 语义。"""
        assert prefill_meta.block_tables is not None
        plan = self._get_prefix_fallback_plan(
            prefill_meta=prefill_meta,
            device=query.device,
            block_size=int(value_cache.shape[-1]),
        )

        workspace = self._get_prefix_fallback_workspace(
            prefill_meta=prefill_meta,
            plan=plan,
            key_dtype=key.dtype,
            value_dtype=value.dtype,
            device=query.device,
        )
        full_key = workspace.full_key
        full_value = workspace.full_value

        if self._can_use_packed_prefix_gather(key_cache, value_cache):
            gather_profile = self._fill_prefix_kv_from_packed_cache_into_full_buffer(
                key_cache=key_cache,
                value_cache=value_cache,
                plan=plan,
                full_key=full_key,
                full_value=full_value,
                profile_enabled=envs.VLLM_BAM_XFORMERS_PREFIX_FALLBACK_PROFILE,
            )
        else:
            prefix_key, prefix_value, gather_profile = self._gather_prefix_kv_from_cache(
                key_cache=key_cache,
                value_cache=value_cache,
                block_tables=prefill_meta.block_tables,
                plan=plan,
            )
            self._copy_prefix_kv_into_full_buffer(
                prefix_key=prefix_key,
                prefix_value=prefix_value,
                plan=plan,
                full_key=full_key,
                full_value=full_value,
            )

        compose_start = compose_end = None
        profile_enabled = envs.VLLM_BAM_XFORMERS_PREFIX_FALLBACK_PROFILE
        if profile_enabled:
            compose_start, compose_end = self._new_cuda_event_pair()
            compose_start.record()
        self._copy_query_kv_into_full_buffer(
            query_key=key,
            query_value=value,
            plan=plan,
            full_key=full_key,
            full_value=full_value,
        )
        compose_ms = 0.0
        if profile_enabled and compose_end is not None:
            compose_end.record()
            compose_ms = self._cuda_elapsed_ms(compose_start, compose_end)

        original_query = query
        if self.num_kv_heads != self.num_heads:
            query = query.view(query.shape[0], self.num_kv_heads,
                               self.num_queries_per_kv, query.shape[-1])
            full_key = full_key[:, :, None, :].expand(
                full_key.shape[0],
                self.num_kv_heads,
                self.num_queries_per_kv,
                full_key.shape[-1],
            )
            full_value = full_value[:, :, None, :].expand(
                full_value.shape[0],
                self.num_kv_heads,
                self.num_queries_per_kv,
                full_value.shape[-1],
            )

        logger.info(
            "[XFORMERS_PREFIX_FALLBACK] query_lens=%s context_lens=%s "
            "kv_lens=%s total_query_tokens=%d total_kv_tokens=%d",
            list(plan.query_lens),
            list(plan.context_lens),
            list(plan.kv_lens),
            query.shape[0],
            plan.total_kv_tokens,
        )

        stage_start = None
        stage_end = None
        if profile_enabled:
            stage_start, stage_end = self._new_cuda_event_pair()
            stage_start.record()
        out = xops.memory_efficient_attention_forward(
            query.unsqueeze(0),
            full_key.unsqueeze(0),
            full_value.unsqueeze(0),
            attn_bias=plan.attn_bias,
            p=0.0,
            scale=self.scale,
        )
        if profile_enabled and stage_end is not None:
            stage_end.record()
            xformers_forward_ms = self._cuda_elapsed_ms(stage_start, stage_end)
            profile = _XFormersPrefixFallbackProfile(
                cache_view_ms=gather_profile.cache_view_ms,
                gather_key_ms=gather_profile.gather_key_ms,
                gather_value_ms=gather_profile.gather_value_ms,
                concat_ms=compose_ms,
                xformers_forward_ms=xformers_forward_ms,
                total_ms=(gather_profile.cache_view_ms +
                          gather_profile.gather_key_ms +
                          gather_profile.gather_value_ms +
                          compose_ms + xformers_forward_ms),
            )
            logger.info(
                "[XFORMERS_PREFIX_FALLBACK_PROFILE] mode=%s cache_view_ms=%.3f "
                "gather_key_ms=%.3f gather_value_ms=%.3f concat_ms=%.3f "
                "xformers_forward_ms=%.3f total_ms=%.3f",
                gather_profile.mode,
                profile.cache_view_ms,
                profile.gather_key_ms,
                profile.gather_value_ms,
                profile.concat_ms,
                profile.xformers_forward_ms,
                profile.total_ms,
            )
        return out.view_as(original_query)

    def _get_zero_alibi_slopes_for_prefix_kernel(
        self,
        *,
        prefill_meta: XFormersMetadata,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """返回一组可跨层复用的全 0 alibi slope。

        这不是为了真正启用 alibi 语义，而是为了在 `sm < 80` 且原本没有
        alibi 的情况下，尝试把 `PagedAttention.forward_prefix()` 切到
        `_fwd_kernel_alibi` 这条不同的 Triton 实现。

        由于 slope 全为 0，注意力分数上额外加的 bias 理论上恒为 0，因此对
        语义没有影响；真正改变的是“选用哪套 Triton kernel”。
        """
        cache_key = (
            device.type,
            device.index,
            int(self.num_heads),
            dtype,
        )
        cached = prefill_meta._cached_zero_alibi_slopes
        if (cached is not None and
                prefill_meta._cached_zero_alibi_slopes_key == cache_key):
            return cached

        zero_alibi_slopes = torch.zeros(
            self.num_heads,
            dtype=dtype,
            device=device,
        )
        prefill_meta._cached_zero_alibi_slopes = zero_alibi_slopes
        prefill_meta._cached_zero_alibi_slopes_key = cache_key
        return zero_alibi_slopes

    def _try_run_paged_prefix_with_zero_alibi(
        self,
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        prefill_meta: XFormersMetadata,
        layer: AttentionLayer,
    ) -> torch.Tensor:
        """实验性地借道 alibi kernel 执行 paged prefix attention。

        当前已知现象是：

        - `sm < 80` + no-alibi 时，原始 prefix kernel 会在某些环境触发
          LLVM/layout 崩溃
        - 但 `_fwd_kernel_alibi` 是另一套实现路径

        因此这里通过“传入全 0 alibi slope”来切换实现分支，验证问题是否只
        集中在 no-alibi kernel 本身。

        之所以保留成独立函数，而不是直接把逻辑塞进主分支，是为了：

        - 让实验路径和当前稳定 fallback 主线明确解耦
        - 后续如果确认这条路可行，可以更容易把它单独收敛成正式实现
        """
        zero_alibi_slopes = self._get_zero_alibi_slopes_for_prefix_kernel(
            prefill_meta=prefill_meta,
            device=query.device,
            dtype=query.dtype,
        )
        logger.info(
            "[XFORMERS_PREFIX_ZERO_ALIBI_EXPERIMENT] num_heads=%d dtype=%s "
            "device_capability=%s",
            self.num_heads,
            str(query.dtype),
            torch.cuda.get_device_capability(query.device),
        )
        return PagedAttention.forward_prefix(
            query,
            key,
            value,
            self.kv_cache_dtype,
            key_cache,
            value_cache,
            prefill_meta.block_tables,
            prefill_meta.query_start_loc,
            prefill_meta.seq_lens_tensor,
            prefill_meta.max_query_len,
            zero_alibi_slopes,
            self.sliding_window,
            layer._k_scale,
            layer._v_scale,
        )

    def _get_prefix_fallback_workspace(
        self,
        *,
        prefill_meta: XFormersMetadata,
        plan: _XFormersPrefixFallbackPlan,
        key_dtype: torch.dtype,
        value_dtype: torch.dtype,
        device: torch.device,
    ) -> _XFormersPrefixFallbackWorkspace:
        """按当前 prefill batch 的稳定形状复用 full KV scratch buffer。

        这里缓存的是“最终要交给 xFormers 的连续 full KV 缓冲区”。
        它的形状只取决于：

        - 当前 batch 的 `total_kv_tokens`
        - `num_kv_heads / head_size`
        - `dtype / device`

        因此完全可以跨层复用。后续每层只需要把 prefix/query 对应的数据原地填进
        去，而不必重新分配一份新的 full KV 张量。
        """
        workspace_key = (
            device.type,
            device.index,
            int(plan.total_kv_tokens),
            int(self.num_kv_heads),
            int(self.head_size),
            key_dtype,
            value_dtype,
        )
        cached_workspace = prefill_meta._cached_prefix_fallback_workspace
        if (cached_workspace is not None
                and prefill_meta._cached_prefix_fallback_workspace_key ==
                workspace_key):
            return cached_workspace

        workspace = _XFormersPrefixFallbackWorkspace(
            full_key=torch.empty(
                (plan.total_kv_tokens, self.num_kv_heads, self.head_size),
                dtype=key_dtype,
                device=device,
            ),
            full_value=torch.empty(
                (plan.total_kv_tokens, self.num_kv_heads, self.head_size),
                dtype=value_dtype,
                device=device,
            ),
        )
        prefill_meta._cached_prefix_fallback_workspace = workspace
        prefill_meta._cached_prefix_fallback_workspace_key = workspace_key
        return workspace

    def _copy_prefix_kv_into_full_buffer(
        self,
        *,
        prefix_key: torch.Tensor,
        prefix_value: torch.Tensor,
        plan: _XFormersPrefixFallbackPlan,
        full_key: torch.Tensor,
        full_value: torch.Tensor,
    ) -> None:
        """把连续 prefix KV 按 plan 中的 prefix 段落回 full KV buffer。

        这一步只在非 packed-direct gather 的兼容路径下需要。
        因为这条路径仍然先得到连续的 `prefix_key/prefix_value` 中间张量，
        所以还需要再做一次 prefix -> full buffer 的 copy。
        """
        for segment in plan.segments:
            if segment.full_prefix_end > segment.full_prefix_start:
                full_key[segment.full_prefix_start:segment.full_prefix_end].copy_(
                    prefix_key[segment.prefix_start:segment.prefix_end],
                    non_blocking=False,
                )
                full_value[
                    segment.full_prefix_start:segment.full_prefix_end].copy_(
                        prefix_value[segment.prefix_start:segment.prefix_end],
                        non_blocking=False,
                    )

    def _copy_query_kv_into_full_buffer(
        self,
        *,
        query_key: torch.Tensor,
        query_value: torch.Tensor,
        plan: _XFormersPrefixFallbackPlan,
        full_key: torch.Tensor,
        full_value: torch.Tensor,
    ) -> None:
        """把本轮 query 对应的新 KV 写入 full KV buffer 的 query 段。

        之所以把 query copy 单独拆出来，是因为当前主热路径已经变成：

        - prefix: 直接从 paged cache gather 到 full buffer
        - query: 把本轮新生成的 key/value 填进 full buffer 尾部对应区间

        这样 prefix 和 query 的搬运逻辑就自然解耦了，后续如果继续把 query
        段也收缩成更直接的消费方式，这里就是独立的切入点。
        """
        for segment in plan.segments:
            if segment.full_query_end > segment.full_query_start:
                full_key[segment.full_query_start:segment.full_query_end].copy_(
                    query_key[segment.query_start:segment.query_end],
                    non_blocking=False,
                )
                full_value[
                    segment.full_query_start:segment.full_query_end].copy_(
                        query_value[segment.query_start:segment.query_end],
                        non_blocking=False,
                    )

    def _get_prefix_fallback_plan(
        self,
        *,
        prefill_meta: XFormersMetadata,
        device: torch.device,
        block_size: int,
    ) -> _XFormersPrefixFallbackPlan:
        """构造并缓存 prefix fallback 的跨层控制面计划。

        当前 V100 路线下，每一层都会走一次 `_run_prefix_attention_fallback()`。
        如果不把控制面收起来，每层都会重复做：

        - 解析 `query_start_loc/context_lens_tensor`
        - 构造 `kv_lens`
        - 生成 `cu_context_lens`
        - 构造 `BlockDiagonalMask`
        - 规划 prefix/query 拼接区间

        这些步骤都不依赖层内 KV 数据，因此完全可以按 prefill batch 只做一次。
        """
        assert prefill_meta.context_lens_tensor is not None
        assert prefill_meta.query_start_loc is not None

        device_key = (device.type, device.index)
        plan_key = (
            device_key,
            self.sliding_window,
            int(block_size),
            int(prefill_meta.num_prefills),
            tuple(int(x) for x in prefill_meta.query_start_loc.tolist()),
            tuple(int(x) for x in prefill_meta.context_lens_tensor.tolist()),
        )
        cached_plan = prefill_meta._cached_prefix_fallback_plan
        if (cached_plan is not None
                and prefill_meta._cached_prefix_fallback_plan_key == plan_key):
            return cached_plan

        query_lens = tuple(
            int(x) for x in (
                prefill_meta.query_start_loc[1:] -
                prefill_meta.query_start_loc[:-1]).tolist())
        context_lens = tuple(
            int(x) for x in prefill_meta.context_lens_tensor.tolist())
        kv_lens = tuple(
            context_len + query_len
            for context_len, query_len in zip(context_lens, query_lens))
        total_context_tokens = int(sum(context_lens))

        cu_context_lens = torch.zeros(
            prefill_meta.num_prefills + 1,
            dtype=torch.int32,
            device=device,
        )
        if prefill_meta.num_prefills > 0:
            torch.cumsum(
                prefill_meta.context_lens_tensor.to(torch.int32),
                dim=0,
                out=cu_context_lens[1:],
            )

        assert prefill_meta.block_tables is not None
        # 这里把“prefix 第 i 个 token 对应哪个 physical block / block 内 offset”
        # 直接预展开成两张小表。这样后面的 Triton gather kernel 就不需要再做：
        # - batch 维 binary search
        # - block_table 行寻址
        # - token -> block/token_offset 的重复推导
        #
        # 这些映射只和当前 prefill metadata 有关，和 layer 无关，因此放进
        # fallback plan 里跨层复用最合适。
        # prefix fallback 当前真实热路径大多是：
        #
        # - 单个 prefill request
        # - 一段连续 prefix hit
        #
        # 对这类场景，如果先把整张 block table 搬回 CPU，再按 token 做 Python
        # for-loop 去展开 block_id / block_offset / full_position，控制面会平白
        # 多一次 host roundtrip。
        #
        # 因此这里对“单请求”主场景做一个专门 fast path：
        #
        # - token offset 直接在 GPU 上用 `torch.arange` 生成
        # - block_id 直接在 GPU 上从 `block_tables[0]` gather
        # - compact/full position 也直接用向量方式生成
        #
        # 多请求场景仍保留保守的 CPU 展开逻辑，避免为了当前主线之外的情形把代
        # 码复杂度抬得太高。
        if prefill_meta.num_prefills == 1 and total_context_tokens > 0:
            single_context_len = int(context_lens[0])
            token_offsets = torch.arange(
                single_context_len,
                dtype=torch.int32,
                device=device,
            )
            block_indices = torch.div(
                token_offsets,
                int(block_size),
                rounding_mode="floor",
            ).to(torch.long)
            block_row = prefill_meta.block_tables[0].to(
                device=device,
                dtype=torch.int32,
            )
            context_block_ids = block_row.index_select(
                0, block_indices).contiguous()
            context_block_offsets = torch.remainder(
                token_offsets, int(block_size)).contiguous()
            context_compact_positions = torch.arange(
                single_context_len,
                dtype=torch.int32,
                device=device,
            )
            # 单请求下，prefix 总是落在 full KV 的起点，因此 full position
            # 与 token offset 相同。
            context_full_positions = token_offsets.contiguous()
        else:
            context_block_ids_list: list[int] = []
            context_block_offsets_list: list[int] = []
            context_compact_positions_list: list[int] = []
            context_full_positions_list: list[int] = []
            block_tables_cpu = prefill_meta.block_tables.to("cpu")
            full_kv_cursor = 0
            for request_idx, (context_len, query_len) in enumerate(
                    zip(context_lens, query_lens)):
                if context_len <= 0:
                    full_kv_cursor += query_len
                    continue
                block_row = block_tables_cpu[request_idx]
                for token_offset in range(context_len):
                    block_index = token_offset // int(block_size)
                    context_block_ids_list.append(
                        int(block_row[block_index].item()))
                    context_block_offsets_list.append(
                        token_offset % int(block_size))
                    context_compact_positions_list.append(
                        len(context_compact_positions_list))
                    context_full_positions_list.append(full_kv_cursor +
                                                       token_offset)
                full_kv_cursor += context_len + query_len

            context_block_ids = torch.tensor(
                context_block_ids_list,
                dtype=torch.int32,
                device=device,
            )
            context_block_offsets = torch.tensor(
                context_block_offsets_list,
                dtype=torch.int32,
                device=device,
            )
            context_compact_positions = torch.tensor(
                context_compact_positions_list,
                dtype=torch.int32,
                device=device,
            )
            context_full_positions = torch.tensor(
                context_full_positions_list,
                dtype=torch.int32,
                device=device,
            )

        segments: list[_XFormersPrefixFallbackSegment] = []
        prefix_cursor = 0
        query_cursor = 0
        full_kv_cursor = 0
        for context_len, query_len in zip(context_lens, query_lens):
            full_prefix_start = full_kv_cursor
            full_prefix_end = full_prefix_start + context_len
            full_query_start = full_prefix_end
            full_query_end = full_query_start + query_len
            segment = _XFormersPrefixFallbackSegment(
                prefix_start=prefix_cursor,
                prefix_end=prefix_cursor + context_len,
                query_start=query_cursor,
                query_end=query_cursor + query_len,
                full_prefix_start=full_prefix_start,
                full_prefix_end=full_prefix_end,
                full_query_start=full_query_start,
                full_query_end=full_query_end,
            )
            segments.append(segment)
            prefix_cursor += context_len
            query_cursor += query_len
            full_kv_cursor = full_query_end

        attn_bias = BlockDiagonalMask.from_seqlens(
            q_seqlen=list(query_lens),
            kv_seqlen=list(kv_lens),
            device=device,
        ).make_causal_from_bottomright()
        if self.sliding_window is not None:
            attn_bias = attn_bias.make_local_attention_from_bottomright(
                self.sliding_window)

        plan = _XFormersPrefixFallbackPlan(
            query_lens=query_lens,
            context_lens=context_lens,
            kv_lens=kv_lens,
            segments=tuple(segments),
            cu_context_lens=cu_context_lens,
            attn_bias=attn_bias,
            total_context_tokens=total_context_tokens,
            total_kv_tokens=full_kv_cursor,
            context_block_ids=context_block_ids,
            context_block_offsets=context_block_offsets,
            context_compact_positions=context_compact_positions,
            context_full_positions=context_full_positions,
        )
        prefill_meta._cached_prefix_fallback_plan = plan
        prefill_meta._cached_prefix_fallback_plan_key = plan_key
        logger.info(
            "[XFORMERS_PREFIX_FALLBACK_PLAN_BUILD] num_prefills=%d "
            "query_lens=%s context_lens=%s total_context_tokens=%d",
            prefill_meta.num_prefills,
            list(query_lens),
            list(context_lens),
            total_context_tokens,
        )
        return plan

    def _gather_prefix_kv_from_cache(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        plan: _XFormersPrefixFallbackPlan,
    ) -> tuple[torch.Tensor, torch.Tensor, _XFormersPrefixGatherProfile]:
        """把 prefix 命中的 paged KV 真正 gather 回连续张量。"""
        profile_enabled = envs.VLLM_BAM_XFORMERS_PREFIX_FALLBACK_PROFILE
        block_size = int(value_cache.shape[-1])

        if self._can_use_packed_prefix_gather(key_cache, value_cache):
            return self._gather_prefix_kv_from_packed_cache(
                key_cache=key_cache,
                value_cache=value_cache,
                plan=plan,
                profile_enabled=profile_enabled,
            )

        cache_view_start = cache_view_end = None
        if profile_enabled:
            cache_view_start, cache_view_end = self._new_cuda_event_pair()
            cache_view_start.record()
        key_src_cache = key_cache.permute(0, 3, 1, 2, 4).contiguous().view(
            key_cache.shape[0],
            block_size,
            self.num_kv_heads,
            self.head_size,
        )
        value_src_cache = value_cache.permute(0, 3, 1, 2).contiguous()
        if profile_enabled and cache_view_end is not None:
            cache_view_end.record()

        gathered_key = torch.empty(
            (plan.total_context_tokens, self.num_kv_heads, self.head_size),
            dtype=key_src_cache.dtype,
            device=key_src_cache.device,
        )
        gathered_value = torch.empty(
            (plan.total_context_tokens, self.num_kv_heads, self.head_size),
            dtype=value_src_cache.dtype,
            device=value_src_cache.device,
        )

        if plan.total_context_tokens == 0:
            cache_view_ms = 0.0
            if (profile_enabled and cache_view_start is not None
                    and cache_view_end is not None):
                cache_view_ms = self._cuda_elapsed_ms(cache_view_start,
                                                      cache_view_end)
            return gathered_key, gathered_value, _XFormersPrefixGatherProfile(
                mode="view_plus_gather",
                cache_view_ms=cache_view_ms,
                gather_key_ms=0.0,
                gather_value_ms=0.0,
            )

        gather_key_start = gather_key_end = None
        if profile_enabled:
            gather_key_start, gather_key_end = self._new_cuda_event_pair()
            gather_key_start.record()
        ops.gather_cache(
            src_cache=key_src_cache,
            dst=gathered_key,
            block_table=block_tables,
            cu_seq_lens=plan.cu_context_lens,
            batch_size=len(plan.query_lens),
        )
        if profile_enabled and gather_key_end is not None:
            gather_key_end.record()

        gather_value_start = gather_value_end = None
        if profile_enabled:
            gather_value_start, gather_value_end = self._new_cuda_event_pair()
            gather_value_start.record()
        ops.gather_cache(
            src_cache=value_src_cache,
            dst=gathered_value,
            block_table=block_tables,
            cu_seq_lens=plan.cu_context_lens,
            batch_size=len(plan.query_lens),
        )
        cache_view_ms = 0.0
        gather_key_ms = 0.0
        gather_value_ms = 0.0
        if profile_enabled and gather_value_end is not None:
            gather_value_end.record()
            cache_view_ms = self._cuda_elapsed_ms(cache_view_start,
                                                  cache_view_end)
            gather_key_ms = self._cuda_elapsed_ms(gather_key_start,
                                                  gather_key_end)
            gather_value_ms = self._cuda_elapsed_ms(gather_value_start,
                                                    gather_value_end)
        return gathered_key, gathered_value, _XFormersPrefixGatherProfile(
            mode="view_plus_gather",
            cache_view_ms=cache_view_ms,
            gather_key_ms=gather_key_ms,
            gather_value_ms=gather_value_ms,
        )

    def _gather_prefix_kv_from_packed_cache(
        self,
        *,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        plan: _XFormersPrefixFallbackPlan,
        profile_enabled: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, _XFormersPrefixGatherProfile]:
        """直接从 packed paged cache gather prefix KV，绕开中间 cache_view 拷贝。

        这条路径的目标非常明确：

        - 不是改变 fallback 的高层语义
        - 只是把 `permute(...).contiguous()` 这层布局重排去掉

        因此输出仍然保持：

        ```text
        gathered_{key,value}: [total_context_tokens, num_kv_heads, head_size]
        ```

        上层 `concat + xformers_forward` 完全不需要改。
        """
        gathered_key = torch.empty(
            (plan.total_context_tokens, self.num_kv_heads, self.head_size),
            dtype=key_cache.dtype,
            device=key_cache.device,
        )
        gathered_value = torch.empty(
            (plan.total_context_tokens, self.num_kv_heads, self.head_size),
            dtype=value_cache.dtype,
            device=value_cache.device,
        )
        if plan.total_context_tokens == 0:
            return gathered_key, gathered_value, _XFormersPrefixGatherProfile(
                mode="packed_direct",
                cache_view_ms=0.0,
                gather_key_ms=0.0,
                gather_value_ms=0.0,
            )

        total_elements = int(plan.total_context_tokens * self.num_kv_heads *
                             self.head_size)
        block = 256
        key_start = key_end = None
        value_start = value_end = None
        if profile_enabled:
            key_start, key_end = self._new_cuda_event_pair()
            key_start.record()
        _gather_packed_key_cache_kernel[(triton.cdiv(total_elements, block), )](
            key_cache,
            plan.context_block_ids,
            plan.context_block_offsets,
            plan.context_compact_positions,
            gathered_key,
            total_elements,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            pack_size=int(key_cache.shape[-1]),
            stride_cache_block=key_cache.stride(0),
            stride_cache_head=key_cache.stride(1),
            stride_cache_d_outer=key_cache.stride(2),
            stride_cache_token=key_cache.stride(3),
            stride_cache_pack=key_cache.stride(4),
            stride_out_token=gathered_key.stride(0),
            stride_out_head=gathered_key.stride(1),
            stride_out_dim=gathered_key.stride(2),
            BLOCK_SIZE=block,
        )
        if profile_enabled and key_end is not None:
            key_end.record()

        if profile_enabled:
            value_start, value_end = self._new_cuda_event_pair()
            value_start.record()
        _gather_packed_value_cache_kernel[(triton.cdiv(total_elements, block), )](
            value_cache,
            plan.context_block_ids,
            plan.context_block_offsets,
            plan.context_compact_positions,
            gathered_value,
            total_elements,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            stride_cache_block=value_cache.stride(0),
            stride_cache_head=value_cache.stride(1),
            stride_cache_dim=value_cache.stride(2),
            stride_cache_token=value_cache.stride(3),
            stride_out_token=gathered_value.stride(0),
            stride_out_head=gathered_value.stride(1),
            stride_out_dim=gathered_value.stride(2),
            BLOCK_SIZE=block,
        )
        if profile_enabled and value_end is not None:
            value_end.record()

        gather_key_ms = 0.0
        gather_value_ms = 0.0
        if (profile_enabled and key_start is not None and key_end is not None
                and value_start is not None and value_end is not None):
            gather_key_ms = self._cuda_elapsed_ms(key_start, key_end)
            gather_value_ms = self._cuda_elapsed_ms(value_start, value_end)
        return gathered_key, gathered_value, _XFormersPrefixGatherProfile(
            mode="packed_direct",
            cache_view_ms=0.0,
            gather_key_ms=gather_key_ms,
            gather_value_ms=gather_value_ms,
        )

    def _fill_prefix_kv_from_packed_cache_into_full_buffer(
        self,
        *,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        plan: _XFormersPrefixFallbackPlan,
        full_key: torch.Tensor,
        full_value: torch.Tensor,
        profile_enabled: bool,
    ) -> _XFormersPrefixGatherProfile:
        """把 packed paged cache 中的 prefix KV 直接填入最终 full KV buffer。

        这是当前主热路径真正想要的形态：

        - 不再先生成 `gathered_key/gathered_value`
        - 也不再把 prefix 从中间张量二次 copy 到 workspace
        - Triton gather kernel 直接根据 `context_full_positions`
          把每个 prefix token scatter 到 full KV 中正确的位置

        这样 prefix 侧就只保留“一次从 paged cache 取数并写入最终目标”的必要搬运。
        """
        if plan.total_context_tokens == 0:
            return _XFormersPrefixGatherProfile(
                mode="packed_direct_to_workspace",
                cache_view_ms=0.0,
                gather_key_ms=0.0,
                gather_value_ms=0.0,
            )

        total_elements = int(plan.total_context_tokens * self.num_kv_heads *
                             self.head_size)
        block = 256
        key_start = key_end = None
        value_start = value_end = None
        if profile_enabled:
            key_start, key_end = self._new_cuda_event_pair()
            key_start.record()
        _gather_packed_key_cache_kernel[(triton.cdiv(total_elements, block), )](
            key_cache,
            plan.context_block_ids,
            plan.context_block_offsets,
            plan.context_full_positions,
            full_key,
            total_elements,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            pack_size=int(key_cache.shape[-1]),
            stride_cache_block=key_cache.stride(0),
            stride_cache_head=key_cache.stride(1),
            stride_cache_d_outer=key_cache.stride(2),
            stride_cache_token=key_cache.stride(3),
            stride_cache_pack=key_cache.stride(4),
            stride_out_token=full_key.stride(0),
            stride_out_head=full_key.stride(1),
            stride_out_dim=full_key.stride(2),
            BLOCK_SIZE=block,
        )
        if profile_enabled and key_end is not None:
            key_end.record()

        if profile_enabled:
            value_start, value_end = self._new_cuda_event_pair()
            value_start.record()
        _gather_packed_value_cache_kernel[(triton.cdiv(total_elements, block), )](
            value_cache,
            plan.context_block_ids,
            plan.context_block_offsets,
            plan.context_full_positions,
            full_value,
            total_elements,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            stride_cache_block=value_cache.stride(0),
            stride_cache_head=value_cache.stride(1),
            stride_cache_dim=value_cache.stride(2),
            stride_cache_token=value_cache.stride(3),
            stride_out_token=full_value.stride(0),
            stride_out_head=full_value.stride(1),
            stride_out_dim=full_value.stride(2),
            BLOCK_SIZE=block,
        )
        if profile_enabled and value_end is not None:
            value_end.record()

        gather_key_ms = 0.0
        gather_value_ms = 0.0
        if (profile_enabled and key_start is not None and key_end is not None
                and value_start is not None and value_end is not None):
            gather_key_ms = self._cuda_elapsed_ms(key_start, key_end)
            gather_value_ms = self._cuda_elapsed_ms(value_start, value_end)
        return _XFormersPrefixGatherProfile(
            mode="packed_direct_to_workspace",
            cache_view_ms=0.0,
            gather_key_ms=gather_key_ms,
            gather_value_ms=gather_value_ms,
        )

    @staticmethod
    def _can_use_packed_prefix_gather(
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> bool:
        """判断当前 fallback 是否可以直接走 packed direct gather。

        当前先只在最稳定、也是我们真实主线正在使用的配置上启用：

        - Triton 可用
        - key/value cache 都在 CUDA 上
        - dtype 不是 fp8/uint8

        如果这些条件不满足，就自动退回原来的：

        ```text
        cache_view + gather_cache
        ```
        """
        if triton is None:
            return False
        if (not key_cache.is_cuda) or (not value_cache.is_cuda):
            return False
        if key_cache.dtype == torch.uint8 or value_cache.dtype == torch.uint8:
            return False
        return True

    @staticmethod
    def _new_cuda_event_pair() -> tuple[torch.cuda.Event, torch.cuda.Event]:
        """为 fallback profile 创建一对计时 event。"""
        return (torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True))

    @staticmethod
    def _cuda_elapsed_ms(start: torch.cuda.Event,
                         end: torch.cuda.Event) -> float:
        """读取同一条 stream 上一对 event 的耗时。"""
        torch.cuda.synchronize()
        return float(start.elapsed_time(end))

    def _run_memory_efficient_xformers_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: XFormersMetadata,
        attn_type: str = AttentionType.DECODER,
    ) -> torch.Tensor:
        """Attention for 1D query of multiple prompts. Multiple prompt
        tokens are flattened in to `query` input.

        See https://facebookresearch.github.io/xformers/components/ops.html
        for API spec.

        Args:
            output: shape = [num_prefill_tokens, num_heads, head_size]
            query: shape = [num_prefill_tokens, num_heads, head_size]
            key: shape = [num_prefill_tokens, num_kv_heads, head_size]
            value: shape = [num_prefill_tokens, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
            attn_type: Select attention type, between encoder attention,
                       decoder self-attention, or encoder/decoder cross-
                       attention. Defaults to decoder self-attention,
                       which is the vLLM default generally
        """

        original_query = query
        if self.num_kv_heads != self.num_heads:
            # GQA/MQA requires the shape [B, M, G, H, K].
            # Note that the output also has the same shape (which is different
            # from a spec from the doc).
            query = query.view(query.shape[0], self.num_kv_heads,
                               self.num_queries_per_kv, query.shape[-1])
            key = key[:, :,
                      None, :].expand(key.shape[0], self.num_kv_heads,
                                      self.num_queries_per_kv, key.shape[-1])
            value = value[:, :,
                          None, :].expand(value.shape[0], self.num_kv_heads,
                                          self.num_queries_per_kv,
                                          value.shape[-1])

        # Set attention bias if not provided. This typically happens at
        # the very attention layer of every iteration.
        # FIXME(woosuk): This is a hack.
        attn_bias = _get_attn_bias(attn_metadata, attn_type)
        if attn_bias is None:
            if self.alibi_slopes is None:

                # Cross attention block of decoder branch of encoder-decoder
                # model uses seq_lens for dec / encoder_seq_lens for enc
                if (attn_type == AttentionType.ENCODER_DECODER):
                    assert attn_metadata.seq_lens is not None
                    assert attn_metadata.encoder_seq_lens is not None

                    # Cross-attention mask is non-causal
                    attn_bias = BlockDiagonalMask.from_seqlens(
                        attn_metadata.seq_lens,
                        attn_metadata.encoder_seq_lens,
                        device=query.device)

                # Encoder branch of encoder-decoder model uses
                # attn_metadata.encoder_seq_lens
                elif attn_type == AttentionType.ENCODER:

                    assert attn_metadata.encoder_seq_lens is not None

                    # Encoder self-attention mask is non-causal
                    attn_bias = BlockDiagonalMask.from_seqlens(
                        attn_metadata.encoder_seq_lens, device=query.device)

                # Self-attention block of encoder-only model just
                # uses the seq_lens directly.
                elif attn_type == AttentionType.ENCODER_ONLY:
                    assert attn_metadata.seq_lens is not None

                    # Encoder self-attention mask is non-causal
                    attn_bias = BlockDiagonalMask.from_seqlens(
                        attn_metadata.seq_lens, device=query.device)

                # Self-attention block of decoder branch just
                # uses the seq_lens directly
                elif attn_type == AttentionType.DECODER:
                    assert attn_metadata.seq_lens is not None

                    # Decoder self-attention mask is causal
                    attn_bias = BlockDiagonalCausalMask.from_seqlens(
                        attn_metadata.seq_lens, device=query.device)
                else:
                    raise ValueError("Unknown AttentionType: %s", attn_type)

                if self.sliding_window is not None:
                    attn_bias = attn_bias.make_local_attention(
                        self.sliding_window)
                attn_bias = [attn_bias]
            else:
                assert attn_type == AttentionType.DECODER
                assert attn_metadata.seq_lens is not None
                attn_bias = _make_alibi_bias(self.alibi_slopes,
                                             self.num_kv_heads, query.dtype,
                                             attn_metadata.seq_lens)

            _set_attn_bias(attn_metadata, attn_bias, attn_type)

        # No alibi slopes.
        # TODO(woosuk): Too many view operations. Let's try to reduce
        # them in the future for code readability.
        if self.alibi_slopes is None:
            # Add the batch dimension.
            query = query.unsqueeze(0)
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
            out = xops.memory_efficient_attention_forward(
                query,
                key,
                value,
                attn_bias=attn_bias[0],
                p=0.0,
                scale=self.scale)
            return out.view_as(original_query)

        # Attention with alibi slopes.
        # FIXME(woosuk): Because xformers does not support dynamic sequence
        # lengths with custom attention bias, we process each prompt one by
        # one. This is inefficient, especially when we have many short prompts.
        assert attn_metadata.seq_lens is not None
        output = torch.empty_like(original_query)
        start = 0
        for i, seq_len in enumerate(attn_metadata.seq_lens):
            end = start + seq_len
            out = xops.memory_efficient_attention_forward(
                query[None, start:end],
                key[None, start:end],
                value[None, start:end],
                attn_bias=attn_bias[i],
                p=0.0,
                scale=self.scale)
            # TODO(woosuk): Unnecessary copy. Optimize.
            output[start:end].copy_(out.view_as(original_query[start:end]))
            start += seq_len
        return output


def _make_alibi_bias(
    alibi_slopes: torch.Tensor,
    num_kv_heads: int,
    dtype: torch.dtype,
    seq_lens: List[int],
) -> List[AttentionBias]:
    attn_biases: List[AttentionBias] = []
    for seq_len in seq_lens:
        bias = torch.arange(seq_len, dtype=dtype)
        # NOTE(zhuohan): HF uses
        #     `bias = bias[None, :].repeat(seq_len, 1)`
        # here. We find that both biases give the same results, but
        # the bias below more accurately follows the original ALiBi
        # paper.
        # Calculate a matrix where each element represents ith element- jth
        # element.
        bias = bias[None, :] - bias[:, None]

        padded_len = (seq_len + 7) // 8 * 8
        num_heads = alibi_slopes.shape[0]
        bias = torch.empty(
            1,  # batch size
            num_heads,
            seq_len,
            padded_len,
            device=alibi_slopes.device,
            dtype=dtype,
        )[:, :, :, :seq_len].copy_(bias)
        bias.mul_(alibi_slopes[:, None, None])
        attn_biases.append(LowerTriangularMaskWithTensorBias(bias))

    return attn_biases
