# SPDX-License-Identifier: Apache-2.0
"""BaM pages -> LMCache KV tensor 的 GPU refill kernel。

当前 BaM 中一个 LMCache chunk 被存成固定数量的 128KB pages：

  [2, num_layers, slot_tokens, hidden_dim]
    -> [2, num_layers, pages_per_kv_layer, page_token_capacity, hidden_dim]
    -> [page_count, 128KB]

这个文件提供一个 Triton kernel，把 BaM pages 直接拷回 LMCache 期望的
KV tensor `[2, num_layers, actual_tokens, hidden_dim]`。

这样做的意义：

- CPU 只负责 launch kernel，不再用 Python view/reshape/contiguous 表达转换。
- 数据转换在 GPU kernel 中显式完成。
- 后续如果要回填 vLLM paged KV cache，可以在这里继续新增 kernel。
"""

from __future__ import annotations

from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - 运行环境可能禁用 Triton
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _bam_pages_to_lmcache_kernel(
        pages_ptr,
        out_ptr,
        total_elements: tl.constexpr,
        num_layers: tl.constexpr,
        actual_tokens: tl.constexpr,
        hidden_dim: tl.constexpr,
        page_token_capacity: tl.constexpr,
        pages_per_kv_layer: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """按元素拷贝并完成 padding 槽位到真实 token 槽位的映射。

        输出逻辑下标：

          out[kv, layer, token, hidden]

        对应 BaM page 里的逻辑位置：

          page_id =
              kv * num_layers * pages_per_kv_layer
            + layer * pages_per_kv_layer
            + token // page_token_capacity

          offset_in_page =
              (token % page_token_capacity) * hidden_dim + hidden

        因为 `pages_ptr` 会按 out dtype 传入，所以 offset 单位是元素，不是字节。
        """
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_elements

        hidden = offsets % hidden_dim
        token = (offsets // hidden_dim) % actual_tokens
        layer = (offsets // (hidden_dim * actual_tokens)) % num_layers
        kv = offsets // (hidden_dim * actual_tokens * num_layers)

        token_page = token // page_token_capacity
        token_in_page = token - token_page * page_token_capacity
        page_id = (kv * num_layers * pages_per_kv_layer +
                   layer * pages_per_kv_layer + token_page)
        page_elem_offset = token_in_page * hidden_dim + hidden
        page_width_elems = page_token_capacity * hidden_dim
        src_offsets = page_id * page_width_elems + page_elem_offset

        values = tl.load(pages_ptr + src_offsets, mask=mask)
        tl.store(out_ptr + offsets, values, mask=mask)


def triton_refill_available() -> bool:
    return triton is not None


def refill_pages_to_lmcache_tensor(pages: torch.Tensor, metadata: Any,
                                   layout: Any) -> torch.Tensor:
    """用 GPU kernel 将 BaM pages 还原成 LMCache KV tensor。

    参数：
      pages:
        `[pages_per_chunk, 128KB]` 的 CUDA uint8 tensor。
      metadata:
        BaMChunkMetadata，提供 actual_tokens / dtype / shape。
      layout:
        LMCacheBaMPageLayout，提供 layer/page/token 布局参数。

    返回：
      `[2, num_layers, actual_tokens, hidden_dim]` 的 CUDA tensor。

    形状变化顺序可以理解为：
      1. `[page_count, 128KB]`
      2. 视图成与 dtype 对齐的一维 byte/element buffer
      3. 重新映射回 `[2, num_layers, actual_tokens, hidden_dim]`
    """
    if triton is None:
        raise RuntimeError("Triton is not available")
    if not pages.is_cuda:
        raise ValueError("pages must be a CUDA tensor for GPU refill")
    if pages.dtype != torch.uint8:
        raise TypeError(f"pages must be uint8, got {pages.dtype}")

    expected_shape = (layout.pages_per_chunk, layout.page_bytes)
    if tuple(pages.shape) != expected_shape:
        raise ValueError(
            "pages shape mismatch: "
            f"expected={expected_shape}, got={tuple(pages.shape)}")

    actual_tokens = int(metadata.actual_tokens)
    out_shape = (2, int(layout.num_layers), actual_tokens,
                 int(layout.hidden_dim))
    out = torch.empty(out_shape, device=pages.device, dtype=metadata.dtype)

    # pages 的底层字节与目标 dtype 一致。
    # 这里仅创建 dtype 视图，不做数据搬运。
    pages_typed = pages.view(metadata.dtype).view(-1)
    out_flat = out.view(-1)

    total_elements = int(out_flat.numel())
    block_size = 256
    grid = (triton.cdiv(total_elements, block_size), )
    _bam_pages_to_lmcache_kernel[grid](
        pages_typed,
        out_flat,
        total_elements,
        int(layout.num_layers),
        actual_tokens,
        int(layout.hidden_dim),
        int(layout.page_token_capacity),
        int(layout.pages_per_kv_layer),
        BLOCK_SIZE=block_size,
    )
    return out
