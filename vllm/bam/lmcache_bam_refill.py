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

    @triton.jit
    def _bam_pages_to_lmcache_kernel_with_token_offset(
        pages_ptr,
        out_ptr,
        total_elements,
        num_layers,
        actual_tokens,
        total_output_tokens,
        token_offset,
        hidden_dim: tl.constexpr,
        page_token_capacity: tl.constexpr,
        pages_per_kv_layer: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """把一个 chunk 直接写到更大输出 tensor 的指定 token 区间。

        这个 kernel 是为 direct placement v1 准备的第一步能力：

        - 不再强制“一个 chunk -> 一个独立输出 tensor”
        - 允许多个 chunk 依次 refill 到一个合并后的 batch tensor

        这样上层就可以把：

        ```text
        chunk0 pages -> chunk0 tensor
        chunk1 pages -> chunk1 tensor
        ...
        cat([...])   -> merged tensor
        ```

        收敛成：

        ```text
        chunk0 pages -> merged[:, :, token0:token1, :]
        chunk1 pages -> merged[:, :, token1:token2, :]
        ```

        这一步虽然还没有完全消掉“LMCache 标准 KV tensor”这个中间态，但已经
        去掉了逐 chunk 中间 tensor 分配和额外 cat/copy，为后续更真实的
        KVPlacementPlan / fused placement 铺路。

        这里有一个很重要的实现细节：

        - `hidden_dim / page_token_capacity / pages_per_kv_layer` 这类布局参数
          在整个模型生命周期内基本不变，保留为 `tl.constexpr` 可以让 Triton
          继续做静态优化。
        - `token_offset / total_output_tokens / actual_tokens / total_elements`
          会随着 batch 里的 chunk 位置变化而变化。如果把它们也声明成
          `tl.constexpr`，那么同一批 4 个 chunk 只要 offset 不同，就可能触发
          多份 kernel 专用化与编译。

        merged refill 路径里 `token_offset` 恰好是：

        ```text
        0, 256, 512, 768, ...
        ```

        这会把“只是写入目标区间不同”的多个 launch 错误地放大成多份编译变体。
        因此这里显式把这些随请求变化的量改成运行时参数，只让真正稳定的 layout
        参数参与编译期专用化。
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

        out_token = token_offset + token
        dst_offsets = (((kv * num_layers + layer) * total_output_tokens +
                        out_token) * hidden_dim + hidden)

        values = tl.load(pages_ptr + src_offsets, mask=mask)
        tl.store(out_ptr + dst_offsets, values, mask=mask)


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


def refill_pages_to_lmcache_tensor_into(
    pages: torch.Tensor,
    *,
    out: torch.Tensor,
    token_offset: int,
    actual_tokens: int,
    layout: Any,
) -> None:
    """把一个 chunk 的 BaM pages 直接 refill 到更大输出 tensor 的指定区间。

    输入/输出约定：

    - `pages`:
      `[pages_per_chunk, 128KB]` 的 CUDA uint8 tensor
    - `out`:
      `[2, num_layers, total_tokens, hidden_dim]` 的 CUDA tensor
    - `token_offset`:
      当前 chunk 在 `out` 第 3 维里的起始 token 下标
    - `actual_tokens`:
      当前 chunk 的真实 token 数

    这个函数是 direct placement 路线的“中间态收缩”工具：

    - 旧方式：每个 chunk 各自 refill 成一个独立 tensor
    - 新方式：多个 chunk 直接写入一个合并 batch tensor

    这样后续就能只做一次 LMCache connector transfer，减少逐 chunk 的
    Python 组织和 kernel launch 开销。
    """
    if triton is None:
        raise RuntimeError("Triton is not available")
    if not pages.is_cuda:
        raise ValueError("pages must be a CUDA tensor for GPU refill")
    if pages.dtype != torch.uint8:
        raise TypeError(f"pages must be uint8, got {pages.dtype}")
    if not out.is_cuda:
        raise ValueError("out must be a CUDA tensor")
    if out.dim() != 4:
        raise ValueError(
            "out must have shape [2, num_layers, total_tokens, hidden_dim], "
            f"got {tuple(out.shape)}")
    if int(out.shape[0]) != 2 or int(out.shape[1]) != int(layout.num_layers):
        raise ValueError(
            "out shape mismatch for KV refill: "
            f"shape={tuple(out.shape)} num_layers={layout.num_layers}")
    if int(out.shape[3]) != int(layout.hidden_dim):
        raise ValueError(
            "out hidden_dim mismatch: "
            f"shape={tuple(out.shape)} hidden_dim={layout.hidden_dim}")

    expected_shape = (layout.pages_per_chunk, layout.page_bytes)
    if tuple(pages.shape) != expected_shape:
        raise ValueError(
            "pages shape mismatch: "
            f"expected={expected_shape}, got={tuple(pages.shape)}")

    token_offset = int(token_offset)
    actual_tokens = int(actual_tokens)
    total_output_tokens = int(out.shape[2])
    if token_offset < 0:
        raise ValueError(f"token_offset must be non-negative, got {token_offset}")
    if actual_tokens <= 0:
        raise ValueError(f"actual_tokens must be positive, got {actual_tokens}")
    if token_offset + actual_tokens > total_output_tokens:
        raise ValueError(
            "refill target token range overflow: "
            f"offset={token_offset} actual_tokens={actual_tokens} "
            f"total_output_tokens={total_output_tokens}")

    pages_typed = pages.view(out.dtype).view(-1)
    out_flat = out.view(-1)
    # 这里按“当前 chunk 实际要写回的元素数”启动 grid。虽然输出 tensor 是一个
    # merged batch，但每次 launch 仍只覆盖当前 chunk 的有效 token 区间。
    total_elements = int(2 * layout.num_layers * actual_tokens *
                         layout.hidden_dim)
    block_size = 256
    grid = (triton.cdiv(total_elements, block_size), )
    _bam_pages_to_lmcache_kernel_with_token_offset[grid](
        pages_typed,
        out_flat,
        total_elements,
        int(layout.num_layers),
        actual_tokens,
        total_output_tokens,
        token_offset,
        int(layout.hidden_dim),
        int(layout.page_token_capacity),
        int(layout.pages_per_kv_layer),
        BLOCK_SIZE=block_size,
    )
