"""SolidAttention-style speculative selection boundary."""

from __future__ import annotations

from typing import Optional, Sequence

from ...common.selectors import ExplicitSelector, TailSelector


class SolidAttentionSelector:
    name = "solidattention"

    def __init__(self, block_budget: int = 0,
                 selections: Optional[Sequence[Sequence[int]]] = None) -> None:
        self._delegate = (ExplicitSelector(selections, self.name)
                          if selections is not None else
                          TailSelector(block_budget, "solidattention_proxy"))

    def select(self, num_layers: int, num_blocks: int):
        return self._delegate.select(num_layers, num_blocks)
