"""Quest adapter with an explicit boundary around the selector only."""

from __future__ import annotations

from typing import Optional, Sequence

from ...common.selectors import ExplicitSelector, TailSelector


class QuestSelector:
    """Translate Quest block choices into the shared sparse access plan.

    ``selected_blocks_by_layer`` is the preferred input from a Quest
    implementation. The deterministic tail proxy exists only for a smoke test
    and is labeled as an approximation in the resulting source field.
    """

    name = "quest"

    def __init__(self,
                 block_budget: int = 0,
                 selected_blocks_by_layer: Optional[
                     Sequence[Sequence[int]]] = None) -> None:
        self._selected = (None if selected_blocks_by_layer is None else
                          ExplicitSelector(selected_blocks_by_layer, self.name))
        self._proxy = (None if block_budget <= 0 else
                       TailSelector(block_budget, "quest_proxy"))
        if self._selected is None and self._proxy is None:
            raise ValueError("QuestSelector needs choices or a positive budget")

    def select(self, num_layers: int, num_blocks: int):
        if self._selected is not None:
            return self._selected.select(num_layers, num_blocks)
        return self._proxy.select(num_layers, num_blocks)
