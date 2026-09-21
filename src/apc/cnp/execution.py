"""Normal CNP recipe execution over prediction masks, with no oracle reset path."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from apc.cnp.contracts import CNPRecipe, SelectionResult, SetState
from apc.cnp.reference import count_selected, sum_first_selected
from apc.primitives.bank import PrimitiveBank


@dataclass(frozen=True)
class CNPExecutionResult:
    """All learned selection steps and optional deterministic terminal output."""

    state: SetState
    selections: tuple[SelectionResult, ...]
    terminal: torch.Tensor | None


def execute_recipe(
    bank: PrimitiveBank,
    family_to_id: dict[str, int],
    state: SetState,
    recipe: CNPRecipe,
) -> CNPExecutionResult:
    """Run only primitives named by a recipe; pass predicted masks to each next step."""

    current = state
    selections: list[SelectionResult] = []
    terminal: torch.Tensor | None = None
    for index, call in enumerate(recipe.steps):
        is_last = index == len(recipe.steps) - 1
        if call.family == "CONDITIONAL_SELECT":
            if call.family not in family_to_id:
                raise KeyError("No primitive ID configured for CONDITIONAL_SELECT")
            primitive = bank.get(family_to_id[call.family])
            if not hasattr(primitive, "forward"):
                raise TypeError("Configured CNP primitive has no forward method")
            result = primitive(current, call.arguments)
            if not isinstance(result, SelectionResult):
                raise TypeError("CNP select primitive must return SelectionResult")
            current = result.state
            selections.append(result)
            continue
        if not is_last:
            raise ValueError(f"Deterministic terminal family {call.family} must be last")
        if call.family == "COUNT":
            terminal = count_selected(current)
        elif call.family == "SUM_FIRST":
            terminal = sum_first_selected(current)
        else:
            raise ValueError(f"Unsupported CNP family {call.family}")
    return CNPExecutionResult(state=current, selections=tuple(selections), terminal=terminal)
