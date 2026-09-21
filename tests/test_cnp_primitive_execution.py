from __future__ import annotations

import torch

from apc.cnp.contracts import CNPPrimitiveCall, CNPRecipe, SelectArguments, SetState
from apc.cnp.execution import execute_recipe
from apc.cnp.primitive import ConditionalSelectPrimitive
from apc.primitives.bank import PrimitiveBank


def _state(valid: torch.Tensor | None = None) -> SetState:
    valid = valid if valid is not None else torch.tensor([[True, True, False]])
    return SetState(
        values=torch.tensor([[[0.1] * 8, [0.2] * 8, [100.0] * 8]], dtype=torch.float32),
        valid=valid,
        item_ids=torch.tensor([[7, 8, 9]], dtype=torch.int64),
    )


def _arguments(threshold: float = 0.3) -> SelectArguments:
    return SelectArguments(
        query=torch.zeros((1, 8), dtype=torch.float32),
        threshold=torch.tensor([threshold], dtype=torch.float32),
    )


def test_primitive_parameter_count_and_padding_invariance() -> None:
    torch.manual_seed(4)
    primitive = ConditionalSelectPrimitive(17)
    assert primitive.base_parameter_count == 8449
    first = primitive(_state(), _arguments())
    changed_padding = SetState(
        values=torch.tensor([[[0.1] * 8, [0.2] * 8, [-500.0] * 8]], dtype=torch.float32),
        valid=torch.tensor([[True, True, False]]),
        item_ids=torch.tensor([[99, 98, -1]], dtype=torch.int64),
    )
    second = primitive(changed_padding, _arguments())
    assert torch.allclose(first.logits[:, :2], second.logits[:, :2])
    assert not bool(first.selected[0, 2])
    assert primitive.forward_call_count == 2


def test_recipe_executes_only_selected_primitive_and_passes_predicted_mask() -> None:
    torch.manual_seed(5)
    selected = ConditionalSelectPrimitive(1)
    unused = ConditionalSelectPrimitive(2)
    bank = PrimitiveBank([selected, unused])
    recipe = CNPRecipe(
        "select_then_count",
        (CNPPrimitiveCall("CONDITIONAL_SELECT", _arguments()), CNPPrimitiveCall("COUNT")),
    )
    result = execute_recipe(bank, {"CONDITIONAL_SELECT": 1}, _state(), recipe)
    assert result.terminal is not None
    assert selected.forward_call_count == 1
    assert unused.forward_call_count == 0
    assert result.terminal.tolist() == result.state.valid.sum(dim=1).tolist()
