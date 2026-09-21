from __future__ import annotations

import pytest
import torch

from apc.cnp.contracts import CNPPrimitiveCall, CNPRecipe, SelectArguments, SetState


def _state(width: int = 2) -> SetState:
    return SetState(
        values=torch.zeros((1, width, 8), dtype=torch.float32),
        valid=torch.ones((1, width), dtype=torch.bool),
        item_ids=torch.arange(width, dtype=torch.int64).unsqueeze(0),
    )


def _arguments() -> SelectArguments:
    return SelectArguments(
        query=torch.zeros((1, 8), dtype=torch.float32),
        threshold=torch.zeros((1,), dtype=torch.float32),
    )


def test_cnp_call_requires_typed_arguments_only_for_select() -> None:
    call = CNPPrimitiveCall("CONDITIONAL_SELECT", _arguments())
    assert call.arguments is not None
    assert CNPPrimitiveCall("COUNT").arguments is None
    with pytest.raises(ValueError, match="requires"):
        CNPPrimitiveCall("CONDITIONAL_SELECT")
    with pytest.raises(ValueError, match="does not accept"):
        CNPPrimitiveCall("COUNT", _arguments())


def test_set_state_rejects_evaluation_labels_and_invalid_shapes() -> None:
    with pytest.raises(TypeError, match="float32"):
        SetState(
            values=torch.zeros((1, 2, 8), dtype=torch.float64),
            valid=torch.ones((1, 2), dtype=torch.bool),
            item_ids=torch.zeros((1, 2), dtype=torch.int64),
        )
    with pytest.raises(ValueError, match="feature dimension"):
        SetState(
            values=torch.zeros((1, 2, 7), dtype=torch.float32),
            valid=torch.ones((1, 2), dtype=torch.bool),
            item_ids=torch.zeros((1, 2), dtype=torch.int64),
        )
    assert _state().width == 2


def test_recipe_requires_nonempty_explicit_steps() -> None:
    with pytest.raises(ValueError, match="at least one"):
        CNPRecipe("empty", ())
    assert CNPRecipe("one", (CNPPrimitiveCall("COUNT"),)).name == "one"
