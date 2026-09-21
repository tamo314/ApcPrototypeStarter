from __future__ import annotations

import pytest
import torch

from apc.cnp.contracts import SelectArguments, SetState
from apc.cnp.data import audit_split_disjoint, make_record, make_world, records_manifest
from apc.cnp.reference import count_selected, reference_select, sum_first_selected


def test_reference_select_matches_hand_calculated_identity_world_case() -> None:
    state = SetState(
        values=torch.tensor([[[0.0] * 8, [1.0] * 8]], dtype=torch.float32),
        valid=torch.tensor([[True, True]]),
        item_ids=torch.tensor([[0, 1]], dtype=torch.int64),
    )
    arguments = SelectArguments(
        query=torch.zeros((1, 8), dtype=torch.float32),
        threshold=torch.tensor([0.01], dtype=torch.float32),
    )
    world = make_world()
    result = reference_select(state, arguments, world)
    assert result.selected.shape == (1, 2)
    assert result.selected[0, 0]
    assert not result.selected[0, 1]


def test_empty_state_is_a_valid_empty_selection_and_downstream_value() -> None:
    state = SetState(
        values=torch.empty((1, 0, 8), dtype=torch.float32),
        valid=torch.empty((1, 0), dtype=torch.bool),
        item_ids=torch.empty((1, 0), dtype=torch.int64),
    )
    arguments = SelectArguments(torch.zeros((1, 8), dtype=torch.float32), torch.zeros(1))
    selected = reference_select(state, arguments, make_world())
    assert selected.selected.numel() == 0
    assert count_selected(selected.state).tolist() == [0]
    assert sum_first_selected(selected.state).tolist() == [0.0]


def test_records_are_deterministic_and_split_audit_rejects_overlap() -> None:
    first = make_record(
        role="source_train", condition_key="q0", length=4, threshold=0.8, example_index=0
    )
    same = make_record(
        role="source_train", condition_key="q0", length=4, threshold=0.8, example_index=0
    )
    assert first.digest() == same.digest()
    assert records_manifest([first])["records"] == 1
    with pytest.raises(ValueError, match="split overlap"):
        audit_split_disjoint({"source_train": [first], "dev_eval": [first]})
