from __future__ import annotations

import torch

from apc.cnp.development import CausalAccumulator


def test_perfect_correct_predictions_have_unit_gap_on_each_control_support() -> None:
    target = torch.tensor([[True, False, True, False]])
    valid = torch.tensor([[True, True, True, False]])
    accumulator = CausalAccumulator()
    accumulator.add(
        target,
        valid,
        target,
        {
            "wrong_family": torch.tensor([[False, True, False, False]]),
            "wrong_argument": torch.tensor([[True, True, False, False]]),
            "none": valid,
        },
        {
            "wrong_family": torch.tensor([[False, True, False, False]]),
            "wrong_argument": torch.tensor([[True, True, False, False]]),
            "none": valid,
        },
    )

    controls = accumulator.report(minimum_effectful_sets=1)["controls"]
    assert isinstance(controls, dict)
    for control in controls.values():
        assert control["effectful_items"] > 0
        assert control["effectful_sets"] == 1
        assert control["correct"] == 1.0
        assert control["intervention"] == 0.0
        assert control["correct_minus_intervention"] == 1.0


def test_reference_effectful_support_is_independent_of_model_predictions() -> None:
    target = torch.tensor([[True, False, True]])
    valid = torch.tensor([[True, True, True]])
    reference_wrong_argument = torch.tensor([[True, True, True]])
    accumulator = CausalAccumulator()
    accumulator.add(
        target,
        valid,
        target,
        {
            "wrong_family": ~target,
            "wrong_argument": target,
            "none": valid,
        },
        {
            "wrong_family": ~target,
            "wrong_argument": reference_wrong_argument,
            "none": valid,
        },
    )

    controls = accumulator.report(minimum_effectful_sets=1)["controls"]
    assert isinstance(controls, dict)
    wrong_argument = controls["wrong_argument"]
    assert isinstance(wrong_argument, dict)
    assert wrong_argument["effectful_items"] == 1
    assert wrong_argument["correct"] == 1.0
    assert wrong_argument["intervention"] == 1.0
    assert wrong_argument["correct_minus_intervention"] == 0.0
