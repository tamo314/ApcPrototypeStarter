from __future__ import annotations

import torch

from apc.cnp.adaptation import (
    ADAPT_TRAIN_EXEMPLARS,
    SETS_PER_CONDITION,
    SMALL_ARM_SETS_PER_CONDITION,
    _base_weights_hash,
    paired_adaptation_training_records,
    quality_floor,
)
from apc.cnp.contracts import SelectArguments, SetState
from apc.cnp.data import SOURCE_THRESHOLDS
from apc.cnp.primitive import ConditionalSelectPrimitive
from apc.cnp.training import adapter_parameters, clone_local_candidate, one_training_step


def test_adaptation_teacher_arms_are_fixed_and_prefix_shared() -> None:
    full, small = paired_adaptation_training_records("q1+q2+")
    assert len(full) == ADAPT_TRAIN_EXEMPLARS * len(SOURCE_THRESHOLDS) * SETS_PER_CONDITION
    assert len(small) == (
        ADAPT_TRAIN_EXEMPLARS * len(SOURCE_THRESHOLDS) * SMALL_ARM_SETS_PER_CONDITION
    )
    full_digests = {record.digest() for record in full}
    assert all(record.digest() in full_digests for record in small)


def test_quality_floor_requires_both_registered_metrics() -> None:
    assert quality_floor({"balanced_accuracy": 0.95, "mean_set_f1": 0.90})
    assert not quality_floor({"balanced_accuracy": 0.949, "mean_set_f1": 1.0})
    assert not quality_floor({"balanced_accuracy": 1.0, "mean_set_f1": 0.899})


def test_local_base_hash_excludes_trainable_adapter() -> None:
    parent = ConditionalSelectPrimitive(0)
    candidate = clone_local_candidate(parent)
    before = _base_weights_hash(candidate)
    optimizer = torch.optim.AdamW(adapter_parameters(candidate), lr=0.001)
    state = SetState(
        values=torch.rand((1, 2, 8)),
        valid=torch.tensor([[True, True]]),
        item_ids=torch.tensor([[0, 1]]),
    )
    arguments = SelectArguments(
        torch.zeros((1, 8)), torch.tensor([0.5])
    )
    one_training_step(candidate, state, arguments, torch.tensor([[True, False]]), optimizer)
    assert _base_weights_hash(candidate) == before
