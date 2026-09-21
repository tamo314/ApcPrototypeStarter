from __future__ import annotations

from apc.cnp.adaptation import (
    ADAPT_TRAIN_EXEMPLARS,
    SETS_PER_CONDITION,
    SMALL_ARM_SETS_PER_CONDITION,
    paired_adaptation_training_records,
    quality_floor,
)
from apc.cnp.data import SOURCE_THRESHOLDS


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
