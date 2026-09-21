from __future__ import annotations

from apc.cnp.adaptation import (
    ADAPT_TRAIN_EXEMPLARS,
    SETS_PER_CONDITION,
    SMALL_ARM_SETS_PER_CONDITION,
    paired_adaptation_training_records,
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
