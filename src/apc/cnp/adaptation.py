"""Fixed CNP-004 sequential-adaptation data and model helpers.

The public helpers in this module deliberately stop short of experiment execution:
they construct the registered data and training batches so the runner can preserve
paired inputs across LOCAL and baselines without consulting evaluation panels.
"""

from __future__ import annotations

from collections.abc import Iterator

from apc.cnp.data import (
    ADAPTATION_BLOCKS,
    KNOWN_LENGTHS,
    SOURCE_THRESHOLDS,
    CNPRecord,
    DataRole,
    make_adaptation_record,
    make_world,
)
from apc.cnp.development import EVAL_BATCH_SETS

ADAPT_TRAIN_EXEMPLARS = 8
TRANSFER_EXEMPLARS = 16
SETS_PER_CONDITION = 64
SMALL_ARM_SETS_PER_CONDITION = 16


def _length_for_example(example_index: int) -> int:
    return KNOWN_LENGTHS[example_index % len(KNOWN_LENGTHS)]


def adaptation_records(
    *,
    role: DataRole,
    block: str,
    exemplar_count: int,
    sets_per_condition: int,
) -> Iterator[CNPRecord]:
    """Yield one fixed CNP-004 panel; callers never choose examples from labels."""

    if block not in ADAPTATION_BLOCKS:
        raise ValueError(f"unsupported CNP adaptation block: {block}")
    if role not in {"adapt_train", "shadow", "transfer_eval"}:
        raise ValueError(f"unsupported CNP adaptation role: {role}")
    world = make_world()
    for exemplar_index in range(exemplar_count):
        for threshold in SOURCE_THRESHOLDS:
            for example_index in range(sets_per_condition):
                yield make_adaptation_record(
                    role=role,
                    block=block,
                    exemplar_index=exemplar_index,
                    length=_length_for_example(example_index),
                    threshold=threshold,
                    example_index=example_index,
                    world=world,
                )


def paired_adaptation_training_records(block: str) -> tuple[list[CNPRecord], list[CNPRecord]]:
    """Return fixed 64- and prefix-sharing 16-teacher arms for one new condition block."""

    full = list(
        adaptation_records(
            role="adapt_train",
            block=block,
            exemplar_count=ADAPT_TRAIN_EXEMPLARS,
            sets_per_condition=SETS_PER_CONDITION,
        )
    )
    small = list(
        adaptation_records(
            role="adapt_train",
            block=block,
            exemplar_count=ADAPT_TRAIN_EXEMPLARS,
            sets_per_condition=SMALL_ARM_SETS_PER_CONDITION,
        )
    )
    full_by_digest = {record.digest(): record for record in full}
    if not all(record.digest() in full_by_digest for record in small):
        raise RuntimeError("CNP-004 small arm must be a prefix subset of the full arm")
    return full, small


def batched(
    records: list[CNPRecord], batch_size: int = EVAL_BATCH_SETS
) -> Iterator[list[CNPRecord]]:
    """Yield fixed-order batches without shuffling or replacement."""

    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]
