"""Unit tests for every Phase A primitive operation (Milestone A1)."""

from __future__ import annotations

import random

import pytest

from apc.environments.operations import (
    KNOWN_OPERATION_NAMES,
    NOVEL_OPERATION_NAMES,
    AccumulateOp,
    BindOp,
    CompareOp,
    CopyOp,
    CountOp,
    NegateOp,
    Operation,
    ReverseOp,
    SelectOp,
    ShiftOp,
    SortOp,
    get_operation,
    register_operation,
    registered_operation_names,
)
from apc.environments.vocab import relation_tokens

VOCAB_SIZE = 10


def test_known_operation_names_match_phase_a_curriculum() -> None:
    assert set(KNOWN_OPERATION_NAMES) == {
        "COPY",
        "SELECT",
        "COMPARE",
        "COUNT",
        "SHIFT",
        "BIND",
        "NEGATE",
        "ACCUMULATE",
    }


def test_registry_lookup_matches_registered_names() -> None:
    for name in KNOWN_OPERATION_NAMES:
        assert get_operation(name).name == name
    assert set(registered_operation_names()) >= set(KNOWN_OPERATION_NAMES)


def test_get_unknown_operation_raises() -> None:
    with pytest.raises(KeyError):
        get_operation("MODULAR_ADD")


def test_register_duplicate_without_overwrite_raises() -> None:
    with pytest.raises(ValueError):
        register_operation(CopyOp())


def test_register_with_overwrite_replaces() -> None:
    register_operation(CopyOp(), overwrite=True)
    assert get_operation("COPY").name == "COPY"


# --- COPY ---------------------------------------------------------------


def test_copy_is_identity() -> None:
    op = CopyOp()
    seq = (1, 2, 3, 4)
    assert op.apply(seq, VOCAB_SIZE, {}) == seq
    assert op.output_length(len(seq)) == len(seq)


# --- NEGATE ---------------------------------------------------------------


def test_negate_complements_within_vocab() -> None:
    op = NegateOp()
    seq = (0, 1, 9, 5)
    assert op.apply(seq, VOCAB_SIZE, {}) == (9, 8, 0, 4)


def test_negate_is_involution() -> None:
    op = NegateOp()
    seq = (2, 7, 3)
    once = op.apply(seq, VOCAB_SIZE, {})
    twice = op.apply(once, VOCAB_SIZE, {})
    assert twice == seq


# --- SHIFT ---------------------------------------------------------------


def test_shift_cyclically_rotates_left() -> None:
    op = ShiftOp()
    seq = (0, 1, 2, 3, 4)
    assert op.apply(seq, VOCAB_SIZE, {"amount": 2}) == (2, 3, 4, 0, 1)


def test_shift_amount_zero_is_identity() -> None:
    op = ShiftOp()
    seq = (5, 6, 7)
    assert op.apply(seq, VOCAB_SIZE, {"amount": 0}) == seq


def test_shift_sample_params_in_range() -> None:
    op = ShiftOp()
    rng = random.Random(0)
    seq = (1, 2, 3, 4, 5)
    for _ in range(50):
        params = op.sample_params(rng, seq, VOCAB_SIZE)
        assert 0 <= params["amount"] < len(seq)


# --- SELECT ---------------------------------------------------------------


def test_select_output_length_is_half_rounded_down_min_one() -> None:
    op = SelectOp()
    assert op.output_length(1) == 1
    assert op.output_length(2) == 1
    assert op.output_length(6) == 3
    assert op.output_length(7) == 3


def test_select_preserves_order_of_chosen_indices() -> None:
    op = SelectOp()
    seq = (10, 11, 12, 13)
    result = op.apply(seq, VOCAB_SIZE, {"indices": [0, 2]})
    assert result == (10, 12)


def test_select_sample_params_indices_are_sorted_unique_and_in_range() -> None:
    op = SelectOp()
    rng = random.Random(1)
    seq = tuple(range(8))
    for _ in range(50):
        params = op.sample_params(rng, seq, VOCAB_SIZE)
        indices = params["indices"]
        assert len(indices) == op.output_length(len(seq))
        assert indices == sorted(set(indices))
        assert all(0 <= i < len(seq) for i in indices)


# --- COMPARE ---------------------------------------------------------------


def test_compare_emits_relation_tokens() -> None:
    op = CompareOp()
    lt, eq, gt = relation_tokens(VOCAB_SIZE)
    seq = (1, 5, 5, 2)
    assert op.apply(seq, VOCAB_SIZE, {}) == (lt, eq, gt)


def test_compare_output_length_is_input_minus_one() -> None:
    op = CompareOp()
    assert op.output_length(5) == 4


def test_compare_invalid_for_length_one() -> None:
    op = CompareOp()
    assert not op.is_valid_for_length(1)
    assert op.is_valid_for_length(2)


# --- COUNT ---------------------------------------------------------------


def test_count_counts_target_occurrences() -> None:
    op = CountOp()
    seq = (3, 1, 3, 3, 2)
    assert op.apply(seq, VOCAB_SIZE, {"target": 3}) == (3,)
    assert op.apply(seq, VOCAB_SIZE, {"target": 9}) == (0,)


def test_count_clips_to_vocab_range() -> None:
    op = CountOp()
    seq = (4,) * 20
    result = op.apply(seq, VOCAB_SIZE, {"target": 4})
    assert result == (VOCAB_SIZE - 1,)


def test_count_output_length_always_one() -> None:
    op = CountOp()
    assert op.output_length(1) == 1
    assert op.output_length(100) == 1


# --- BIND ---------------------------------------------------------------


def test_bind_looks_up_value_for_key() -> None:
    op = BindOp()
    seq = (1, 8, 2, 9, 3, 7)  # keys 1,2,3 -> values 8,9,7
    assert op.apply(seq, VOCAB_SIZE, {"query_key": 2}) == (9,)


def test_bind_duplicate_key_uses_last_value() -> None:
    op = BindOp()
    seq = (1, 8, 1, 9)  # key 1 appears twice
    assert op.apply(seq, VOCAB_SIZE, {"query_key": 1}) == (9,)


def test_bind_requires_even_length() -> None:
    op = BindOp()
    assert op.is_valid_for_length(4)
    assert not op.is_valid_for_length(3)
    assert not op.is_valid_for_length(1)


def test_bind_sample_params_query_key_always_present() -> None:
    op = BindOp()
    rng = random.Random(2)
    seq = (5, 1, 6, 2, 7, 3)
    keys = seq[0::2]
    for _ in range(50):
        params = op.sample_params(rng, seq, VOCAB_SIZE)
        assert params["query_key"] in keys


# --- ACCUMULATE ---------------------------------------------------------------


def test_accumulate_running_sum_modulo_vocab() -> None:
    op = AccumulateOp()
    seq = (3, 4, 9, 1)
    # running sums: 3, 7, 16 % 10 = 6, 17 % 10 = 7
    assert op.apply(seq, VOCAB_SIZE, {}) == (3, 7, 6, 7)


def test_accumulate_preserves_length() -> None:
    op = AccumulateOp()
    assert op.output_length(7) == 7


# --- SORT (Task 009 novel operation) ----------------------------------------


def test_sort_sorts_ascending() -> None:
    op = SortOp()
    seq = (4, 1, 3, 1, 9, 0)
    assert op.apply(seq, VOCAB_SIZE, {}) == (0, 1, 1, 3, 4, 9)


def test_sort_preserves_length() -> None:
    op = SortOp()
    assert op.output_length(7) == 7


def test_sort_sample_params_is_empty() -> None:
    op = SortOp()
    rng = random.Random(0)
    assert op.sample_params(rng, (3, 1, 2), VOCAB_SIZE) == {}


def test_novel_operation_names_is_sort_and_reverse_disjoint_from_known() -> None:
    assert NOVEL_OPERATION_NAMES == ("SORT", "REVERSE")
    assert set(NOVEL_OPERATION_NAMES).isdisjoint(KNOWN_OPERATION_NAMES)


def test_sort_is_registered_but_not_a_known_operation() -> None:
    assert get_operation("SORT").name == "SORT"
    assert "SORT" in registered_operation_names()
    assert "SORT" not in KNOWN_OPERATION_NAMES


# --- REVERSE (Task 012 second novel operation) ------------------------------


def test_reverse_reverses_sequence() -> None:
    op = ReverseOp()
    seq = (4, 1, 3, 1, 9, 0)
    assert op.apply(seq, VOCAB_SIZE, {}) == (0, 9, 1, 3, 1, 4)


def test_reverse_preserves_length() -> None:
    op = ReverseOp()
    assert op.output_length(7) == 7


def test_reverse_sample_params_is_empty() -> None:
    op = ReverseOp()
    rng = random.Random(0)
    assert op.sample_params(rng, (3, 1, 2), VOCAB_SIZE) == {}


def test_reverse_is_registered_but_not_a_known_operation() -> None:
    assert get_operation("REVERSE").name == "REVERSE"
    assert "REVERSE" in registered_operation_names()
    assert "REVERSE" not in KNOWN_OPERATION_NAMES


# --- shared Operation contract ---------------------------------------------


@pytest.mark.parametrize("name", KNOWN_OPERATION_NAMES)
def test_every_known_operation_is_registered_and_typed(name: str) -> None:
    op = get_operation(name)
    assert isinstance(op, Operation)
    assert op.name == name
    assert op.min_input_length >= 1


@pytest.mark.parametrize("name", NOVEL_OPERATION_NAMES)
def test_every_novel_operation_is_registered_and_typed(name: str) -> None:
    op = get_operation(name)
    assert isinstance(op, Operation)
    assert op.name == name
    assert op.min_input_length >= 1
