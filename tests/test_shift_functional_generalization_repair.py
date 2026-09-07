# ruff: noqa: E501
"""Focused, CPU-only, GPU-free coverage for B-C005R3-009's pure logic.

Matches this repo's existing precedent (R3-006/R3-007/R3-008): anything that
needs a real Core/bank/router/checkpoint (`run_shift_functional_generalization_repair`,
`ensure_pretrained_core`, `_train_shift_candidate`, `diagnose_stratum_errors`,
`closed_loop_shift_em`, `load_bank_from_checkpoint`) is exercised only via the
milestone script (`scripts/run_phase_b_b2_post_d2_repair.py --task
B-C005R3-009`) and the ad hoc CPU smoke run used during development, not in
pytest. What pytest covers here needs no Core, no GPU, no pretrained
checkpoint: config validation, the pure `(length, amount)` grid/schedule
logic, the params-controlled example builder (real but lightweight generator
code), and the freeze-audit/variant-selection helpers against small
hand-built `PrimitiveBank`/`ShiftRelativePrimitive` objects.
"""

from __future__ import annotations

import pytest
import torch

from apc.evaluation.shift_functional_generalization_repair import (
    VARIANTS,
    ShiftFunctionalGeneralizationRepairConfig,
    _build_variant_selection,
    _build_weighted_schedule,
    _generate_shift_example_with_params,
    _mean,
    _shift_params,
    _state_dict_hash,
    build_freeze_audit,
    shift_legal_grid,
)
from apc.primitives.bank import PrimitiveBank
from apc.primitives.primitive import (
    CrossPositionPrimitive,
    CrossPositionPrimitiveConfig,
    PrimitiveStatus,
    ShiftRelativePrimitive,
    ShiftRelativePrimitiveConfig,
)

# ---------------------------------------------------------------------------
# ShiftFunctionalGeneralizationRepairConfig validation.
# ---------------------------------------------------------------------------


def test_config_defaults_are_valid() -> None:
    config = ShiftFunctionalGeneralizationRepairConfig()
    assert config.development_seeds == (10, 11, 12, 13, 14)
    assert config.bank_size == 16
    assert config.variants == VARIANTS
    assert config.operator_train_steps == 12000


def test_config_rejects_empty_development_seeds() -> None:
    with pytest.raises(ValueError, match="development_seeds"):
        ShiftFunctionalGeneralizationRepairConfig(development_seeds=())


def test_config_rejects_bad_bank_size() -> None:
    with pytest.raises(ValueError, match="bank_size"):
        ShiftFunctionalGeneralizationRepairConfig(bank_size=100)


def test_config_rejects_bad_sequence_length_range() -> None:
    with pytest.raises(ValueError, match="sequence_length_range"):
        ShiftFunctionalGeneralizationRepairConfig(sequence_length_range=(10, 6))


def test_config_rejects_nonpositive_steps() -> None:
    with pytest.raises(ValueError, match="operator_train_steps"):
        ShiftFunctionalGeneralizationRepairConfig(operator_train_steps=0)


def test_config_rejects_nonpositive_batch_size() -> None:
    with pytest.raises(ValueError, match="operator_batch_size"):
        ShiftFunctionalGeneralizationRepairConfig(operator_batch_size=0)


def test_config_rejects_nonpositive_example_counts() -> None:
    with pytest.raises(ValueError, match="gate_query_examples"):
        ShiftFunctionalGeneralizationRepairConfig(gate_query_examples=0)


def test_config_rejects_bad_gate_thresholds() -> None:
    with pytest.raises(ValueError, match="gate_mean_query_em_threshold"):
        ShiftFunctionalGeneralizationRepairConfig(gate_mean_query_em_threshold=0.0)
    with pytest.raises(ValueError, match="gate_ref_adequacy_tau"):
        ShiftFunctionalGeneralizationRepairConfig(gate_ref_adequacy_tau=1.5)


def test_config_rejects_negative_regression_budget() -> None:
    with pytest.raises(ValueError, match="gate_other_task_regression_pp_max"):
        ShiftFunctionalGeneralizationRepairConfig(gate_other_task_regression_pp_max=-1.0)


def test_config_rejects_unknown_variant() -> None:
    with pytest.raises(ValueError, match="variants"):
        ShiftFunctionalGeneralizationRepairConfig(variants=("not_a_real_variant",))


# ---------------------------------------------------------------------------
# shift_legal_grid / params-controlled example builder / _shift_params.
# ---------------------------------------------------------------------------


def test_shift_legal_grid_enumerates_every_amount_per_length() -> None:
    grid = shift_legal_grid((2, 3))
    assert grid == ((2, 0), (2, 1), (3, 0), (3, 1), (3, 2))


def test_shift_legal_grid_matches_shift_op_domain_size() -> None:
    # amount in [0, length) per ShiftOp.sample_params -> `length` values per length.
    grid = shift_legal_grid((6, 10))
    assert len(grid) == sum(range(6, 11))


def test_generate_shift_example_with_params_produces_pinned_length_and_amount() -> None:
    ex = _generate_shift_example_with_params(12345, length=7, amount=3, vocab_size=10, split="dev")
    assert len(ex.input_tokens) == 7
    assert _shift_params(ex) == (7, 3)


def test_generate_shift_example_with_params_applies_correct_cyclic_shift() -> None:
    ex = _generate_shift_example_with_params(999, length=6, amount=2, vocab_size=10, split="dev")
    expected = ex.input_tokens[2:] + ex.input_tokens[:2]
    assert ex.target_tokens == expected


def test_generate_shift_example_with_params_is_deterministic() -> None:
    a = _generate_shift_example_with_params(42, length=8, amount=5, vocab_size=10, split="train")
    b = _generate_shift_example_with_params(42, length=8, amount=5, vocab_size=10, split="train")
    assert a.input_tokens == b.input_tokens
    assert a.target_tokens == b.target_tokens


def test_generate_shift_example_with_params_amount_zero_is_identity() -> None:
    ex = _generate_shift_example_with_params(7, length=6, amount=0, vocab_size=10, split="dev")
    assert ex.target_tokens == ex.input_tokens


# ---------------------------------------------------------------------------
# _build_weighted_schedule: deterministic, weak-stratum-weighted round robin.
# ---------------------------------------------------------------------------


def _diag_rows(pairs: list[tuple[int, int, float]]) -> list[dict]:
    return [
        {"length": length, "amount": amount, "wraps_around": amount > 0, "n": 10, "exact_match": em}
        for length, amount, em in pairs
    ]


def test_build_weighted_schedule_has_requested_length() -> None:
    rows = _diag_rows([(6, 0, 1.0), (6, 1, 0.5), (7, 0, 0.9)])
    schedule = _build_weighted_schedule(rows, schedule_len=100, seed=0)
    assert len(schedule) == 100
    assert all(item in {(6, 0), (6, 1), (7, 0)} for item in schedule)


def test_build_weighted_schedule_oversamples_the_weaker_stratum() -> None:
    rows = _diag_rows([(6, 0, 1.0), (6, 1, 0.0)])
    schedule = _build_weighted_schedule(rows, schedule_len=1000, seed=0)
    weak_count = schedule.count((6, 1))
    strong_count = schedule.count((6, 0))
    assert weak_count > strong_count


def test_build_weighted_schedule_is_deterministic() -> None:
    rows = _diag_rows([(6, 0, 1.0), (6, 1, 0.5), (7, 2, 0.2)])
    a = _build_weighted_schedule(rows, schedule_len=50, seed=3)
    b = _build_weighted_schedule(rows, schedule_len=50, seed=3)
    assert a == b


def test_build_weighted_schedule_rejects_all_empty_strata() -> None:
    rows = [{"length": 6, "amount": 0, "wraps_around": False, "n": 0, "exact_match": 1.0}]
    with pytest.raises(ValueError, match="at least one stratum"):
        _build_weighted_schedule(rows, schedule_len=10, seed=0)


def test_build_weighted_schedule_skips_zero_n_strata() -> None:
    rows = [
        {"length": 6, "amount": 0, "wraps_around": False, "n": 0, "exact_match": 0.0},
        {"length": 6, "amount": 1, "wraps_around": True, "n": 5, "exact_match": 1.0},
    ]
    schedule = _build_weighted_schedule(rows, schedule_len=20, seed=0)
    assert all(item == (6, 1) for item in schedule)


# ---------------------------------------------------------------------------
# build_freeze_audit / _state_dict_hash against small hand-built banks --
# no Core, no GPU, no pretrained checkpoint needed.
# ---------------------------------------------------------------------------


def _small_bank() -> tuple[PrimitiveBank, int, int]:
    bank = PrimitiveBank()
    select = bank.new_cross_position_primitive(
        CrossPositionPrimitiveConfig(operation="SELECT", d_model=16, d_operator=8, n_head=2, d_operator_ff=16, vocab_size=6, max_sequence_length=8, arg_dim=4),
        status=PrimitiveStatus.STABLE,
    )
    shift = bank.new_shift_relative_primitive(
        ShiftRelativePrimitiveConfig(d_model=16, d_operator=8, n_head=2, d_operator_ff=16, vocab_size=6, max_sequence_length=8, arg_dim=4),
        status=PrimitiveStatus.STABLE,
    )
    return bank, select.primitive_id, shift.primitive_id


def test_build_freeze_audit_passes_when_only_shift_changes() -> None:
    bank_before, select_id, shift_id = _small_bank()
    bank_after = PrimitiveBank()
    bank_after.add_primitive(bank_before.get(select_id))
    new_shift = ShiftRelativePrimitive(
        shift_id,
        ShiftRelativePrimitiveConfig(d_model=16, d_operator=8, n_head=2, d_operator_ff=16, vocab_size=6, max_sequence_length=8, arg_dim=4),
        status=PrimitiveStatus.STABLE,
    )
    torch.nn.init.ones_(new_shift.readout.weight)
    bank_after.add_primitive(new_shift)

    audit = build_freeze_audit(bank_before, bank_after, shift_id)
    assert audit["all_other_primitives_unchanged"] is True
    assert audit["shift_primitive_changed_by_training"] is True
    assert audit["freeze_audit_passed"] is True


def test_build_freeze_audit_fails_when_other_primitive_also_changes() -> None:
    bank_before, select_id, shift_id = _small_bank()
    bank_after = PrimitiveBank()
    mutated_select = CrossPositionPrimitive(
        select_id,
        CrossPositionPrimitiveConfig(operation="SELECT", d_model=16, d_operator=8, n_head=2, d_operator_ff=16, vocab_size=6, max_sequence_length=8, arg_dim=4),
        status=PrimitiveStatus.STABLE,
    )
    torch.nn.init.ones_(mutated_select.readout.weight)
    bank_after.add_primitive(mutated_select)
    bank_after.add_primitive(bank_before.get(shift_id))

    audit = build_freeze_audit(bank_before, bank_after, shift_id)
    assert audit["all_other_primitives_unchanged"] is False
    assert audit["freeze_audit_passed"] is False


def test_build_freeze_audit_fails_when_shift_did_not_change() -> None:
    bank_before, _select_id, shift_id = _small_bank()
    bank_after = PrimitiveBank()
    for pid in bank_before.ids():
        bank_after.add_primitive(bank_before.get(pid))

    audit = build_freeze_audit(bank_before, bank_after, shift_id)
    assert audit["shift_primitive_changed_by_training"] is False
    assert audit["freeze_audit_passed"] is False


def test_state_dict_hash_is_stable_and_sensitive_to_weights() -> None:
    _bank, _select_id, shift_id = _small_bank()
    bank2, _select_id2, shift_id2 = _small_bank()
    a = _state_dict_hash(_bank.get(shift_id))
    b = _state_dict_hash(bank2.get(shift_id2))
    assert a != b  # independently initialized -> different weights
    assert a == _state_dict_hash(_bank.get(shift_id))  # stable/repeatable


# ---------------------------------------------------------------------------
# _build_variant_selection / _mean.
# ---------------------------------------------------------------------------


def test_mean_of_empty_is_none() -> None:
    assert _mean([]) is None
    assert _mean([1.0, 3.0]) == 2.0


def test_build_variant_selection_picks_strict_winner() -> None:
    selection = _build_variant_selection({"iid_baseline": [0.5, 0.5], "error_weighted_stratified": [0.9, 0.9]})
    assert selection["chosen_variant"] == "error_weighted_stratified"
    assert selection["tie_break_applied"] is False


def test_build_variant_selection_tie_break_prefers_stratified() -> None:
    selection = _build_variant_selection({"iid_baseline": [0.9, 0.9], "error_weighted_stratified": [0.9, 0.9]})
    assert selection["chosen_variant"] == "error_weighted_stratified"
    assert selection["tie_break_applied"] is True
