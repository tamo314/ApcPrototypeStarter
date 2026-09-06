# ruff: noqa: E501
"""Focused, CPU-only, GPU-free coverage for B-C005R3-008's pure logic.

Matches this repo's existing precedent (D2/R3-002/R3-004/R3-006/R3-007):
anything that needs a real Core/bank/router/checkpoint
(`run_bind_argument_scorer_repair`, `train_bind_head_repair` itself, since it
calls `extract_task_representations(core, ...)`) is exercised only via the
milestone script (`scripts/run_phase_b_b2_post_d2_repair.py --task
B-C005R3-008`), not in pytest.

Unlike that, the schema/value-coverage audits use only the real (lightweight,
CPU-only) example generator (mirrors R3-007's `audit_select_argument_schema`
precedent), the stratified-sampling function is pure list/dict logic, and the
freeze-audit/checkpoint-hash/aggregation helpers need only plain
`ArgumentScorer`/`nn.Linear` objects and plain dicts -- no Core, bank, or
GPU -- so they all get real (not mocked) coverage here.
"""

from __future__ import annotations

import pytest
import torch

from apc.evaluation.bind_argument_scorer_repair import (
    VARIANTS,
    BindArgumentScorerRepairConfig,
    _build_gate,
    _build_other_op_regression,
    _build_variant_selection,
    _head,
    _mean,
    audit_bind_argument_schema,
    audit_training_value_coverage,
    build_checkpoint_hashes,
    build_freeze_audit,
    stratify_bind_examples_by_query_key,
)
from apc.evaluation.recurrence_benchmark import generate_benchmark_examples
from apc.primitives.argument_scoring import ArgumentScorer, ArgumentScorerConfig

# ---------------------------------------------------------------------------
# BindArgumentScorerRepairConfig validation.
# ---------------------------------------------------------------------------


def test_config_defaults_are_valid() -> None:
    config = BindArgumentScorerRepairConfig()
    assert config.bank_size == 128
    assert config.development_seeds == (10, 11, 12, 13, 14)
    assert config.variants == VARIANTS
    assert config.repair_training_budget <= config.repair_pool_examples


def test_config_rejects_empty_development_seeds() -> None:
    with pytest.raises(ValueError, match="development_seeds"):
        BindArgumentScorerRepairConfig(development_seeds=())


def test_config_rejects_bad_bank_size() -> None:
    with pytest.raises(ValueError, match="bank_size"):
        BindArgumentScorerRepairConfig(bank_size=100)


def test_config_rejects_nonpositive_example_counts() -> None:
    with pytest.raises(ValueError, match="support_examples"):
        BindArgumentScorerRepairConfig(support_examples=0)


def test_config_rejects_nonpositive_steps() -> None:
    with pytest.raises(ValueError, match="router_steps and repair_steps"):
        BindArgumentScorerRepairConfig(repair_steps=0)


def test_config_rejects_bad_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        BindArgumentScorerRepairConfig(top_k=6)


def test_config_rejects_bad_threshold() -> None:
    with pytest.raises(ValueError, match="adequacy_exact_match_threshold"):
        BindArgumentScorerRepairConfig(adequacy_exact_match_threshold=1.5)


def test_config_rejects_training_budget_exceeding_pool() -> None:
    with pytest.raises(ValueError, match="repair_training_budget must not exceed"):
        BindArgumentScorerRepairConfig(repair_pool_examples=32, repair_training_budget=64)


def test_config_rejects_bad_gate_thresholds() -> None:
    with pytest.raises(ValueError, match="gate_argument_accuracy_threshold"):
        BindArgumentScorerRepairConfig(gate_argument_accuracy_threshold=0.0)
    with pytest.raises(ValueError, match="gate_family_top1_threshold"):
        BindArgumentScorerRepairConfig(gate_family_top1_threshold=1.5)


def test_config_rejects_negative_regression_budget() -> None:
    with pytest.raises(ValueError, match="gate_other_op_regression_pp_max"):
        BindArgumentScorerRepairConfig(gate_other_op_regression_pp_max=-1.0)


def test_config_rejects_unknown_variant() -> None:
    with pytest.raises(ValueError, match="variants"):
        BindArgumentScorerRepairConfig(variants=("not_a_real_variant",))


# ---------------------------------------------------------------------------
# Schema / value-coverage audits (real generator code; no Core/bank/GPU needed).
# ---------------------------------------------------------------------------


def test_audit_bind_argument_schema_matches_known_generator_invariants() -> None:
    result = audit_bind_argument_schema(seed=10)
    assert result["generator_always_scalar"] is True
    assert result["generator_query_key_always_present_at_a_key_position"] is True
    assert result["round_trip_exact"] is True
    assert result["n_examples"] > 0


def test_audit_bind_argument_schema_is_deterministic_across_repeated_calls() -> None:
    first = audit_bind_argument_schema(seed=11)
    second = audit_bind_argument_schema(seed=11)
    assert first == second


def test_audit_training_value_coverage_counts_real_examples() -> None:
    examples = list(
        generate_benchmark_examples(10 * 30_000 + 128, 32, operation="BIND", split="dev")
    )
    result = audit_training_value_coverage(examples, vocab_size=10)
    assert result["n_training_examples"] == 32
    assert sum(result["per_value_count"].values()) == 32
    assert result["n_values_never_seen"] + sum(1 for c in result["per_value_count"].values() if c > 0) == 10


def test_audit_training_value_coverage_handles_empty_input() -> None:
    result = audit_training_value_coverage([], vocab_size=10)
    assert result["n_training_examples"] == 0
    assert result["n_values_never_seen"] == 10
    assert result["min_count_over_seen_values"] == 0
    assert result["max_count_over_seen_values"] == 0


# ---------------------------------------------------------------------------
# Stratified sampling (pure list/dict logic over real Example objects).
# ---------------------------------------------------------------------------


def test_stratify_covers_every_observed_value_up_to_cap() -> None:
    pool = list(
        generate_benchmark_examples(12 * 750_000, 256, operation="BIND", split="dev")
    )
    selected = stratify_bind_examples_by_query_key(pool, target_total=100)
    assert len(selected) <= max(100, len({0}))  # sanity: never absurdly larger than pool
    assert len(selected) <= len(pool)
    from apc.primitives.argument_scoring import extract_raw_argument_values

    observed_in_pool = {
        extract_raw_argument_values("BIND", ex.task_spec.steps[0].arguments)[0] for ex in pool
    }
    observed_in_selection = {
        extract_raw_argument_values("BIND", ex.task_spec.steps[0].arguments)[0] for ex in selected
    }
    assert observed_in_selection == observed_in_pool


def test_stratify_every_selected_example_drawn_from_pool() -> None:
    pool = list(
        generate_benchmark_examples(13 * 750_000, 128, operation="BIND", split="dev")
    )
    selected = stratify_bind_examples_by_query_key(pool, target_total=64)
    pool_ids = {id(ex) for ex in pool}
    assert all(id(ex) in pool_ids for ex in selected)


def test_stratify_is_deterministic() -> None:
    pool = list(
        generate_benchmark_examples(14 * 750_000, 128, operation="BIND", split="dev")
    )
    first = stratify_bind_examples_by_query_key(pool, target_total=64)
    second = stratify_bind_examples_by_query_key(pool, target_total=64)
    assert [ex.target_tokens for ex in first] == [ex.target_tokens for ex in second]


def test_stratify_empty_pool_returns_empty() -> None:
    assert stratify_bind_examples_by_query_key([], target_total=64) == []


# ---------------------------------------------------------------------------
# Freeze audit / checkpoint hashes (real ArgumentScorer, no Core needed).
# ---------------------------------------------------------------------------


def _build_scorer(seed: int) -> ArgumentScorer:
    torch.manual_seed(seed)
    return ArgumentScorer(ArgumentScorerConfig(d_model=8, arg_vocab_size=10))


def test_head_helper_returns_linear_for_every_operation() -> None:
    scorer = _build_scorer(0)
    for op in ("SHIFT", "SELECT", "COUNT", "BIND"):
        head = _head(scorer, op)
        assert isinstance(head, torch.nn.Linear)


def test_freeze_audit_detects_only_bind_head_changed() -> None:
    base = _build_scorer(1)
    import copy

    repaired = copy.deepcopy(base)
    with torch.no_grad():
        _head(repaired, "BIND").weight.add_(1.0)

    audit = build_freeze_audit(base, repaired)
    assert audit["bind_head_changed_by_training"] is True
    assert audit["all_other_heads_unchanged"] is True
    assert audit["freeze_audit_passed"] is True


def test_freeze_audit_fails_if_another_head_also_changed() -> None:
    base = _build_scorer(2)
    import copy

    repaired = copy.deepcopy(base)
    with torch.no_grad():
        _head(repaired, "BIND").weight.add_(1.0)
        _head(repaired, "SHIFT").weight.add_(1.0)

    audit = build_freeze_audit(base, repaired)
    assert audit["other_head_unchanged_by_operation"]["SHIFT"] is False
    assert audit["freeze_audit_passed"] is False


def test_freeze_audit_fails_if_bind_head_never_changed() -> None:
    base = _build_scorer(3)
    import copy

    repaired = copy.deepcopy(base)  # no mutation at all

    audit = build_freeze_audit(base, repaired)
    assert audit["bind_head_changed_by_training"] is False
    assert audit["freeze_audit_passed"] is False


def test_checkpoint_hashes_differ_only_for_bind() -> None:
    base = _build_scorer(4)
    import copy

    repaired = copy.deepcopy(base)
    with torch.no_grad():
        _head(repaired, "BIND").weight.add_(1.0)

    hashes = build_checkpoint_hashes(base, repaired)
    assert hashes["bind_head_hash"]["before"] != hashes["bind_head_hash"]["after"]
    for op in ("SHIFT", "SELECT", "COUNT"):
        assert hashes["other_head_hash"][op]["before"] == hashes["other_head_hash"][op]["after"]
    assert hashes["base_scorer_full_state_hash"] != hashes["repaired_scorer_full_state_hash"]


# ---------------------------------------------------------------------------
# Aggregation helpers (pure dict/list logic).
# ---------------------------------------------------------------------------


def test_mean_of_empty_is_none() -> None:
    assert _mean([]) is None


def test_mean_basic() -> None:
    assert _mean([1.0, 2.0, 3.0]) == 2.0


def test_build_variant_selection_picks_higher_score() -> None:
    result = _build_variant_selection(
        {"larger_iid_sample": [0.9, 0.9], "stratified_value_coverage": [0.95, 0.97]}
    )
    assert result["chosen_variant"] == "stratified_value_coverage"
    assert result["tie_break_applied"] is False


def test_build_variant_selection_tie_break_prefers_stratified() -> None:
    result = _build_variant_selection(
        {"larger_iid_sample": [0.95, 0.95], "stratified_value_coverage": [0.95, 0.95]}
    )
    assert result["chosen_variant"] == "stratified_value_coverage"
    assert result["tie_break_applied"] is True


def test_build_other_op_regression_zero_when_identical() -> None:
    baseline = {"SHIFT": [{"argument_accuracy": 1.0}], "SELECT": [{"argument_accuracy": 0.9}], "COUNT": [{"argument_accuracy": 0.8}]}
    repaired = {"SHIFT": [{"argument_accuracy": 1.0}], "SELECT": [{"argument_accuracy": 0.9}], "COUNT": [{"argument_accuracy": 0.8}]}
    result = _build_other_op_regression(baseline, repaired)
    assert result == {"SHIFT": 0.0, "SELECT": 0.0, "COUNT": 0.0}


def test_build_other_op_regression_detects_drop() -> None:
    baseline = {"SHIFT": [{"argument_accuracy": 1.0}], "SELECT": [{"argument_accuracy": 1.0}], "COUNT": [{"argument_accuracy": 1.0}]}
    repaired = {"SHIFT": [{"argument_accuracy": 0.98}], "SELECT": [{"argument_accuracy": 1.0}], "COUNT": [{"argument_accuracy": 1.0}]}
    result = _build_other_op_regression(baseline, repaired)
    assert result["SHIFT"] == pytest.approx(2.0)
    assert result["SELECT"] == 0.0


def _fake_cell(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "argument_accuracy": 0.97,
        "primitive_call_top1": 0.97,
        "physical_primitive_top1": 1.0,
        "primitive_call_topk": 1.0,
        "unselected_forward_calls": 0,
        "leak_audit_passed": True,
    }
    base.update(overrides)
    return base


def test_build_gate_passes_when_all_thresholds_met() -> None:
    cells = [_fake_cell() for _ in range(5)]
    config = BindArgumentScorerRepairConfig()
    gate = _build_gate(cells, {"SHIFT": 0.0, "SELECT": 0.0, "COUNT": 0.0}, 0.0, config)
    assert gate["result"] == "VALIDATION_PASS"
    assert gate["argument_accuracy"]["pass"] is True


def test_build_gate_fails_on_low_argument_accuracy() -> None:
    cells = [_fake_cell(argument_accuracy=0.5) for _ in range(5)]
    config = BindArgumentScorerRepairConfig()
    gate = _build_gate(cells, {"SHIFT": 0.0, "SELECT": 0.0, "COUNT": 0.0}, 0.0, config)
    assert gate["result"] == "FAIL"
    assert gate["argument_accuracy"]["pass"] is False


def test_build_gate_fails_on_regression_budget_exceeded() -> None:
    cells = [_fake_cell() for _ in range(5)]
    config = BindArgumentScorerRepairConfig()
    gate = _build_gate(cells, {"SHIFT": 2.0, "SELECT": 0.0, "COUNT": 0.0}, 2.0, config)
    assert gate["result"] == "FAIL"
    assert gate["other_operation_regression_pp"]["pass"] is False


def test_build_gate_fails_on_unselected_forward_calls() -> None:
    cells = [_fake_cell(unselected_forward_calls=1) for _ in range(5)]
    config = BindArgumentScorerRepairConfig()
    gate = _build_gate(cells, {"SHIFT": 0.0, "SELECT": 0.0, "COUNT": 0.0}, 0.0, config)
    assert gate["result"] == "FAIL"
    assert gate["unselected_forward_calls_total"] == 5


def test_build_gate_fails_on_leak_audit_failure() -> None:
    cells = [_fake_cell(leak_audit_passed=False) for _ in range(5)]
    config = BindArgumentScorerRepairConfig()
    gate = _build_gate(cells, {"SHIFT": 0.0, "SELECT": 0.0, "COUNT": 0.0}, 0.0, config)
    assert gate["result"] == "FAIL"
    assert gate["leak_audit_all_passed"] is False
