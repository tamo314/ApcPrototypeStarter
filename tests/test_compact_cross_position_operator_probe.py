"""Tests for the compact cross-position operator probe (Phase A.1 diagnostic
Task A1-R005E-005).

Fast, small-step tests only -- mirroring `tests/test_frozen_high_capacity_
operator_benchmark.py`'s convention of exercising the wiring (pure helpers,
group construction, config validation, operator forward shape, per-operation
benchmark, multi-seed aggregation) with a tiny model and a handful of
training steps. The benchmark's own scientific claim (5 seeds, the full
30000-step core / 8000-step operator budget) is run via `scripts/
compact_cross_position_operator_probe.py`, not asserted here.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from apc.evaluation.compact_cross_position_operator_probe import (
    CORRECT_EXACT_MATCH_THRESHOLD,
    DEFAULT_GROUP_SIZE,
    DEFAULT_SEEDS,
    EFFECTFUL_WRONG_ARGUMENT_CEILING,
    MIN_CAUSAL_GAP,
    MIN_GATE_SEEDS,
    MIN_GROUP_SIZE,
    NONE_CEILING,
    TOKEN_ACCURACY_GATED_OPERATIONS,
    TOKEN_ACCURACY_THRESHOLD,
    CompactCrossPositionOperator,
    CompactOperatorConfig,
    CompactOperatorGroup,
    PrimitiveTrainConfig,
    _argument_candidates,
    _correct_argument_value,
    _flatten_groups,
    _operation_output,
    _valid_content_lengths,
    compact_operator_config_from_dict,
    generate_compact_operator_counterfactual_groups,
    run_compact_operator_probe,
    run_compact_operator_probe_multi_seed,
)
from apc.evaluation.shared_core_generalization import SharedCoreGateTrainConfig

OPERATIONS = ("SHIFT", "SELECT", "COUNT", "BIND")


def _tiny_config(**overrides: object) -> CompactOperatorConfig:
    base = CompactOperatorConfig(
        seed=0,
        vocab_size=6,
        sequence_length_range=(4, 6),
        operation_names=("SHIFT", "BIND"),
        group_size=3,
        model={
            "d_model": 16,
            "n_layer": 2,
            "n_head": 2,
            "d_ff": 32,
            "max_seq_len": 32,
            "dropout": 0.0,
        },
        core_train=SharedCoreGateTrainConfig(
            steps=3, batch_size=4, eval_every=1, progress_eval_examples=2, device="cpu"
        ),
        d_operator=8,
        n_operator_head=2,
        d_operator_ff=16,
        arg_dim=6,
        max_sequence_length=8,
        operator_train=PrimitiveTrainConfig(
            steps=3, batch_size=4, eval_every=1, progress_eval_examples=2
        ),
        num_unseen_eval_groups=20,
        min_unseen_eval_examples=8,
    )
    return dataclasses.replace(base, **overrides) if overrides else base


# --- _valid_content_lengths / _operation_output / _argument_candidates -----


def test_valid_content_lengths_restricts_bind_to_even() -> None:
    assert _valid_content_lengths("BIND", (3, 7)) == (4, 6)
    assert _valid_content_lengths("SHIFT", (3, 7)) == (3, 4, 5, 6, 7)


def test_operation_output_shift_rotates() -> None:
    assert _operation_output("SHIFT", (0, 1, 2, 3), 1, vocab_size=6) == (1, 2, 3, 0)


def test_operation_output_select_gathers_in_argument_order() -> None:
    assert _operation_output("SELECT", (5, 6, 7, 8), [0, 2], vocab_size=10) == (5, 7)


def test_operation_output_count_clips_to_vocab_size_minus_one() -> None:
    assert _operation_output("COUNT", (1, 1, 1, 1), 1, vocab_size=3) == (2,)


def test_operation_output_bind_last_match_wins() -> None:
    # keys at 0, 2 both equal 5; values at 1, 3 are 9, 7 -- last pair wins.
    assert _operation_output("BIND", (5, 9, 5, 7), 5, vocab_size=10) == (7,)


def test_operation_output_rejects_unsupported_operation() -> None:
    with pytest.raises(ValueError, match="unsupported operation"):
        _operation_output("COPY", (1, 2, 3), None, vocab_size=6)


def test_argument_candidates_bind_is_distinct_present_keys_only() -> None:
    rng = __import__("random").Random(0)
    candidates = _argument_candidates("BIND", rng, (3, 9, 3, 7), vocab_size=10)
    assert set(candidates) == {3}  # only key actually present, deduplicated


def test_argument_candidates_select_matches_output_length() -> None:
    rng = __import__("random").Random(0)
    candidates = _argument_candidates("SELECT", rng, (0, 1, 2, 3), vocab_size=10)
    assert all(len(c) == 2 for c in candidates)  # output_length(4) == max(1, 4 // 2)


# --- CompactOperatorGroup ----------------------------------------------


def test_group_rejects_too_few_members() -> None:
    groups = generate_compact_operator_counterfactual_groups(
        0, 1, operation="SHIFT", step=0, split="train", vocab_size=6, sequence_length_range=(4, 6)
    )
    group = groups[0]
    with pytest.raises(ValueError, match="MIN_GROUP_SIZE|>= 2"):
        CompactOperatorGroup(
            operation=group.operation,
            input_tokens=group.input_tokens,
            examples=group.examples[:1],
            argument_values=group.argument_values[:1],
        )


def test_group_rejects_duplicate_outputs() -> None:
    groups = generate_compact_operator_counterfactual_groups(
        0, 1, operation="SHIFT", step=0, split="train", vocab_size=6, sequence_length_range=(4, 6)
    )
    group = groups[0]
    with pytest.raises(ValueError, match="pairwise distinct"):
        CompactOperatorGroup(
            operation=group.operation,
            input_tokens=group.input_tokens,
            examples=(group.examples[0], group.examples[0]),
            argument_values=(group.argument_values[0], group.argument_values[0]),
        )


def test_wrong_argument_index_wraps_around() -> None:
    groups = generate_compact_operator_counterfactual_groups(
        0, 1, operation="SHIFT", step=0, split="train", vocab_size=6, sequence_length_range=(4, 6)
    )
    group = groups[0]
    last = len(group.examples) - 1
    assert group.wrong_argument_index(last) == 0


# --- generate_compact_operator_counterfactual_groups ------------------------


@pytest.mark.parametrize("operation", OPERATIONS)
def test_generated_groups_meet_min_size_and_are_pairwise_distinct(operation: str) -> None:
    groups = generate_compact_operator_counterfactual_groups(
        0,
        30,
        operation=operation,
        step=0,
        split="train",
        vocab_size=6,
        sequence_length_range=(4, 6),
    )
    for group in groups:
        assert len(group.examples) >= MIN_GROUP_SIZE
        outputs = [e.target_tokens for e in group.examples]
        assert len(set(outputs)) == len(outputs)
        if operation == "BIND":
            assert len(group.input_tokens) % 2 == 0


def test_generated_groups_are_deterministic_given_same_seed_step_split() -> None:
    a = generate_compact_operator_counterfactual_groups(
        1, 5, operation="COUNT", step=2, split="train", vocab_size=6, sequence_length_range=(4, 6)
    )
    b = generate_compact_operator_counterfactual_groups(
        1, 5, operation="COUNT", step=2, split="train", vocab_size=6, sequence_length_range=(4, 6)
    )
    assert [g.input_tokens for g in a] == [g.input_tokens for g in b]
    assert [g.argument_values for g in a] == [g.argument_values for g in b]


def test_generated_groups_differ_across_steps() -> None:
    a = generate_compact_operator_counterfactual_groups(
        1, 5, operation="SHIFT", step=0, split="train", vocab_size=6, sequence_length_range=(4, 6)
    )
    b = generate_compact_operator_counterfactual_groups(
        1, 5, operation="SHIFT", step=1, split="train", vocab_size=6, sequence_length_range=(4, 6)
    )
    assert [g.input_tokens for g in a] != [g.input_tokens for g in b]


def test_generate_groups_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="operation"):
        generate_compact_operator_counterfactual_groups(
            0, 1, operation="COPY", step=0, split="train"
        )
    with pytest.raises(ValueError, match="n_groups"):
        generate_compact_operator_counterfactual_groups(
            0, 0, operation="SHIFT", step=0, split="train"
        )
    with pytest.raises(ValueError, match="step"):
        generate_compact_operator_counterfactual_groups(
            0, 1, operation="SHIFT", step=-1, split="train"
        )
    with pytest.raises(ValueError, match="group_size"):
        generate_compact_operator_counterfactual_groups(
            0, 1, operation="SHIFT", step=0, split="train", group_size=1
        )


def test_flatten_groups_wrong_argument_is_a_sibling_members_value() -> None:
    groups = generate_compact_operator_counterfactual_groups(
        0, 3, operation="SHIFT", step=0, split="train", vocab_size=6, sequence_length_range=(4, 6)
    )
    examples, wrong_argument_by_id, wrong_target_tokens_by_id = _flatten_groups(groups)
    assert len(examples) == sum(len(g.examples) for g in groups)
    for group in groups:
        for index, example in enumerate(group.examples):
            partner = group.wrong_argument_index(index)
            assert wrong_argument_by_id[id(example)] == group.argument_values[partner]
            assert wrong_target_tokens_by_id[id(example)] == group.examples[partner].target_tokens
            # The wrong argument's true output must actually differ (group invariant).
            assert wrong_target_tokens_by_id[id(example)] != example.target_tokens


def test_correct_argument_value_matches_generation() -> None:
    groups = generate_compact_operator_counterfactual_groups(
        0, 1, operation="BIND", step=0, split="train", vocab_size=6, sequence_length_range=(4, 6)
    )
    group = groups[0]
    for example, value in zip(group.examples, group.argument_values, strict=True):
        assert _correct_argument_value("BIND", example) == value


# --- Config validation / round-trip -----------------------------------------


def test_config_validation_rejects_empty_operation_names() -> None:
    with pytest.raises(ValueError, match="operation_names"):
        _tiny_config(operation_names=())


def test_config_validation_rejects_non_divisible_operator_width() -> None:
    with pytest.raises(ValueError, match="divisible"):
        _tiny_config(d_operator=15, n_operator_head=2)


def test_config_validation_rejects_max_sequence_length_below_range() -> None:
    with pytest.raises(ValueError, match="max_sequence_length"):
        _tiny_config(max_sequence_length=3, sequence_length_range=(4, 6))


def test_config_from_dict_round_trips_defaults_and_overrides() -> None:
    raw = {
        "seed": 2,
        "vocab_size": 6,
        "sequence_length_range": [4, 6],
        "operation_names": ["SHIFT"],
        "group_size": 3,
        "model": {"d_model": 16, "n_layer": 2, "n_head": 2, "d_ff": 32, "max_seq_len": 32},
        "core_train": {"steps": 3, "batch_size": 4},
        "d_operator": 8,
        "n_operator_head": 2,
        "d_operator_ff": 16,
        "arg_dim": 6,
        "max_sequence_length": 8,
        "operator_train": {"steps": 3},
        "num_unseen_eval_groups": 20,
        "min_unseen_eval_examples": 8,
    }
    config = compact_operator_config_from_dict(raw)
    assert config.seed == 2
    assert config.operation_names == ("SHIFT",)
    assert config.model["d_model"] == 16
    assert config.core_train.steps == 3
    assert config.d_operator == 8
    assert config.operator_train.steps == 3

    defaulted = compact_operator_config_from_dict({})
    assert defaulted == CompactOperatorConfig()


def test_default_operation_names_are_the_four_parameterized_operations() -> None:
    assert set(CompactOperatorConfig().operation_names) == {
        "SHIFT",
        "SELECT",
        "COUNT",
        "BIND",
    }


def test_default_seed_policy_matches_task_text() -> None:
    assert DEFAULT_SEEDS == (0, 1, 2, 3, 4)
    assert MIN_GATE_SEEDS == 5


def test_default_operator_is_much_smaller_than_high_capacity_default() -> None:
    # A1-R005E-005's own "small projections": d_operator=32 vs. A1-R005E-004's
    # d_operator=256, one attention step vs. n_layers=3 self-attention blocks.
    defaults = CompactOperatorConfig()
    assert defaults.d_operator == 32
    assert defaults.d_operator_ff == 64


# --- CompactCrossPositionOperator forward shape/masking ---------------------


def _tiny_operator(operation: str) -> CompactCrossPositionOperator:
    return CompactCrossPositionOperator(
        operation,
        d_model=8,
        d_operator=12,
        n_head=2,
        d_operator_ff=16,
        vocab_size=6,
        max_sequence_length=8,
        arg_dim=4,
    )


def test_operator_forward_shape_shift_variable_length() -> None:
    operator = _tiny_operator("SHIFT")
    content_features = torch.randn(3, 5, 8)  # batch=3, Lmax=5
    content_lengths = [5, 3, 4]
    output_lengths = [5, 3, 4]
    argument_values = [1, 0, 2]
    logits = operator(content_features, content_lengths, output_lengths, argument_values)
    assert logits.shape == (3, 5, 6)


def test_operator_forward_shape_bind_single_output() -> None:
    operator = _tiny_operator("BIND")
    content_features = torch.randn(2, 4, 8)
    content_lengths = [4, 4]
    output_lengths = [1, 1]
    argument_values = [2, 3]
    logits = operator(content_features, content_lengths, output_lengths, argument_values)
    assert logits.shape == (2, 1, 6)


def test_operator_forward_none_arm_zeros_argument_token_and_runs() -> None:
    operator = _tiny_operator("COUNT")
    content_features = torch.randn(2, 4, 8)
    logits = operator(content_features, [4, 4], [1, 1], None)
    assert logits.shape == (2, 1, 6)
    assert torch.isfinite(logits).all()


def test_operator_forward_rejects_bad_width_head_combo() -> None:
    with pytest.raises(ValueError, match="divisible"):
        CompactCrossPositionOperator(
            "SHIFT",
            d_model=8,
            d_operator=10,
            n_head=3,
            d_operator_ff=16,
            vocab_size=6,
            max_sequence_length=8,
            arg_dim=4,
        )


def test_operator_forward_rejects_unknown_operation() -> None:
    with pytest.raises(ValueError, match="operation"):
        CompactCrossPositionOperator(
            "COPY",
            d_model=8,
            d_operator=8,
            n_head=2,
            d_operator_ff=16,
            vocab_size=6,
            max_sequence_length=8,
            arg_dim=4,
        )


def test_operator_param_count_is_far_smaller_than_high_capacity_operator() -> None:
    # A1-R005E-004's own default HighCapacityOperator (d_operator=256,
    # n_layers=3, d_operator_ff=1024, arg_dim=16, d_model=192) has
    # ~2.44M parameters (docs/DECISIONS.md ADR-0041). This module's default
    # config (d_operator=32, d_operator_ff=64) should land at least two
    # orders of magnitude smaller, at the same d_model=192/arg_dim=16.
    operator = CompactCrossPositionOperator(
        "SHIFT",
        d_model=192,
        d_operator=32,
        n_head=4,
        d_operator_ff=64,
        vocab_size=10,
        max_sequence_length=32,
        arg_dim=16,
    )
    param_count = sum(p.numel() for p in operator.parameters())
    assert param_count < 50_000


# --- End-to-end tiny benchmark run ------------------------------------------


def test_run_compact_operator_probe_report_shape() -> None:
    config = _tiny_config()
    report = run_compact_operator_probe(config)
    assert set(report.per_operation) == {"SHIFT", "BIND"}
    for operation, op_report in report.per_operation.items():
        assert op_report.operation == operation
        assert 0.0 <= op_report.correct_exact_match <= 1.0
        assert 0.0 <= op_report.effectful_wrong_argument_exact_match <= 1.0
        assert 0.0 <= op_report.none_exact_match <= 1.0
        assert op_report.num_unseen_eval_examples >= config.min_unseen_eval_examples
        assert op_report.operator_param_count > 0
        assert op_report.passed == (
            op_report.correct_exact_match_passed
            and (
                op_report.token_accuracy_passed
                if op_report.token_accuracy_passed is not None
                else True
            )
            and op_report.effectful_wrong_argument_passed
            and op_report.none_passed
            and op_report.causal_gap_passed
        )
        if operation in TOKEN_ACCURACY_GATED_OPERATIONS:
            assert op_report.token_accuracy_passed is not None
        else:
            assert op_report.token_accuracy_passed is None


def test_run_compact_operator_probe_writes_metrics(tmp_path: object) -> None:
    from pathlib import Path

    metrics_dir = Path(str(tmp_path)) / "seed_0"
    config = _tiny_config(operation_names=("SHIFT",))
    run_compact_operator_probe(config, metrics_dir=metrics_dir)
    assert (metrics_dir / "SHIFT_core_metrics.jsonl").exists()
    assert (metrics_dir / "SHIFT_operator_metrics.jsonl").exists()


def test_multi_seed_aggregation_matches_manual_formula() -> None:
    base_config = _tiny_config(operation_names=("SHIFT",))
    seeds = (0, 1)
    result = run_compact_operator_probe_multi_seed(base_config, seeds=seeds)
    assert result.seeds == seeds
    assert result.meets_seed_policy is False  # only 2 < MIN_GATE_SEEDS

    op_reports = [report.per_operation["SHIFT"] for report in result.per_seed]
    summary = result.per_operation_summary["SHIFT"]
    mean_correct = sum(r.correct_exact_match for r in op_reports) / len(op_reports)
    mean_wrong = sum(r.effectful_wrong_argument_exact_match for r in op_reports) / len(op_reports)
    mean_none = sum(r.none_exact_match for r in op_reports) / len(op_reports)
    expected_gap = mean_correct - max(mean_wrong, mean_none)

    assert summary.mean_correct_exact_match == pytest.approx(mean_correct)
    assert summary.mean_exact_match_causal_gap == pytest.approx(expected_gap)
    assert summary.correct_exact_match_passed == (mean_correct >= CORRECT_EXACT_MATCH_THRESHOLD)
    assert summary.effectful_wrong_argument_passed == (
        mean_wrong <= EFFECTFUL_WRONG_ARGUMENT_CEILING
    )
    assert summary.none_passed == (mean_none <= NONE_CEILING)
    assert summary.causal_gap_passed == (expected_gap >= MIN_CAUSAL_GAP)
    assert summary.operator_param_count == op_reports[0].operator_param_count
    assert result.passed == all(s.passed for s in result.per_operation_summary.values())


def test_threshold_constants_match_task_text() -> None:
    assert CORRECT_EXACT_MATCH_THRESHOLD == 0.90
    assert TOKEN_ACCURACY_THRESHOLD == 0.98
    assert EFFECTFUL_WRONG_ARGUMENT_CEILING == 0.30
    assert MIN_CAUSAL_GAP == 0.50
    assert NONE_CEILING == 0.30
    assert TOKEN_ACCURACY_GATED_OPERATIONS == ("SHIFT", "SELECT")
    assert DEFAULT_GROUP_SIZE == 3
    assert MIN_GROUP_SIZE == 2
