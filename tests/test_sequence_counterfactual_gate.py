"""Tests for the SHIFT/SELECT sequence counterfactual gate (Phase A.1
Post-Correction Task A1-R005D-008).

Fast, small-step tests only -- mirroring `tests/
test_count_counterfactual_gate.py`/`tests/test_bind_counterfactual_gate.py`'s
convention of exercising the wiring (group construction, config validation,
primitive-bank construction, report shape, multi-seed aggregation) with a
tiny model and a handful of training steps, for both `SHIFT` and `SELECT`.
The gate's own scientific claim (5 seeds, the full training budget) is run
via `scripts/sequence_counterfactual_gate.py`, not asserted here.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import pytest
import torch

from apc.environments.task_spec import operation_id
from apc.evaluation.sequence_counterfactual_gate import (
    DEFAULT_GROUP_SIZE,
    DEFAULT_SEEDS,
    EXACT_MATCH_THRESHOLD,
    MIN_EXACT_MATCH_CAUSAL_GAP,
    MIN_GATE_SEEDS,
    MIN_GROUP_SIZE,
    NONE_CEILING,
    SELECT_OPERATION,
    SEQUENCE_OPERATIONS,
    SHIFT_OPERATION,
    TOKEN_ACCURACY_THRESHOLD,
    PrimitiveTrainConfig,
    SequenceCounterfactualGateConfig,
    SequenceCounterfactualGateReport,
    SequenceCounterfactualGroup,
    _build_primitive_bank,
    _flatten_groups,
    _mean_token_accuracy,
    _token_accuracy,
    generate_sequence_counterfactual_groups,
    run_sequence_counterfactual_gate,
    run_sequence_counterfactual_gate_multi_seed,
    sequence_counterfactual_gate_config_from_dict,
)
from apc.evaluation.shared_core_generalization import SharedCoreGateTrainConfig
from apc.primitives.conditioning import (
    ConditionedPrimitive,
    OrderPreservingIndexSetArgumentEncoder,
)
from apc.primitives.primitive import PrimitiveStatus


def _tiny_config(
    operation: str = SHIFT_OPERATION, **overrides: object
) -> SequenceCounterfactualGateConfig:
    base = SequenceCounterfactualGateConfig(
        operation=operation,
        seed=0,
        vocab_size=6,
        sequence_length_range=(4, 6),
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
        primitive_rank=4,
        arg_dim=6,
        max_sequence_length=8,
        primitive_train=PrimitiveTrainConfig(
            steps=3, batch_size=4, eval_every=1, progress_eval_examples=2
        ),
        num_unseen_eval_groups=20,
        min_unseen_eval_examples=8,
    )
    return dataclasses.replace(base, **overrides) if overrides else base


# --- _token_accuracy / _mean_token_accuracy -------------------------------


def test_token_accuracy_perfect_match() -> None:
    assert _token_accuracy((1, 2, 3), (1, 2, 3)) == 1.0


def test_token_accuracy_partial_match() -> None:
    assert _token_accuracy((1, 2, 3, 4), (1, 0, 3, 0)) == pytest.approx(0.5)


def test_token_accuracy_shorter_prediction_counts_missing_as_wrong() -> None:
    assert _token_accuracy((1, 2, 3, 4), (1, 2)) == pytest.approx(0.5)


def test_token_accuracy_longer_prediction_ignores_extra_tail() -> None:
    assert _token_accuracy((1, 2), (1, 2, 9, 9, 9)) == pytest.approx(1.0)


def test_token_accuracy_rejects_empty_target() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        _token_accuracy((), (1,))


# --- generate_sequence_counterfactual_groups / SequenceCounterfactualGroup


@pytest.mark.parametrize("operation", SEQUENCE_OPERATIONS)
def test_generated_groups_have_pairwise_distinct_outputs(operation: str) -> None:
    groups = generate_sequence_counterfactual_groups(
        0, 50, operation=operation, step=0, split="train"
    )
    for group in groups:
        outputs = [example.target_tokens for example in group.examples]
        assert len(set(outputs)) == len(outputs)


@pytest.mark.parametrize("operation", SEQUENCE_OPERATIONS)
def test_generated_groups_share_the_same_content(operation: str) -> None:
    groups = generate_sequence_counterfactual_groups(
        0, 20, operation=operation, step=0, split="train"
    )
    for group in groups:
        for example in group.examples:
            assert example.input_tokens == group.input_tokens


@pytest.mark.parametrize("operation", SEQUENCE_OPERATIONS)
def test_generated_groups_meet_minimum_size(operation: str) -> None:
    groups = generate_sequence_counterfactual_groups(
        0, 100, operation=operation, step=0, split="train"
    )
    assert all(len(group.examples) >= MIN_GROUP_SIZE for group in groups)


@pytest.mark.parametrize("operation", SEQUENCE_OPERATIONS)
def test_generated_groups_default_to_the_documented_group_size_most_of_the_time(
    operation: str,
) -> None:
    groups = generate_sequence_counterfactual_groups(
        0, 300, operation=operation, step=0, split="train"
    )
    at_default_size = sum(1 for group in groups if len(group.examples) == DEFAULT_GROUP_SIZE)
    assert at_default_size / len(groups) > 0.7


def test_shift_groups_are_rotations_of_the_content() -> None:
    groups = generate_sequence_counterfactual_groups(
        0, 20, operation=SHIFT_OPERATION, step=0, split="train"
    )
    for group in groups:
        content = group.input_tokens
        for example, amount in zip(group.examples, group.argument_values, strict=True):
            expected = content[amount % len(content) :] + content[: amount % len(content)]
            assert example.target_tokens == expected


def test_select_groups_output_length_matches_operation() -> None:
    from apc.environments.operations import get_operation

    groups = generate_sequence_counterfactual_groups(
        0, 20, operation=SELECT_OPERATION, step=0, split="train"
    )
    select_op = get_operation(SELECT_OPERATION)
    for group in groups:
        expected_len = select_op.output_length(len(group.input_tokens))
        for example, indices in zip(group.examples, group.argument_values, strict=True):
            assert len(indices) == expected_len
            assert len(example.target_tokens) == expected_len


def test_generate_groups_is_deterministic_given_the_same_arguments() -> None:
    first = generate_sequence_counterfactual_groups(
        0, 10, operation=SHIFT_OPERATION, step=3, split="train"
    )
    second = generate_sequence_counterfactual_groups(
        0, 10, operation=SHIFT_OPERATION, step=3, split="train"
    )
    assert [g.input_tokens for g in first] == [g.input_tokens for g in second]
    assert [g.argument_values for g in first] == [g.argument_values for g in second]


def test_generate_groups_differ_by_operation_at_the_same_seed_step() -> None:
    shift_groups = generate_sequence_counterfactual_groups(
        0, 10, operation=SHIFT_OPERATION, step=0, split="train"
    )
    select_groups = generate_sequence_counterfactual_groups(
        0, 10, operation=SELECT_OPERATION, step=0, split="train"
    )
    assert [g.input_tokens for g in shift_groups] != [g.input_tokens for g in select_groups]


def test_generate_groups_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="operation"):
        generate_sequence_counterfactual_groups(0, 1, operation="COUNT", step=0, split="train")
    with pytest.raises(ValueError, match="n_groups"):
        generate_sequence_counterfactual_groups(
            0, 0, operation=SHIFT_OPERATION, step=0, split="train"
        )
    with pytest.raises(ValueError, match="step"):
        generate_sequence_counterfactual_groups(
            0, 1, operation=SHIFT_OPERATION, step=-1, split="train"
        )
    with pytest.raises(ValueError, match="group_size"):
        generate_sequence_counterfactual_groups(
            0, 1, operation=SHIFT_OPERATION, step=0, split="train", group_size=1
        )


def test_group_rejects_unsupported_operation() -> None:
    groups = generate_sequence_counterfactual_groups(
        0, 1, operation=SHIFT_OPERATION, step=0, split="train"
    )
    group = groups[0]
    with pytest.raises(ValueError, match="operation must be one of"):
        SequenceCounterfactualGroup(
            operation="COUNT",
            input_tokens=group.input_tokens,
            examples=group.examples,
            argument_values=group.argument_values,
        )


def test_group_rejects_mismatched_examples_and_argument_values() -> None:
    groups = generate_sequence_counterfactual_groups(
        0, 1, operation=SHIFT_OPERATION, step=0, split="train"
    )
    group = groups[0]
    with pytest.raises(ValueError, match="len\\(examples\\) == len\\(argument_values\\)"):
        SequenceCounterfactualGroup(
            operation=group.operation,
            input_tokens=group.input_tokens,
            examples=group.examples,
            argument_values=group.argument_values[:-1],
        )


def test_group_rejects_fewer_than_min_group_size_members() -> None:
    groups = generate_sequence_counterfactual_groups(
        0, 1, operation=SHIFT_OPERATION, step=0, split="train"
    )
    group = groups[0]
    with pytest.raises(ValueError, match="requires >="):
        SequenceCounterfactualGroup(
            operation=group.operation,
            input_tokens=group.input_tokens,
            examples=group.examples[:1],
            argument_values=group.argument_values[:1],
        )


def test_group_rejects_non_distinct_outputs() -> None:
    groups = generate_sequence_counterfactual_groups(
        0, 1, operation=SHIFT_OPERATION, step=0, split="train"
    )
    group = groups[0]
    with pytest.raises(ValueError, match="pairwise distinct outputs"):
        SequenceCounterfactualGroup(
            operation=group.operation,
            input_tokens=group.input_tokens,
            examples=(group.examples[0], group.examples[0]),
            argument_values=(group.argument_values[0], group.argument_values[0]),
        )


def test_wrong_argument_index_points_to_a_different_member() -> None:
    groups = generate_sequence_counterfactual_groups(
        0, 20, operation=SHIFT_OPERATION, step=0, split="train"
    )
    for group in groups:
        for index in range(len(group.examples)):
            partner = group.wrong_argument_index(index)
            assert partner != index
            assert group.examples[partner].target_tokens != group.examples[index].target_tokens


# --- _flatten_groups -----------------------------------------------------------


@pytest.mark.parametrize("operation", SEQUENCE_OPERATIONS)
def test_flatten_groups_preserves_total_example_count(operation: str) -> None:
    groups = generate_sequence_counterfactual_groups(
        0, 10, operation=operation, step=0, split="train"
    )
    examples, wrong_call_by_id, wrong_target_tokens_by_id = _flatten_groups(groups)
    assert len(examples) == sum(len(g.examples) for g in groups)
    assert len(wrong_call_by_id) == len(examples)
    assert len(wrong_target_tokens_by_id) == len(examples)


@pytest.mark.parametrize("operation", SEQUENCE_OPERATIONS)
def test_flatten_groups_wrong_call_is_always_effectful(operation: str) -> None:
    groups = generate_sequence_counterfactual_groups(
        0, 20, operation=operation, step=0, split="train"
    )
    examples, _, wrong_target_tokens_by_id = _flatten_groups(groups)
    for example in examples:
        assert wrong_target_tokens_by_id[id(example)] != example.target_tokens


def test_flatten_groups_wrong_call_is_a_well_formed_shift_call() -> None:
    groups = generate_sequence_counterfactual_groups(
        0, 5, operation=SHIFT_OPERATION, step=0, split="train"
    )
    examples, wrong_call_by_id, _ = _flatten_groups(groups)
    for example in examples:
        call = wrong_call_by_id[id(example)]
        assert call.operation == "SHIFT"
        assert "amount" in call.arguments


def test_flatten_groups_wrong_call_is_a_well_formed_select_call() -> None:
    groups = generate_sequence_counterfactual_groups(
        0, 5, operation=SELECT_OPERATION, step=0, split="train"
    )
    examples, wrong_call_by_id, _ = _flatten_groups(groups)
    for example in examples:
        call = wrong_call_by_id[id(example)]
        assert call.operation == "SELECT"
        assert "indices" in call.arguments


# --- SequenceCounterfactualGateConfig validation --------------------------------


def test_config_rejects_unsupported_operation() -> None:
    with pytest.raises(ValueError, match="operation must be one of"):
        _tiny_config(operation="COUNT")


def test_config_rejects_group_size_below_minimum() -> None:
    with pytest.raises(ValueError, match="group_size"):
        _tiny_config(group_size=1)


def test_config_rejects_max_sequence_length_smaller_than_length_range() -> None:
    with pytest.raises(ValueError, match="max_sequence_length"):
        _tiny_config(sequence_length_range=(4, 20), max_sequence_length=8)


def test_config_rejects_non_positive_num_unseen_eval_groups() -> None:
    with pytest.raises(ValueError, match="num_unseen_eval_groups"):
        _tiny_config(num_unseen_eval_groups=0)


def test_config_rejects_non_positive_min_unseen_eval_examples() -> None:
    with pytest.raises(ValueError, match="min_unseen_eval_examples"):
        _tiny_config(min_unseen_eval_examples=0)


def test_config_defaults_to_shift_and_documented_group_size() -> None:
    config = SequenceCounterfactualGateConfig()
    assert config.operation == SHIFT_OPERATION
    assert config.group_size == DEFAULT_GROUP_SIZE


# --- sequence_counterfactual_gate_config_from_dict ------------------------------


def test_config_from_dict_fills_in_defaults() -> None:
    config = sequence_counterfactual_gate_config_from_dict({"seed": 3})
    assert config.seed == 3
    assert config.operation == SHIFT_OPERATION
    assert config.group_size == DEFAULT_GROUP_SIZE


def test_config_from_dict_reads_operation() -> None:
    config = sequence_counterfactual_gate_config_from_dict({"operation": "SELECT"})
    assert config.operation == SELECT_OPERATION


def test_config_from_dict_parses_nested_train_configs() -> None:
    raw = {
        "core_train": {"steps": 5},
        "primitive_train": {"steps": 7, "lr": 0.001},
        "arg_dim": 12,
    }
    config = sequence_counterfactual_gate_config_from_dict(raw)
    assert config.core_train.steps == 5
    assert config.primitive_train.steps == 7
    assert config.primitive_train.lr == 0.001
    assert config.arg_dim == 12


# --- _build_primitive_bank ------------------------------------------------------


def test_build_primitive_bank_registers_exactly_one_shift_primitive() -> None:
    bank = _build_primitive_bank(
        d_model=16,
        rank=4,
        operation=SHIFT_OPERATION,
        vocab_size=6,
        max_sequence_length=8,
        arg_dim=6,
    )
    assert len(bank) == 1
    primitive = bank.get(operation_id(SHIFT_OPERATION))
    assert isinstance(primitive, ConditionedPrimitive)
    assert primitive.config.d_model == 16
    assert primitive.config.rank == 4
    assert primitive.status == PrimitiveStatus.CANDIDATE
    assert primitive.enabled is True


def test_build_primitive_bank_select_uses_order_preserving_encoder() -> None:
    bank = _build_primitive_bank(
        d_model=16,
        rank=4,
        operation=SELECT_OPERATION,
        vocab_size=6,
        max_sequence_length=8,
        arg_dim=6,
    )
    primitive = bank.get(operation_id(SELECT_OPERATION))
    assert isinstance(primitive, ConditionedPrimitive)
    assert isinstance(primitive.argument_encoder, OrderPreservingIndexSetArgumentEncoder)


def test_build_primitive_bank_primitive_is_trainable() -> None:
    bank = _build_primitive_bank(
        d_model=16,
        rank=4,
        operation=SHIFT_OPERATION,
        vocab_size=6,
        max_sequence_length=8,
        arg_dim=6,
    )
    assert not bank.get(operation_id(SHIFT_OPERATION)).is_frozen()


# --- run_sequence_counterfactual_gate --------------------------------------------


@pytest.mark.parametrize("operation", SEQUENCE_OPERATIONS)
def test_run_returns_report_with_expected_fields(operation: str) -> None:
    report = run_sequence_counterfactual_gate(_tiny_config(operation))
    assert isinstance(report, SequenceCounterfactualGateReport)
    assert report.operation == operation
    assert report.core_steps_trained == 3
    assert report.primitive_steps_trained == 3
    assert report.bank_size == 1
    assert report.num_unseen_eval_groups == 20
    assert report.num_unseen_eval_examples >= 8
    for value in (
        report.correct_exact_match,
        report.correct_token_accuracy,
        report.effectful_wrong_argument_exact_match,
        report.effectful_wrong_argument_token_accuracy,
        report.none_exact_match,
        report.none_token_accuracy,
        report.argument_effect_rate,
    ):
        assert 0.0 <= value <= 1.0


def test_run_argument_effect_rate_is_always_one_by_construction() -> None:
    report = run_sequence_counterfactual_gate(_tiny_config(SHIFT_OPERATION))
    assert report.argument_effect_rate == pytest.approx(1.0)


def test_run_freezes_every_stable_core_parameter() -> None:
    report = run_sequence_counterfactual_gate(_tiny_config(SHIFT_OPERATION))
    assert report.core_trainable_param_count == 0
    assert report.core_param_count > 0


def test_run_primitive_bank_has_nonzero_parameter_count() -> None:
    report = run_sequence_counterfactual_gate(_tiny_config(SHIFT_OPERATION))
    assert report.primitive_param_count > 0


def test_run_exact_match_causal_gap_matches_correct_minus_max_controls() -> None:
    report = run_sequence_counterfactual_gate(_tiny_config(SHIFT_OPERATION))
    expected = report.correct_exact_match - max(
        report.effectful_wrong_argument_exact_match, report.none_exact_match
    )
    assert report.exact_match_causal_gap == pytest.approx(expected)


def test_run_token_accuracy_causal_gap_matches_correct_minus_max_controls() -> None:
    report = run_sequence_counterfactual_gate(_tiny_config(SHIFT_OPERATION))
    expected = report.correct_token_accuracy - max(
        report.effectful_wrong_argument_token_accuracy, report.none_token_accuracy
    )
    assert report.token_accuracy_causal_gap == pytest.approx(expected)


def test_run_is_deterministic_given_the_same_seed() -> None:
    first = run_sequence_counterfactual_gate(_tiny_config(SHIFT_OPERATION))
    second = run_sequence_counterfactual_gate(_tiny_config(SHIFT_OPERATION))
    assert first.correct_exact_match == second.correct_exact_match
    assert first.correct_token_accuracy == second.correct_token_accuracy


def test_run_rejects_undersized_eval_batch() -> None:
    with pytest.raises(ValueError, match="min_unseen_eval_examples"):
        run_sequence_counterfactual_gate(
            _tiny_config(SHIFT_OPERATION, num_unseen_eval_groups=1, min_unseen_eval_examples=10_000)
        )


def test_run_length_stratified_correct_covers_the_configured_length_range() -> None:
    report = run_sequence_counterfactual_gate(_tiny_config(SHIFT_OPERATION))
    lengths = {int(length) for length in report.length_stratified_correct}
    assert lengths
    assert lengths <= {4, 5, 6}
    for stats in report.length_stratified_correct.values():
        assert stats["count"] > 0
        assert 0.0 <= stats["exact_match"] <= 1.0
        assert 0.0 <= stats["token_accuracy"] <= 1.0


def test_report_to_dict_round_trips_through_json() -> None:
    report = run_sequence_counterfactual_gate(_tiny_config(SHIFT_OPERATION))
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["correct_exact_match"] == report.correct_exact_match
    assert payload["operation"] == SHIFT_OPERATION


def test_no_cuda_required_for_cpu_device_override() -> None:
    report = run_sequence_counterfactual_gate(_tiny_config(SHIFT_OPERATION))
    assert report.device == str(torch.device("cpu"))


# --- run_sequence_counterfactual_gate_multi_seed --------------------------------


def test_multi_seed_runs_one_report_per_seed_and_writes_run_dir(tmp_path: Path) -> None:
    seeds = (0, 1)
    result = run_sequence_counterfactual_gate_multi_seed(
        _tiny_config(SHIFT_OPERATION), seeds=seeds, run_dir=tmp_path
    )
    assert result.operation == SHIFT_OPERATION
    assert result.seeds == seeds
    assert len(result.per_seed) == 2
    for seed in seeds:
        assert (tmp_path / f"seed_{seed}" / "report.json").exists()


def test_multi_seed_overrides_base_config_seed_per_entry() -> None:
    result = run_sequence_counterfactual_gate_multi_seed(
        _tiny_config(SHIFT_OPERATION), seeds=(0, 1, 2)
    )
    assert [report.config.seed for report in result.per_seed] == [0, 1, 2]


def test_multi_seed_mean_exact_match_causal_gap_matches_mean_arms() -> None:
    result = run_sequence_counterfactual_gate_multi_seed(
        _tiny_config(SHIFT_OPERATION), seeds=(0, 1)
    )
    expected = result.mean_correct_exact_match - max(
        result.mean_effectful_wrong_argument_exact_match, result.mean_none_exact_match
    )
    assert result.mean_exact_match_causal_gap == pytest.approx(expected)


def test_multi_seed_family_count_passed_reflects_bank_size() -> None:
    result = run_sequence_counterfactual_gate_multi_seed(_tiny_config(SHIFT_OPERATION), seeds=(0,))
    assert result.family_count_passed is True
    assert all(report.bank_size == 1 for report in result.per_seed)


def test_multi_seed_passed_requires_all_four_conditions() -> None:
    fail_exact_match = run_sequence_counterfactual_gate_multi_seed(
        _tiny_config(SHIFT_OPERATION), seeds=(0,), exact_match_threshold=2.0
    )
    assert fail_exact_match.exact_match_passed is False
    assert fail_exact_match.passed is False

    fail_token_accuracy = run_sequence_counterfactual_gate_multi_seed(
        _tiny_config(SHIFT_OPERATION), seeds=(0,), token_accuracy_threshold=2.0
    )
    assert fail_token_accuracy.token_accuracy_passed is False
    assert fail_token_accuracy.passed is False

    fail_none = run_sequence_counterfactual_gate_multi_seed(
        _tiny_config(SHIFT_OPERATION), seeds=(0,), none_ceiling=-1.0
    )
    assert fail_none.none_passed is False
    assert fail_none.passed is False

    fail_gap = run_sequence_counterfactual_gate_multi_seed(
        _tiny_config(SHIFT_OPERATION), seeds=(0,), min_exact_match_causal_gap=2.0
    )
    assert fail_gap.causal_gap_passed is False
    assert fail_gap.passed is False


def test_multi_seed_meets_seed_policy_reflects_seed_count() -> None:
    short = run_sequence_counterfactual_gate_multi_seed(_tiny_config(SHIFT_OPERATION), seeds=(0, 1))
    assert short.meets_seed_policy is False
    full = run_sequence_counterfactual_gate_multi_seed(
        _tiny_config(SHIFT_OPERATION), seeds=tuple(range(MIN_GATE_SEEDS))
    )
    assert full.meets_seed_policy is True


def test_multi_seed_report_to_dict_round_trips_through_json(tmp_path: Path) -> None:
    result = run_sequence_counterfactual_gate_multi_seed(
        _tiny_config(SHIFT_OPERATION), seeds=(0, 1), run_dir=tmp_path
    )
    payload = json.loads(json.dumps(result.to_dict()))
    assert payload["passed"] == result.passed
    assert len(payload["per_seed"]) == 2


def test_multi_seed_runs_select_operation_end_to_end(tmp_path: Path) -> None:
    result = run_sequence_counterfactual_gate_multi_seed(
        _tiny_config(SELECT_OPERATION), seeds=(0, 1), run_dir=tmp_path
    )
    assert result.operation == SELECT_OPERATION
    assert len(result.per_seed) == 2


# --- module constants -----------------------------------------------------------


def test_default_seeds_matches_min_gate_seeds_length() -> None:
    assert len(DEFAULT_SEEDS) == MIN_GATE_SEEDS


def test_thresholds_match_a1_r005d_008() -> None:
    assert EXACT_MATCH_THRESHOLD == 0.85
    assert TOKEN_ACCURACY_THRESHOLD == 0.95
    assert MIN_EXACT_MATCH_CAUSAL_GAP == 0.50
    assert NONE_CEILING == 0.30


def test_select_candidate_count_matches_math_comb() -> None:
    """Sanity check the module docstring's own claim about SELECT's
    candidate-pool size staying small at these lengths."""
    from apc.environments.operations import get_operation

    select_op = get_operation(SELECT_OPERATION)
    for length in range(6, 11):
        k = select_op.output_length(length)
        assert math.comb(length, k) < 300


def test_mean_token_accuracy_averages_per_example() -> None:
    from apc.environments.generator import Example, OracleMetadata
    from apc.environments.program import Program

    def _example(target: tuple[int, ...]) -> Example:
        return Example(
            input_tokens=(0,) * len(target),
            target_tokens=target,
            program=Program(steps=()),
            operation_graph=(),
            category="known",
            split="test",
            vocab_size=6,
            task_spec=None,
            oracle_metadata=OracleMetadata(label="K", primitive_operations=()),
            symbol_permutation=None,
        )

    examples = [_example((1, 2)), _example((3, 4))]
    predictions = [(1, 2), (3, 0)]
    assert _mean_token_accuracy(examples, predictions) == pytest.approx(0.75)
