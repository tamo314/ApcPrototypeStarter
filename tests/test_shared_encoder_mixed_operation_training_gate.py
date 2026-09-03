"""Tests for the balanced mixed-operation training gate (Phase A.1
diagnostic Task A1-R005E-S002).

Fast, small-step tests only -- mirroring `tests/test_joint_representation_
compact_operator_probe.py`'s convention of exercising the training loop with
a tiny model and a handful of steps. The benchmark's own scientific claim
(5 seeds, the full 152000-step balanced budget) is run via `scripts/
shared_encoder_mixed_operation_training_gate.py`, not asserted here.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path

import pytest
import torch

from apc.evaluation.compact_cross_position_operator_probe import (
    generate_compact_operator_counterfactual_groups,
)
from apc.evaluation.parameter_free_primitive_gate import PrimitiveTrainConfig
from apc.evaluation.shared_encoder_architecture_gate import (
    SharedEncoderArchitectureConfig,
    _build_operator,
    build_shared_encoder_architecture,
)
from apc.evaluation.shared_encoder_mixed_operation_training_gate import (
    DEFAULT_SEEDS,
    DEFAULT_TOTAL_STEPS,
    MIN_GATE_SEEDS,
    TASK_BLIND_ATOL,
    OperationSharedTrainingReport,
    SharedMixedTrainingConfig,
    SharedMixedTrainingReport,
    _build_operation_schedule,
    _evaluate_arm_shared,
    _train_shared,
    run_shared_mixed_training_gate,
    run_shared_mixed_training_gate_multi_seed,
    shared_mixed_training_config_from_dict,
)


def _tiny_config(**overrides: object) -> SharedMixedTrainingConfig:
    base = SharedMixedTrainingConfig(
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
        device="cpu",
        d_operator=8,
        n_operator_head=2,
        d_operator_ff=16,
        arg_dim=6,
        max_sequence_length=8,
        operation_sample_weights={"SHIFT": 1, "BIND": 1},
        joint_train=PrimitiveTrainConfig(
            steps=6, batch_size=4, eval_every=2, progress_eval_examples=2
        ),
        num_unseen_eval_groups=20,
        min_unseen_eval_examples=8,
    )
    return dataclasses.replace(base, **overrides) if overrides else base


# --- Config validation / round-trip -----------------------------------------


def test_config_rejects_empty_operation_names() -> None:
    with pytest.raises(ValueError, match="operation_names"):
        _tiny_config(operation_names=(), operation_sample_weights={})


def test_config_rejects_non_divisible_operator_width() -> None:
    with pytest.raises(ValueError, match="divisible"):
        _tiny_config(d_operator=15, n_operator_head=2)


def test_config_rejects_max_sequence_length_below_range() -> None:
    with pytest.raises(ValueError, match="max_sequence_length"):
        _tiny_config(max_sequence_length=3, sequence_length_range=(4, 6))


def test_config_rejects_weight_mismatch_with_operation_names() -> None:
    with pytest.raises(ValueError, match="operation_sample_weights"):
        _tiny_config(operation_sample_weights={"SHIFT": 1})


def test_config_rejects_non_positive_weight() -> None:
    with pytest.raises(ValueError, match="operation_sample_weights"):
        _tiny_config(operation_sample_weights={"SHIFT": 0, "BIND": 1})


def test_config_rejects_steps_below_cycle_length() -> None:
    with pytest.raises(ValueError, match="cycle length"):
        _tiny_config(joint_train=PrimitiveTrainConfig(steps=1, batch_size=4))


def test_config_rejects_non_positive_num_unseen_eval_groups() -> None:
    with pytest.raises(ValueError, match="num_unseen_eval_groups"):
        _tiny_config(num_unseen_eval_groups=0)


def test_config_from_dict_round_trips_defaults_and_overrides() -> None:
    raw = {
        "seed": 2,
        "vocab_size": 6,
        "sequence_length_range": [4, 6],
        "operation_names": ["SHIFT", "BIND"],
        "group_size": 3,
        "model": {"d_model": 16, "n_layer": 2, "n_head": 2, "d_ff": 32, "max_seq_len": 32},
        "device": "cpu",
        "d_operator": 8,
        "n_operator_head": 2,
        "d_operator_ff": 16,
        "arg_dim": 6,
        "max_sequence_length": 8,
        "operation_sample_weights": {"SHIFT": 2, "BIND": 1},
        "joint_train": {"steps": 6, "batch_size": 4},
        "num_unseen_eval_groups": 20,
        "min_unseen_eval_examples": 8,
    }
    config = shared_mixed_training_config_from_dict(raw)
    assert config.seed == 2
    assert config.operation_names == ("SHIFT", "BIND")
    assert config.operation_sample_weights == {"SHIFT": 2, "BIND": 1}
    assert config.model["d_model"] == 16
    assert config.device == "cpu"
    assert config.joint_train.steps == 6

    defaulted = shared_mixed_training_config_from_dict({})
    assert defaulted == SharedMixedTrainingConfig()


def test_default_operation_names_are_the_four_parameterized_operations() -> None:
    assert set(SharedMixedTrainingConfig().operation_names) == {
        "SHIFT",
        "SELECT",
        "COUNT",
        "BIND",
    }


def test_default_seed_policy_matches_every_other_gate() -> None:
    assert DEFAULT_SEEDS == (0, 1, 2, 3, 4)
    assert MIN_GATE_SEEDS == 5


def test_default_total_steps_is_four_times_e006a_per_operator_budget() -> None:
    assert DEFAULT_TOTAL_STEPS == 4 * 38000
    assert SharedMixedTrainingConfig().joint_train.steps == DEFAULT_TOTAL_STEPS


def test_default_operation_sample_weights_are_balanced() -> None:
    defaults = SharedMixedTrainingConfig()
    assert defaults.operation_sample_weights == {name: 1 for name in defaults.operation_names}


def test_default_operator_dimensions_match_s001_unchanged() -> None:
    shared_defaults = SharedMixedTrainingConfig()
    architecture_defaults = SharedEncoderArchitectureConfig()
    assert shared_defaults.d_operator == architecture_defaults.d_operator
    assert shared_defaults.n_operator_head == architecture_defaults.n_operator_head
    assert shared_defaults.d_operator_ff == architecture_defaults.d_operator_ff
    assert shared_defaults.arg_dim == architecture_defaults.arg_dim
    assert shared_defaults.max_sequence_length == architecture_defaults.max_sequence_length
    assert shared_defaults.model == architecture_defaults.model


def test_to_architecture_config_matches_shared_fields() -> None:
    config = _tiny_config()
    architecture_config = config.to_architecture_config()
    assert architecture_config.seed == config.seed
    assert architecture_config.vocab_size == config.vocab_size
    assert architecture_config.operation_names == config.operation_names
    assert architecture_config.model == config.model
    assert architecture_config.d_operator == config.d_operator


# --- Balanced operation schedule ---------------------------------------------


def test_build_operation_schedule_exact_balance_for_equal_weights() -> None:
    schedule = _build_operation_schedule(("SHIFT", "BIND"), {"SHIFT": 1, "BIND": 1}, 6, seed=0)
    assert len(schedule) == 6
    assert schedule.count("SHIFT") == 3
    assert schedule.count("BIND") == 3


def test_build_operation_schedule_respects_weight_ratio() -> None:
    schedule = _build_operation_schedule(("SHIFT", "BIND"), {"SHIFT": 2, "BIND": 1}, 12, seed=0)
    assert schedule.count("SHIFT") == 8
    assert schedule.count("BIND") == 4


def test_build_operation_schedule_is_deterministic() -> None:
    first = _build_operation_schedule(("SHIFT", "BIND"), {"SHIFT": 1, "BIND": 1}, 20, seed=3)
    second = _build_operation_schedule(("SHIFT", "BIND"), {"SHIFT": 1, "BIND": 1}, 20, seed=3)
    assert first == second


def test_build_operation_schedule_differs_across_seeds() -> None:
    a = _build_operation_schedule(("SHIFT", "BIND"), {"SHIFT": 1, "BIND": 1}, 20, seed=0)
    b = _build_operation_schedule(("SHIFT", "BIND"), {"SHIFT": 1, "BIND": 1}, 20, seed=1)
    assert a != b


def test_build_operation_schedule_truncates_partial_final_cycle() -> None:
    schedule = _build_operation_schedule(("SHIFT", "BIND"), {"SHIFT": 1, "BIND": 1}, 5, seed=0)
    assert len(schedule) == 5
    assert set(schedule) <= {"SHIFT", "BIND"}


# --- Training actually updates the shared encoder ----------------------------


def test_train_shared_updates_encoder_parameters() -> None:
    config = _tiny_config()
    architecture = build_shared_encoder_architecture(config.to_architecture_config())
    before = copy.deepcopy(architecture.core.model.token_emb.weight.data)

    accumulator = _train_shared(architecture, config)

    after = architecture.core.model.token_emb.weight.data
    assert not torch.allclose(before, after)
    assert sum(accumulator.step_counts.values()) == config.joint_train.steps
    for operation in config.operation_names:
        assert accumulator.step_counts[operation] > 0
        assert accumulator.examples_seen[operation] > 0
        assert accumulator.final_loss[operation] == accumulator.final_loss[operation]  # not NaN


def test_train_shared_only_updates_operators_when_selected() -> None:
    # An all-SHIFT schedule (config names only SHIFT) must never move a BIND
    # operator's own parameters, since it never receives a gradient -- the
    # functional proof that a single shared optimizer naturally skips
    # non-selected operators (module docstring, "One optimizer, naturally
    # selective updates").
    shift_only_config = _tiny_config(
        operation_names=("SHIFT",),
        operation_sample_weights={"SHIFT": 1},
        joint_train=PrimitiveTrainConfig(steps=4, batch_size=4, eval_every=2),
    )
    architecture = build_shared_encoder_architecture(shift_only_config.to_architecture_config())
    # Attach a BIND operator that is never scheduled, to prove it is
    # untouched (mirrors production wiring where all four operators exist
    # but only the scheduled one updates on a given step).
    bind_operator = _build_operator(
        architecture.core, shift_only_config.to_architecture_config(), "BIND"
    )
    architecture.operators["BIND"] = bind_operator
    bind_before = copy.deepcopy(bind_operator.content_in_proj.weight.data)

    _train_shared(architecture, shift_only_config)

    bind_after = architecture.operators["BIND"].content_in_proj.weight.data
    assert torch.allclose(bind_before, bind_after)


def test_train_shared_writes_metrics_with_expected_keys(tmp_path: Path) -> None:
    config = _tiny_config()
    architecture = build_shared_encoder_architecture(config.to_architecture_config())
    metrics_path = tmp_path / "mixed_training_metrics.jsonl"
    _train_shared(architecture, config, metrics_path=metrics_path)
    lines = metrics_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3  # steps=6, eval_every=2 -> checkpoints at 2, 4, 6
    for line in lines:
        record = json.loads(line)
        assert set(record) == {
            "step",
            "current_operation",
            "operation_step_counts",
            "operation_mean_loss_so_far",
            "operation_mean_encoder_grad_norm_so_far",
            "operation_mean_operator_grad_norm_so_far",
            "operation_progress_correct_exact_match",
        }
        assert set(record["operation_progress_correct_exact_match"]) == set(
            config.operation_names
        )


# --- _evaluate_arm_shared -----------------------------------------------------


def test_evaluate_arm_shared_restores_train_mode() -> None:
    config = _tiny_config()
    architecture = build_shared_encoder_architecture(config.to_architecture_config())
    groups = generate_compact_operator_counterfactual_groups(
        0, 2, operation="SHIFT", step=0, split="train", vocab_size=6, sequence_length_range=(4, 6)
    )
    examples = [example for group in groups for example in group.examples]
    _evaluate_arm_shared(architecture, "SHIFT", examples, argument_provider=lambda e: 0)
    assert architecture.core.model.training
    assert architecture.operators["SHIFT"].training


# --- Full gate report --------------------------------------------------------


def test_run_shared_mixed_training_gate_report_shape() -> None:
    config = _tiny_config()
    report = run_shared_mixed_training_gate(config)
    assert isinstance(report, SharedMixedTrainingReport)
    assert set(report.per_operation) == {"SHIFT", "BIND"}
    assert report.total_steps == config.joint_train.steps
    assert report.encoder_update_count == config.joint_train.steps
    assert sum(report.operation_step_counts.values()) == config.joint_train.steps
    assert sum(report.operation_examples_seen.values()) > 0
    assert report.operation_realized_sampling_proportions["SHIFT"] == pytest.approx(0.5)
    assert report.operation_realized_sampling_proportions["BIND"] == pytest.approx(0.5)
    assert report.core_param_count > 0
    assert report.core_trainable_param_count == report.core_param_count

    for operation, op_report in report.per_operation.items():
        assert isinstance(op_report, OperationSharedTrainingReport)
        assert op_report.operation == operation
        assert 0.0 <= op_report.correct_exact_match <= 1.0
        assert 0.0 <= op_report.effectful_wrong_argument_exact_match <= 1.0
        assert 0.0 <= op_report.none_exact_match <= 1.0
        assert op_report.num_unseen_eval_examples >= config.min_unseen_eval_examples
        assert op_report.operator_param_count > 0
        assert op_report.task_blind_max_abs_diff >= 0.0
        assert op_report.steps_trained == report.operation_step_counts[operation]
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
    assert report.passed == all(r.passed for r in report.per_operation.values())
    assert report.task_blind_invariant_passed == all(
        r.task_blind_invariant_passed for r in report.per_operation.values()
    )


def test_run_shared_mixed_training_gate_raises_below_min_unseen_examples() -> None:
    config = _tiny_config(min_unseen_eval_examples=10_000)
    with pytest.raises(ValueError, match="min_unseen_eval_examples"):
        run_shared_mixed_training_gate(config)


def test_run_shared_mixed_training_gate_writes_metrics(tmp_path: Path) -> None:
    config = _tiny_config()
    metrics_path = tmp_path / "seed_0" / "mixed_training_metrics.jsonl"
    run_shared_mixed_training_gate(config, metrics_path=metrics_path)
    assert metrics_path.exists()


def test_multi_seed_aggregation_matches_manual_formula() -> None:
    base_config = _tiny_config()
    seeds = (0, 1)
    result = run_shared_mixed_training_gate_multi_seed(base_config, seeds=seeds)
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
    assert result.passed == all(s.passed for s in result.per_operation_summary.values())
    assert result.core_param_count == result.per_seed[0].core_param_count


def test_multi_seed_writes_per_seed_reports_and_metrics(tmp_path: Path) -> None:
    base_config = _tiny_config()
    run_dir = tmp_path
    run_shared_mixed_training_gate_multi_seed(base_config, seeds=(0, 1), run_dir=run_dir)
    for seed in (0, 1):
        report_path = run_dir / f"seed_{seed}" / "report.json"
        metrics_path = run_dir / f"seed_{seed}" / "mixed_training_metrics.jsonl"
        assert report_path.is_file()
        assert metrics_path.is_file()
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload["config"]["seed"] == seed


def test_multi_seed_reports_seed_policy() -> None:
    base_config = _tiny_config()
    result = run_shared_mixed_training_gate_multi_seed(base_config, seeds=(0, 1))
    assert result.meets_seed_policy is False
    result_full = run_shared_mixed_training_gate_multi_seed(
        base_config, seeds=(0, 1, 2, 3, 4)
    )
    assert result_full.meets_seed_policy is True


def test_threshold_constants_reused_from_e005() -> None:
    from apc.evaluation.shared_encoder_mixed_operation_training_gate import (
        CORRECT_EXACT_MATCH_THRESHOLD,
        EFFECTFUL_WRONG_ARGUMENT_CEILING,
        MIN_CAUSAL_GAP,
        NONE_CEILING,
        TOKEN_ACCURACY_GATED_OPERATIONS,
        TOKEN_ACCURACY_THRESHOLD,
    )

    assert CORRECT_EXACT_MATCH_THRESHOLD == 0.90
    assert TOKEN_ACCURACY_THRESHOLD == 0.98
    assert EFFECTFUL_WRONG_ARGUMENT_CEILING == 0.30
    assert MIN_CAUSAL_GAP == 0.50
    assert NONE_CEILING == 0.30
    assert TOKEN_ACCURACY_GATED_OPERATIONS == ("SHIFT", "SELECT")


def test_task_blind_atol_matches_every_earlier_module() -> None:
    assert TASK_BLIND_ATOL == 1e-5
