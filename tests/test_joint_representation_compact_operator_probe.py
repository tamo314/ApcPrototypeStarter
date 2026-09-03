"""Tests for the joint task-blind representation + compact operator probe
(Phase A.1 diagnostic Task A1-R005E-006A).

Fast, small-step tests only -- mirroring `tests/test_compact_cross_position_
operator_probe.py`'s convention of exercising the wiring (config validation,
core/operator construction, joint training, evaluation, R_access
aggregation) with a tiny model and a handful of training steps. The
benchmark's own scientific claim (5 seeds, the full 38000-step joint budget)
is run via `scripts/joint_representation_compact_operator_probe.py`, not
asserted here.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path

import pytest
import torch

from apc.evaluation.compact_cross_position_operator_probe import (
    CompactOperatorConfig,
    generate_compact_operator_counterfactual_groups,
)
from apc.evaluation.joint_representation_compact_operator_probe import (
    DEFAULT_SEEDS,
    MIN_GATE_SEEDS,
    R_ACCESS_EPS,
    TASK_BLIND_ATOL,
    JointOperatorConfig,
    PrimitiveTrainConfig,
    _batch_features,
    _build_joint_core,
    _build_operator,
    _evaluate_arm,
    _task_blind_invariance_max_abs_diff,
    _train_joint,
    joint_operator_config_from_dict,
    representation_accessibility_recovery,
    run_joint_operator_probe,
    run_joint_operator_probe_multi_seed,
)

OPERATIONS = ("SHIFT", "SELECT", "COUNT", "BIND")


def _tiny_config(**overrides: object) -> JointOperatorConfig:
    base = JointOperatorConfig(
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
        joint_train=PrimitiveTrainConfig(
            steps=3, batch_size=4, eval_every=1, progress_eval_examples=2
        ),
        num_unseen_eval_groups=20,
        min_unseen_eval_examples=8,
    )
    return dataclasses.replace(base, **overrides) if overrides else base


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
        "device": "cpu",
        "d_operator": 8,
        "n_operator_head": 2,
        "d_operator_ff": 16,
        "arg_dim": 6,
        "max_sequence_length": 8,
        "joint_train": {"steps": 3},
        "num_unseen_eval_groups": 20,
        "min_unseen_eval_examples": 8,
    }
    config = joint_operator_config_from_dict(raw)
    assert config.seed == 2
    assert config.operation_names == ("SHIFT",)
    assert config.model["d_model"] == 16
    assert config.device == "cpu"
    assert config.d_operator == 8
    assert config.joint_train.steps == 3

    defaulted = joint_operator_config_from_dict({})
    assert defaulted == JointOperatorConfig()


def test_default_operation_names_are_the_four_parameterized_operations() -> None:
    assert set(JointOperatorConfig().operation_names) == {"SHIFT", "SELECT", "COUNT", "BIND"}


def test_default_seed_policy_matches_task_text() -> None:
    assert DEFAULT_SEEDS == (0, 1, 2, 3, 4)
    assert MIN_GATE_SEEDS == 5


def test_default_operator_dimensions_match_e005_unchanged() -> None:
    # Critical control (A1-R005E-006A's own task text): "Reuse the same
    # compact operator class and dimensions as E-005. Do not make the
    # operator stronger."
    joint_defaults = JointOperatorConfig()
    compact_defaults = CompactOperatorConfig()
    assert joint_defaults.d_operator == compact_defaults.d_operator
    assert joint_defaults.n_operator_head == compact_defaults.n_operator_head
    assert joint_defaults.d_operator_ff == compact_defaults.d_operator_ff
    assert joint_defaults.arg_dim == compact_defaults.arg_dim
    assert joint_defaults.max_sequence_length == compact_defaults.max_sequence_length


def test_default_model_architecture_matches_e005_frozen_core_unchanged() -> None:
    joint_defaults = JointOperatorConfig()
    compact_defaults = CompactOperatorConfig()
    assert joint_defaults.model == compact_defaults.model


def test_default_joint_train_steps_is_sum_of_e005_two_phase_budget() -> None:
    compact_defaults = CompactOperatorConfig()
    joint_defaults = JointOperatorConfig()
    assert (
        joint_defaults.joint_train.steps
        == compact_defaults.core_train.steps + compact_defaults.operator_train.steps
    )


# --- Core/operator construction and grad-enabled batch features -------------


def test_build_joint_core_is_not_frozen() -> None:
    config = _tiny_config()
    core = _build_joint_core(config)
    assert all(p.requires_grad for p in core.model.parameters())


def test_batch_features_no_grad_false_allows_backprop_into_encoder() -> None:
    config = _tiny_config()
    core = _build_joint_core(config)
    operator = _build_operator(core, config, "SHIFT")
    groups = generate_compact_operator_counterfactual_groups(
        0, 2, operation="SHIFT", step=0, split="train", vocab_size=6, sequence_length_range=(4, 6)
    )
    examples = [example for group in groups for example in group.examples]
    content_features, content_lengths, output_lengths = _batch_features(
        core, examples, "SHIFT", no_grad=False
    )
    assert content_features.requires_grad
    argument_values = [0] * len(examples)
    logits = operator(content_features, content_lengths, output_lengths, argument_values)
    loss = logits.sum()
    loss.backward()
    assert core.model.token_emb.weight.grad is not None
    assert torch.isfinite(core.model.token_emb.weight.grad).all()


def test_batch_features_no_grad_true_disables_backprop() -> None:
    config = _tiny_config()
    core = _build_joint_core(config)
    groups = generate_compact_operator_counterfactual_groups(
        0, 2, operation="SHIFT", step=0, split="train", vocab_size=6, sequence_length_range=(4, 6)
    )
    examples = [example for group in groups for example in group.examples]
    content_features, _, _ = _batch_features(core, examples, "SHIFT", no_grad=True)
    assert not content_features.requires_grad


def test_evaluate_arm_restores_train_mode_on_both_core_and_operator() -> None:
    config = _tiny_config()
    core = _build_joint_core(config)
    operator = _build_operator(core, config, "SHIFT")
    groups = generate_compact_operator_counterfactual_groups(
        0, 2, operation="SHIFT", step=0, split="train", vocab_size=6, sequence_length_range=(4, 6)
    )
    examples = [example for group in groups for example in group.examples]
    _evaluate_arm(core, operator, examples, "SHIFT", argument_provider=lambda e: 0)
    assert core.model.training
    assert operator.training


# --- Joint training actually updates the encoder -----------------------------


def test_train_joint_updates_encoder_parameters() -> None:
    # The whole point of A1-R005E-006A vs A1-R005E-005: the encoder is no
    # longer frozen, so its own parameters must actually change during
    # training, unlike apc.evaluation.compact_cross_position_operator_probe's
    # frozen core.
    config = _tiny_config(joint_train=PrimitiveTrainConfig(steps=5, batch_size=4, eval_every=1))
    core = _build_joint_core(config)
    operator = _build_operator(core, config, "SHIFT")
    before = copy.deepcopy(core.model.token_emb.weight.data)

    final_loss, examples_seen, encoder_grad_norm, operator_grad_norm = _train_joint(
        core, operator, config, "SHIFT"
    )

    after = core.model.token_emb.weight.data
    assert not torch.allclose(before, after)
    assert examples_seen > 0
    assert final_loss == final_loss  # not NaN
    assert encoder_grad_norm >= 0.0
    assert operator_grad_norm >= 0.0


def test_train_joint_writes_metrics_with_grad_norms(tmp_path: Path) -> None:
    config = _tiny_config(joint_train=PrimitiveTrainConfig(steps=2, batch_size=4, eval_every=1))
    core = _build_joint_core(config)
    operator = _build_operator(core, config, "SHIFT")
    metrics_path = tmp_path / "SHIFT_joint_metrics.jsonl"
    _train_joint(core, operator, config, "SHIFT", metrics_path=metrics_path)
    lines = metrics_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        record = json.loads(line)
        assert set(record) == {
            "step",
            "loss",
            "encoder_grad_norm",
            "operator_grad_norm",
            "progress_correct_exact_match",
        }


# --- Task-blind invariance ---------------------------------------------------


def test_task_blind_invariance_is_near_zero_for_trained_core() -> None:
    config = _tiny_config(joint_train=PrimitiveTrainConfig(steps=3, batch_size=4, eval_every=1))
    core = _build_joint_core(config)
    operator = _build_operator(core, config, "SHIFT")
    _train_joint(core, operator, config, "SHIFT")

    groups = generate_compact_operator_counterfactual_groups(
        0, 1, operation="SHIFT", step=0, split="test", vocab_size=6, sequence_length_range=(4, 6)
    )
    max_abs_diff = _task_blind_invariance_max_abs_diff(core, "SHIFT", groups[0])
    assert max_abs_diff <= TASK_BLIND_ATOL
    assert max_abs_diff >= 0.0


# --- R_access ----------------------------------------------------------------


def test_representation_accessibility_recovery_formula() -> None:
    # R_access = (m_C10 - m_C00) / max(eps, m_C01 - m_C00)
    assert representation_accessibility_recovery(0.8, 0.2, 0.9) == pytest.approx(
        (0.8 - 0.2) / (0.9 - 0.2)
    )


def test_representation_accessibility_recovery_uses_eps_floor() -> None:
    # When C01 does not materially exceed C00, the denominator is floored at
    # eps rather than dividing by (near-)zero.
    value = representation_accessibility_recovery(0.5, 0.2, 0.2)
    assert value == pytest.approx((0.5 - 0.2) / R_ACCESS_EPS)


# --- End-to-end tiny benchmark run ------------------------------------------


def test_run_joint_operator_probe_report_shape() -> None:
    config = _tiny_config()
    report = run_joint_operator_probe(config)
    assert set(report.per_operation) == {"SHIFT", "BIND"}
    for operation, op_report in report.per_operation.items():
        assert op_report.operation == operation
        assert 0.0 <= op_report.correct_exact_match <= 1.0
        assert 0.0 <= op_report.effectful_wrong_argument_exact_match <= 1.0
        assert 0.0 <= op_report.none_exact_match <= 1.0
        assert op_report.num_unseen_eval_examples >= config.min_unseen_eval_examples
        assert op_report.operator_param_count > 0
        assert op_report.core_param_count > 0
        assert op_report.core_trainable_param_count == op_report.core_param_count
        assert op_report.task_blind_max_abs_diff >= 0.0
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


def test_run_joint_operator_probe_writes_metrics(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "seed_0"
    config = _tiny_config(operation_names=("SHIFT",))
    run_joint_operator_probe(config, metrics_dir=metrics_dir)
    assert (metrics_dir / "SHIFT_joint_metrics.jsonl").exists()


def test_multi_seed_aggregation_matches_manual_formula_without_r_access() -> None:
    base_config = _tiny_config(operation_names=("SHIFT",))
    seeds = (0, 1)
    result = run_joint_operator_probe_multi_seed(
        base_config, seeds=seeds, e004_summary_path=None, e005_summary_path=None
    )
    assert result.seeds == seeds
    assert result.meets_seed_policy is False  # only 2 < MIN_GATE_SEEDS
    assert result.e004_summary_path is None
    assert result.e005_summary_path is None

    op_reports = [report.per_operation["SHIFT"] for report in result.per_seed]
    summary = result.per_operation_summary["SHIFT"]
    mean_correct = sum(r.correct_exact_match for r in op_reports) / len(op_reports)
    mean_wrong = sum(r.effectful_wrong_argument_exact_match for r in op_reports) / len(op_reports)
    mean_none = sum(r.none_exact_match for r in op_reports) / len(op_reports)
    expected_gap = mean_correct - max(mean_wrong, mean_none)

    assert summary.mean_correct_exact_match == pytest.approx(mean_correct)
    assert summary.mean_exact_match_causal_gap == pytest.approx(expected_gap)
    assert summary.r_access_correct_exact_match is None
    assert summary.r_access_causal_gap is None
    assert summary.c00_frozen_compact_correct_exact_match is None
    assert summary.c01_frozen_high_cap_correct_exact_match is None
    assert result.passed == all(s.passed for s in result.per_operation_summary.values())


def test_multi_seed_aggregation_computes_r_access_from_summary_files(tmp_path: Path) -> None:
    e005_summary = {
        "per_operation_summary": {
            "SHIFT": {"mean_correct_exact_match": 0.1, "mean_exact_match_causal_gap": 0.05}
        }
    }
    e004_summary = {
        "per_operation_summary": {
            "SHIFT": {"mean_correct_exact_match": 0.9, "mean_exact_match_causal_gap": 0.8}
        }
    }
    e005_path = tmp_path / "e005_summary.json"
    e004_path = tmp_path / "e004_summary.json"
    e005_path.write_text(json.dumps(e005_summary), encoding="utf-8")
    e004_path.write_text(json.dumps(e004_summary), encoding="utf-8")

    base_config = _tiny_config(operation_names=("SHIFT",))
    result = run_joint_operator_probe_multi_seed(
        base_config, seeds=(0, 1), e004_summary_path=e004_path, e005_summary_path=e005_path
    )
    summary = result.per_operation_summary["SHIFT"]
    assert summary.c00_frozen_compact_correct_exact_match == pytest.approx(0.1)
    assert summary.c01_frozen_high_cap_correct_exact_match == pytest.approx(0.9)
    expected_r_access_correct = (summary.mean_correct_exact_match - 0.1) / (0.9 - 0.1)
    assert summary.r_access_correct_exact_match == pytest.approx(expected_r_access_correct)
    expected_r_access_gap = (summary.mean_exact_match_causal_gap - 0.05) / (0.8 - 0.05)
    assert summary.r_access_causal_gap == pytest.approx(expected_r_access_gap)


def test_multi_seed_aggregation_raises_on_missing_summary_file(tmp_path: Path) -> None:
    base_config = _tiny_config(operation_names=("SHIFT",))
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError):
        run_joint_operator_probe_multi_seed(
            base_config, seeds=(0,), e004_summary_path=missing, e005_summary_path=missing
        )


def test_threshold_constants_reused_from_e005() -> None:
    from apc.evaluation.joint_representation_compact_operator_probe import (
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
