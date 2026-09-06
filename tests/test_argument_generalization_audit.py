# ruff: noqa: E501
"""Focused invariant coverage for B-C005D2-004 (no GPU checkpoints required)."""

from __future__ import annotations

import pytest

from apc.evaluation.argument_generalization_audit import (
    ArgumentGeneralizationAuditConfig,
    _argument_distance,
    _argument_structure_breakdown,
    _calibration_summary,
    _classify_operations,
    _content_dependent_ambiguity,
    _data_scale_summary,
    _group_metrics,
    _operation_breakdown,
    _training_values,
)


def _config(**overrides: object) -> ArgumentGeneralizationAuditConfig:
    return ArgumentGeneralizationAuditConfig(**overrides)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Config validation.
# ---------------------------------------------------------------------------


def test_config_rejects_development_overlap_with_sealed_partitions() -> None:
    with pytest.raises(ValueError, match="development seeds"):
        _config(development_seeds=(20,), regate_sealed_seeds=(21,))


def test_config_rejects_non_standard_bank_size() -> None:
    with pytest.raises(ValueError, match="bank_size"):
        _config(bank_size=17)


def test_config_rejects_out_of_range_threshold() -> None:
    with pytest.raises(ValueError, match="thresholds"):
        _config(no_failure_threshold=1.5)


def test_config_rejects_non_positive_probe_examples() -> None:
    with pytest.raises(ValueError, match="probe_train_examples"):
        _config(probe_train_examples=0)


# ---------------------------------------------------------------------------
# Structural argument descriptors (D2-004.2).
# ---------------------------------------------------------------------------


def test_argument_distance_is_minimal_circular_step_for_scalar_operations() -> None:
    # _wrong_arguments always adds exactly one modulo the domain for the three
    # scalar operations, so the circular distance should always be 1.
    assert _argument_distance("SHIFT", [3], [4], modulus=8) == 1.0
    assert _argument_distance("COUNT", [9], [0], modulus=10) == 1.0  # wraparound
    assert _argument_distance("BIND", [0], [1], modulus=10) == 1.0


def test_argument_distance_for_select_is_symmetric_difference_size() -> None:
    assert _argument_distance("SELECT", [1, 3], [2, 4], modulus=8) == 4.0
    assert _argument_distance("SELECT", [1, 3], [1, 4], modulus=8) == 2.0


class _FakeExample:
    def __init__(self, input_tokens: tuple[int, ...]) -> None:
        self.input_tokens = input_tokens


def test_content_dependent_ambiguity_count_checks_token_presence() -> None:
    example = _FakeExample((1, 2, 3, 2))
    assert _content_dependent_ambiguity("COUNT", example, [2]) is True
    assert _content_dependent_ambiguity("COUNT", example, [9]) is False


def test_content_dependent_ambiguity_bind_checks_key_positions_only() -> None:
    # keys live at even positions (0, 2); value 2 only appears at an odd
    # (value) position, so it should NOT count as a content-plausible key.
    example = _FakeExample((1, 5, 3, 2))
    assert _content_dependent_ambiguity("BIND", example, [1]) is True
    assert _content_dependent_ambiguity("BIND", example, [2]) is False


def test_content_dependent_ambiguity_is_none_for_position_arguments() -> None:
    example = _FakeExample((1, 2, 3))
    assert _content_dependent_ambiguity("SHIFT", example, [1]) is None
    assert _content_dependent_ambiguity("SELECT", example, [1]) is None


class _FakeTaskSpec:
    def __init__(self, arguments: dict[str, object]) -> None:
        self.steps = [_FakeStep(arguments)]


class _FakeStep:
    def __init__(self, arguments: dict[str, object]) -> None:
        self.arguments = arguments


class _FakeTrainingExample:
    def __init__(self, arguments: dict[str, object]) -> None:
        self.task_spec = _FakeTaskSpec(arguments)


def test_training_values_flattens_raw_argument_values() -> None:
    examples = {
        "COUNT": [
            _FakeTrainingExample({"target": 3}),
            _FakeTrainingExample({"target": 3}),
            _FakeTrainingExample({"target": 5}),
        ]
    }
    assert _training_values(examples, "COUNT") == [3, 3, 5]
    assert _training_values(examples, "BIND") == []


# ---------------------------------------------------------------------------
# Aggregation helpers (D2-004.1 / D2-004.2 / D2-004.3).
# ---------------------------------------------------------------------------


def _row(
    operation: str,
    partition: str = "regate_sealed",
    state: str = "R2_frozen_post_repair",
    *,
    argument_correct: bool,
    top5: bool = True,
    family_top1: bool = True,
    family_margin: float = 1.0,
    combined_margin: float = 1.0,
    argument_score_margin: float | None = 0.5,
    p_correct: float | None = 0.6,
    p_specific_wrong: float | None = 0.4,
    p_best_wrong: float | None = 0.4,
    seen_in_training: bool | None = True,
    content_dependent_ambiguity: bool | None = None,
    sequence_length: int = 8,
    argument_distance: float = 1.0,
    argument_value_type: str = "single_valued",
) -> dict[str, object]:
    return {
        "partition": partition,
        "router_state": state,
        "operation": operation,
        "family_top1": family_top1,
        "argument_correct": argument_correct,
        "primitive_call_top1": argument_correct,
        "top5_included": top5,
        "family_score_margin": family_margin,
        "combined_score_margin": combined_margin,
        "argument_score_margin": argument_score_margin,
        "p_correct_argument": p_correct,
        "p_specific_wrong_argument": p_specific_wrong,
        "p_best_wrong_argument": p_best_wrong,
        "seen_in_training": seen_in_training,
        "content_dependent_ambiguity": content_dependent_ambiguity,
        "sequence_length": sequence_length,
        "argument_distance": argument_distance,
        "argument_value_type": argument_value_type,
    }


def test_group_metrics_reports_none_for_empty_group() -> None:
    metrics = _group_metrics([])
    assert metrics == {"n": 0, "argument_accuracy": None, "primitive_call_top1": None}


def test_operation_breakdown_splits_by_cell_and_operation() -> None:
    rows = [
        _row("BIND", argument_correct=True),
        _row("BIND", argument_correct=False),
        _row("COUNT", partition="development", state="R0_frozen_pre_repair", argument_correct=True, argument_score_margin=None, p_correct=None, p_specific_wrong=None, p_best_wrong=None),
    ]
    breakdown = _operation_breakdown(rows)
    assert breakdown["regate_sealed/R2_frozen_post_repair"]["BIND"]["argument_accuracy"] == pytest.approx(0.5)
    assert breakdown["regate_sealed/R2_frozen_post_repair"]["BIND"]["n"] == 2
    assert breakdown["development/R0_frozen_pre_repair"]["COUNT"]["argument_score_margin_mean"] is None


def test_argument_structure_breakdown_splits_seen_vs_rare() -> None:
    rows = [
        _row("BIND", argument_correct=True, seen_in_training=True),
        _row("BIND", argument_correct=True, seen_in_training=True),
        _row("BIND", argument_correct=False, seen_in_training=False),
    ]
    structure = _argument_structure_breakdown(rows)
    cell = structure["seen_vs_rare"]["regate_sealed/R2_frozen_post_repair"]["BIND"]
    assert cell["seen"]["argument_accuracy"] == pytest.approx(1.0)
    assert cell["rare"]["argument_accuracy"] == pytest.approx(0.0)
    assert cell["seen"]["n"] == 2
    assert cell["rare"]["n"] == 1


def test_argument_structure_breakdown_content_ambiguity_only_covers_count_and_bind() -> None:
    rows = [
        _row("COUNT", argument_correct=True, content_dependent_ambiguity=True),
        _row("COUNT", argument_correct=False, content_dependent_ambiguity=False),
        _row("SHIFT", argument_correct=True, content_dependent_ambiguity=None),
    ]
    structure = _argument_structure_breakdown(rows)
    ambiguity = structure["content_dependent_ambiguity"]["regate_sealed/R2_frozen_post_repair"]
    assert "SHIFT" not in ambiguity
    assert ambiguity["COUNT"]["content_plausible_wrong_argument"]["argument_accuracy"] == pytest.approx(1.0)
    assert ambiguity["COUNT"]["content_absent_wrong_argument"]["argument_accuracy"] == pytest.approx(0.0)


def test_calibration_summary_splits_by_correctness() -> None:
    rows = [
        _row("BIND", argument_correct=True, p_correct=0.9),
        _row("BIND", argument_correct=False, p_correct=0.2),
    ]
    calibration = _calibration_summary(rows)
    cell = calibration["regate_sealed/R2_frozen_post_repair"]["BIND"]
    assert cell["mean_p_correct_when_argument_correct"] == pytest.approx(0.9)
    assert cell["mean_p_correct_when_argument_incorrect"] == pytest.approx(0.2)


def test_calibration_summary_skips_pre_repair_rows_with_no_scorer() -> None:
    rows = [_row("BIND", state="R0_frozen_pre_repair", argument_correct=True, p_correct=None, p_specific_wrong=None, p_best_wrong=None, argument_score_margin=None)]
    calibration = _calibration_summary(rows)
    assert calibration == {}


# ---------------------------------------------------------------------------
# D2-004.4 failure classification.
# ---------------------------------------------------------------------------


def test_classify_operations_no_failure_when_sealed_accuracy_is_high() -> None:
    config = _config()
    breakdown = {"regate_sealed/R2_frozen_post_repair": {"BIND": {"argument_accuracy": 0.97}}}
    result = _classify_operations(breakdown, {}, {}, config)
    assert result["BIND"]["classification"] == "NO_FAILURE"


def test_classify_operations_value_coverage_failure_when_rare_values_lag() -> None:
    config = _config()
    breakdown = {"regate_sealed/R2_frozen_post_repair": {"COUNT": {"argument_accuracy": 0.5}}}
    structure = {
        "seen_vs_rare": {
            "regate_sealed/R2_frozen_post_repair": {
                "COUNT": {
                    "seen": {"argument_accuracy": 0.9, "n": 20},
                    "rare": {"argument_accuracy": 0.2, "n": 20},
                }
            }
        }
    }
    result = _classify_operations(breakdown, structure, {}, config)
    assert result["COUNT"]["classification"] == "VALUE_COVERAGE_FAILURE"


def test_classify_operations_select_gets_argument_encoding_failure() -> None:
    config = _config()
    breakdown = {"regate_sealed/R2_frozen_post_repair": {"SELECT": {"argument_accuracy": 0.5}}}
    result = _classify_operations(breakdown, {}, {}, config)
    assert result["SELECT"]["classification"] == "ARGUMENT_ENCODING_FAILURE"
    assert "structural_note" in result["SELECT"]["evidence"]


def test_classify_operations_generalization_failure_when_data_scale_helps() -> None:
    config = _config()
    breakdown = {"regate_sealed/R2_frozen_post_repair": {"BIND": {"argument_accuracy": 0.5}}}
    data_scale = {"BIND": {"mean_improvement": 0.2}}
    result = _classify_operations(breakdown, {}, data_scale, config)
    assert result["BIND"]["classification"] == "ARGUMENT_SCORER_GENERALIZATION_FAILURE"


def test_classify_operations_task_representation_failure_when_dev_gap_persists_without_data_help() -> None:
    config = _config()
    breakdown = {
        "regate_sealed/R2_frozen_post_repair": {"BIND": {"argument_accuracy": 0.5}},
        "development/R2_frozen_post_repair": {"BIND": {"argument_accuracy": 0.97}},
    }
    data_scale = {"BIND": {"mean_improvement": 0.0}}
    result = _classify_operations(breakdown, {}, data_scale, config)
    assert result["BIND"]["classification"] == "TASK_REPRESENTATION_FAILURE"


def test_classify_operations_unresolved_when_no_sealed_rows() -> None:
    config = _config()
    result = _classify_operations({}, {}, {}, config)
    assert result["BIND"]["classification"] == "UNRESOLVED"


def test_data_scale_summary_reports_mean_improvement() -> None:
    rows = [
        {
            "operation": "BIND",
            "actual_r2_argument_accuracy": 0.5,
            "data_scale_probe_argument_accuracy": 0.7,
        },
        {
            "operation": "BIND",
            "actual_r2_argument_accuracy": 0.6,
            "data_scale_probe_argument_accuracy": 0.6,
        },
    ]
    summary = _data_scale_summary(rows)
    assert summary["BIND"]["mean_improvement"] == pytest.approx(0.1)
