"""Unit tests for the argument-blind baseline audit (Task A1-R005E-S004)."""

from __future__ import annotations

import pytest

from apc.environments.generator import Example, OracleMetadata
from apc.environments.program import Program, ProgramStep
from apc.environments.task_spec import TaskSpec
from apc.evaluation.argument_blind_baseline_audit import (
    AuditConfig,
    audit_argument_blind_baselines,
    audit_config_from_dict,
    compute_content_only_baselines,
    compute_majority_target_baseline,
    format_audit_table_markdown,
)


def _make_dummy_example(input_tokens: tuple[int, ...], target_tokens: tuple[int, ...]) -> Example:
    prog = Program(steps=(ProgramStep(operation="COUNT", params={"target": 0}),))
    return Example(
        input_tokens=input_tokens,
        target_tokens=target_tokens,
        program=prog,
        operation_graph=None,
        category="known",
        split="test",
        vocab_size=10,
        task_spec=TaskSpec.from_program(prog),
        oracle_metadata=OracleMetadata(label="K", primitive_operations=("COUNT",)),
        symbol_permutation=None,
    )


def test_compute_majority_target_baseline() -> None:
    examples = [
        _make_dummy_example((1, 2, 3), (0,)),
        _make_dummy_example((1, 2, 3), (0,)),
        _make_dummy_example((1, 2, 3), (1,)),
        _make_dummy_example((1, 2, 3), (2,)),
    ]
    base = compute_majority_target_baseline(examples)
    assert base.top_target == (0,)
    assert pytest.approx(base.majority_accuracy) == 0.50
    assert "(0,)" in base.target_frequencies
    assert pytest.approx(base.target_frequencies["(0,)"]) == 0.50


def test_compute_content_only_baselines_count() -> None:
    examples = [
        # Content has zero 0s, target is 0
        _make_dummy_example((1, 2, 3, 4), (0,)),
        # Content has two 1s, target is 2
        _make_dummy_example((1, 1, 3, 4), (2,)),
        # Content has zero 0s, target is 0
        _make_dummy_example((5, 6, 7, 8), (0,)),
    ]
    baselines = compute_content_only_baselines("COUNT", examples, vocab_size=10)
    assert "predict_count_zero" in baselines.heuristics
    assert pytest.approx(baselines.heuristics["predict_count_zero"]) == 2 / 3
    assert "predict_content_mode_count" in baselines.heuristics
    assert baselines.max_content_only_accuracy >= 2 / 3


def test_compute_content_only_baselines_bind() -> None:
    # Key-value pairs: (key0, val0, key1, val1)
    examples = [
        # Last pair value is 7
        _make_dummy_example((1, 4, 2, 7), (7,)),
        # Last pair value is 9
        _make_dummy_example((3, 2, 5, 9), (9,)),
        # Last pair value is 8, but target is 1
        _make_dummy_example((0, 1, 4, 8), (1,)),
    ]
    baselines = compute_content_only_baselines("BIND", examples, vocab_size=10)
    assert "predict_last_pair_value" in baselines.heuristics
    assert pytest.approx(baselines.heuristics["predict_last_pair_value"]) == 2 / 3
    assert "predict_first_pair_value" in baselines.heuristics
    assert pytest.approx(baselines.heuristics["predict_first_pair_value"]) == 1 / 3
    assert "uniform_random_present_pair" in baselines.heuristics
    # 2 pairs per example -> expected random accuracy is 0.50
    assert pytest.approx(baselines.heuristics["uniform_random_present_pair"]) == 0.50


def test_audit_config_roundtrip() -> None:
    cfg = AuditConfig(
        seeds=(0, 1),
        num_eval_groups=100,
        vocab_size=10,
        sequence_length_range=(6, 8),
        operations=("COUNT",),
        historical_none_ceiling=0.25,
    )
    d = cfg.to_dict()
    restored = audit_config_from_dict(d)
    assert restored.seeds == (0, 1)
    assert restored.num_eval_groups == 100
    assert restored.operations == ("COUNT",)
    assert restored.historical_none_ceiling == 0.25


def test_audit_argument_blind_baselines_end_to_end() -> None:
    # Lightweight end-to-end check
    cfg = AuditConfig(
        seeds=(0,),
        num_eval_groups=50,
        operations=("COUNT", "BIND"),
    )
    report = audit_argument_blind_baselines(cfg)

    assert "COUNT" in report.per_operation_audit
    assert "BIND" in report.per_operation_audit

    count_audit = report.per_operation_audit["COUNT"]
    bind_audit = report.per_operation_audit["BIND"]

    # In COUNT, target 0 is the majority class and its accuracy > 0.30
    assert count_audit.majority_baseline.majority_accuracy > 0.30
    assert count_audit.ceiling_is_below_natural_baseline is True

    # In BIND, predicting the last pair value achieves > 0.30
    assert bind_audit.content_only_baselines.max_content_only_accuracy > 0.30
    assert bind_audit.ceiling_is_below_natural_baseline is True

    table_md = format_audit_table_markdown(report.per_operation_audit)
    assert "| Operation | Majority Target Base |" in table_md
    assert "| COUNT |" in table_md
    assert "| BIND |" in table_md

    assert "Formal Recommendation" in report.recommendations
    assert "baseline-relative criterion" in report.recommendations
