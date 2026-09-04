"""Unit tests for shared vs specialized retention analysis (Task A1-R005E-S003)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from apc.evaluation.shared_vs_specialized_retention_analysis import (
    DEFAULT_E004_SUMMARY_PATH,
    DEFAULT_E005_SUMMARY_PATH,
    DEFAULT_E006A_SUMMARY_PATH,
    DEFAULT_S002_SUMMARY_PATH,
    RetentionConfig,
    analyze_shared_vs_specialized_retention,
    calculate_retention,
    classify_retention_band,
    format_evidence_table_markdown,
    load_summary,
    retention_config_from_dict,
)


def test_calculate_retention_basic() -> None:
    # Exact match ratio
    assert pytest.approx(calculate_retention(0.5, 0.5), rel=1e-5) == 1.0
    assert pytest.approx(calculate_retention(0.9, 1.0), rel=1e-5) == 0.9
    assert pytest.approx(calculate_retention(0.35, 0.5), rel=1e-5) == 0.7

    # Epsilon protection against zero / near-zero specialized denominator
    res_zero = calculate_retention(0.5, 0.0, eps=1e-4)
    assert pytest.approx(res_zero, rel=1e-5) == 0.5 / 1e-4


def test_classify_retention_band() -> None:
    # Strong band (>= 0.90)
    assert classify_retention_band(1.0) == "strong"
    assert classify_retention_band(0.95) == "strong"
    assert classify_retention_band(0.90) == "strong"

    # Mixed band (0.70 <= ratio < 0.90)
    assert classify_retention_band(0.899999) == "mixed"
    assert classify_retention_band(0.80) == "mixed"
    assert classify_retention_band(0.70) == "mixed"

    # Specialized band (< 0.70)
    assert classify_retention_band(0.699999) == "specialized"
    assert classify_retention_band(0.50) == "specialized"
    assert classify_retention_band(0.0) == "specialized"


def test_retention_config_roundtrip() -> None:
    cfg = RetentionConfig(
        s002_summary_path="dummy_s002.json",
        e006a_summary_path="dummy_e006a.json",
        e005_summary_path="dummy_e005.json",
        e004_summary_path="dummy_e004.json",
        operation_names=("SHIFT", "SELECT"),
        eps=1e-5,
    )
    d = cfg.to_dict()
    assert d["s002_summary_path"] == "dummy_s002.json"
    assert d["operation_names"] == ["SHIFT", "SELECT"]
    assert d["eps"] == 1e-5

    restored = retention_config_from_dict(d)
    assert restored == cfg


def test_load_summary_missing_file() -> None:
    with pytest.raises(FileNotFoundError, match="Summary file not found"):
        load_summary("non_existent_summary_12345.json")


def _make_mock_summary(
    seeds: list[int],
    op_metrics: dict[str, dict[str, float]],
) -> dict[str, Any]:
    return {
        "seeds": seeds,
        "per_operation_summary": {
            op: {
                "operation": op,
                "mean_correct_exact_match": metrics.get("correct", 0.5),
                "stdev_correct_exact_match": metrics.get("stdev", 0.01),
                "min_correct_exact_match": metrics.get("min_correct", 0.45),
                "max_correct_exact_match": metrics.get("max_correct", 0.55),
                "mean_correct_token_accuracy": metrics.get("token_acc", 0.8),
                "mean_effectful_wrong_argument_exact_match": metrics.get("wrong", 0.0),
                "mean_none_exact_match": metrics.get("none", 0.05),
                "mean_exact_match_causal_gap": metrics.get("gap", 0.45),
                "mean_token_accuracy_causal_gap": metrics.get("token_gap", 0.6),
                "task_blind_invariant_passed": True,
                "mean_argument_effect_rate": 1.0,
                "operator_param_count": 18000,
            }
            for op, metrics in op_metrics.items()
        },
    }


def test_analyze_shared_vs_specialized_retention_mock(tmp_path: Path) -> None:
    seeds = [0, 1, 2, 3, 4]
    s002_data = _make_mock_summary(
        seeds,
        {
            "SHIFT": {"correct": 0.50, "gap": 0.50, "stdev": 0.02},
            "SELECT": {"correct": 0.99, "gap": 0.95, "stdev": 0.001},
            "COUNT": {"correct": 0.99, "gap": 0.67, "stdev": 0.005},
            "BIND": {"correct": 0.99, "gap": 0.69, "stdev": 0.001},
        },
    )
    e006a_data = _make_mock_summary(
        seeds,
        {
            "SHIFT": {"correct": 0.50, "gap": 0.50, "stdev": 0.10},
            "SELECT": {"correct": 1.00, "gap": 0.96, "stdev": 0.000},
            "COUNT": {"correct": 0.99, "gap": 0.67, "stdev": 0.002},
            "BIND": {"correct": 0.99, "gap": 0.67, "stdev": 0.001},
        },
    )
    e005_data = _make_mock_summary(
        seeds,
        {
            "SHIFT": {"correct": 0.04, "gap": 0.04},
            "SELECT": {"correct": 0.33, "gap": 0.31},
            "COUNT": {"correct": 0.47, "gap": 0.18},
            "BIND": {"correct": 0.37, "gap": 0.07},
        },
    )
    e004_data = _make_mock_summary(
        seeds,
        {
            "SHIFT": {"correct": 0.62, "gap": 0.62},
            "SELECT": {"correct": 0.92, "gap": 0.89},
            "COUNT": {"correct": 0.72, "gap": 0.40},
            "BIND": {"correct": 0.88, "gap": 0.59},
        },
    )

    s002_path = tmp_path / "s002.json"
    e006a_path = tmp_path / "e006a.json"
    e005_path = tmp_path / "e005.json"
    e004_path = tmp_path / "e004.json"

    s002_path.write_text(json.dumps(s002_data), encoding="utf-8")
    e006a_path.write_text(json.dumps(e006a_data), encoding="utf-8")
    e005_path.write_text(json.dumps(e005_data), encoding="utf-8")
    e004_path.write_text(json.dumps(e004_data), encoding="utf-8")

    cfg = RetentionConfig(
        s002_summary_path=str(s002_path),
        e006a_summary_path=str(e006a_path),
        e005_summary_path=str(e005_path),
        e004_summary_path=str(e004_path),
    )
    report = analyze_shared_vs_specialized_retention(cfg)

    assert report.meets_seed_policy is True
    assert report.global_strong_support is True
    assert report.num_operations_strong == 4
    assert report.num_operations_mixed == 0
    assert report.num_operations_specialized == 0

    shift = report.per_operation_summary["SHIFT"]
    assert pytest.approx(shift.r_shared_correct, rel=1e-3) == 1.0
    assert pytest.approx(shift.r_shared_gap, rel=1e-3) == 1.0
    assert shift.overall_retention_band == "strong"
    assert shift.passed_strong_shared_support is True
    assert shift.variance_reduction_ratio == 5.0  # 0.10 / 0.02


def test_analyze_real_summaries() -> None:
    if not (
        DEFAULT_S002_SUMMARY_PATH.exists()
        and DEFAULT_E006A_SUMMARY_PATH.exists()
        and DEFAULT_E005_SUMMARY_PATH.exists()
        and DEFAULT_E004_SUMMARY_PATH.exists()
    ):
        pytest.skip("Real summary artifacts not present in runs/")

    cfg = RetentionConfig()
    report = analyze_shared_vs_specialized_retention(cfg)

    assert report.meets_seed_policy is True
    assert report.global_strong_support is True
    assert report.num_operations_strong == 4
    assert len(report.per_operation_summary) == 4

    for op in ("SHIFT", "SELECT", "COUNT", "BIND"):
        s = report.per_operation_summary[op]
        assert s.r_shared_correct >= 0.90
        assert s.r_shared_gap >= 0.90
        assert s.overall_retention_band == "strong"
        assert s.passed_strong_shared_support is True

    # Check Markdown table contains key content
    table = report.evidence_table_markdown
    assert "| Operation | S002 Correct | E-006A Correct |" in table
    assert "| SHIFT |" in table
    assert "| SELECT |" in table
    assert "| COUNT |" in table
    assert "| BIND |" in table


def test_format_evidence_table_markdown(tmp_path: Path) -> None:
    seeds = [0, 1, 2, 3, 4]
    mock_data = _make_mock_summary(
        seeds,
        {"SELECT": {"correct": 0.99, "gap": 0.95}},
    )
    p = tmp_path / "mock.json"
    p.write_text(json.dumps(mock_data), encoding="utf-8")
    cfg = RetentionConfig(
        s002_summary_path=str(p),
        e006a_summary_path=str(p),
        e005_summary_path=str(p),
        e004_summary_path=str(p),
        operation_names=("SELECT",),
    )
    report = analyze_shared_vs_specialized_retention(cfg)
    table = format_evidence_table_markdown(report.per_operation_summary)
    assert "| SELECT |" in table
    assert "**100.00%**" in table
