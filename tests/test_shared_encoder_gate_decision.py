"""Unit tests for the shared encoder gate branch decision (Task A1-R005E-S005)."""

from __future__ import annotations

import json
from typing import Any

from apc.evaluation.shared_encoder_gate_decision import (
    DEFAULT_S003_SUMMARY_PATH,
    DEFAULT_S004_SUMMARY_PATH,
    evaluate_shared_encoder_gate_decision,
)


def _make_mock_s003(
    shift_correct: float = 0.5193,
    num_strong: int = 4,
    num_specialized: int = 0,
    scb_retention: float = 0.998,
) -> dict[str, Any]:
    return {
        "num_operations_strong": num_strong,
        "num_operations_mixed": 0,
        "num_operations_specialized": num_specialized,
        "per_operation_summary": {
            "SHIFT": {
                "operation": "SHIFT",
                "s002_correct_exact_match": shift_correct,
                "s002_causal_gap": shift_correct,
                "r_shared_correct": 0.9998,
                "r_shared_gap": 1.0,
                "overall_retention_band": "strong" if num_specialized == 0 else "specialized",
            },
            "SELECT": {
                "operation": "SELECT",
                "s002_correct_exact_match": 0.9998,
                "s002_causal_gap": 0.9605,
                "r_shared_correct": scb_retention,
                "r_shared_gap": scb_retention,
                "overall_retention_band": "strong",
            },
            "COUNT": {
                "operation": "COUNT",
                "s002_correct_exact_match": 0.9955,
                "s002_causal_gap": 0.6708,
                "r_shared_correct": scb_retention,
                "r_shared_gap": scb_retention,
                "overall_retention_band": "strong",
            },
            "BIND": {
                "operation": "BIND",
                "s002_correct_exact_match": 0.9998,
                "s002_causal_gap": 0.6986,
                "r_shared_correct": scb_retention,
                "r_shared_gap": scb_retention,
                "overall_retention_band": "strong",
            },
        },
    }


def _make_mock_s004() -> dict[str, Any]:
    return {
        "per_operation": {
            "COUNT": {
                "ceiling_is_below_natural_baseline": True,
                "natural_argument_blind_baseline": 0.3356,
                "recommended_relative_threshold": 0.3856,
            },
            "BIND": {
                "ceiling_is_below_natural_baseline": True,
                "natural_argument_blind_baseline": 0.3400,
                "recommended_relative_threshold": 0.3900,
            },
        }
    }


def test_outcome_b_broad_success_except_shift() -> None:
    s003 = _make_mock_s003(shift_correct=0.5193)
    s004 = _make_mock_s004()

    report = evaluate_shared_encoder_gate_decision(s003, s004)
    assert report.outcome == "B"
    assert report.branch_b_supported is True
    assert report.trigger_s006_shift_probe is True
    assert report.adopt_baseline_relative_none is True
    assert "Outcome B" in report.decision_summary_markdown


def test_outcome_a_clean_sweep() -> None:
    s003 = _make_mock_s003(shift_correct=0.95)
    s004 = _make_mock_s004()

    report = evaluate_shared_encoder_gate_decision(s003, s004)
    assert report.outcome == "A"
    assert report.branch_b_supported is True
    assert report.trigger_s006_shift_probe is False


def test_outcome_c_multi_op_collapse() -> None:
    s003 = _make_mock_s003(num_specialized=2)
    s004 = _make_mock_s004()

    report = evaluate_shared_encoder_gate_decision(s003, s004)
    assert report.outcome == "C"
    assert report.branch_b_supported is False
    assert report.trigger_s006_shift_probe is False


def test_outcome_d_scb_failure() -> None:
    s003 = _make_mock_s003(scb_retention=0.65)
    s004 = _make_mock_s004()

    report = evaluate_shared_encoder_gate_decision(s003, s004)
    assert report.outcome == "D"
    assert report.branch_b_supported is False
    assert report.trigger_s006_shift_probe is False


def test_real_summaries_decision() -> None:
    if not (DEFAULT_S003_SUMMARY_PATH.exists() and DEFAULT_S004_SUMMARY_PATH.exists()):
        return

    s003_data = json.loads(DEFAULT_S003_SUMMARY_PATH.read_text(encoding="utf-8"))
    s004_data = json.loads(DEFAULT_S004_SUMMARY_PATH.read_text(encoding="utf-8"))

    report = evaluate_shared_encoder_gate_decision(s003_data, s004_data)
    assert report.outcome == "B"
    assert report.branch_b_supported is True
    assert report.trigger_s006_shift_probe is True
    assert report.adopt_baseline_relative_none is True
    assert "Triggers S006 Operator Probe" in report.decision_summary_markdown
