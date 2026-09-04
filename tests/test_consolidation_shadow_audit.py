"""Unit tests for Task A1-B007X-001 Consolidation Metric and Shadow Audit."""

from __future__ import annotations

from pathlib import Path

import pytest

from apc.evaluation.composition_library_benchmark import DESIGNATED_COMPOSITIONS
from apc.evaluation.consolidation_shadow_audit import (
    ConsolidationShadowAuditConfig,
    _audit_b006_parameters,
    run_consolidation_shadow_audit,
    shadow_audit_config_from_dict,
)
from apc.evaluation.unified_oracle_causal_benchmark import ALL_CANONICAL_OPERATIONS


def test_shadow_audit_config_roundtrip() -> None:
    """Verify configuration serialization and parsing roundtrip."""
    cfg = ConsolidationShadowAuditConfig(
        seed=42,
        vocab_size=12,
        num_canonical_eval_examples=50,
        num_composition_eval_examples=40,
        max_forgetting_threshold=0.015,
    )
    d = cfg.to_dict()
    assert d["seed"] == 42
    assert d["vocab_size"] == 12
    assert d["num_canonical_eval_examples"] == 50
    assert d["num_composition_eval_examples"] == 40
    assert d["max_forgetting_threshold"] == 0.015

    parsed = shadow_audit_config_from_dict(d)
    assert parsed.seed == 42
    assert parsed.vocab_size == 12
    assert parsed.num_canonical_eval_examples == 50
    assert parsed.num_composition_eval_examples == 40
    assert parsed.max_forgetting_threshold == 0.015


def test_b006_parameter_audit_existing_run(tmp_path: Path) -> None:
    """Verify parameter audit from mock or existing B006 run artifacts."""
    # Test from real directory if present
    real_b006_dir = Path("runs/phase_a1_consolidation_benchmark")
    audit = _audit_b006_parameters(real_b006_dir, seed=0)
    assert audit.temporary_parameter_count == 17098
    assert audit.candidate_parameter_count == 17098
    assert audit.parameter_ratio == pytest.approx(1.0, abs=1e-4)
    assert audit.classification == "functional_consolidation"
    assert audit.is_parameter_compression is False

    # Test fallback when directory does not exist
    fallback_audit = _audit_b006_parameters(tmp_path / "nonexistent", seed=99)
    assert fallback_audit.b006_artifact_present is False
    assert fallback_audit.temporary_parameter_count == 17098
    assert fallback_audit.candidate_parameter_count == 17098
    assert fallback_audit.parameter_ratio == 1.0
    assert fallback_audit.classification == "functional_consolidation"
    assert fallback_audit.is_parameter_compression is False


def test_run_consolidation_shadow_audit_tiny_cpu(tmp_path: Path) -> None:
    """Run tiny end-to-end shadow audit on CPU.

    Ensures all 8 canonical ops and 6 compositions are evaluated.
    """
    tiny_cfg = ConsolidationShadowAuditConfig(
        seed=123,
        vocab_size=10,
        sequence_length_range=(6, 8),
        group_size=2,
        model={
            "d_model": 32,
            "n_layer": 1,
            "n_head": 2,
            "d_ff": 64,
            "max_seq_len": 24,
            "dropout": 0.0,
        },
        device="cpu",
        d_operator=16,
        n_operator_head=2,
        d_operator_ff=32,
        max_sequence_length=16,
        num_canonical_eval_examples=4,
        num_composition_eval_examples=4,
        max_forgetting_threshold=0.50,  # generous for untrained tiny test
        core_train_steps=4,
        bank_train_steps=4,
        b006_benchmark_dir=str(tmp_path / "nonexistent"),
    )

    report = run_consolidation_shadow_audit(tiny_cfg, seed_dir=tmp_path / "seed_123")

    # Verify B006 parameter audit
    assert report.b006_parameter_audit.temporary_parameter_count == 17098
    assert report.b006_parameter_audit.candidate_parameter_count == 17098
    assert report.b006_parameter_audit.parameter_ratio == 1.0
    assert report.b006_parameter_audit.classification == "functional_consolidation"

    # Verify all 8 canonical operations are present individually
    assert len(report.canonical_metrics) == 8
    for op in ALL_CANONICAL_OPERATIONS:
        assert op in report.canonical_metrics
        metric = report.canonical_metrics[op]
        assert 0.0 <= metric.before_exact_match <= 1.0
        assert 0.0 <= metric.after_exact_match <= 1.0
        assert 0.0 <= metric.forgetting <= 1.0

    # Verify all 6 representative compositions are present individually
    assert len(report.composition_metrics) == len(DESIGNATED_COMPOSITIONS)
    for comp in DESIGNATED_COMPOSITIONS:
        comp_name = f"{comp[0]}->{comp[1]}"
        assert comp_name in report.composition_metrics
        metric = report.composition_metrics[comp_name]
        assert 0.0 <= metric.before_exact_match <= 1.0
        assert 0.0 <= metric.after_exact_match <= 1.0
        assert 0.0 <= metric.forgetting <= 1.0

    # Check report dictionary serialization
    rep_dict = report.to_dict()
    assert "canonical_metrics" in rep_dict
    assert "composition_metrics" in rep_dict
    assert "b006_parameter_audit" in rep_dict
