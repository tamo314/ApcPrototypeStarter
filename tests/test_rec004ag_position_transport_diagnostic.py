"""Focused invariants for REC-004AG position transport diagnostic."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def _module() -> object:
    path = Path("scripts/diagnose_rec004ag_position_transport.py")
    spec = importlib.util.spec_from_file_location("rec004ag", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_architecture_coordinate_mapping_determinism_and_values() -> None:
    module = _module()
    mapping, details = module.compute_architecture_coordinate_mapping(
        target_length=10, source_length=9
    )

    expected = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 4,
        6: 5,
        7: 6,
        8: 7,
        9: 8,
    }
    assert mapping == expected
    assert len(details) == 10

    # Ensure no oracle or target token references
    for row in details:
        assert "target_token" not in row
        assert "oracle" not in row
        assert "prediction" not in row

    # Verify tie-breaking: target 4 (4/9=0.4444) and target 5 (5/9=0.5556)
    # both map to source 4 (4/8=0.5)
    assert details[4]["source_position"] == 4
    assert details[5]["source_position"] == 4


def test_selection_evidence_threshold_enforcement() -> None:
    module = _module()
    thresholds = {
        "routing_improvement_min": 0.05,
        "margin_improvement_min": 0.25,
        "direct_error_recovery_min": 0.05,
    }

    # Case 1: Below threshold
    failing_metrics = {
        "normal_validation_length10": {
            "baseline": {
                "top1_correct_key_routing_rate": 0.44,
                "correct_key_margin": -7.0,
                "sequence_em": 0.08,
            },
            "position_transport_cf": {
                "top1_correct_key_routing_rate": 0.30,  # lower
                "correct_key_margin": -7.5,  # lower
                "sequence_em": 0.00,  # lower
                "direct_error_token_recovery_rate": 0.03,
            },
        },
        "length10_confirmation": {
            "baseline": {
                "top1_correct_key_routing_rate": 0.44,
                "correct_key_margin": -7.1,
                "sequence_em": 0.08,
            },
            "position_transport_cf": {
                "top1_correct_key_routing_rate": 0.29,
                "correct_key_margin": -7.5,
                "sequence_em": 0.00,
                "direct_error_token_recovery_rate": 0.03,
            },
        },
    }
    decision, evidence = module.evaluate_selection_evidence(failing_metrics, thresholds)
    assert decision == "POSITION_ROUTING_TARGET_NOT_SUPPORTED"
    assert not evidence["all_passed"]

    # Case 2: Above threshold
    passing_metrics = {
        "normal_validation_length10": {
            "baseline": {
                "top1_correct_key_routing_rate": 0.40,
                "correct_key_margin": -7.0,
                "sequence_em": 0.08,
            },
            "position_transport_cf": {
                "top1_correct_key_routing_rate": 0.50,  # +0.10
                "correct_key_margin": -6.5,  # +0.50
                "sequence_em": 0.15,  # > 0.08
                "direct_error_token_recovery_rate": 0.20,
            },
        },
        "length10_confirmation": {
            "baseline": {
                "top1_correct_key_routing_rate": 0.40,
                "correct_key_margin": -7.0,
                "sequence_em": 0.08,
            },
            "position_transport_cf": {
                "top1_correct_key_routing_rate": 0.48,  # +0.08
                "correct_key_margin": -6.6,  # +0.40
                "sequence_em": 0.12,  # > 0.08
                "direct_error_token_recovery_rate": 0.15,
            },
        },
    }
    decision_pass, evidence_pass = module.evaluate_selection_evidence(passing_metrics, thresholds)
    assert decision_pass == "POSITION_ROUTING_TARGET_SUPPORTED"
    assert evidence_pass["all_passed"]


def test_stratum_metrics_entropy_and_tv_distance() -> None:
    module = _module()
    device = torch.device("cpu")
    batch = 2
    length = 10
    vocab = 50

    logits = torch.randn(batch, length, vocab, device=device)
    scores = torch.randn(batch, 1, length, length, device=device)
    qk = torch.randn(batch, 1, length, length, device=device)
    pos = torch.randn(batch, 1, length, length, device=device)
    target = torch.randint(0, vocab, (batch, length), device=device)
    base_logits = logits.clone()
    base_attn = scores.softmax(-1)

    m = module.collect_stratum_metrics(logits, scores, qk, pos, target, base_logits, base_attn)

    assert "attention_entropy" in m
    assert "attention_tv_distance_from_baseline" in m
    assert "direct_error_token_worsening_rate" in m
    assert "unchanged_token_count" in m
    assert m["attention_tv_distance_from_baseline"] == 0.0
    assert m["attention_entropy"] >= 0.0
    assert m["unchanged_token_count"] == batch * length
