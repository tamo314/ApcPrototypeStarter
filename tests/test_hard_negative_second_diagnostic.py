"""Focused invariant coverage for B-C005D2-001."""

from __future__ import annotations

import pytest

from apc.evaluation.hard_negative_second_diagnostic import (
    HardNegativeSecondDiagnosticConfig,
    _comparison,
    _quantiles,
    _verdict,
)
from apc.meta.phase_b_protocol import HardNegativeLevel


def _row(partition: str, state: str, *, top1: bool, level: str) -> dict[str, object]:
    return {
        "partition": partition,
        "router_state": state,
        "checkpoint_label": "label",
        "bank_size": 128,
        "level": level,
        "family_top1": top1,
        "argument_correct": top1,
        "top1_correct": top1,
        "top5_included": True,
        "score_margin": 1.0 if top1 else -1.0,
        "correct_rank": 1 if top1 else 2,
        "key_cosine_similarity": 0.5,
        "semantic_relation_id": "SHIFT->CYCLE_FOUR",
        "relation_type": "related_learned_primitive_key",
        "target_family": "SHIFT",
        "competitor_family": "CYCLE_FOUR",
        "target_arguments": {"amount": 1},
    }


def test_sealed_partitions_cannot_be_development() -> None:
    with pytest.raises(ValueError, match="development seeds"):
        HardNegativeSecondDiagnosticConfig(development_seeds=(20,), regate_sealed_seeds=(21,))


def test_comparison_contains_required_descriptors_and_checkpoint_warning() -> None:
    l3 = HardNegativeLevel.L3_SEMANTICALLY_RELATED.value
    rows = [
        _row("development", "R2_frozen_post_repair", top1=True, level=l3),
        _row("original_sealed", "R0_frozen_pre_repair", top1=False, level=l3),
        _row("regate_sealed", "R2_frozen_post_repair", top1=False, level=l3),
    ]
    comparison = _comparison(rows)
    item = comparison["comparisons"][
        "development_post_repair_vs_regate_sealed/L3_semantically_related"
    ]
    assert "Cross-checkpoint" in item["warning"]
    assert (
        item["target_competitor_pair_frequencies"]["development"][0]["value"] == "SHIFT->CYCLE_FOUR"
    )
    assert _quantiles([])["mean"] is None


def test_verdict_flags_the_declared_development_difficulty_mismatch() -> None:
    l3 = HardNegativeLevel.L3_SEMANTICALLY_RELATED.value
    rows = [
        _row("development", "R0_frozen_pre_repair", top1=True, level=l3),
        _row("original_sealed", "R0_frozen_pre_repair", top1=False, level=l3),
        _row("development", "R2_frozen_post_repair", top1=True, level=l3),
        _row("regate_sealed", "R2_frozen_post_repair", top1=False, level=l3),
    ]
    verdict = _verdict(rows)
    assert verdict["classification"] == "NON_REPRESENTATIVE"
    assert verdict["flags"] == ["DEVELOPMENT_DIFFICULTY_MISMATCH"]
