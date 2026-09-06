# ruff: noqa: E501
"""Focused invariant coverage for B-C005D2-003 (no GPU checkpoints required)."""

from __future__ import annotations

import pytest

from apc.evaluation.semantic_relation_audit import (
    SemanticRelationAuditConfig,
    _bin_label,
    _class_split,
    _classify_matching,
    _classify_relation_holdout,
    _difficulty_matched_comparison,
    _difficulty_probe_accuracy,
    _final_verdict,
    _fit_difficulty_probe,
    _quantile_edges,
    _relation_group_summary,
    _relation_taxonomy,
)


def _config(**overrides: object) -> SemanticRelationAuditConfig:
    return SemanticRelationAuditConfig(**overrides)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Config validation.
# ---------------------------------------------------------------------------


def test_config_rejects_development_overlap_with_sealed_partitions() -> None:
    with pytest.raises(ValueError, match="development seeds"):
        _config(development_seeds=(20,), regate_sealed_seeds=(21,))


def test_config_rejects_non_standard_bank_size() -> None:
    with pytest.raises(ValueError, match="bank_size"):
        _config(bank_size=17)


def test_config_rejects_too_few_cosine_bins() -> None:
    with pytest.raises(ValueError, match="cosine_bin_count"):
        _config(cosine_bin_count=1)


# ---------------------------------------------------------------------------
# D2-003.1 relation taxonomy.
# ---------------------------------------------------------------------------


def test_relation_taxonomy_is_the_four_canonical_registry_pairs() -> None:
    taxonomy = _relation_taxonomy()
    relation_ids = {item["relation_id"] for item in taxonomy["relations"]}
    assert relation_ids == {"SHIFT->CYCLE_FOUR", "SELECT->BIND", "COUNT->BIND", "BIND->COUNT"}
    assert "outcome" not in taxonomy["note"].lower() or "not derived" in taxonomy["note"].lower()


# ---------------------------------------------------------------------------
# Quantile binning helpers.
# ---------------------------------------------------------------------------


def test_quantile_edges_split_uniform_values_into_equal_mass_bins() -> None:
    values = [float(v) for v in range(100)]
    edges = _quantile_edges(values, 4)
    assert len(edges) == 3
    assert edges == sorted(edges)


def test_bin_label_places_values_relative_to_edges() -> None:
    edges = [1.0, 2.0]
    assert _bin_label(0.5, edges) == 0
    assert _bin_label(1.5, edges) == 1
    assert _bin_label(2.5, edges) == 2


def test_quantile_edges_empty_input_returns_no_edges() -> None:
    assert _quantile_edges([], 3) == []


# ---------------------------------------------------------------------------
# Relation-group summary.
# ---------------------------------------------------------------------------


def _row(
    partition: str,
    state: str,
    relation: str,
    *,
    top1: bool,
    key_cosine: float = 0.5,
    score_margin: float = 1.0,
) -> dict[str, object]:
    return {
        "partition": partition,
        "router_state": state,
        "semantic_relation_id": relation,
        "top1_correct": top1,
        "top5_included": True,
        "score_margin": score_margin if top1 else -abs(score_margin),
        "key_cosine_similarity": key_cosine,
    }


def test_relation_group_summary_reports_spread_across_relations() -> None:
    rows = [
        _row("regate_sealed", "R2_frozen_post_repair", "SHIFT->CYCLE_FOUR", top1=True),
        _row("regate_sealed", "R2_frozen_post_repair", "SHIFT->CYCLE_FOUR", top1=True),
        _row("regate_sealed", "R2_frozen_post_repair", "COUNT->BIND", top1=False),
        _row("regate_sealed", "R2_frozen_post_repair", "COUNT->BIND", top1=False),
    ]
    summary = _relation_group_summary(rows)
    cell = summary["regate_sealed/R2_frozen_post_repair"]
    assert cell["per_relation"]["SHIFT->CYCLE_FOUR"]["top1"] == 1.0
    assert cell["per_relation"]["COUNT->BIND"]["top1"] == 0.0
    assert cell["cross_relation_spread"] == pytest.approx(1.0)


def test_relation_group_summary_spread_is_none_for_a_single_relation() -> None:
    rows = [_row("development", "R0_frozen_pre_repair", "SHIFT->CYCLE_FOUR", top1=True)]
    summary = _relation_group_summary(rows)
    assert summary["development/R0_frozen_pre_repair"]["cross_relation_spread"] is None


# ---------------------------------------------------------------------------
# Difficulty probe (2D key-cosine-similarity / score-margin -> correctness).
# ---------------------------------------------------------------------------


def _separable_rows(seed_offset: int, *, n: int = 40) -> list[dict[str, object]]:
    rows = []
    for i in range(n):
        correct = i % 2 == 0
        cosine = 0.9 if correct else 0.1
        margin = 3.0 if correct else -3.0
        rows.append(
            _row(
                "development",
                "R0_frozen_pre_repair",
                "SHIFT->CYCLE_FOUR",
                top1=correct,
                key_cosine=cosine + 0.01 * ((i + seed_offset) % 3),
                score_margin=margin,
            )
        )
    return rows


def test_class_split_separates_correct_and_incorrect_rows() -> None:
    rows = _separable_rows(0)
    positive, negative = _class_split(rows)
    assert len(positive) == 20
    assert len(negative) == 20


def test_fit_difficulty_probe_returns_none_when_one_class_is_absent() -> None:
    rows = [_row("development", "R0_frozen_pre_repair", "SHIFT->CYCLE_FOUR", top1=True) for _ in range(10)]
    probe = _fit_difficulty_probe(rows, steps=50, lr=0.1, seed=0, shuffle_labels=False)
    assert probe is None
    assert _difficulty_probe_accuracy(probe, rows) is None


def test_fit_difficulty_probe_separates_well_clustered_difficulty_classes() -> None:
    train_rows = _separable_rows(0)
    eval_rows = _separable_rows(1)
    probe = _fit_difficulty_probe(train_rows, steps=300, lr=0.1, seed=0, shuffle_labels=False)
    assert probe is not None
    accuracy = _difficulty_probe_accuracy(probe, eval_rows)
    assert accuracy is not None
    assert accuracy >= 0.95


# ---------------------------------------------------------------------------
# Difficulty-matched comparison.
# ---------------------------------------------------------------------------


def test_difficulty_matched_comparison_explains_away_gap_from_bin_composition() -> None:
    # Both partitions have IDENTICAL per-bin top1 (easy=1.0, hard=0.5), but development
    # is mostly easy examples while sealed is mostly hard ones -- the raw gap is pure
    # composition shift, and matching on (relation, cosine-bin) should erase it.
    development_rows = (
        [_row("development", "R0", "SHIFT->CYCLE_FOUR", top1=True, key_cosine=0.1) for _ in range(10)]
        + [_row("development", "R0", "SHIFT->CYCLE_FOUR", top1=True, key_cosine=0.9)]
        + [_row("development", "R0", "SHIFT->CYCLE_FOUR", top1=False, key_cosine=0.9)]
    )
    sealed_rows = (
        [_row("regate_sealed", "R2", "SHIFT->CYCLE_FOUR", top1=True, key_cosine=0.1) for _ in range(2)]
        + [_row("regate_sealed", "R2", "SHIFT->CYCLE_FOUR", top1=True, key_cosine=0.9) for _ in range(5)]
        + [_row("regate_sealed", "R2", "SHIFT->CYCLE_FOUR", top1=False, key_cosine=0.9) for _ in range(5)]
    )

    config = _config(cosine_bin_count=2)
    matched = _difficulty_matched_comparison(development_rows, sealed_rows, config, label="test")
    assert matched["shared_cell_count"] >= 1
    assert matched["raw_gap"] is not None and matched["raw_gap"] > 0.05
    assert matched["matched_gap"] is not None
    assert abs(matched["matched_gap"]) < abs(matched["raw_gap"])


def test_classify_matching_labels() -> None:
    config = _config()
    assert _classify_matching({"raw_gap": 0.0, "matched_gap": 0.0, "shared_cell_count": 1}, config) == (
        "SEED_SPLIT_SUFFICIENT"
    )
    assert _classify_matching({"raw_gap": 0.5, "matched_gap": 0.0, "shared_cell_count": 1}, config) == (
        "DIFFICULTY_MATCHING_REQUIRED"
    )
    assert _classify_matching({"raw_gap": 0.5, "matched_gap": 0.5, "shared_cell_count": 1}, config) == (
        "MODEL_GENERALIZATION_FAILURE_AFTER_MATCHING"
    )
    assert _classify_matching({"raw_gap": None, "matched_gap": None, "shared_cell_count": 0}, config) == (
        "UNRESOLVED"
    )


# ---------------------------------------------------------------------------
# Relation-holdout probe classification and final verdict combination.
# ---------------------------------------------------------------------------


def test_classify_relation_holdout_flags_large_generalization_gap() -> None:
    config = _config()
    per_relation = {
        "SHIFT->CYCLE_FOUR": {"cross_relation_generalization_accuracy": 0.5, "in_relation_accuracy": 0.95},
        "SELECT->BIND": {"cross_relation_generalization_accuracy": 0.55, "in_relation_accuracy": 0.9},
    }
    classification, evidence = _classify_relation_holdout(per_relation, config)
    assert classification == "SEMANTIC_RELATION_HOLDOUT_REQUIRED"
    assert evidence["n_relations_compared"] == 2


def test_classify_relation_holdout_sufficient_when_gap_is_small() -> None:
    config = _config()
    per_relation = {
        "SHIFT->CYCLE_FOUR": {"cross_relation_generalization_accuracy": 0.9, "in_relation_accuracy": 0.92},
    }
    classification, _ = _classify_relation_holdout(per_relation, config)
    assert classification == "SEED_SPLIT_SUFFICIENT"


def test_classify_relation_holdout_unresolved_when_no_relation_is_comparable() -> None:
    config = _config()
    per_relation = {"SHIFT->CYCLE_FOUR": {"cross_relation_generalization_accuracy": None, "in_relation_accuracy": None}}
    classification, evidence = _classify_relation_holdout(per_relation, config)
    assert classification == "UNRESOLVED"
    assert "reason" in evidence


def test_final_verdict_flags_semantic_relation_holdout_required() -> None:
    config = _config()
    relation_summary = {
        "regate_sealed/R2_frozen_post_repair": {"cross_relation_spread": 0.6, "per_relation": {}}
    }
    holdout = {
        "SHIFT->CYCLE_FOUR": {"cross_relation_generalization_accuracy": 0.5, "in_relation_accuracy": 0.95}
    }
    matched_post = {"raw_gap": 0.3, "matched_gap": 0.3, "shared_cell_count": 2}
    matched_pre = {"raw_gap": 0.0, "matched_gap": 0.0, "shared_cell_count": 2}
    verdict = _final_verdict(relation_summary, holdout, matched_post, matched_pre, config)
    assert verdict["semantic_relation_holdout_required"] is True
    assert "SEMANTIC_RELATION_HOLDOUT_REQUIRED" in verdict["labels"]


def test_final_verdict_defaults_to_unresolved_when_nothing_can_be_classified() -> None:
    config = _config()
    relation_summary: dict[str, object] = {}
    holdout: dict[str, object] = {}
    matched = {"raw_gap": None, "matched_gap": None, "shared_cell_count": 0}
    verdict = _final_verdict(relation_summary, holdout, matched, matched, config)
    assert verdict["labels"] == ["UNRESOLVED"]
    assert verdict["semantic_relation_holdout_required"] is False
