"""Focused CPU checks for the B-C004 evaluation-only hard-negative builder."""

from __future__ import annotations

import pytest
import torch

from apc.evaluation.hard_negative_routing_benchmark import (
    HardNegativeSafetyConfig,
    build_hard_negative_candidates,
)
from apc.meta.phase_b_protocol import HardNegativeLevel


def _keys() -> dict[int, torch.Tensor]:
    return {
        0: torch.tensor([1.0, 0.0, 0.0, 0.0]),
        1: torch.tensor([0.0, 1.0, 0.0, 0.0]),
        2: torch.tensor([0.0, 0.0, 1.0, 0.0]),
        3: torch.tensor([0.0, 0.0, 0.0, 1.0]),
    }


def _build(level: HardNegativeLevel, seed: int = 7):
    return build_hard_negative_candidates(
        level=level,
        target_id=0,
        target_operation="SHIFT",
        candidate_ids=[0, 1, 2, 3],
        keys_by_id=_keys(),
        operation_by_id={0: "SHIFT", 1: "CYCLE_FOUR", 2: "COUNT", 3: "BIND"},
        seed=seed,
    )


def test_hard_negative_generation_is_deterministic_and_non_mutating() -> None:
    keys = _keys()
    before = {key: value.clone() for key, value in keys.items()}
    candidates_a, competitor_a = _build(HardNegativeLevel.L2_NEAR_NEIGHBOR)
    candidates_b, competitor_b = _build(HardNegativeLevel.L2_NEAR_NEIGHBOR)
    ids_a = [candidate.candidate_id for candidate in candidates_a]
    ids_b = [candidate.candidate_id for candidate in candidates_b]
    assert ids_a == ids_b
    assert torch.equal(competitor_a.score_key, competitor_b.score_key)
    assert all(torch.equal(keys[key], before[key]) for key in keys)


def test_l3_uses_related_learned_primitive_provenance() -> None:
    _, competitor = _build(HardNegativeLevel.L3_SEMANTICALLY_RELATED)
    assert competitor.execute_primitive_id == 1
    assert competitor.provenance == "related_learned_primitive_key"
    assert competitor.semantic_relation == "SHIFT->CYCLE_FOUR"


def test_l4_reuses_existing_primitive_with_no_bank_candidate_duplication() -> None:
    candidates, competitor = _build(HardNegativeLevel.L4_CONFUSABLE_FAMILY)
    assert competitor.execute_primitive_id == 0
    assert competitor.argument_override == {"__wrong_argument__": True}
    assert competitor.candidate_id.startswith("virtual:0:")
    # The logical virtual candidate adds no new persistent primitive ID.
    assert {candidate.execute_primitive_id for candidate in candidates} == {0, 1, 2, 3}
    assert len(candidates) == 5


def test_levels_provide_distinct_competitor_construction() -> None:
    competitors = [_build(level)[1] for level in HardNegativeLevel]
    assert len({candidate.provenance for candidate in competitors}) == len(HardNegativeLevel)
    assert torch.dot(competitors[0].score_key, _keys()[0]).abs() < 1e-5


def test_safety_gate_config_keeps_verification_budget_bounded() -> None:
    """B-C005 accepts only the predeclared top-k and adequacy threshold range."""
    config = HardNegativeSafetyConfig(
        seeds=(0,), bank_sizes=(16,), support_examples=2, query_examples=3, top_k=5
    )
    assert config.adequacy_exact_match_threshold == 0.95
    with pytest.raises(ValueError, match="top_k"):
        HardNegativeSafetyConfig(top_k=6)
    with pytest.raises(ValueError, match="adequacy_exact_match_threshold"):
        HardNegativeSafetyConfig(adequacy_exact_match_threshold=0.0)
