"""Unit tests for Controller Ablations and Failure Analysis (Phase A.2 Task A2-C011).

Verifies:
1. Exact feature masking across all required evidence perturbation functions.
2. Router-confidence-only threshold policy logic.
3. CPU smoke test of decision ablations across K/C/N/R stream.
4. CPU smoke test of router incremental growth under R2 (Bounded Replay) vs R1 (No Replay).
5. CPU smoke test of plastic policy variants (compact-first vs compact-only vs always-overcomplete).
"""

from __future__ import annotations

import pytest

from apc.evaluation.controller_ablation_benchmark import (
    ControllerAblationConfig,
    evaluate_decision_ablations_across_stream,
    evaluate_plastic_policy_ablations,
    evaluate_router_replay_ablation,
    modify_evidence_no_composition,
    modify_evidence_no_recurrence_similarity,
    modify_evidence_no_support_functional_score,
    modify_evidence_router_confidence_only,
    predict_router_confidence_only,
)
from apc.meta.adequacy import MAX_CLAMPED_LOSS, AdequacyEvidence
from apc.meta.episode_log import ControllerAction


def test_evidence_perturbation_functions() -> None:
    """Verify exact feature masking for each ablation function."""
    base_ev = AdequacyEvidence(
        direct_em=1.0,
        direct_loss=0.05,
        direct_token_acc=1.0,
        direct_primitive_id=0,
        direct_margin=0.8,
        composition_em=0.95,
        composition_loss=0.08,
        composition_depth=2,
        composition_recipe=("COPY", "NEGATE"),
        composition_improvement_em=0.5,
        composition_improvement_loss=2.0,
        router_confidence=0.85,
        router_margin=0.6,
        router_entropy=0.2,
        recurrence_key_similarity=0.9,
    )

    # 1. No composition evidence
    ev_nocomp = modify_evidence_no_composition(base_ev)
    assert ev_nocomp.direct_em == 1.0
    assert ev_nocomp.composition_em == 0.0
    assert ev_nocomp.composition_loss == MAX_CLAMPED_LOSS
    assert ev_nocomp.composition_recipe is None
    assert ev_nocomp.composition_improvement_em == 0.0
    assert ev_nocomp.router_confidence == 0.85

    # 2. No support-set functional score
    ev_nosupp = modify_evidence_no_support_functional_score(base_ev)
    assert ev_nosupp.direct_em == 0.0
    assert ev_nosupp.direct_loss == MAX_CLAMPED_LOSS
    assert ev_nosupp.direct_token_acc == 0.0
    assert ev_nosupp.composition_em == 0.95
    assert ev_nosupp.router_confidence == 0.85

    # 3. Router confidence only
    ev_router = modify_evidence_router_confidence_only(base_ev)
    assert ev_router.direct_em == 0.0
    assert ev_router.composition_em == 0.0
    assert ev_router.router_confidence == 0.85
    assert ev_router.recurrence_key_similarity == 0.0

    # 5. No recurrence similarity
    ev_norec = modify_evidence_no_recurrence_similarity(base_ev)
    assert ev_norec.direct_em == 1.0
    assert ev_norec.composition_em == 0.95
    assert ev_norec.recurrence_key_similarity == 0.0


def test_router_confidence_only_policy() -> None:
    """Verify decision thresholding on router confidence only."""
    ev_high = AdequacyEvidence(
        direct_em=0.0,
        direct_loss=10.0,
        direct_token_acc=0.0,
        direct_primitive_id=0,
        router_confidence=0.85,
    )
    pred_high = predict_router_confidence_only(ev_high, confidence_threshold=0.50)
    assert pred_high.action == ControllerAction.DIRECT_REUSE
    assert pred_high.probabilities[ControllerAction.DIRECT_REUSE] == 0.85

    ev_low = AdequacyEvidence(
        direct_em=0.0,
        direct_loss=10.0,
        direct_token_acc=0.0,
        direct_primitive_id=0,
        router_confidence=0.35,
    )
    pred_low = predict_router_confidence_only(ev_low, confidence_threshold=0.50)
    assert pred_low.action == ControllerAction.PLASTIC_SEARCH
    assert pred_low.probabilities[ControllerAction.PLASTIC_SEARCH] == 0.65


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_cpu_smoke_decision_ablations(tmp_path: pytest.TempPathFactory) -> None:
    """Fast CPU test verifying controller decision ablations on a mini stream."""
    cfg = ControllerAblationConfig(
        seeds=(0,),
        num_k=14,
        num_c=12,
        num_n=6,
        num_r=8,
        support_size=8,
        eval_size=8,
        device_str="cpu",
    )

    reports = evaluate_decision_ablations_across_stream(seeds=(0,), config=cfg)
    assert "baseline" in reports
    assert "no_composition_evidence" in reports
    assert "no_support_functional_score" in reports
    assert "router_confidence_only" in reports
    assert "no_recurrence_similarity" in reports

    # Baseline should have high overall action accuracy
    assert reports["baseline"]["mean_overall_action_accuracy"] >= 0.90
    # No composition evidence should show high false plastic on C (false expansion)
    assert reports["no_composition_evidence"]["mean_c_false_plastic_rate"] > 0.50
    assert reports["no_composition_evidence"]["false_expansion_observed"] is True


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_router_replay_ablation_smoke() -> None:
    """Verify router growth comparison under R2 vs R1 on CPU."""
    res = evaluate_router_replay_ablation(seeds=(0,), device_str="cpu")
    assert "r2_bounded_replay" in res
    assert "r1_no_bounded_replay" in res
    r2 = res["r2_bounded_replay"]
    r1 = res["r1_no_bounded_replay"]
    # R2 preserves old class routing with <= 2pp drop
    assert r2["mean_old_class_drop"] <= 0.02
    # R1 causes catastrophic forgetting (> 50pp drop)
    assert r1["mean_old_class_drop"] > 0.50
    assert r1["routing_forgetting_observed"] is True


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plastic_policy_ablations_smoke() -> None:
    """Verify compact-first, compact-only, and always-overcomplete lifecycle policies on CPU."""
    res = evaluate_plastic_policy_ablations(device_str="cpu", seed=42)
    assert "compact_first_baseline" in res
    assert "compact_only_ablation" in res
    assert "always_overcomplete_ablation" in res

    p_base = res["compact_first_baseline"]
    p_comp = res["compact_only_ablation"]
    p_over = res["always_overcomplete_ablation"]

    # Compact-first uses compact peak parameters on easy task
    assert p_base["easy_task"]["peak_params"] <= 25_000
    # Compact-only fails on hard task where compact fails
    assert p_comp["hard_task"]["adaptation_failure"] is True
    # Always-overcomplete uses overcomplete peak parameters (~137k)
    assert p_over["easy_task"]["peak_params"] > 100_000
    assert p_over["easy_task"]["param_inflation_factor"] > 5.0
