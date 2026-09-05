"""Unit tests for Learned Adequacy and Novelty Controller (Phase A.2 Task A2-C006).

Verifies:
1. Exact AUROC computation via Mann-Whitney U statistic with tie handling.
2. Zero oracle leakage (controller consumes only 13 numerical evidence features).
3. Deterministic prediction given identical evidence vectors.
4. Serialization and deserialization roundtrip (.save and .load).
5. Canonical K/C/N/R fixture action mapping:
   - K -> DIRECT_REUSE
   - C -> COMPOSE
   - N -> PLASTIC_SEARCH
   - R -> DIRECT_REUSE
6. Evaluation metrics computation and acceptance criteria checking.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from apc.meta.adequacy import AdequacyEvidence
from apc.meta.episode_log import ControllerAction
from apc.meta.learned_controller import (
    AdequacyControllerConfig,
    ControllerPrediction,
    LearnedAdequacyController,
    build_default_trained_controller,
    compute_auroc,
    evaluate_controller_metrics,
    train_adequacy_classifier,
)
from tests.test_adequacy_evidence import _build_test_kcnr_environment, compute_adequacy_evidence


def test_compute_auroc_exactness() -> None:
    """Verify AUROC on known distributions with perfect, inverted, and tied scores."""
    # 1. Perfect separation: pos > neg -> AUROC = 1.0
    pos = [0.8, 0.9, 1.0]
    neg = [0.1, 0.2, 0.3]
    assert compute_auroc(pos, neg) == pytest.approx(1.0)

    # 2. Inverted separation: pos < neg -> AUROC = 0.0
    pos = [0.1, 0.2, 0.3]
    neg = [0.8, 0.9, 1.0]
    assert compute_auroc(pos, neg) == pytest.approx(0.0)

    # 3. Exact ties: all scores identical -> AUROC = 0.5
    pos = [0.5, 0.5]
    neg = [0.5, 0.5]
    assert compute_auroc(pos, neg) == pytest.approx(0.5)

    # 4. Known intermediate case:
    # pos: [2, 4], neg: [1, 3]
    # pairs: (2,1): pos>neg (1); (2,3): pos<neg (0); (4,1): pos>neg (1); (4,3): pos>neg (1)
    # total U = 3, n_pos*n_neg = 4 -> AUROC = 0.75
    assert compute_auroc([2.0, 4.0], [1.0, 3.0]) == pytest.approx(0.75)


def test_compute_auroc_empty_raises() -> None:
    """Empty scores should raise ValueError."""
    with pytest.raises(ValueError, match="must be non-empty"):
        compute_auroc([], [1.0])
    with pytest.raises(ValueError, match="must be non-empty"):
        compute_auroc([1.0], [])


def test_controller_zero_oracle_leakage() -> None:
    """The controller reads only 13 numeric features; redacting external fields has zero effect."""
    core, bank, router, op_to_id, fixtures = _build_test_kcnr_environment()
    ev = compute_adequacy_evidence(core, bank, router, op_to_id, fixtures["K"])

    controller = build_default_trained_controller(seed=42)
    pred1 = controller.predict(ev)

    # Mutate metadata dictionary in evidence
    ev_mutated = AdequacyEvidence.from_dict(ev.to_dict())
    ev_mutated.metadata["oracle_leak_attempt"] = "FORBIDDEN"

    pred2 = controller.predict(ev_mutated)

    assert pred1.action == pred2.action
    assert pred1.novelty_score == pytest.approx(pred2.novelty_score, abs=1e-7)
    for act in ControllerAction:
        assert pred1.probabilities[act] == pytest.approx(pred2.probabilities[act], abs=1e-7)


def test_controller_prediction_determinism() -> None:
    """Repeated calls with identical evidence produce identical predictions."""
    core, bank, router, op_to_id, fixtures = _build_test_kcnr_environment()
    ev = compute_adequacy_evidence(core, bank, router, op_to_id, fixtures["C"])

    controller = build_default_trained_controller(seed=42)
    pred1 = controller.predict(ev)
    pred2 = controller.predict(ev)

    assert pred1.action == pred2.action
    assert pred1.novelty_score == pred2.novelty_score
    assert pred1.probabilities == pred2.probabilities
    assert pred1.feature_vector == pred2.feature_vector


def test_controller_serialization_roundtrip(tmp_path: Path) -> None:
    """Saving and loading controller preserves all weights and outputs."""
    controller = build_default_trained_controller(seed=42)
    ckpt_path = tmp_path / "controller.json"
    controller.save(ckpt_path)

    loaded = LearnedAdequacyController.load(ckpt_path)

    # Compare predictions on a test evidence vector
    dummy_ev = AdequacyEvidence(
        direct_em=0.95,
        direct_loss=0.05,
        direct_token_acc=1.0,
        direct_primitive_id=0,
        router_confidence=0.90,
    )
    p_orig = controller.predict(dummy_ev)
    p_load = loaded.predict(dummy_ev)

    assert p_orig.action == p_load.action
    assert p_orig.novelty_score == pytest.approx(p_load.novelty_score, abs=1e-7)
    for act in ControllerAction:
        assert p_orig.probabilities[act] == pytest.approx(p_load.probabilities[act], abs=1e-7)


def test_kcnr_fixture_action_selection() -> None:
    """Controller must correctly map K, C, N, R fixtures to their target actions."""
    core, bank, router, op_to_id, fixtures = _build_test_kcnr_environment()
    controller = build_default_trained_controller(seed=42)

    # 1. K (Known): REVERSE
    ev_k = compute_adequacy_evidence(core, bank, router, op_to_id, fixtures["K"])
    pred_k = controller.predict(ev_k)
    assert pred_k.action == ControllerAction.DIRECT_REUSE
    assert pred_k.probabilities[ControllerAction.DIRECT_REUSE] > 0.70
    assert pred_k.probabilities[ControllerAction.PLASTIC_SEARCH] < 0.10

    # 2. C (Composition): REVERSE -> NEGATE
    ev_c = compute_adequacy_evidence(core, bank, router, op_to_id, fixtures["C"])
    pred_c = controller.predict(ev_c)
    assert pred_c.action == ControllerAction.COMPOSE
    assert pred_c.probabilities[ControllerAction.COMPOSE] > 0.70
    assert pred_c.probabilities[ControllerAction.PLASTIC_SEARCH] < 0.10

    # 3. N (Novel): CYCLE_FOUR
    ev_n = compute_adequacy_evidence(core, bank, router, op_to_id, fixtures["N"])
    pred_n = controller.predict(ev_n)
    assert pred_n.action == ControllerAction.PLASTIC_SEARCH
    assert pred_n.probabilities[ControllerAction.PLASTIC_SEARCH] > 0.70

    # 4. R (Recurrence): SWAP_PAIRS
    ev_r = compute_adequacy_evidence(core, bank, router, op_to_id, fixtures["R"])
    pred_r = controller.predict(ev_r)
    assert pred_r.action == ControllerAction.DIRECT_REUSE
    assert pred_r.probabilities[ControllerAction.DIRECT_REUSE] > 0.70


def test_evaluate_controller_metrics_logic() -> None:
    """Verify acceptance criteria calculation over a balanced synthetic episode stream."""
    episodes: list[tuple[str, ControllerPrediction]] = []

    # 10 K episodes: all predict DIRECT_REUSE
    for _ in range(10):
        pred = ControllerPrediction(
            action=ControllerAction.DIRECT_REUSE,
            probabilities={
                ControllerAction.DIRECT_REUSE: 0.95,
                ControllerAction.COMPOSE: 0.03,
                ControllerAction.PLASTIC_SEARCH: 0.02,
            },
            novelty_score=0.02,
            feature_vector=[1.0] * 13,
        )
        episodes.append(("K", pred))

    # 10 C episodes: all predict COMPOSE
    for _ in range(10):
        pred = ControllerPrediction(
            action=ControllerAction.COMPOSE,
            probabilities={
                ControllerAction.DIRECT_REUSE: 0.05,
                ControllerAction.COMPOSE: 0.92,
                ControllerAction.PLASTIC_SEARCH: 0.03,
            },
            novelty_score=0.03,
            feature_vector=[0.0] * 13,
        )
        episodes.append(("C", pred))

    # 10 N episodes: all predict PLASTIC_SEARCH
    for _ in range(10):
        pred = ControllerPrediction(
            action=ControllerAction.PLASTIC_SEARCH,
            probabilities={
                ControllerAction.DIRECT_REUSE: 0.02,
                ControllerAction.COMPOSE: 0.03,
                ControllerAction.PLASTIC_SEARCH: 0.95,
            },
            novelty_score=0.95,
            feature_vector=[0.0] * 13,
        )
        episodes.append(("N", pred))

    # 10 R episodes: all predict DIRECT_REUSE
    for _ in range(10):
        pred = ControllerPrediction(
            action=ControllerAction.DIRECT_REUSE,
            probabilities={
                ControllerAction.DIRECT_REUSE: 0.94,
                ControllerAction.COMPOSE: 0.03,
                ControllerAction.PLASTIC_SEARCH: 0.03,
            },
            novelty_score=0.03,
            feature_vector=[1.0] * 13,
        )
        episodes.append(("R", pred))

    metrics = evaluate_controller_metrics(episodes)
    assert metrics.all_passed is True
    assert metrics.kc_vs_n_auroc == pytest.approx(1.0)
    assert metrics.k_false_plastic_rate == 0.0
    assert metrics.c_false_plastic_rate == 0.0
    assert metrics.n_plastic_trigger_rate == 1.0
    assert metrics.r_direct_reuse_rate == 1.0
    assert metrics.c_action_accuracy == 1.0
    assert metrics.overall_action_accuracy == 1.0


def test_train_adequacy_classifier_convergence() -> None:
    """Verify train_adequacy_classifier converges on synthetic profiles."""
    training_data: list[tuple[list[float], ControllerAction]] = [
        (
            [1.0, 0.1, 1.0, 0.8, 1.0, 0.1, 1.0, 0.0, 0.0, 0.9, 0.8, 0.1, 0.9],
            ControllerAction.DIRECT_REUSE,
        ),
        (
            [0.1, 2.5, 0.1, 0.0, 1.0, 0.1, 2.0, 0.9, 2.4, 0.6, 0.2, 0.6, 0.4],
            ControllerAction.COMPOSE,
        ),
        (
            [0.0, 3.0, 0.0, 0.0, 0.0, 3.0, 1.0, 0.0, 0.0, 0.3, 0.1, 1.1, 0.2],
            ControllerAction.PLASTIC_SEARCH,
        ),
    ]
    cfg = AdequacyControllerConfig(train_epochs=100, lr=0.05)
    controller = train_adequacy_classifier(training_data, config=cfg, seed=42)

    for feats, target_act in training_data:
        ev = AdequacyEvidence(
            direct_em=feats[0],
            direct_loss=feats[1],
            direct_token_acc=feats[2],
            direct_primitive_id=0,
            composition_em=feats[4],
            composition_loss=feats[5],
            composition_depth=int(feats[6]),
            composition_improvement_em=feats[7],
            composition_improvement_loss=feats[8],
            router_confidence=feats[9],
            router_margin=feats[10],
            router_entropy=feats[11],
            recurrence_key_similarity=feats[12],
        )
        pred = controller.predict(ev)
        assert pred.action == target_act
