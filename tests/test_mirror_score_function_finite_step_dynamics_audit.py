"""Tests for B-C005REC-004S finite-step score-function dynamics audit."""

from __future__ import annotations

import torch

from apc.evaluation import mirror_score_function_finite_step_dynamics_audit as audit
from apc.primitives.primitive import (
    CrossPositionLengthBiasPrimitive,
    CrossPositionLengthBiasPrimitiveConfig,
)


def _primitive() -> CrossPositionLengthBiasPrimitive:
    return CrossPositionLengthBiasPrimitive(
        99,
        CrossPositionLengthBiasPrimitiveConfig(
            operation="MIRROR_HALVES",
            d_model=8,
            d_operator=4,
            n_head=1,
            d_operator_ff=8,
            vocab_size=5,
            max_sequence_length=10,
            length_ref=10,
        ),
    )


def test_score_forward_from_params_matches_primitive_forward() -> None:
    primitive = _primitive()
    content = torch.randn(2, 6, 8)
    params = {name: p.detach() for name, p in primitive.named_parameters()}

    res = audit._score_forward_from_params(primitive, content, 6, params)
    scores = res["scores"]

    assert scores.shape == (2, 1, 6, 6)
    assert torch.isfinite(scores).all()


def test_score_jvp_and_finite_difference_parity_at_small_alpha() -> None:
    primitive = _primitive()
    content = torch.randn(2, 6, 8)
    named = dict(primitive.named_parameters())
    params = {name: p.detach() for name, p in named.items()}

    # Create small synthetic delta on Q and K
    deltas = {name: torch.zeros_like(p) for name, p in named.items()}
    deltas["cross_attn.in_proj_weight"] = 0.01 * torch.randn_like(
        named["cross_attn.in_proj_weight"]
    )
    deltas["position_bias_hidden.weight"] = 0.01 * torch.randn_like(
        named["position_bias_hidden.weight"]
    )

    tangent = audit.rec004q._score_jvp(primitive, content, 6, deltas)

    s_before = audit._score_forward_from_params(primitive, content, 6, params)["scores"]
    alpha = 0.001
    params_pert = {name: p + alpha * deltas[name] for name, p in params.items()}
    s_after = audit._score_forward_from_params(primitive, content, 6, params_pert)["scores"]

    exact_diff = s_after - s_before
    pred_diff = alpha * tangent

    cos = audit._safe_cosine(exact_diff, pred_diff)
    assert cos is not None
    assert cos > 0.99


def test_decision_logic_overshoot() -> None:
    checkpoint_results = {
        "I03_SCORE_ONLY_12000": {
            "results_by_probe": {
                "continuity": {
                    "length10_position4": {
                        "overshoot_inversion_fraction_at_1.0": 0.70,
                        "early_improvement_fraction_at_0.1": 0.85,
                        "median_r_1.0": 0.3,
                        "median_cosine_1.0": 0.7,
                        "sign_agreement_1.0": 0.3,
                    }
                }
            }
        },
        "I03_CP_SCORE_12000": {
            "results_by_probe": {
                "continuity": {
                    "length10_position4": {
                        "overshoot_inversion_fraction_at_1.0": 0.65,
                        "early_improvement_fraction_at_0.1": 0.90,
                        "median_r_1.0": 0.3,
                        "median_cosine_1.0": 0.7,
                        "sign_agreement_1.0": 0.3,
                    }
                }
            }
        },
        "I04_P_7000": {
            "results_by_probe": {
                "continuity": {
                    "length10_position4": {
                        "overshoot_inversion_fraction_at_1.0": 0.10,
                        "early_improvement_fraction_at_0.1": 0.95,
                        "median_r_1.0": 0.1,
                        "median_cosine_1.0": 0.9,
                        "sign_agreement_1.0": 0.9,
                    }
                }
            }
        },
    }
    locality_check = {"all_passed": True}
    decision = audit._evaluate_decision(checkpoint_results, locality_check)
    assert decision["label"] == "OPTIMIZER_SIZED_SCORE_FUNCTION_OVERSHOOT_SUPPORTED"


def test_decision_logic_locally_faithful() -> None:
    checkpoint_results = {
        "I03_SCORE_ONLY_12000": {
            "results_by_probe": {
                "continuity": {
                    "length10_position4": {
                        "overshoot_inversion_fraction_at_1.0": 0.10,
                        "early_improvement_fraction_at_0.1": 0.90,
                        "median_r_1.0": 0.2,
                        "median_cosine_1.0": 0.90,
                        "sign_agreement_1.0": 0.85,
                    }
                }
            }
        },
        "I03_CP_SCORE_12000": {
            "results_by_probe": {
                "continuity": {
                    "length10_position4": {
                        "overshoot_inversion_fraction_at_1.0": 0.15,
                        "early_improvement_fraction_at_0.1": 0.90,
                        "median_r_1.0": 0.25,
                        "median_cosine_1.0": 0.85,
                        "sign_agreement_1.0": 0.82,
                    }
                }
            }
        },
        "I04_P_7000": {
            "results_by_probe": {
                "continuity": {
                    "length10_position4": {
                        "overshoot_inversion_fraction_at_1.0": 0.05,
                        "early_improvement_fraction_at_0.1": 0.95,
                        "median_r_1.0": 0.1,
                        "median_cosine_1.0": 0.95,
                        "sign_agreement_1.0": 0.95,
                    }
                }
            }
        },
    }
    locality_check = {"all_passed": True}
    decision = audit._evaluate_decision(checkpoint_results, locality_check)
    assert decision["label"] == "ONE_STEP_SCORE_DYNAMICS_LOCALLY_FAITHFUL"


def test_decision_logic_parity_gate_failure() -> None:
    checkpoint_results = {}
    locality_check = {"all_passed": False}
    decision = audit._evaluate_decision(checkpoint_results, locality_check)
    assert decision["label"] == "FINITE_STEP_DIAGNOSTIC_IMPLEMENTATION_OR_LOCALITY_UNRESOLVED"
