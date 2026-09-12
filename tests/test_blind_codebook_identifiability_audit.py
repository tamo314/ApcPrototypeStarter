"""Tests for C-D001Z Blind Codebook Identifiability & Symmetry-Breaking Audit.

Verifies:
1. Integrity boundary (zero training updates, zero model inits, zero sealed data reads).
2. Structured JSON artifact integrity and schema consistency.
3. Mathematical correctness of group actions (S_V permutation group, O(d) orthogonal group).
4. Symmetric World A/B construction: exact observable equivalence (H, D, y) with divergent z*.
5. Four pre-registered controls:
   - Control 1: no-anchor (H(Z|obs) > 0, baseline upper bound fails floor)
   - Control 2: 1-anchor (unanchored keys fail floor)
   - Control 3: partial-anchor (unanchored keys fail floor)
   - Control 4: full-codebook (H(Z|obs) = 0, B_det_emb achieves 100% ceiling)
6. Five-dimensional oracle scoring of static vocabulary anchors (0/5 non-oracle).
7. Impossibility-Dominance Dilemma theorem and Charter retraction verdict.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from apc.evaluation.blind_codebook_audit import (
    AnchorSet,
    BlindCodebookAudit,
    OrthogonalTransformAction,
    PermutationGroupAction,
    SymmetricWorldPair,
    analyze_impossibility_dominance_dilemma,
    evaluate_five_dimensional_oracle_score,
)

ARTIFACT_PATH = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "phase_c"
    / "artifacts"
    / "blind_codebook_identifiability_audit.json"
)


@pytest.fixture
def audit_data() -> dict[str, Any]:
    assert ARTIFACT_PATH.is_file(), f"Artifact missing: {ARTIFACT_PATH}"
    with open(ARTIFACT_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_integrity_boundary_zero_execution(audit_data: dict[str, Any]) -> None:
    boundary = audit_data["integrity_boundary"]
    for k, v in boundary.items():
        assert v == 0, f"Integrity boundary violated in {k}: {v} != 0"


def test_artifact_top_level_metadata(audit_data: dict[str, Any]) -> None:
    assert audit_data["task_id"] == "C-D001Z"
    assert audit_data["document_id"] == "DOC-PHASE-C-D001Z-AUDIT"
    assert audit_data["decision"] == "ROUTING_IDENTIFIABILITY_QUALIFIED_STOP"
    assert audit_data["audit_verdict"] == "BLIND_GROUNDING_IDENTIFIABILITY_IMPOSSIBILITY_PROVEN"
    assert audit_data["charter_candidate_decision"] == "RETRACT_CHARTER_CANDIDATE"
    assert audit_data["charter_state"] == "READY_FOR_REVIEW_NOT_APPROVED"
    assert audit_data["research_execution_authority"] == "NOT_AUTHORIZED"


def test_permutation_group_action_algebra() -> None:
    tokens = [1, 2, 3, 4]
    p_id = PermutationGroupAction.identity(tokens)
    assert p_id.apply_sequence(tokens) == tokens

    p_trans = PermutationGroupAction.transposition(tokens, 1, 2)
    assert p_trans.apply_sequence([1, 2, 3, 4]) == [2, 1, 3, 4]
    assert p_trans.inverse_sequence([2, 1, 3, 4]) == [1, 2, 3, 4]

    # Composition: (1 2) o (1 2) = identity
    p_comp = p_trans.compose(p_trans)
    assert p_comp.apply_sequence(tokens) == tokens

    # Orbit size: 4! = 24
    assert PermutationGroupAction.orbit_size(4) == 24


def test_orthogonal_group_action_isometry() -> None:
    theta = math.pi / 3.0  # 60 degrees
    rot = OrthogonalTransformAction.rotation_2d(theta, dim=2)

    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.0, 1.0])

    # Check distance preservation
    assert rot.is_isometry(v1, v2)

    # Check inner product preservation: <Q v1, Q v2> == <v1, v2> == 0
    qv1 = rot.apply_vector(v1)
    qv2 = rot.apply_vector(v2)
    assert abs(float(np.dot(qv1, qv2))) < 1e-6

    # Check inverse: Q^T Q = I
    rot_inv = rot.inverse()
    v_orig = rot_inv.apply_vector(qv1)
    assert np.allclose(v_orig, v1)


def test_symmetric_world_pair_observables_identical_diverging_coordinates() -> None:
    # Keys: 10, 20. Values: 100, 200.
    codebook_a = {
        10: np.array([1.0, 0.0, 0.0, 0.0]),
        20: np.array([0.0, 1.0, 0.0, 0.0]),
        100: np.array([0.0, 0.0, 1.0, 0.0]),
        200: np.array([0.0, 0.0, 0.0, 1.0]),
    }
    # Permutation pi = (10 20)(100 200)
    pi = PermutationGroupAction({10: 20, 20: 10, 100: 200, 200: 100})
    pair = SymmetricWorldPair(codebook_a, pi)

    # Sequence in World A: [k1, v1, k2, v2]
    seq_a = [10, 100, 20, 200]
    descriptor = {"op": "BindOp", "args": {"query_key": 10}, "tie_break": "FIRST"}

    result = pair.construct_indistinguishable_instance(seq_a, descriptor)

    assert result["observables_match"] is True
    assert result["h_max_diff"] < 1e-6
    assert result["y_a"] == result["y_b"] == 100
    assert result["coordinates_diverge"] is True
    assert result["z_star_a"] == 1
    assert result["z_star_b"] == 3


def test_preregistered_controls_simulation(audit_data: dict[str, Any]) -> None:
    key_tokens = [10, 20, 30, 40]
    val_tokens = [100, 200, 300, 400]
    audit = BlindCodebookAudit(key_tokens, val_tokens, dim=8)

    # 1. No-Anchor
    a_no = AnchorSet(level="NO_ANCHOR", anchored_tokens=(), total_tokens=8)
    res_no = audit.evaluate_control("CTRL-Z1-NO-ANCHOR", a_no)
    assert res_no["verdict"] == "IDENTIFIABILITY_IMPOSSIBLE"
    assert res_no["h_z_given_observables_bits"] == 2.0  # log2(4)
    assert res_no["best_deterministic_baseline_upper_bound"] == 0.25
    assert res_no["counterfactual_swap_covariance"] == 0.0

    # 2. 1-Anchor
    a_1 = AnchorSet(level="ONE_ANCHOR", anchored_tokens=(10,), total_tokens=8)
    res_1 = audit.evaluate_control("CTRL-Z2-1-ANCHOR", a_1)
    assert res_1["verdict"] == "PARTIALLY_UNIDENTIFIABLE_FAILS_FLOOR"
    assert math.isclose(res_1["h_z_given_observables_bits"], 1.1887218755408671)
    assert res_1["best_deterministic_baseline_upper_bound"] == 0.50
    assert res_1["counterfactual_swap_covariance"] == 0.25

    # 3. Partial-Anchor (2 keys)
    a_part = AnchorSet(level="PARTIAL_ANCHOR", anchored_tokens=(10, 20), total_tokens=8)
    res_part = audit.evaluate_control("CTRL-Z3-PARTIAL-ANCHOR", a_part)
    assert res_part["verdict"] == "PARTIALLY_UNIDENTIFIABLE_FAILS_FLOOR"
    assert res_part["h_z_given_observables_bits"] == 0.50
    assert res_part["best_deterministic_baseline_upper_bound"] == 0.75
    assert res_part["counterfactual_swap_covariance"] == 0.50

    # 4. Full-Codebook (all 8 tokens)
    a_full = AnchorSet(
        level="FULL_CODEBOOK", anchored_tokens=tuple(audit.all_tokens), total_tokens=8
    )
    res_full = audit.evaluate_control("CTRL-Z4-FULL-CODEBOOK", a_full)
    assert res_full["verdict"] == "CEILING_DOMINATED_BY_B_DET_EMB"
    assert res_full["h_z_given_observables_bits"] == 0.0
    assert res_full["best_deterministic_baseline_upper_bound"] == 1.0
    assert res_full["counterfactual_swap_covariance"] == 1.0

    # Cross-check with JSON artifact
    ctrls = audit_data["preregistered_controls"]
    assert ctrls["control_1_no_anchor"]["verdict"] == res_no["verdict"]
    assert ctrls["control_2_1_anchor"]["verdict"] == res_1["verdict"]
    assert ctrls["control_3_partial_anchor"]["verdict"] == res_part["verdict"]
    assert ctrls["control_4_full_codebook"]["verdict"] == res_full["verdict"]


def test_five_dimensional_oracle_audit() -> None:
    oracle_eval = evaluate_five_dimensional_oracle_score("STATIC_VOCABULARY_ANCHORS")
    assert oracle_eval["oracle_score"] == 0
    assert oracle_eval["is_oracle"] is False
    assert oracle_eval["classification"] == "NON_ORACLE"
    for dim, score in oracle_eval["dimensions"].items():
        assert score == 0, f"Dimension {dim} unexpectedly scored {score}"


def test_impossibility_dominance_dilemma_theorem(audit_data: dict[str, Any]) -> None:
    dilemma = analyze_impossibility_dominance_dilemma()
    assert dilemma["charter_verdict"] == "RETRACT_CHARTER_CANDIDATE"
    assert dilemma["residual_learning_margin_delta"] == 0.0

    artifact_dilemma = audit_data["impossibility_dominance_dilemma"]
    assert artifact_dilemma["charter_verdict"] == "RETRACT_CHARTER_CANDIDATE"
    assert artifact_dilemma["residual_learning_margin_delta"] == 0.0
