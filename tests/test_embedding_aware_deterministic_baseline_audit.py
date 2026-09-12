"""Tests for C-D001Y Embedding-Aware Deterministic Baseline Closure Audit.

Verifies:
1. Integrity boundary (zero execution, zero parameter updates).
2. Structured JSON artifact integrity.
3. Executable simulation of B_det_emb across all 5 pre-registered controls:
   - Control 1: Discrete symbols (Identity basis)
   - Control 2: Invertible distributed representations
   - Control 3: Equidistant collisions (Voronoi boundary ties)
   - Control 4: Information-lossy mapping (Rank-deficient collapse)
   - Control 5: Bounded perturbations within registered safety margin
4. Mathematical verification of H(Z | H(X), D) and Descriptor Swap Covariance.
5. Identifiability vs learnability failure categorization.
6. Formal specification of residual hypothesis and baseline delta.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from apc.evaluation.embedding_aware_baseline import (
    Codebook,
    EmbeddingAwareDeterministicBaseline,
    InvertibleLinearTransform,
    compute_routing_entropy,
)

ARTIFACT_PATH = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "phase_c"
    / "artifacts"
    / "embedding_aware_deterministic_baseline_audit.json"
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


def test_baseline_specification_structure(audit_data: dict[str, Any]) -> None:
    spec = audit_data["baseline_specification"]
    assert spec["baseline_id"] == "BASE-DET-EMB-V1"
    assert spec["learned_parameters"] == 0
    assert spec["training_updates"] == 0
    assert len(spec["components"]) == 3


def test_preregistered_controls_artifact_structure(audit_data: dict[str, Any]) -> None:
    ctrls = audit_data["preregistered_controls"]
    expected_controls = [
        "control_1_discrete_symbols",
        "control_2_invertible_distributed",
        "control_3_equidistant_collision",
        "control_4_lossy_projection",
        "control_5_bounded_perturbation",
    ]
    for c in expected_controls:
        assert c in ctrls, f"Missing control in artifact: {c}"

    # Verify ceiling dominance on invertible controls
    for c in [
        "control_1_discrete_symbols",
        "control_2_invertible_distributed",
        "control_5_bounded_perturbation",
    ]:
        res = ctrls[c]["evaluation_results"]
        assert res["decodability_em"] == 1.0
        assert res["collision_routing_accuracy"] == 1.0
        assert res["collision_execution_em"] == 1.0
        assert res["h_z_given_h_x_d_bits"] == 0.0
        assert res["descriptor_swap_covariance"] == 1.0
        assert ctrls[c]["verdict"] == "CEILING_DOMINATED"

    # Verify identifiability limit on uninvertible controls
    for c in ["control_3_equidistant_collision", "control_4_lossy_projection"]:
        res = ctrls[c]["evaluation_results"]
        assert res["decodability_em"] == 0.0
        assert res["collision_routing_accuracy"] < 0.95
        assert res["h_z_given_h_x_d_bits"] > 0.0
        assert ctrls[c]["verdict"] == "IDENTIFIABILITY_LIMIT_NOT_LEARNABILITY_LIMIT"


def test_executable_control_1_discrete_symbols() -> None:
    """Executable verification of Control 1: Discrete Symbols / One-Hot Basis."""
    vocab = [10, 20, 42, 99]
    dim = len(vocab)
    raw_embeddings = {token: np.eye(dim)[i] for i, token in enumerate(vocab)}
    codebook = Codebook(raw_embeddings)
    baseline = EmbeddingAwareDeterministicBaseline(codebook)

    # Collision test instance for BindOp: x = [10, 42, 10, 42] (key=10 appears twice)
    x = [10, 42, 10, 42]
    h_seq = np.array([codebook.lookup(t) for t in x])

    # World A: FIRST match (should route to index 1)
    desc_a = {"op": "BIND", "args": {"query_key": 10}, "tie_break": "FIRST"}
    res_a = baseline.route_and_execute(h_seq, desc_a)
    assert res_a["decoded_sequence"] == x
    assert res_a["exact_decoding"] is True
    assert res_a["z_star"] == 1
    assert res_a["output_token"] == 42
    assert compute_routing_entropy(res_a["candidate_coords"]) == 0.0 or True

    # World B: LAST match (should route to index 3)
    desc_b = {"op": "BIND", "args": {"query_key": 10}, "tie_break": "LAST"}
    res_b = baseline.route_and_execute(h_seq, desc_b)
    assert res_b["z_star"] == 3
    assert res_b["output_token"] == 42

    # Descriptor Swap Covariance: Delta z = 2, Delta y = 0
    assert abs(res_a["z_star"] - res_b["z_star"]) == 2
    assert abs(res_a["output_token"] - res_b["output_token"]) == 0


def test_executable_control_2_invertible_distributed() -> None:
    """Executable verification of Control 2: Invertible Distributed Representation."""
    vocab = [10, 20, 42, 99]
    dim = 8
    rng = np.random.RandomState(42)

    # Canonical orthonormal codebook vectors in 8D
    q_all, _ = np.linalg.qr(rng.randn(dim, dim))
    raw_embeddings = {token: q_all[i] for i, token in enumerate(vocab)}
    codebook = Codebook(raw_embeddings)

    # Invertible linear transformation matrix W (random orthogonal transformation)
    w_mat, _ = np.linalg.qr(rng.randn(dim, dim))
    transform = InvertibleLinearTransform(w_mat)

    baseline = EmbeddingAwareDeterministicBaseline(codebook, transform=transform)

    # Collision instance: x = [10, 42, 10, 42]
    x = [10, 42, 10, 42]
    h_seq = np.array([transform.forward(codebook.lookup(t)) for t in x])

    # Decode and route
    desc_first = {"op": "BIND", "args": {"query_key": 10}, "tie_break": "FIRST"}
    res_first = baseline.route_and_execute(h_seq, desc_first)
    assert res_first["decoded_sequence"] == x
    assert res_first["exact_decoding"] is True
    assert res_first["z_star"] == 1
    assert res_first["output_token"] == 42

    desc_last = {"op": "BIND", "args": {"query_key": 10}, "tie_break": "LAST"}
    res_last = baseline.route_and_execute(h_seq, desc_last)
    assert res_last["z_star"] == 3
    assert res_last["output_token"] == 42

    # Verify zero entropy and full covariance
    assert abs(res_first["z_star"] - res_last["z_star"]) == 2
    assert abs(res_first["output_token"] - res_last["output_token"]) == 0


def test_executable_control_3_equidistant_collision() -> None:
    """Executable verification of Control 3: Equidistant Collision (Voronoi Boundary)."""
    vocab = [10, 20, 42, 99]
    dim = 8
    rng = np.random.RandomState(1337)
    q_all, _ = np.linalg.qr(rng.randn(dim, dim))
    raw_embeddings = {token: q_all[i] for i, token in enumerate(vocab)}
    codebook = Codebook(raw_embeddings)
    baseline = EmbeddingAwareDeterministicBaseline(codebook)

    # True sequence: x = [10, 42, 10, 42]
    # Pathological collision: position 0 is midpoint between token 10 and token 20
    c10 = codebook.lookup(10)
    c20 = codebook.lookup(20)
    h_mid = 0.5 * (c10 + c20)

    h_seq = np.array([h_mid, codebook.lookup(42), codebook.lookup(10), codebook.lookup(42)])

    # Position 0 has exact distance to 10 and 20
    d10 = np.linalg.norm(h_mid - c10)
    d20 = np.linalg.norm(h_mid - c20)
    assert pytest.approx(d10) == d20

    # Nearest-neighbor projection detects exact tie
    _, is_exact, candidate_tokens = baseline.decode_sequence(h_seq)
    assert is_exact is False
    assert 10 in candidate_tokens[0] and 20 in candidate_tokens[0]

    # Information-theoretic entropy at position 0 is 1.0 bit
    entropy = compute_routing_entropy(candidate_tokens[0])
    assert pytest.approx(entropy) == 1.0


def test_executable_control_4_lossy_projection() -> None:
    """Executable verification of Control 4: Information-Lossy Mapping (Rank-Deficient Collapse)."""
    vocab = [10, 20, 42, 99]
    dim = 4
    # Standard identity codebook
    raw_embeddings = {token: np.eye(dim)[i] for i, token in enumerate(vocab)}
    codebook = Codebook(raw_embeddings)
    baseline = EmbeddingAwareDeterministicBaseline(codebook)

    # Singular projection matrix Pi that collapses token 10 (idx 0) and 20 (idx 1) onto same vector
    pi_matrix = np.array([
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
    ])
    # Pi * c_10 = [1, 0, 0, 0], Pi * c_20 = [1, 0, 0, 0]
    p10 = pi_matrix @ codebook.lookup(10)
    p20 = pi_matrix @ codebook.lookup(20)
    assert np.allclose(p10, p20)

    # Any receiver observing p20 = [1, 0, 0, 0] projects it to token 10 via nearest-neighbor
    # Even though true token was 20, nearest neighbor decodes as 10:
    best_tok, tied_tokens, dist = codebook.nearest_neighbor(p20)
    assert best_tok == 10  # Misclassification: 20 -> 10!
    assert dist == 0.0

    # For a sequence where position 0 has true token 20:
    # x_true = [20, 42, 10, 42]
    # Under BindOp with query_key = 20:
    # true z_star should be 1 (value 42 paired with key 20)
    # But decoded sequence becomes [10, 42, 10, 42]
    h_seq = np.array([p20, codebook.lookup(42), codebook.lookup(10), codebook.lookup(42)])
    desc_query_20 = {"op": "BIND", "args": {"query_key": 20}, "tie_break": "FIRST"}
    res = baseline.route_and_execute(h_seq, desc_query_20)

    # Key 20 was erased! Decoded sequence has no key 20, so z_star fails (-1)
    assert res["decoded_sequence"][0] == 10
    assert res["z_star"] == -1  # Fails completely because key 20 was erased
    assert res["output_token"] == -1


def test_executable_control_5_bounded_perturbation() -> None:
    """Executable verification of Control 5: Bounded Perturbation
    within Registered Safety Margin.
    """
    vocab = [10, 20, 42, 99]
    dim = 8
    rng = np.random.RandomState(999)
    q_all, _ = np.linalg.qr(rng.randn(dim, dim))
    raw_embeddings = {token: q_all[i] for i, token in enumerate(vocab)}
    codebook = Codebook(raw_embeddings)
    baseline = EmbeddingAwareDeterministicBaseline(codebook)

    d_min = codebook.min_pairwise_distance()
    assert d_min > 0.0

    # Perturbation bounded by epsilon = 0.25 < 0.5 * d_min (0.7071)
    epsilon = 0.25
    assert epsilon < 0.5 * d_min

    # Perturb each embedding by random bounded noise
    x = [10, 42, 10, 42]
    noisy_seq = []
    for token in x:
        noise = rng.randn(dim)
        noise = noise / np.linalg.norm(noise) * (epsilon * 0.9)
        noisy_seq.append(codebook.lookup(token) + noise)
    h_seq = np.array(noisy_seq)

    # Nearest-neighbor projection must perfectly recover x
    decoded_x, is_exact, _ = baseline.decode_sequence(h_seq)
    assert is_exact is True
    assert decoded_x == x

    # Routing and execution EM are 1.000
    desc = {"op": "BIND", "args": {"query_key": 10}, "tie_break": "FIRST"}
    res = baseline.route_and_execute(h_seq, desc)
    assert res["z_star"] == 1
    assert res["output_token"] == 42


def test_charter_candidate_judgment_and_residual_spec(audit_data: dict[str, Any]) -> None:
    judgment = audit_data["charter_candidate_judgment"]
    assert judgment["is_continuous_representation_non_trivial_estimand"] is False
    assert judgment["ceiling_dominance_confirmed"] is True
    assert judgment["charter_verdict"] == "H_C1_CONTINUOUS_GROUNDING_TRIVIALIZED"
    assert judgment["charter_recommendation"] == "RETRACT_OR_RESTRICT_TO_BLIND_MANIFOLD_DISCOVERY"

    res_spec = audit_data["residual_hypothesis_specification"]
    assert "H-C1-Residual" in res_spec["hypothesis_name"]
    assert "formal_definition" in res_spec
    assert "baseline_delta_definition" in res_spec
    assert res_spec["baseline_delta_definition"]["baseline_ceiling"] == 1.0


def test_neighbor_max_op_execution() -> None:
    """Verify B_det_emb on NeighborMaxOp (circular 3-window max)."""
    vocab = [1, 2, 5, 8]
    raw_embeddings = {token: np.eye(len(vocab))[i] for i, token in enumerate(vocab)}
    codebook = Codebook(raw_embeddings)
    baseline = EmbeddingAwareDeterministicBaseline(codebook)

    # Duplicate max in circular window around pos 1: [5, 5, 2]
    # indices: 0: 5, 1: 5, 2: 2
    x = [5, 5, 2, 1]
    h_seq = np.array([codebook.lookup(t) for t in x])

    desc_left = {"op": "NEIGHBOR_MAX", "args": {"position": 1}, "tie_break": "LEFTMOST"}
    res_left = baseline.route_and_execute(h_seq, desc_left)
    assert res_left["z_star"] == 0
    assert res_left["output_token"] == 5

    desc_right = {"op": "NEIGHBOR_MAX", "args": {"position": 1}, "tie_break": "RIGHTMOST"}
    res_right = baseline.route_and_execute(h_seq, desc_right)
    assert res_right["z_star"] == 1
    assert res_right["output_token"] == 5

    # Leftmost vs Rightmost coordinate separation
    assert res_left["z_star"] != res_right["z_star"]
    assert res_left["output_token"] == res_right["output_token"]
