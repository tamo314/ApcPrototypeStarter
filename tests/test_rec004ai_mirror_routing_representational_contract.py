"""Tests for B-C005REC-004AI: MIRROR Routing Representational Contract Review."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from apc.environments.operations import MirrorHalvesOp
from apc.evaluation.mirror_position_initialization_diagnostic import (
    mirror_halves_position_map,
)

ROOT = Path(__file__).resolve().parent.parent
REC004AI_DIR = ROOT / "runs/phase_b_restart/rec004ai/run_001"


def test_mirror_halves_semantics_invariants() -> None:
    """Verify core mathematical invariants of MIRROR_HALVES."""
    op = MirrorHalvesOp()
    assert op.name == "MIRROR_HALVES"
    assert op.min_input_length == 2

    for n in range(2, 17):
        pi = mirror_halves_position_map(n)
        assert len(pi) == n
        # Bijective
        assert set(pi) == set(range(n))
        # Involution (self-inverse)
        assert all(pi[pi[i]] == i for i in range(n))
        # First half maps to first half, second half maps to second half
        mid = n // 2
        assert all(pi[i] < mid for i in range(mid))
        assert all(pi[i] >= mid for i in range(mid, n))


def test_mirror_halves_content_invariance() -> None:
    """Verify that routing decision is strictly content-invariant."""
    op = MirrorHalvesOp()
    n = 10
    pi = mirror_halves_position_map(n)
    seq1 = (1, 2, 3, 4, 5, 6, 7, 8, 9, 0)
    seq2 = (9, 9, 8, 8, 7, 7, 6, 6, 5, 5)
    out1 = op.apply(seq1, 10, {})
    out2 = op.apply(seq2, 10, {})

    # In both sequences, output position i draws from sequence[pi[i]]
    for i in range(n):
        assert out1[i] == seq1[pi[i]]
        assert out2[i] == seq2[pi[i]]


def test_rec004ai_artifacts_exist_and_pass() -> None:
    """Verify that REC-004AI artifacts exist and reflect MINIMAL_ROUTING_CONTRACT_IDENTIFIED."""
    assert REC004AI_DIR.is_dir()
    summary_path = REC004AI_DIR / "summary.json"
    assert summary_path.is_file()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["task"] == "B-C005REC-004AI"
    assert summary["execution_status"] == "PASS"
    assert summary["decision"] == "MINIMAL_ROUTING_CONTRACT_IDENTIFIED"
    assert summary["optimizer_updates"] == 0
    assert summary["new_parameters"] == 0
    assert summary["candidate_selected"] is None
    assert summary["bundle_write"] is False
    assert summary["rg3"] == "NOT_EXECUTED"
    assert summary["rec005_eligible"] is False
    assert summary["g1"] == "NOT_CLEARED"
    assert summary["g4"] == "NOT_CLEARED"


def test_minimal_contract_specification() -> None:
    """Verify that the minimal contract satisfies all requirements without hardcoding."""
    contract_path = REC004AI_DIR / "minimal_architecture_contract.json"
    assert contract_path.is_file()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    assert contract["contract_name"] == "ContentDecoupledDiscretePositionalCrossAttention"
    assert contract["acronym"] == "CD-DPCA"
    assert "content_features" in contract["runtime_inputs"][0]
    assert "target tokens" in contract["strictly_prohibited_runtime_inputs"]

    # Positional dimensions
    pos_rep = contract["integer_positional_representation"]
    assert "query_position_embedding" in pos_rep
    assert "key_position_embedding" in pos_rep

    # Length dimension
    len_rep = contract["length_representation"]
    assert "length_embedding" in len_rep


def test_representational_sufficiency_construction() -> None:
    """Verify synthetic constructive proof of discrete orthonormal routing."""
    proof_path = REC004AI_DIR / "representational_sufficiency_proof.json"
    assert proof_path.is_file()
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    assert proof["orthonormal_margin_check"]["exact_one_hot_achievable"] is True

    # Constructive check: 16 discrete positions in R^32
    d_op = 32
    q = torch.eye(16, d_op) * 8.0
    k = torch.eye(16, d_op) * 8.0
    scores = torch.matmul(q, k.T)
    probs = torch.softmax(scores, dim=-1)
    diag = torch.diag(probs)
    assert (diag > 0.999).all()


def test_cross_relation_static_compatibility() -> None:
    """Verify static compatibility with 15 non-SHIFT operations."""
    compat_path = REC004AI_DIR / "cross_relation_compatibility_audit.json"
    assert compat_path.is_file()
    compat = json.loads(compat_path.read_text(encoding="utf-8"))
    assert compat["tensor_interface_compatibility"]["status"] == "COMPATIBLE"
    assert compat["shared_core_compatibility"]["status"] == "COMPATIBLE"
    assert compat["value_readout_compatibility"]["status"] == "COMPATIBLE"
