"""Unit tests for NRQ-004 Frozen Bundle Compatibility Reconstruction & Reproduction."""

from __future__ import annotations

import json
from pathlib import Path

from apc.evaluation.nrq004_bundle_reconstruction import (
    A1_B004_MODEL_VOCAB_SIZE,
    A1_B004_NUM_OPERATIONS,
    audit_and_reconstruct_bundles,
    build_a1_b004_tokens,
    evaluate_seed_depth2_controls,
)


def test_a1_b004_tokens_schema() -> None:
    """Ensure the reconstructed token schema exactly matches Phase A.1 Task A1-B004."""
    tokens = build_a1_b004_tokens(10)
    assert tokens.num_operations == A1_B004_NUM_OPERATIONS
    assert tokens.model_vocab_size == A1_B004_MODEL_VOCAB_SIZE
    assert tokens.env_vocab_size == 10
    assert tokens.arg_span == 10


def test_bundle_provenance_audit(tmp_path: Path) -> None:
    """Verify that provenance auditing correctly identifies intact seeds and broken seed 0."""
    provenance_map = audit_and_reconstruct_bundles(
        seeds=(0, 1, 2, 3, 4),
        repo_root=Path("."),
        destination_namespace=tmp_path / "bundles",
    )

    # Seed 0 must be marked BUNDLE_LOSS due to overwrite
    assert provenance_map[0].bundle_status == "BUNDLE_LOSS"
    assert provenance_map[0].core_status == "OVERWRITTEN_INCOMPATIBLE"
    assert provenance_map[0].bank_status == "INTACT"

    # Seeds 1-4 must be marked COHERENT_VERIFIED
    for s in (1, 2, 3, 4):
        assert provenance_map[s].bundle_status == "COHERENT_VERIFIED"
        assert provenance_map[s].core_status == "INTACT"
        assert provenance_map[s].bank_status == "INTACT"
        assert provenance_map[s].core_token_emb_shape == [A1_B004_MODEL_VOCAB_SIZE, 192]


def test_reconstructed_bundle_manifests(tmp_path: Path) -> None:
    """Check that destination namespace contains correct manifests and copies."""
    bundle_dest = tmp_path / "bundles"
    provenance_map = audit_and_reconstruct_bundles(
        seeds=(0, 1),
        repo_root=Path("."),
        destination_namespace=bundle_dest,
    )

    # Seed 1 manifest
    s1_manifest = bundle_dest / "seed_1" / "manifest.json"
    assert s1_manifest.is_file()
    data1 = json.loads(s1_manifest.read_text(encoding="utf-8"))
    assert data1["bundle_status"] == "COHERENT_VERIFIED"
    assert data1["token_vocab_size"] == A1_B004_MODEL_VOCAB_SIZE
    assert data1["core_sha256"] == provenance_map[1].core_sha256

    # Seed 0 status
    s0_status = bundle_dest / "seed_0" / "status.json"
    assert s0_status.is_file()
    data0 = json.loads(s0_status.read_text(encoding="utf-8"))
    assert data0["bundle_status"] == "BUNDLE_LOSS"


def test_seed1_positive_control_reproduction(tmp_path: Path) -> None:
    """Verify that Seed 1 passes all 6 depth-2 positive controls under reconstructed bundle."""
    bundle_dest = tmp_path / "bundles"
    prov_map = audit_and_reconstruct_bundles(
        seeds=(1,),
        repo_root=Path("."),
        destination_namespace=bundle_dest,
    )

    # Fast evaluation with 50 examples
    eval_res = evaluate_seed_depth2_controls(
        seed=1,
        provenance=prov_map[1],
        bundle_dir=bundle_dest,
        num_eval_examples=50,
        num_adaptation_examples=16,
    )

    assert eval_res.seed_passed is True
    assert eval_res.compositions_passed == 6
    assert eval_res.mean_recovered_exact_match >= 0.95
    assert eval_res.mean_functional_agreement >= 0.99
    assert eval_res.failure_attribution == "NONE_PASSED"


def test_seed0_failure_attribution(tmp_path: Path) -> None:
    """Verify that Seed 0 is definitively attributed to BUNDLE_LOSS."""
    bundle_dest = tmp_path / "bundles"
    prov_map = audit_and_reconstruct_bundles(
        seeds=(0,),
        repo_root=Path("."),
        destination_namespace=bundle_dest,
    )

    eval_res = evaluate_seed_depth2_controls(
        seed=0,
        provenance=prov_map[0],
        bundle_dir=bundle_dest,
        num_eval_examples=30,
        num_adaptation_examples=16,
    )

    assert eval_res.seed_passed is False
    assert eval_res.failure_attribution == "BUNDLE_LOSS"
