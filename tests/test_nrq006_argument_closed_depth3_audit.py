"""Unit tests for NRQ-006 Argument-Closed Deterministic Full-Registry Depth-3 Closure Audit."""

from __future__ import annotations

import json
from pathlib import Path

from apc.evaluation.nrq006_argument_closed_depth3_audit import (
    CANDIDATE_DEPTH_LE_2_PROGRAMS,
    build_dataset_manifest_hash,
    derive_deterministic_seed,
    run_full_registry_audit,
    run_nrq006_audit,
    save_nrq006_artifacts,
)


def test_deterministic_seed_derivation() -> None:
    """Verify SHA-256 seed derivation is deterministic and independent of PYTHONHASHSEED."""
    s1 = derive_deterministic_seed("SHIFT->REVERSE->SELECT", data_seed=101, salt="test")
    s2 = derive_deterministic_seed("SHIFT->REVERSE->SELECT", data_seed=101, salt="test")
    assert s1 == s2
    assert isinstance(s1, int)
    assert 0 <= s1 < (2**31 - 1)

    s3 = derive_deterministic_seed("SHIFT->REVERSE->SELECT", data_seed=102, salt="test")
    assert s1 != s3


def test_grammar_candidate_program_cardinality() -> None:
    """Verify candidate programs under argument grammar are finite and non-empty."""
    assert len(CANDIDATE_DEPTH_LE_2_PROGRAMS) > 0
    # Depth-1 and Depth-2 sequences are included
    depths = {len(seq) for seq, _, _ in CANDIDATE_DEPTH_LE_2_PROGRAMS}
    assert depths == {1, 2}


def test_symbolic_audit_partitioning() -> None:
    """Verify 512 depth-3 recipes partition into 42 invalid, 370 reducible, 100 irreducible."""
    summary = run_full_registry_audit()
    assert summary.total_audited == 512
    assert summary.structurally_invalid_count == 42
    assert summary.reducible_count == 370
    assert summary.irreducible_count == 100
    assert summary.equivalence_classes_count == 60
    assert len(summary.canonical_selected_classes) == 60

    # Check that canonical NRQ-005 recipes are included in irreducible panel
    nrq005_recipes = {
        ("SHIFT", "REVERSE", "SELECT"),
        ("SHIFT", "NEGATE", "SELECT"),
        ("REVERSE", "NEGATE", "SELECT"),
        ("SHIFT", "REVERSE", "BIND"),
        ("NEGATE", "SHIFT", "SELECT"),
        ("REVERSE", "SHIFT", "BIND"),
    }
    irred_set = set(summary.irreducible_recipes)
    assert nrq005_recipes.issubset(irred_set)


def test_dataset_manifest_hash_determinism() -> None:
    """Verify dataset manifest hash is bitwise identical across independent calls."""
    sample_recipes = [("SHIFT", "REVERSE", "SELECT"), ("SHIFT", "NEGATE", "SELECT")]
    hash1, manifest1 = build_dataset_manifest_hash(
        sample_recipes, data_seeds=(101,), support_n=8, eval_n=10
    )
    hash2, manifest2 = build_dataset_manifest_hash(
        sample_recipes, data_seeds=(101,), support_n=8, eval_n=10
    )
    assert hash1 == hash2
    assert manifest1 == manifest2


def test_fast_nrq006_audit_execution(tmp_path: Path) -> None:
    """Execute fast audit on Seed 1 with 1 data seed and verify report artifacts."""
    bundle_base = Path("runs/nrq004_reconstructed_bundles")
    report, manifest_doc = run_nrq006_audit(
        bundle_base=bundle_base,
        bundle_seeds=(1,),
        data_seeds=(101,),
        support_n=16,
        eval_n=20,
        max_workers=1,
        process_id=1,
    )

    assert report.task_id == "NRQ-006"
    assert report.audit_summary["equivalence_classes_count"] == 60
    assert report.oracle_floor_passed_length_adequate is True
    assert report.mean_oracle_em_length_adequate >= 0.90
    assert report.mean_exhaustive_em_length_adequate >= 0.90
    assert report.failure_classes_count > 0
    assert report.adr0165_status == "QUALIFIED"

    # Test artifact saving
    rev_path, sum_path, man_path = save_nrq006_artifacts(
        report, manifest_doc, repo_root=tmp_path, process_id=1
    )
    assert rev_path.is_file()
    assert sum_path.is_file()
    assert man_path.is_file()

    rev_data = json.loads(rev_path.read_text(encoding="utf-8"))
    assert rev_data["task_id"] == "NRQ-006"
    assert rev_data["dataset_manifest_hash"] == report.dataset_manifest_hash
    assert rev_data["adr_decision"] == "ADR0165_QUALIFIED_BY_FULL_REGISTRY_AUDIT"
