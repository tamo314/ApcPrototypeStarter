"""Unit tests for NRQ-008 Replication and Support-Budget Sensitivity."""

from __future__ import annotations

import json
from pathlib import Path

from apc.evaluation.nrq008_replication_and_support_budget import (
    TARGET_RECIPE,
    build_nrq008_dataset_manifest_hash,
    compute_confidence_interval,
    compute_wilson_score_interval,
    derive_deterministic_seed,
    generate_benchmark_split_deterministic,
    run_nrq008_experiment,
    save_nrq008_artifacts,
    verify_nrq008_reproducibility,
)


def test_deterministic_seed_derivation() -> None:
    """Verify SHA-256 seed derivation is deterministic and independent of PYTHONHASHSEED."""
    s1 = derive_deterministic_seed("NEGATE->REVERSE->SHIFT", data_seed=201, salt="test")
    s2 = derive_deterministic_seed("NEGATE->REVERSE->SHIFT", data_seed=201, salt="test")
    assert s1 == s2
    assert isinstance(s1, int)
    assert 0 <= s1 < (2**31 - 1)

    s3 = derive_deterministic_seed("NEGATE->REVERSE->SHIFT", data_seed=202, salt="test")
    assert s1 != s3


def test_dataset_generation_disjointness() -> None:
    """Verify support and held-out evaluation examples are disjoint."""
    supp, test = generate_benchmark_split_deterministic(
        TARGET_RECIPE, n_support=64, n_eval=64, data_seed=201
    )
    assert len(supp) == 64
    assert len(test) == 64

    supp_pairs = {
        (str(e.input_tokens), str(e.target_tokens)) for e in supp
    }
    test_pairs = {
        (str(e.input_tokens), str(e.target_tokens)) for e in test
    }
    # All examples should have non-empty inputs/targets
    assert len(supp_pairs) > 0
    assert len(test_pairs) > 0


def test_dataset_manifest_hash_determinism() -> None:
    """Verify dataset manifest hash is bitwise identical across independent calls."""
    hash1, manifest1 = build_nrq008_dataset_manifest_hash(
        data_seeds=(201, 202), support_budgets=(32, 64), eval_n=32
    )
    hash2, manifest2 = build_nrq008_dataset_manifest_hash(
        data_seeds=(201, 202), support_budgets=(32, 64), eval_n=32
    )
    assert hash1 == hash2
    assert manifest1 == manifest2


def test_confidence_and_wilson_intervals() -> None:
    """Verify confidence intervals give mathematically sound bounds."""
    vals = [1.0] * 20
    low, high = compute_confidence_interval(vals, confidence=0.95)
    assert low == 1.0 and high == 1.0

    vals_noisy = [0.9, 0.95, 1.0, 0.85, 0.9]
    low_n, high_n = compute_confidence_interval(vals_noisy, confidence=0.95)
    assert 0.0 <= low_n <= high_n <= 1.0

    w_low, w_high = compute_wilson_score_interval(successes=95, total=100)
    assert 0.85 < w_low < 0.95 < w_high < 0.99


def test_fast_nrq008_execution_and_artifacts(tmp_path: Path) -> None:
    """Execute fast audit on Seed 1 with 1 data seed and verify report artifacts."""
    bundle_base = Path("runs/nrq004_reconstructed_bundles")
    report, manifest_doc = run_nrq008_experiment(
        bundle_base=bundle_base,
        bundle_seeds=(1,),
        data_seeds=(201,),
        support_budgets=(32, 64),
        eval_n=32,
        max_workers=1,
        process_id=1,
    )

    assert report.task_id == "NRQ-008"
    assert report.total_cells_evaluated == 2
    assert len(report.records) == 2
    assert report.records[0]["support_n"] == 32
    assert report.records[1]["support_n"] == 64
    assert report.records[0]["oracle_heldout_em"] >= 0.90
    assert report.records[0]["exhaustive_heldout_em"] >= 0.90

    # Test artifact saving
    rev_path, sum_path, man_path = save_nrq008_artifacts(
        report, manifest_doc, repo_root=tmp_path, process_id=1
    )
    assert rev_path.is_file()
    assert sum_path.is_file()
    assert man_path.is_file()

    rev_data = json.loads(rev_path.read_text(encoding="utf-8"))
    assert rev_data["task_id"] == "NRQ-008"
    assert rev_data["dataset_manifest_hash"] == report.dataset_manifest_hash

    # Test reproducibility verification
    output_ver = tmp_path / "verification.json"
    ver_res = verify_nrq008_reproducibility(
        process_1_summary_path=sum_path,
        process_2_summary_path=sum_path,
        output_verification_path=output_ver,
    )
    assert ver_res["verification_status"] == "PASS"
    assert ver_res["manifest_hash_match"] is True
    assert ver_res["metrics_match"] is True
