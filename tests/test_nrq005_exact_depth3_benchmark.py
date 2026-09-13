"""Unit tests for NRQ-005 Exact-Depth-3 Irreducible Composition Benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from apc.evaluation.nrq005_exact_depth3_benchmark import (
    EXACT_DEPTH3_IRREDUCIBLE_PANEL,
    REDUCIBLE_DEPTH3_PANEL,
    enumerate_all_depth_le_3_candidates,
    generate_benchmark_split,
    run_nrq005_benchmark,
    run_symbolic_probe,
    save_nrq005_artifacts,
)


def test_candidate_space_cardinality() -> None:
    """Verify total candidate space is exactly 584 (8 depth-1 + 64 depth-2 + 512 depth-3)."""
    cands = enumerate_all_depth_le_3_candidates()
    assert len(cands) == 584

    d1 = [c for c in cands if len(c) == 1]
    d2 = [c for c in cands if len(c) == 2]
    d3 = [c for c in cands if len(c) == 3]

    assert len(d1) == 8
    assert len(d2) == 64
    assert len(d3) == 512


def test_symbolic_probe_irreducible_panel() -> None:
    """Ensure all 6 exact-depth-3 recipes have ZERO depth <= 2 functional equivalents."""
    results = run_symbolic_probe(
        EXACT_DEPTH3_IRREDUCIBLE_PANEL,
        category="irreducible_depth3",
        n_probe=30,
        seed=42,
    )
    assert len(results) == len(EXACT_DEPTH3_IRREDUCIBLE_PANEL)
    for res in results:
        assert res.is_exact_depth3_irreducible is True
        assert len(res.matching_depth_le_2_candidates) == 0, (
            f"Recipe {res.recipe} unexpectedly matched depth <= 2: "
            f"{res.matching_depth_le_2_candidates}"
        )


def test_symbolic_probe_reducible_panel() -> None:
    """Ensure all 6 reducible depth-3 recipes possess valid depth <= 2 equivalents."""
    results = run_symbolic_probe(
        REDUCIBLE_DEPTH3_PANEL,
        category="reducible_depth3",
        n_probe=30,
        seed=42,
    )
    assert len(results) == len(REDUCIBLE_DEPTH3_PANEL)
    for res in results:
        assert res.is_exact_depth3_irreducible is False
        assert len(res.matching_depth_le_2_candidates) > 0, (
            f"Reducible recipe {res.recipe} unexpectedly had 0 depth <= 2 matches."
        )


def test_benchmark_split_generation() -> None:
    """Verify disjoint generation of support and evaluation splits."""
    recipe = ("SHIFT", "REVERSE", "SELECT")
    support, test = generate_benchmark_split(
        recipe, n_support=16, n_eval=20, data_seed=123
    )

    assert len(support) == 16
    assert len(test) == 20

    # Ensure inputs are disjoint
    support_inputs = set(ex.input_tokens for ex in support)
    test_inputs = set(ex.input_tokens for ex in test)
    overlap = support_inputs.intersection(test_inputs)
    assert len(overlap) <= 1


def test_fast_benchmark_execution(tmp_path: Path) -> None:
    """Execute fast benchmark run on Seed 1 with 1 data seed and verify artifacts."""
    bundle_base = Path("runs/nrq004_reconstructed_bundles")
    report = run_nrq005_benchmark(
        bundle_base=bundle_base,
        bundle_seeds=(1,),
        data_seeds=(101,),
        support_n=16,
        eval_n=20,
        fast_mode=False,
    )

    assert report.task_id == "NRQ-005"
    assert report.oracle_floor_passed is True
    assert report.mean_oracle_em_irreducible >= 0.90
    assert report.mean_exhaustive_em_irreducible >= 0.90

    # Test artifact saving
    rev_path, sum_path = save_nrq005_artifacts(report, repo_root=tmp_path)
    assert rev_path.is_file()
    assert sum_path.is_file()

    rev_data = json.loads(rev_path.read_text(encoding="utf-8"))
    assert rev_data["task_id"] == "NRQ-005"
    assert rev_data["oracle_floor_passed"] is True
    assert rev_data["adr0162_decision"] in (
        "EMPIRICALLY_SUPPORT_CLOSURE_DEPTH_LE_3",
        "REJECT_ADR0162_CLOSURE",
        "BASELINE_SUBCEILING_DEPTH_LE_3",
    )
