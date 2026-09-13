#!/usr/bin/env python
"""CLI runner for NRQ-005 — Exact-Depth-3 Irreducible Composition Benchmark.

Usage:
    python scripts/nrq005_exact_depth3_benchmark.py [--fast]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src is on pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from apc.evaluation.nrq005_exact_depth3_benchmark import (
    run_nrq005_benchmark,
    save_nrq005_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NRQ-005: Exact-Depth-3 Irreducible Composition Benchmark"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run in fast mode (evaluates 2 data seeds instead of 5 for quick check)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    print("=" * 80)
    print("NRQ-005: Missing-Artifact-Robust Exact-Depth-3 Irreducible Composition Benchmark")
    print("=" * 80)

    report = run_nrq005_benchmark(
        bundle_base=repo_root / "runs" / "nrq004_reconstructed_bundles",
        fast_mode=args.fast,
    )

    review_file, summary_file = save_nrq005_artifacts(report, repo_root=repo_root)
    print(f"\n[Artifact] Review record written to: {review_file}")
    print(f"[Artifact] Run summary written to: {summary_file}")

    print("\n" + "=" * 80)
    print("NRQ-005 BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    print(f"Protocol Type: {report.protocol_type}")
    print(f"Seed 0 Exclusion: {report.seed0_exclusion_reason}")
    print(f"Bundles Evaluated: {report.bundle_seeds_evaluated}")
    print(f"Data Seeds Evaluated: {report.data_seeds_evaluated}")
    print(f"Support N: {report.support_n}, Eval N: {report.eval_n}")
    print(f"Oracle Floor Passed (>= {report.oracle_floor_threshold}): {report.oracle_floor_passed}")
    print(f"Mean Oracle EM (Irreducible Depth-3): {report.mean_oracle_em_irreducible:.4f}")
    print(f"Mean Beam Search EM (Irreducible Depth-3): {report.mean_beam_em_irreducible:.4f}")
    print(
        f"Mean Exhaustive Search EM (Irreducible Depth-3): "
        f"{report.mean_exhaustive_em_irreducible:.4f}"
    )
    print(
        f"Mean Exhaustive Recipe Recovery: "
        f"{report.mean_exhaustive_recovery_irreducible:.4f}"
    )
    print(
        f"Mean Oracle EM (Reducible Depth-3 Controls): "
        f"{report.mean_oracle_em_reducible_depth3:.4f}"
    )
    print(
        f"Mean Exhaustive EM (Reducible Depth-3 Controls): "
        f"{report.mean_exhaustive_em_reducible_depth3:.4f}"
    )
    print(f"Mean Oracle EM (Depth-2 Controls): {report.mean_oracle_em_depth2:.4f}")
    print(f"Mean Exhaustive EM (Depth-2 Controls): {report.mean_exhaustive_em_depth2:.4f}")

    print("\n--- PER-BUNDLE IRREDUCIBLE DEPTH-3 PERFORMANCE ---")
    for b_seed, s_info in sorted(report.per_bundle_summary.items()):
        print(
            f"Bundle {b_seed}: Oracle EM = {s_info['mean_oracle_em']:.4f} | "
            f"Beam EM = {s_info['mean_beam_em']:.4f} | "
            f"Exhaustive EM = {s_info['mean_exhaustive_em']:.4f} | "
            f"Exhaustive Recovery = {s_info['mean_exhaustive_recovery']:.4f}"
        )

    print("\n--- BEAM-BUDGET SENSITIVITY SWEEP (Irreducible Depth-3 Panel) ---")
    for pt in report.beam_budget_sensitivity:
        print(
            f"Width = {pt['beam_width']:2d} | Recovery = {pt['mean_recipe_recovery']:.2%} | "
            f"EM = {pt['mean_functional_em']:.4f} | "
            f"Eval Cands = {pt['mean_candidates_evaluated']:.1f} | "
            f"Pruned = {pt['mean_candidates_pruned']:.1f} | "
            f"Time = {pt['mean_time_seconds']:.3f}s"
        )

    print("\n" + "=" * 80)
    print(f"DECISION: {report.adr0162_decision}")
    print(f"RATIONALE: {report.decision_rationale}")
    print("=" * 80)


if __name__ == "__main__":
    main()
