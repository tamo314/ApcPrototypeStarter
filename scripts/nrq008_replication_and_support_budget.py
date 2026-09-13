#!/usr/bin/env python
"""CLI runner for NRQ-008 Replication and Support-Budget Sensitivity.

Usage:
    python scripts/nrq008_replication_and_support_budget.py --process-id 1
    python scripts/nrq008_replication_and_support_budget.py --process-id 2 \\
        --verify-against runs/nrq008_replication_and_support_budget/summary_process_1.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src is on pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from apc.evaluation.nrq008_replication_and_support_budget import (
    DEFAULT_BUNDLE_SEEDS,
    DEFAULT_DATA_SEEDS,
    DEFAULT_EVAL_N,
    DEFAULT_SUPPORT_BUDGETS,
    run_nrq008_experiment,
    save_nrq008_artifacts,
    verify_nrq008_reproducibility,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NRQ-008: Replication and Support-Budget Sensitivity of Sole Cell"
    )
    parser.add_argument(
        "--bundle-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_BUNDLE_SEEDS),
        help="Bundle seeds to evaluate (default: 1 2 3 4)",
    )
    parser.add_argument(
        "--data-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_DATA_SEEDS),
        help="Data seeds to evaluate (default: 201..220)",
    )
    parser.add_argument(
        "--support-budgets",
        type=int,
        nargs="+",
        default=list(DEFAULT_SUPPORT_BUDGETS),
        help="Support budgets to evaluate (default: 32 64 128 256)",
    )
    parser.add_argument(
        "--eval-n",
        type=int,
        default=DEFAULT_EVAL_N,
        help="Number of evaluation examples per cell (default: 1024)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Max parallel bundle workers (default: 4)",
    )
    parser.add_argument(
        "--process-id",
        type=int,
        default=1,
        help="Process identifier (1 or 2) for inter-process reproducibility testing",
    )
    parser.add_argument(
        "--verify-against",
        type=str,
        default="",
        help="Path to previous process summary JSON to verify exact reproducibility",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    bundle_base = repo_root / "runs" / "nrq004_reconstructed_bundles"

    print("=" * 80)
    print("NRQ-008: Replication and Support-Budget Sensitivity of Sole APC-over-Exhaustive Cell")
    print(f"Process ID: {args.process_id}")
    print(f"Bundles: {args.bundle_seeds} (1-3 Ceiling Controls, 4 Challenge Bundle)")
    d_seeds = args.data_seeds
    print(f"Data Seeds: {len(d_seeds)} seeds ({min(d_seeds)}..{max(d_seeds)})")
    print(f"Support Budgets: {args.support_budgets}")
    print(f"Held-out Evaluation N: {args.eval_n}")
    print("=" * 80)

    report, manifest_doc = run_nrq008_experiment(
        bundle_base=bundle_base,
        bundle_seeds=tuple(args.bundle_seeds),
        data_seeds=tuple(args.data_seeds),
        support_budgets=tuple(args.support_budgets),
        eval_n=args.eval_n,
        max_workers=args.max_workers,
        process_id=args.process_id,
    )

    review_file, summary_file, manifest_file = save_nrq008_artifacts(
        report,
        manifest_doc,
        repo_root=repo_root,
        process_id=args.process_id,
    )

    print(f"\n[Artifact] Review record written to: {review_file}")
    print(f"[Artifact] Run summary written to: {summary_file}")
    print(f"[Artifact] Dataset manifest written to: {manifest_file}")

    print("\n" + "=" * 80)
    print("NRQ-008 BENCHMARK & REPLICATION SUMMARY")
    print("=" * 80)
    print(f"Task ID: {report.task_id}")
    print(f"Process ID: {report.process_id}")
    print(f"Dataset Manifest Hash: {report.dataset_manifest_hash}")
    print(f"Total Cells Evaluated: {report.total_cells_evaluated}")
    print(f"Scientific Decision: {report.scientific_decision}")
    print(f"Rationale: {report.decision_rationale}")

    print("\n--- BUDGET SENSITIVITY SUMMARY ---")
    for _b_key, b_info in sorted(report.budget_summaries.items()):
        print(
            f"Budget N={b_info['support_n']:3d} | "
            f"Oracle Heldout EM: {b_info['mean_oracle_heldout_em']:.4f} | "
            f"Exhaustive EM: {b_info['mean_exhaustive_heldout_em']:.4f} | "
            f"Beam EM: {b_info['mean_beam_heldout_em']:.4f} | "
            f"Delta (Or - Exh): {b_info['mean_oracle_minus_exhaustive_em']:.4f} | "
            f"Exh >= 0.95: {b_info['exhaustive_ge_95_count']}/{b_info['total_cells']} "
            f"({b_info['exhaustive_ge_95_fraction']*100:.1f}%)"
        )

    print("\n--- MAX BUDGET N=256 PER-BUNDLE PERFORMANCE & 95% CI ---")
    for b_seed in report.bundle_seeds_evaluated:
        b_info = report.max_budget_256_bundle_stats[f"bundle_{b_seed}"]
        or_ci = b_info["oracle_em_ci_95"]
        ex_ci = b_info["exhaustive_em_ci_95"]
        or_str = f"Oracle EM = {b_info['mean_oracle_em']:.4f} [{or_ci[0]:.4f}, {or_ci[1]:.4f}]"
        ex_str = f"Exh EM = {b_info['mean_exhaustive_em']:.4f} [{ex_ci[0]:.4f}, {ex_ci[1]:.4f}]"
        print(
            f"Bundle {b_seed} ({b_info['bundle_role']:16s}): "
            f"{or_str} | {ex_str} | "
            f"Delta = {b_info['mean_oracle_minus_exhaustive']:.4f} | "
            f"Falsifies: {b_info['falsifies_closure_cell']}"
        )

    if args.verify_against:
        verify_path = Path(args.verify_against)
        if not verify_path.is_file():
            print(f"\n[ERROR] Verification reference not found: {verify_path}")
            sys.exit(1)

        print("\n" + "=" * 80)
        print("INTER-PROCESS REPRODUCIBILITY VERIFICATION")
        print("=" * 80)
        output_ver_file = (
            repo_root
            / "runs"
            / "nrq008_replication_and_support_budget"
            / "process_reproducibility_verification.json"
        )
        ver_record = verify_nrq008_reproducibility(
            process_1_summary_path=verify_path,
            process_2_summary_path=summary_file,
            output_verification_path=output_ver_file,
        )
        print(f"Verification Status: {ver_record['verification_status']}")
        print(f"Manifest Hash Match: {ver_record['manifest_hash_match']}")
        print(f"Decision Match: {ver_record['scientific_decision_match']}")
        print(f"Metrics Match: {ver_record['metrics_match']}")
        print(f"Max Delta Oracle EM: {ver_record['max_delta_oracle_em']:.2e}")
        print(f"Max Delta Exhaustive EM: {ver_record['max_delta_exhaustive_em']:.2e}")
        print(f"Max Delta Beam EM: {ver_record['max_delta_beam_em']:.2e}")
        print(f"Recipe Mismatches: {ver_record['recipe_mismatches']}")
        print(f"[Artifact] Reproducibility verification written to: {output_ver_file}")

        if ver_record["verification_status"] != "PASS":
            print("[FAIL] Verification failed!")
            sys.exit(1)


if __name__ == "__main__":
    main()
