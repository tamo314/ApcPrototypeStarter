#!/usr/bin/env python
"""CLI runner for NRQ-006 — Argument-Closed Deterministic Full-Registry Depth-3 Closure Audit.

Usage:
    python scripts/nrq006_argument_closed_depth3_audit.py --process-id 1
    python scripts/nrq006_argument_closed_depth3_audit.py --process-id 2 \
        --verify-against runs/nrq006_depth3_audit/summary_process_1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure src is on pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from apc.evaluation.nrq006_argument_closed_depth3_audit import (
    DEFAULT_BUNDLE_SEEDS,
    DEFAULT_DATA_SEEDS,
    DEFAULT_EVAL_N,
    DEFAULT_SUPPORT_N,
    run_nrq006_audit,
    save_nrq006_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NRQ-006: Argument-Closed Deterministic Full-Registry Depth-3 Closure Audit"
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
        help="Data seeds to evaluate (default: 101 102 103 104 105)",
    )
    parser.add_argument(
        "--support-n",
        type=int,
        default=DEFAULT_SUPPORT_N,
        help="Number of support examples (default: 32)",
    )
    parser.add_argument(
        "--eval-n",
        type=int,
        default=DEFAULT_EVAL_N,
        help="Number of evaluation examples (default: 50)",
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
    print("=" * 80)
    print("NRQ-006: Argument-Closed Deterministic Full-Registry Depth-3 Closure Audit")
    print(f"Process ID: {args.process_id}")
    print("=" * 80)

    report, manifest_doc = run_nrq006_audit(
        bundle_base=repo_root / "runs" / "nrq004_reconstructed_bundles",
        bundle_seeds=tuple(args.bundle_seeds),
        data_seeds=tuple(args.data_seeds),
        support_n=args.support_n,
        eval_n=args.eval_n,
        max_workers=args.max_workers,
        process_id=args.process_id,
    )

    review_file, summary_file, manifest_file = save_nrq006_artifacts(
        report,
        manifest_doc,
        repo_root=repo_root,
        process_id=args.process_id,
    )

    print(f"\n[Artifact] Review record written to: {review_file}")
    print(f"[Artifact] Run summary written to: {summary_file}")
    print(f"[Artifact] Dataset manifest written to: {manifest_file}")

    print("\n" + "=" * 80)
    print("NRQ-006 AUDIT & BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Task ID: {report.task_id}")
    print(f"Process ID: {report.process_id}")
    print(f"Dataset Manifest Hash: {report.dataset_manifest_hash}")
    print(f"Total 3-Op Recipes Audited: {report.audit_summary['total_audited']}")
    print(f"Structurally Invalid Count: {report.audit_summary['structurally_invalid_count']}")
    print(f"Reducible Recipes Count: {report.audit_summary['reducible_count']}")
    print(f"Irreducible Recipes Count: {report.audit_summary['irreducible_count']}")
    print(f"Irreducible Equivalence Classes: {report.audit_summary['equivalence_classes_count']}")
    print(f"Canonical Classes Evaluated: {len(report.selected_canonical_classes)}")
    print(f"Bundles Evaluated: {report.bundle_seeds_evaluated}")
    print(f"Data Seeds Evaluated: {report.data_seeds_evaluated}")
    print(f"Support N: {report.support_n}, Eval N: {report.eval_n}")

    print("\n--- PERFORMANCE METRICS ---")
    print(f"All 57 Classes Mean Oracle EM: {report.mean_oracle_em_all:.4f}")
    print(f"All 57 Classes Mean Exhaustive Baseline EM: {report.mean_exhaustive_em_all:.4f}")
    print(
        f"Length-Adequate (41 Classes) Mean Oracle EM: "
        f"{report.mean_oracle_em_length_adequate:.4f}"
    )
    print(
        f"Length-Adequate (41 Classes) Mean Baseline EM: "
        f"{report.mean_exhaustive_em_length_adequate:.4f}"
    )
    print(
        f"Length-Adequate Mean Recipe Recovery: "
        f"{report.mean_exhaustive_recovery_length_adequate:.4f}"
    )
    print(f"Oracle Floor Passed (Length-Adequate): {report.oracle_floor_passed_length_adequate}")
    print(f"Failure Classes Count: {report.failure_classes_count}")

    if report.failure_classes:
        print("\n--- FAILURE CLASSES (Subthreshold Oracle Floor / Baseline) ---")
        for fc in report.failure_classes:
            print(
                f"  Class: {fc['class_name']:30s} | "
                f"Oracle EM: {fc['mean_oracle_em']:.4f} | "
                f"Exh EM: {fc['mean_exhaustive_em']:.4f} | "
                f"Reason: {fc['failure_reason']}"
            )

    print("\n--- PER-BUNDLE SUMMARY (Length-Adequate Panel) ---")
    for b_seed, b_info in sorted(report.per_bundle_summary.items()):
        print(
            f"Bundle {b_seed}: Oracle EM = {b_info['mean_oracle_em']:.4f} | "
            f"Exhaustive EM = {b_info['mean_exhaustive_em']:.4f} | "
            f"All >= 0.95: {b_info['baseline_all_ge_95']}"
        )

    print("\n" + "=" * 80)
    print(f"ADR-0165 STATUS: {report.adr0165_status}")
    print(f"DECISION: {report.adr_decision}")
    print(f"RATIONALE: {report.decision_rationale}")
    print("=" * 80)

    # Verification against previous process
    if args.verify_against:
        verify_path = Path(args.verify_against)
        if not verify_path.is_file():
            print(f"\n[ERROR] Verification file not found: {verify_path}")
            sys.exit(1)

        prev_data = json.loads(verify_path.read_text(encoding="utf-8"))
        print("\n" + "=" * 80)
        print("INTER-PROCESS REPRODUCIBILITY VERIFICATION")
        print("=" * 80)

        # 1. Dataset Manifest Hash
        hash_match = report.dataset_manifest_hash == prev_data["dataset_manifest_hash"]
        print(f"Dataset Manifest Hash Match: {hash_match}")
        print(f"  Process {report.process_id}: {report.dataset_manifest_hash}")
        print(
            f"  Reference Process {prev_data['process_id']}: "
            f"{prev_data['dataset_manifest_hash']}"
        )
        if not hash_match:
            print("[FAIL] Manifest hash mismatch across independent processes!")
            sys.exit(1)

        # 2. Selected Classes Match
        classes_match = report.selected_canonical_classes == prev_data["selected_canonical_classes"]
        print(
            f"Selected Canonical Classes Match: {classes_match} "
            f"({len(report.selected_canonical_classes)} classes)"
        )
        if not classes_match:
            print("[FAIL] Selected canonical classes mismatch across independent processes!")
            sys.exit(1)

        # 3. Primary Metrics Match
        diff_oracle = abs(report.mean_oracle_em_all - prev_data["mean_oracle_em_all"])
        diff_exh = abs(report.mean_exhaustive_em_all - prev_data["mean_exhaustive_em_all"])
        diff_adeq_oracle = abs(
            report.mean_oracle_em_length_adequate - prev_data["mean_oracle_em_length_adequate"]
        )
        diff_adeq_exh = abs(
            report.mean_exhaustive_em_length_adequate
            - prev_data["mean_exhaustive_em_length_adequate"]
        )

        metrics_match = (
            diff_oracle < 1e-6
            and diff_exh < 1e-6
            and diff_adeq_oracle < 1e-6
            and diff_adeq_exh < 1e-6
        )
        print(f"Primary Metrics Match: {metrics_match}")
        print(f"  Delta Mean Oracle EM: {diff_oracle:.2e}")
        print(f"  Delta Mean Exhaustive EM: {diff_exh:.2e}")
        print(f"  Delta Length-Adequate Oracle EM: {diff_adeq_oracle:.2e}")
        print(f"  Delta Length-Adequate Exhaustive EM: {diff_adeq_exh:.2e}")

        if not metrics_match:
            print("[FAIL] Metric mismatch across independent processes!")
            sys.exit(1)

        print(
            "\n[VERIFICATION PASS] Inter-process determinism and exact reproducibility confirmed!"
        )
        repro_file = (
            repo_root / "runs" / "nrq006_depth3_audit" / "process_reproducibility_verification.json"
        )
        with open(repro_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "verification_status": "PASS",
                    "process_1_id": prev_data["process_id"],
                    "process_2_id": report.process_id,
                    "dataset_manifest_hash": report.dataset_manifest_hash,
                    "manifest_hash_match": hash_match,
                    "classes_match": classes_match,
                    "metrics_match": metrics_match,
                    "mean_oracle_em_all": report.mean_oracle_em_all,
                    "mean_exhaustive_em_all": report.mean_exhaustive_em_all,
                    "mean_oracle_em_length_adequate": report.mean_oracle_em_length_adequate,
                    "mean_exhaustive_em_length_adequate": report.mean_exhaustive_em_length_adequate,
                },
                f,
                indent=2,
            )
        print(f"[Artifact] Verification record written to: {repro_file}")


if __name__ == "__main__":
    main()
