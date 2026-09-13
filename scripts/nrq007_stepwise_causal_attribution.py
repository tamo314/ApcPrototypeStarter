"""CLI runner for Task NRQ-007 Stepwise Causal Attribution.

Usage:
    python scripts/nrq007_stepwise_causal_attribution.py
    python scripts/nrq007_stepwise_causal_attribution.py --output-dir runs/nrq007_causal_attribution
"""

from __future__ import annotations

import argparse
from pathlib import Path

from apc.evaluation.nrq007_stepwise_causal_attribution import (
    run_nrq007_causal_attribution,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Task NRQ-007 Stepwise Causal Attribution."
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("runs/nrq006_depth3_audit/summary_process_1.json"),
        help="Path to NRQ-006 summary_process_1.json",
    )
    parser.add_argument(
        "--bundle-base",
        type=Path,
        default=Path("runs/nrq004_reconstructed_bundles"),
        help="Base directory for intact reconstructed bundles.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/nrq007_causal_attribution"),
        help="Directory to save attribution artifacts.",
    )
    parser.add_argument(
        "--support-n",
        type=int,
        default=32,
        help="Support set size (default: 32).",
    )
    parser.add_argument(
        "--eval-n",
        type=int,
        default=50,
        help="Evaluation set size (default: 50).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=== Starting NRQ-007 Stepwise Causal Attribution ===")
    print(f"NRQ-006 Summary: {args.summary_path}")
    print(f"Bundle Base: {args.bundle_base}")
    print(f"Output Directory: {args.output_dir}")

    report, reagg = run_nrq007_causal_attribution(
        summary_path=args.summary_path,
        bundle_base=args.bundle_base,
        support_n=args.support_n,
        eval_n=args.eval_n,
        output_dir=args.output_dir,
    )

    print("\n=== Reaggregation Reconciliation ===")
    print(f"Total canonical classes: {reagg.total_classes}")
    print(f"Total cells: {reagg.total_cells}")
    print(f"Length-adequate classes: {reagg.length_adequate_classes_count}")
    print(f"Length-contracted classes: {reagg.length_contracted_classes_count}")
    print(f"Failure classes (mean threshold): {reagg.mean_failure_classes_count}")
    print(f"Passing classes (all-cell gate): {reagg.all_cell_passing_classes_count}")
    print(f"Failing classes (all-cell gate): {reagg.all_cell_failing_classes_count}")

    print("\n=== Attribution Results (35 Failure Classes) ===")
    for attr, count in report.attribution_class_counts.items():
        print(f"  {attr}: {count} classes")

    print("\n=== Decision & Gate Status ===")
    print(f"ADR-0166 Status Amendment: {report.adr0166_status_amendment}")
    print(f"Decision: {report.adr_decision}")
    print("Execution complete. Artifacts saved successfully.")


if __name__ == "__main__":
    main()
