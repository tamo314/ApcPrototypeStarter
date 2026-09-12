#!/usr/bin/env python3
"""Task B-C005REC-004AP: Stratified Gradient Alignment Diagnostic Runner."""

from __future__ import annotations

import argparse
from pathlib import Path

from apc.evaluation.mirror_cd_dpca_gradient_alignment_strata import (
    CDDPCAGradientAlignmentStrataConfig,
    run_gradient_alignment_strata_task,
)

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task B-C005REC-004AP CD-DPCA Stratified Gradient Alignment Diagnostic"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs/phase_b_restart/rec004ap/run_001",
        help="Run artifact output directory",
    )
    parser.add_argument(
        "--rec004al-dir",
        type=Path,
        default=ROOT / "runs/phase_b_restart/rec004al/run_001",
        help="REC-004AL run directory",
    )
    parser.add_argument(
        "--rec004ak-dir",
        type=Path,
        default=ROOT / "runs/phase_b_restart/rec004ak/run_001",
        help="REC-004AK run directory",
    )
    args = parser.parse_args()

    config = CDDPCAGradientAlignmentStrataConfig(
        output_dir=args.output_dir,
        rec004al_dir=args.rec004al_dir,
        rec004ak_dir=args.rec004ak_dir,
    )
    print("Starting Task B-C005REC-004AP Stratified Gradient Alignment Diagnostic...")
    summary = run_gradient_alignment_strata_task(config)
    print(f"Task completed with status: {summary['status']}")
    print(f"Decision: {summary['decision']}")
    print(f"Decision Reason: {summary['decision_reason']}")

    dist = summary["strata_distribution"]
    pct_a = dist["fractions"]["A"] * 100
    pct_b = dist["fractions"]["B"] * 100
    pct_c = dist["fractions"]["C"] * 100
    print(
        f"Strata Distribution: A(unique)={dist['counts']['A']} ({pct_a:.1f}%), "
        f"B(other)={dist['counts']['B']} ({pct_b:.1f}%), "
        f"C(key7)={dist['counts']['C']} ({pct_c:.1f}%)"
    )

    t6000 = summary["terminal_metrics_step6000"]
    d_a = t6000["stratum_A"]["predicted_margin_change"]
    m_a = t6000["stratum_A"]["per_example_median"]
    d_b = t6000["stratum_B"]["predicted_margin_change"]
    m_b = t6000["stratum_B"]["per_example_median"]
    d_c = t6000["stratum_C"]["predicted_margin_change"]
    m_c = t6000["stratum_C"]["per_example_median"]
    d_p = t6000["pooled"]["predicted_margin_change"]
    print(
        f"Terminal Step 6000 Alignment:\n"
        f"  Stratum A: delta_M={d_a:+.4f}, median={m_a:+.4f}\n"
        f"  Stratum B: delta_M={d_b:+.4f}, median={m_b:+.4f}\n"
        f"  Stratum C: delta_M={d_c:+.4f}, median={m_c:+.4f}\n"
        f"  Pooled:    delta_M={d_p:+.4f}"
    )


if __name__ == "__main__":
    main()
