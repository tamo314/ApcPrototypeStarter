#!/usr/bin/env python3
"""Runner script for Task B-C005REC-004AO: CD-DPCA Position-4 False-Attractor Diagnostic."""

from __future__ import annotations

import argparse
from pathlib import Path

from apc.evaluation.mirror_cd_dpca_gradient_accessibility import (
    CDDPCAGradientAccessibilityConfig,
    run_gradient_accessibility_diagnostic_task,
)

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task B-C005REC-004AO CD-DPCA Gradient Accessibility Diagnostic"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs/phase_b_restart/rec004ao/run_001",
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

    config = CDDPCAGradientAccessibilityConfig(
        output_dir=args.output_dir,
        rec004al_dir=args.rec004al_dir,
        rec004ak_dir=args.rec004ak_dir,
    )
    print("Starting Task B-C005REC-004AO Gradient Accessibility Diagnostic...")
    summary = run_gradient_accessibility_diagnostic_task(config)
    print(f"Task completed with status: {summary['status']}")
    print(f"Decision: {summary['decision']}")
    print(f"Decision Reason: {summary['decision_reason']}")
    pos4_traj = summary["position_4_trajectory"]
    print(
        f"Position 4 Terminal State: top1={pos4_traj['terminal_top1_key']}, "
        f"p(corr)={pos4_traj['terminal_correct_key_prob']:.4e}, "
        f"margin={pos4_traj['terminal_margin']:+.4f}, "
        f"acc={pos4_traj['terminal_accuracy']:.4f}"
    )
    grad_acc = summary["gradient_accessibility"]
    pct = grad_acc["negative_margin_change_checkpoint_fraction"] * 100
    print(
        f"Gradient Metrics: ||g_routing||={grad_acc['terminal_routing_grad_norm']:.4f}, "
        f"delta_M={grad_acc['terminal_predicted_margin_change']:+.4f}, "
        f"cos_align={grad_acc['terminal_cosine_alignment']:+.4f}, "
        f"negative_checkpoints={pct:.1f}%"
    )


if __name__ == "__main__":
    main()
