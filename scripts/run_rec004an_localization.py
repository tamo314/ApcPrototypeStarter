#!/usr/bin/env python3
"""Runner script for Task B-C005REC-004AN: CD-DPCA Length-10 Optimization Failure Localization."""

from __future__ import annotations

import argparse
from pathlib import Path

from apc.evaluation.mirror_cd_dpca_failure_localization import (
    CDDPCALocalizationConfig,
    run_failure_localization_task,
)

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task B-C005REC-004AN CD-DPCA Failure Localization"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs/phase_b_restart/rec004an/run_001",
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

    config = CDDPCALocalizationConfig(
        output_dir=args.output_dir,
        rec004al_dir=args.rec004al_dir,
        rec004ak_dir=args.rec004ak_dir,
    )
    print("Starting Task B-C005REC-004AN Failure Localization...")
    summary = run_failure_localization_task(config)
    print(f"Task completed with status: {summary['status']}")
    print(f"Decision: {summary['decision']}")
    print(f"Trajectory Pattern: {summary['trajectory_pattern']}")
    pos4_pct = summary["position_localization"]["position_4_error_fraction"] * 100
    print(f"Position 4 Error Fraction: {pos4_pct:.2f}%")
    print(f"Decision Reason: {summary['decision_reason']}")


if __name__ == "__main__":
    main()
