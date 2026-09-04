#!/usr/bin/env python
"""CLI entry point for Shared-vs-Specialized Retention Analysis (Phase A.1
diagnostic Task A1-R005E-S003,
`docs/CODEX_TASKS_A1_R005E_SHARED_ENCODER_GATE.md`).

Loads the S002 shared encoder summary and the E-006A specialized encoder summary,
computes per-operation retention ratios (R_shared_correct, R_shared_gap),
classifies qualitative retention bands (strong / mixed / specialized), and
compares against historical controls E-005 (C00) and E-004 (C01).

Writes the standard artifact layout to `--run-dir`:
  - config.yaml
  - system.json
  - report.json
  - summary.json

Usage:
    python scripts/shared_vs_specialized_retention_analysis.py \\
        --config configs/phase_a1_shared_vs_specialized_retention_analysis.yaml \\
        --run-dir runs/phase_a1_shared_vs_specialized_retention_analysis
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from apc.evaluation.shared_vs_specialized_retention_analysis import (
    RetentionConfig,
    analyze_shared_vs_specialized_retention,
    retention_config_from_dict,
)
from apc.utils.config import load_config, save_config
from apc.utils.system_info import get_system_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/phase_a1_shared_vs_specialized_retention_analysis.yaml",
        help="Path to phase_a1_shared_vs_specialized_retention_analysis.yaml",
    )
    parser.add_argument(
        "--run-dir",
        default="runs/phase_a1_shared_vs_specialized_retention_analysis",
        help="Output directory for retention analysis artifacts",
    )
    parser.add_argument(
        "--s002-summary",
        default=None,
        help="Override path to S002 summary.json",
    )
    parser.add_argument(
        "--e006a-summary",
        default=None,
        help="Override path to E-006A summary.json",
    )
    parser.add_argument(
        "--e005-summary",
        default=None,
        help="Override path to E-005 summary.json",
    )
    parser.add_argument(
        "--e004-summary",
        default=None,
        help="Override path to E-004 summary.json",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_config: dict[str, Any] = {}
    if Path(args.config).exists():
        raw_config = load_config(args.config)

    if args.s002_summary is not None:
        raw_config["s002_summary_path"] = args.s002_summary
    if args.e006a_summary is not None:
        raw_config["e006a_summary_path"] = args.e006a_summary
    if args.e005_summary is not None:
        raw_config["e005_summary_path"] = args.e005_summary
    if args.e004_summary is not None:
        raw_config["e004_summary_path"] = args.e004_summary

    config = retention_config_from_dict(raw_config) if raw_config else RetentionConfig()

    save_config(config.to_dict(), run_dir / "config.yaml")

    system_info = get_system_info(seed=0)
    (run_dir / "system.json").write_text(json.dumps(system_info, indent=2), encoding="utf-8")

    report = analyze_shared_vs_specialized_retention(config)

    report_dict = report.to_dict()
    (run_dir / "report.json").write_text(json.dumps(report_dict, indent=2), encoding="utf-8")

    summary_dict = {
        "num_operations_strong": report.num_operations_strong,
        "num_operations_mixed": report.num_operations_mixed,
        "num_operations_specialized": report.num_operations_specialized,
        "global_strong_support": report.global_strong_support,
        "meets_seed_policy": report.meets_seed_policy,
        "s002_seeds": report.s002_seeds,
        "e006a_seeds": report.e006a_seeds,
        "per_operation_summary": {
            op: {
                "operation": s.operation,
                "s002_correct_exact_match": s.s002_correct_exact_match,
                "e006a_correct_exact_match": s.e006a_correct_exact_match,
                "r_shared_correct": s.r_shared_correct,
                "s002_causal_gap": s.s002_causal_gap,
                "e006a_causal_gap": s.e006a_causal_gap,
                "r_shared_gap": s.r_shared_gap,
                "overall_retention_band": s.overall_retention_band,
                "passed_strong_shared_support": s.passed_strong_shared_support,
                "s002_vs_e005_correct_ratio": s.s002_vs_e005_correct_ratio,
                "s002_vs_e004_correct_percentage": s.s002_vs_e004_correct_percentage,
                "variance_reduction_ratio": s.variance_reduction_ratio,
            }
            for op, s in report.per_operation_summary.items()
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary_dict, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("A1-R005E-S003: SHARED-VS-SPECIALIZED RETENTION EVIDENCE TABLE")
    print("=" * 80)
    print(report.evidence_table_markdown)
    print("=" * 80)
    print(
        f"Global Strong Support: {report.global_strong_support} "
        f"({report.num_operations_strong}/4 strong)"
    )
    print(f"Meets Seed Policy: {report.meets_seed_policy}")
    print(f"Artifacts saved to: {run_dir.resolve()}\n")


if __name__ == "__main__":
    main()
