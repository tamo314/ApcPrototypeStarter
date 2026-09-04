#!/usr/bin/env python
"""CLI entry point for the COUNT/BIND Argument-Blind Baseline Audit (Phase A.1
diagnostic Task A1-R005E-S004,
`docs/CODEX_TASKS_A1_R005E_SHARED_ENCODER_GATE.md`).

Computes majority-target and content-only argument-blind baselines for COUNT and
BIND across evaluation sets, compares them against observed None-arm accuracies
from E-004, E-005, E-006A, and S002, and recommends future gate criteria.

Writes standard artifacts to `--run-dir`:
  - config.yaml
  - system.json
  - report.json
  - summary.json

Usage:
    python scripts/argument_blind_baseline_audit.py \\
        --config configs/phase_a1_argument_blind_baseline_audit.yaml \\
        --run-dir runs/phase_a1_argument_blind_baseline_audit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from apc.evaluation.argument_blind_baseline_audit import (
    AuditConfig,
    audit_argument_blind_baselines,
    audit_config_from_dict,
)
from apc.utils.config import load_config, save_config
from apc.utils.system_info import get_system_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/phase_a1_argument_blind_baseline_audit.yaml",
        help="Path to phase_a1_argument_blind_baseline_audit.yaml",
    )
    parser.add_argument(
        "--run-dir",
        default="runs/phase_a1_argument_blind_baseline_audit",
        help="Output directory for audit artifacts",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_config: dict[str, Any] = {}
    if Path(args.config).exists():
        raw_config = load_config(args.config)

    config = audit_config_from_dict(raw_config) if raw_config else AuditConfig()

    save_config(config.to_dict(), run_dir / "config.yaml")

    system_info = get_system_info(seed=0)
    (run_dir / "system.json").write_text(json.dumps(system_info, indent=2), encoding="utf-8")

    report = audit_argument_blind_baselines(config)

    report_dict = report.to_dict()
    (run_dir / "report.json").write_text(json.dumps(report_dict, indent=2), encoding="utf-8")

    summary_dict = {
        "operations": list(config.operations),
        "historical_none_ceiling": config.historical_none_ceiling,
        "per_operation": {
            op: {
                "operation": op,
                "majority_baseline_accuracy": a.majority_baseline.majority_accuracy,
                "majority_top_target": a.majority_baseline.top_target,
                "best_content_only_heuristic": a.content_only_baselines.best_heuristic_name,
                "best_content_only_accuracy": (
                    a.content_only_baselines.max_content_only_accuracy
                ),
                "natural_argument_blind_baseline": a.natural_argument_blind_baseline,
                "ceiling_is_below_natural_baseline": a.ceiling_is_below_natural_baseline,
                "mean_observed_none_across_gates": a.mean_observed_none_across_gates,
                "observed_none_consistent_with_natural_baseline": (
                    a.observed_none_consistent_with_natural_baseline
                ),
                "recommended_relative_threshold": a.recommended_relative_threshold,
            }
            for op, a in report.per_operation_audit.items()
        },
        "all_ceilings_below_natural_baseline": all(
            a.ceiling_is_below_natural_baseline for a in report.per_operation_audit.values()
        ),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary_dict, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("A1-R005E-S004: COUNT/BIND ARGUMENT-BLIND BASELINE AUDIT EVIDENCE TABLE")
    print("=" * 80)
    print(report.summary_table_markdown)
    print("=" * 80)
    print("\n" + report.recommendations)
    print(f"\nArtifacts saved to: {run_dir.resolve()}\n")


if __name__ == "__main__":
    main()
