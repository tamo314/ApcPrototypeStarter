#!/usr/bin/env python
"""CLI entry point for the Shared-Gate Branch Decision (Phase A.1 diagnostic Task
A1-R005E-S005, `docs/CODEX_TASKS_A1_R005E_SHARED_ENCODER_GATE.md`).

Synthesizes S003 retention analysis and S004 argument-blind baseline audit to
formally decide the branch outcome (Outcome B), authorize S006, and record
the decision artifacts.

Writes artifacts to `--run-dir`:
  - system.json
  - report.json
  - summary.json

Usage:
    python scripts/shared_encoder_gate_decision.py \\
        --run-dir runs/phase_a1_shared_encoder_gate_decision
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from apc.evaluation.shared_encoder_gate_decision import (
    DEFAULT_S003_SUMMARY_PATH,
    DEFAULT_S004_SUMMARY_PATH,
    evaluate_shared_encoder_gate_decision,
)
from apc.utils.system_info import get_system_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--s003-summary",
        default=str(DEFAULT_S003_SUMMARY_PATH),
        help="Path to S003 summary.json",
    )
    parser.add_argument(
        "--s004-summary",
        default=str(DEFAULT_S004_SUMMARY_PATH),
        help="Path to S004 summary.json",
    )
    parser.add_argument(
        "--run-dir",
        default="runs/phase_a1_shared_encoder_gate_decision",
        help="Output directory for decision artifacts",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    s003_p = Path(args.s003_summary)
    s004_p = Path(args.s004_summary)

    if not s003_p.exists():
        raise FileNotFoundError(f"S003 summary not found: {s003_p}")
    if not s004_p.exists():
        raise FileNotFoundError(f"S004 summary not found: {s004_p}")

    s003_data = json.loads(s003_p.read_text(encoding="utf-8"))
    s004_data = json.loads(s004_p.read_text(encoding="utf-8"))

    report = evaluate_shared_encoder_gate_decision(s003_data, s004_data)

    system_info = get_system_info(seed=0)
    (run_dir / "system.json").write_text(json.dumps(system_info, indent=2), encoding="utf-8")

    report_dict = report.to_dict()
    (run_dir / "report.json").write_text(json.dumps(report_dict, indent=2), encoding="utf-8")

    summary_dict = {
        "outcome": report.outcome,
        "outcome_title": report.outcome_title,
        "branch_b_supported": report.branch_b_supported,
        "trigger_s006_shift_probe": report.trigger_s006_shift_probe,
        "adopt_baseline_relative_none": report.adopt_baseline_relative_none,
        "per_operation_verdicts": report.per_operation_verdicts,
        "recommendations": report.recommendations,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary_dict, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("A1-R005E-S005: SHARED QUERYABLE REPRESENTATION GATE BRANCH DECISION")
    print("=" * 80)
    print(report.decision_summary_markdown)
    print("=" * 80)
    print(f"\nArtifacts saved to: {run_dir.resolve()}\n")


if __name__ == "__main__":
    main()
