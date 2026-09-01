#!/usr/bin/env python
"""CLI entry point for Task A1-R005D-001 (`docs/CODEX_TASKS_A1_R005_RETRY.md`,
`docs/EXPERIMENT_PLAN_A1_R005_RETRY.md` D1): re-analyze the existing, frozen
A1-R005 run (`apc.evaluation.parameterized_primitive_gate`, STOP GATE, H2c,
`docs/DECISIONS.md` ADR-0029) without retraining.

Reads `--run-dir` (default target: `runs/phase_a1_parameterized_primitive_gate`,
A1-R005's own frozen artifact directory) read-only -- it is never modified --
and writes this re-analysis's own report to `--output-dir`.

Usage:
    python scripts/r005d001_reanalyze_parameterized_primitive_gate.py \\
        --run-dir runs/phase_a1_parameterized_primitive_gate \\
        --output-dir runs/a1_r005d_001_reanalysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apc.environments.operations import PARAMETERIZED_OPERATION_NAMES
from apc.evaluation.parameterized_primitive_gate_reanalysis import reanalyze_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Existing parameterized_primitive_gate-shaped run directory to "
        "re-analyze (read-only; never modified)",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory to write this re-analysis's own report to"
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2,3,4",
        help="Comma-separated seeds to re-analyze (default: '0,1,2,3,4')",
    )
    args = parser.parse_args()

    seeds = tuple(int(token) for token in args.seeds.split(","))
    result = reanalyze_run(args.run_dir, seeds=seeds)

    missing_operations = set(PARAMETERIZED_OPERATION_NAMES) - set(result.operation_names)
    if missing_operations:
        raise SystemExit(
            f"re-analyzed run is missing operation(s) {sorted(missing_operations)} "
            f"required by A1-R005D-001's acceptance criteria; got {result.operation_names}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    print(json.dumps(result.to_dict(), indent=2))
    print(f"A1-R005D-001 re-analysis written to: {report_path}")
    print(f"Source run (unmodified): {args.run_dir}")


if __name__ == "__main__":
    main()
