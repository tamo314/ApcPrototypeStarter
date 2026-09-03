#!/usr/bin/env python
"""CLI entry point for the joint task-blind representation + compact
operator probe (Phase A.1 diagnostic Task A1-R005E-006A).

Per seed, per operation: constructs a fresh, unfrozen, task-blind Stable
Core (`apc.evaluation.joint_representation_compact_operator_probe.
_build_joint_core`, same architecture as every earlier module in this
diagnostic chain), jointly trains it with a dedicated `apc.evaluation.
compact_cross_position_operator_probe.CompactCrossPositionOperator` (A1-
R005E-005's own compact operator class and dimensions, unchanged) on
counterfactual argument groups, then evaluates the Correct / effectful Wrong
argument / None causal ablation on a large unseen batch, and computes the
`R_access` representation-accessibility-recovery diagnostic against
A1-R005E-004's/005's own saved summaries. See the module docstring for full
design rationale. Writes the run artifact layout described in `README.md`:
`config.yaml`, `report.json`, `summary.json`, `system.json`, and one
`seed_<n>/{report.json, <OPERATION>_joint_metrics.jsonl}` per seed.

Usage:
    python scripts/joint_representation_compact_operator_probe.py \\
        --config configs/phase_a1_joint_representation_compact_operator_probe.yaml \\
        --run-dir runs/phase_a1_joint_representation_compact_operator_probe
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apc.evaluation.joint_representation_compact_operator_probe import (
    DEFAULT_E004_SUMMARY_PATH,
    DEFAULT_E005_SUMMARY_PATH,
    DEFAULT_SEEDS,
    joint_operator_config_from_dict,
    run_joint_operator_probe_multi_seed,
)
from apc.utils.config import load_config, save_config
from apc.utils.system_info import get_system_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a phase_a1_joint_representation_compact_operator_probe.yaml-shaped config",
    )
    parser.add_argument(
        "--run-dir", required=True, help="Output directory for this run's artifacts"
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help=(
            "Comma-separated seed override, e.g. '0,1,2,3,4' "
            "(default: the config's top-level 'seeds' key, or 0-4)"
        ),
    )
    parser.add_argument(
        "--e004-summary",
        default=str(DEFAULT_E004_SUMMARY_PATH),
        help=(
            "Path to A1-R005E-004's saved summary.json, used for the R_access diagnostic "
            "(default: the standard run-dir location). Pass an empty string to skip R_access."
        ),
    )
    parser.add_argument(
        "--e005-summary",
        default=str(DEFAULT_E005_SUMMARY_PATH),
        help=(
            "Path to A1-R005E-005's saved summary.json, used for the R_access diagnostic "
            "(default: the standard run-dir location). Pass an empty string to skip R_access."
        ),
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_config = load_config(args.config)
    base_config = joint_operator_config_from_dict(raw_config)
    if args.seeds is not None:
        seeds = tuple(int(token) for token in args.seeds.split(","))
    else:
        seeds = tuple(raw_config.get("seeds", DEFAULT_SEEDS))

    resolved_config = base_config.to_dict()
    resolved_config["seeds"] = list(seeds)
    save_config(resolved_config, run_dir / "config.yaml")

    system_info = get_system_info(seed=base_config.seed)
    (run_dir / "system.json").write_text(json.dumps(system_info, indent=2), encoding="utf-8")

    e004_summary_path = Path(args.e004_summary) if args.e004_summary else None
    e005_summary_path = Path(args.e005_summary) if args.e005_summary else None

    result = run_joint_operator_probe_multi_seed(
        base_config,
        seeds=seeds,
        run_dir=run_dir,
        e004_summary_path=e004_summary_path,
        e005_summary_path=e005_summary_path,
    )
    (run_dir / "report.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    summary = {
        "seeds": list(result.seeds),
        "passed": result.passed,
        "meets_seed_policy": result.meets_seed_policy,
        "e004_summary_path": result.e004_summary_path,
        "e005_summary_path": result.e005_summary_path,
        "per_operation_summary": {
            name: summary.to_dict() for name, summary in result.per_operation_summary.items()
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    verdict = "PASS" if result.passed else "FAIL"
    print(f"A1-R005E-006A Joint representation + compact operator probe: {verdict}")
    print(f"Report written to: {run_dir / 'report.json'}")


if __name__ == "__main__":
    main()
