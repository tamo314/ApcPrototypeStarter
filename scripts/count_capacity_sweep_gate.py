#!/usr/bin/env python
"""CLI entry point for the minimal capacity/training sweep gate (Phase A.1
Post-Correction Task A1-R005D-005).

Runs `apc.evaluation.count_capacity_sweep_gate.run_count_capacity_sweep_gate`
against a `phase_a1_count_capacity_sweep_gate.yaml`-shaped baseline config
(A1-R005D-004's own baseline, unmodified): the staged sweep ("steps x2",
"steps x4", "rank 16", "arg_dim x2", each isolating exactly one knob) runs in
order and stops at the first stage whose full 5-seed gate passes
A1-R005D-004's own acceptance thresholds. Writes the run artifact layout:
`config.yaml`, `sweep_report.json`, `summary.json`, `system.json`, and one
`<stage_name>/{report.json, seed_<n>/{report.json, core_metrics.jsonl,
primitive_metrics.jsonl}}` per stage actually run.

Usage:
    python scripts/count_capacity_sweep_gate.py \\
        --config configs/phase_a1_count_capacity_sweep_gate.yaml \\
        --run-dir runs/phase_a1_count_capacity_sweep_gate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apc.evaluation.count_capacity_sweep_gate import run_count_capacity_sweep_gate
from apc.evaluation.count_counterfactual_gate import (
    DEFAULT_SEEDS,
    count_counterfactual_gate_config_from_dict,
)
from apc.utils.config import load_config, save_config
from apc.utils.system_info import get_system_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a phase_a1_count_capacity_sweep_gate.yaml-shaped config",
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
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_config = load_config(args.config)
    base_config = count_counterfactual_gate_config_from_dict(raw_config)
    if args.seeds is not None:
        seeds = tuple(int(token) for token in args.seeds.split(","))
    else:
        seeds = tuple(raw_config.get("seeds", DEFAULT_SEEDS))

    resolved_config = base_config.to_dict()
    resolved_config["seeds"] = list(seeds)
    save_config(resolved_config, run_dir / "config.yaml")

    system_info = get_system_info(seed=base_config.seed)
    (run_dir / "system.json").write_text(json.dumps(system_info, indent=2), encoding="utf-8")

    result = run_count_capacity_sweep_gate(base_config, seeds=seeds, run_dir=run_dir)
    (run_dir / "sweep_report.json").write_text(
        json.dumps(result.to_dict(), indent=2), encoding="utf-8"
    )

    summary = {
        "seeds": list(seeds),
        "stages_run": [stage.name for stage in result.stages],
        "selected_stage": result.selected_stage,
        "sweep_exhausted_without_pass": result.sweep_exhausted_without_pass,
        "per_stage": {
            stage.name: {
                "swept_field": stage.swept_field,
                "swept_value": stage.swept_value,
                "primitive_param_count": stage.primitive_param_count,
                "mean_correct_exact_match": stage.multi_seed.mean_correct_exact_match,
                "mean_effectful_wrong_argument_exact_match": (
                    stage.multi_seed.mean_effectful_wrong_argument_exact_match
                ),
                "mean_none_exact_match": stage.multi_seed.mean_none_exact_match,
                "mean_causal_gap": stage.multi_seed.mean_causal_gap,
                "passed": stage.multi_seed.passed,
            }
            for stage in result.stages
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if result.selected_stage is not None:
        print(
            f"A1-R005D-005 capacity/training sweep: PASS at stage "
            f"'{result.selected_stage}' -- smallest robust passing configuration found"
        )
    else:
        print(
            "A1-R005D-005 capacity/training sweep: no stage passed "
            "-- per docs/CODEX_TASKS_A1_R005_RETRY.md, continue to A1-R005D-006 "
            "(not run automatically by this script)"
        )
    print(f"Sweep report written to: {run_dir / 'sweep_report.json'}")


if __name__ == "__main__":
    main()
