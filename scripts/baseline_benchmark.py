#!/usr/bin/env python
"""CLI entry point for the Phase A baseline comparison (Task 013).

Runs `apc.evaluation.baselines.run_all_baselines` (B0-B3 plus B4, the full
APC loop from Task 012) end to end under one shared config -- see
`apc.evaluation.baselines`'s module docstring for what "shared" means
here -- and writes, per baseline, the same run artifact layout other
scripts in this repo use: `config.yaml`, `report.json`, `system.json`, a
combined `summary.json` across all five, and (unless `--no-plots`) a
top-level `plots/` with the cross-baseline comparison figures from
`apc.evaluation.baseline_plots`.

Usage:
    python scripts/baseline_benchmark.py \\
        --config configs/phase_a_sequential.yaml --run-dir runs/phase_a_baselines
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apc.evaluation.baselines import BaselineConfig, baseline_config_from_dict, run_all_baselines
from apc.utils.config import load_config, save_config
from apc.utils.system_info import get_system_info


def _summarize(name: str, report: object) -> dict[str, object]:
    summary: dict[str, object] = {
        "pretrain_exact_match": report.pretrain_exact_match,  # type: ignore[attr-defined]
        "max_forgetting": report.max_forgetting,  # type: ignore[attr-defined]
        "mean_backward_transfer": report.mean_backward_transfer,  # type: ignore[attr-defined]
        "stable_core_parameter_count": report.stable_core_parameter_count,  # type: ignore[attr-defined]
        "resident_total_parameter_count_final": (
            report.resident_total_parameter_count_final  # type: ignore[attr-defined]
        ),
        "resident_primitive_parameter_count_final": (
            report.resident_primitive_parameter_count_final  # type: ignore[attr-defined]
        ),
        "total_train_steps": report.total_train_steps,  # type: ignore[attr-defined]
        "wall_clock_seconds": report.wall_clock_seconds,  # type: ignore[attr-defined]
    }
    if name == "B4":
        summary["num_learn_consolidate_release_cycles"] = (
            report.num_learn_consolidate_release_cycles  # type: ignore[attr-defined]
        )
        summary["num_reused_without_new_cycle"] = (
            report.num_reused_without_new_cycle  # type: ignore[attr-defined]
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, help="Path to a phase_a_sequential.yaml-shaped config"
    )
    parser.add_argument(
        "--run-dir", required=True, help="Output directory for this run's artifacts"
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="Skip rendering plots (no matplotlib needed)"
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_config = load_config(args.config)
    config: BaselineConfig = baseline_config_from_dict(raw_config)
    save_config(config.to_dict(), run_dir / "config.yaml")

    system_info = get_system_info(seed=config.sequential.seed)
    (run_dir / "system.json").write_text(json.dumps(system_info, indent=2), encoding="utf-8")

    results = run_all_baselines(config)

    summary: dict[str, object] = {}
    for name, report in results.items():
        baseline_dir = run_dir / name
        baseline_dir.mkdir(parents=True, exist_ok=True)
        (baseline_dir / "report.json").write_text(
            json.dumps(report.to_dict(), indent=2), encoding="utf-8"
        )
        summary[name] = _summarize(name, report)
        print(f"Report written to: {baseline_dir / 'report.json'}")

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if not args.no_plots:
        from apc.evaluation.baseline_plots import plot_all_baselines

        plot_paths = plot_all_baselines(results, run_dir / "plots")
        for path in plot_paths:
            print(f"Plot written to: {path}")


if __name__ == "__main__":
    main()
