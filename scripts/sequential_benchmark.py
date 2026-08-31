#!/usr/bin/env python
"""CLI entry point for the Phase A sequential benchmark (Task 012).

Runs `apc.evaluation.sequential_benchmark.run_sequential_benchmark` end to
end (pretraining included -- this does not load a checkpoint from another
script) and writes the run artifact layout described in `README.md`:
`config.yaml`, `report.json`, `system.json`, and (unless `--no-plots`)
`plots/` with the figures `docs/EXPERIMENT_PLAN.md` section 9 requires.

Usage:
    python scripts/sequential_benchmark.py \\
        --config configs/phase_a_sequential.yaml --run-dir runs/phase_a_sequential
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apc.evaluation.sequential_benchmark import (
    SequentialBenchmarkConfig,
    run_sequential_benchmark,
    sequential_config_from_dict,
)
from apc.utils.config import load_config, save_config
from apc.utils.system_info import get_system_info


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
    config: SequentialBenchmarkConfig = sequential_config_from_dict(raw_config)
    save_config(config.to_dict(), run_dir / "config.yaml")

    system_info = get_system_info(seed=config.seed)
    (run_dir / "system.json").write_text(json.dumps(system_info, indent=2), encoding="utf-8")

    report = run_sequential_benchmark(config)
    report_dict = report.to_dict()
    (run_dir / "report.json").write_text(json.dumps(report_dict, indent=2), encoding="utf-8")

    summary = {
        "pretrain_exact_match": report.pretrain_exact_match,
        "num_learn_consolidate_release_cycles": report.num_learn_consolidate_release_cycles,
        "num_reused_without_new_cycle": report.num_reused_without_new_cycle,
        "max_forgetting": report.max_forgetting,
        "mean_backward_transfer": report.mean_backward_transfer,
        "persistent_parameter_count_final": report.persistent_parameter_count_final,
        "temporary_peak_parameter_count": report.temporary_peak_parameter_count,
        "generalization_gap_known_vs_composition": report.generalization_gap_known_vs_composition,
        "novelty_gap_composition_vs_operation": report.novelty_gap_composition_vs_operation,
        "wall_clock_seconds": report.wall_clock_seconds,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Report written to: {run_dir / 'report.json'}")

    if not args.no_plots:
        from apc.evaluation.sequential_benchmark_plots import plot_all

        plot_paths = plot_all(report, run_dir / "plots")
        for path in plot_paths:
            print(f"Plot written to: {path}")


if __name__ == "__main__":
    main()
