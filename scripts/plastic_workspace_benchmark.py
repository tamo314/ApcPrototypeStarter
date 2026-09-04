#!/usr/bin/env python
"""CLI entry point for Plastic Workspace Residual Learning Benchmark
(Task A1-B005 / Milestone B-M5).

Verifies that novel operations (unsolvable by existing bank primitives or compositions)
trigger temporary plastic capacity, which adapts rapidly as a residual without updating
the frozen Stable Core or persistent primitives.

Usage:
    python scripts/plastic_workspace_benchmark.py \
        --config configs/phase_a1_plastic_workspace_benchmark.yaml \
        --run-dir runs/phase_a1_plastic_workspace_benchmark
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from apc.evaluation.plastic_workspace_benchmark import (
    DEFAULT_SEEDS,
    PlasticWorkspaceBenchmarkConfig,
    plastic_workspace_config_from_dict,
    run_plastic_workspace_benchmark_multi_seed,
)
from apc.utils.config import load_config, save_config
from apc.utils.system_info import get_system_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to phase_a1_plastic_workspace_benchmark.yaml",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Output directory for this run's artifacts",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seed override, e.g. '0,1,2,3,4'",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_config = load_config(args.config)
    base_config = plastic_workspace_config_from_dict(raw_config)
    if args.seeds is not None:
        seeds = tuple(int(s.strip()) for s in args.seeds.split(","))
    else:
        seeds = tuple(raw_config.get("seeds", DEFAULT_SEEDS))

    resolved_config = base_config.to_dict()
    resolved_config["seeds"] = list(seeds)
    save_config(resolved_config, run_dir / "config.yaml")

    system_info = get_system_info(seed=base_config.seed)
    (run_dir / "system.json").write_text(json.dumps(system_info, indent=2), encoding="utf-8")

    def config_factory(s: int) -> PlasticWorkspaceBenchmarkConfig:
        return dataclasses.replace(base_config, seed=s)

    result = run_plastic_workspace_benchmark_multi_seed(
        seeds, config_factory, output_dir=run_dir
    )

    summary = {
        "seeds": list(result.seeds),
        "overall_passed": result.overall_passed,
        "meets_seed_policy": result.meets_seed_policy,
        "mean_overall_residual_exact_match": result.mean_overall_residual_exact_match,
        "mean_overall_scratch_exact_match": result.mean_overall_scratch_exact_match,
        "mean_overall_frozen_bank_exact_match": result.mean_overall_frozen_bank_exact_match,
        "per_operation_means": result.per_operation_means,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    verdict = "PASS" if result.overall_passed else "FAIL"
    print(f"\nTask A1-B005 Plastic Workspace Residual Learning Benchmark: {verdict}")
    print(f"Summary written to: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
