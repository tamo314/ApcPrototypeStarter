#!/usr/bin/env python
"""CLI entry point for Composition Library Execution Benchmark (Phase A.1 Post-Diagnostic
Task A1-B003, STOP GATE).

Evaluates designated multi-step compositions over the frozen shared task-blind Stable Core
and PrimitiveBank with strict sparse call tracking and zero plastic capacity.

Usage:
    python scripts/composition_library_benchmark.py \
        --config configs/phase_a1_composition_library_benchmark.yaml \
        --run-dir runs/phase_a1_composition_library_benchmark
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from apc.evaluation.composition_library_benchmark import (
    DEFAULT_SEEDS,
    CompositionBenchmarkConfig,
    composition_benchmark_config_from_dict,
    run_composition_library_benchmark_multi_seed,
)
from apc.utils.config import load_config, save_config
from apc.utils.system_info import get_system_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to phase_a1_composition_library_benchmark.yaml",
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
    base_config = composition_benchmark_config_from_dict(raw_config)
    if args.seeds is not None:
        seeds = tuple(int(s.strip()) for s in args.seeds.split(","))
    else:
        seeds = tuple(raw_config.get("seeds", DEFAULT_SEEDS))

    resolved_config = base_config.to_dict()
    resolved_config["seeds"] = list(seeds)
    save_config(resolved_config, run_dir / "config.yaml")

    system_info = get_system_info(seed=base_config.seed)
    (run_dir / "system.json").write_text(json.dumps(system_info, indent=2), encoding="utf-8")

    def config_factory(s: int) -> CompositionBenchmarkConfig:
        return dataclasses.replace(base_config, seed=s)

    result = run_composition_library_benchmark_multi_seed(
        seeds, config_factory, output_dir=run_dir
    )

    summary = {
        "seeds": list(result.seeds),
        "overall_passed": result.overall_passed,
        "meets_seed_policy": result.meets_seed_policy,
        "mean_overall_exact_match": result.mean_overall_exact_match,
        "per_composition_means": result.per_composition_means,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    verdict = "PASS" if result.overall_passed else "FAIL"
    print(f"Task A1-B003 Composition Library Execution Benchmark: {verdict}")
    print(f"Summary written to: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
