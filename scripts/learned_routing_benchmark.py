#!/usr/bin/env python
"""CLI entry point for Learned Routing & Full Closed Loop Benchmark (Task A1-B008 / Milestone B-M8).

Replaces oracle routing with a learned top-k router conditioned on z_task.
Demonstrates autonomous sparse execution, compute savings, and end-to-end continual learning.

Usage:
    python scripts/learned_routing_benchmark.py \
        --config configs/phase_a1_learned_routing_benchmark.yaml \
        --run-dir runs/phase_a1_learned_routing_benchmark
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from apc.evaluation.learned_routing_benchmark import (
    DEFAULT_SEEDS,
    LearnedRoutingBenchmarkConfig,
    routing_config_from_dict,
    run_learned_routing_benchmark_multi_seed,
)
from apc.utils.config import load_config, save_config
from apc.utils.system_info import get_system_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to phase_a1_learned_routing_benchmark.yaml",
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
    base_config = routing_config_from_dict(raw_config)
    if args.seeds is not None:
        seeds = tuple(int(s.strip()) for s in args.seeds.split(","))
    else:
        seeds = tuple(raw_config.get("seeds", DEFAULT_SEEDS))

    resolved_config = base_config.to_dict()
    resolved_config["seeds"] = list(seeds)
    save_config(resolved_config, run_dir / "config.yaml")

    system_info = get_system_info(seed=base_config.seed)
    (run_dir / "system.json").write_text(json.dumps(system_info, indent=2), encoding="utf-8")

    def config_factory(s: int) -> LearnedRoutingBenchmarkConfig:
        return dataclasses.replace(base_config, seed=s)

    result = run_learned_routing_benchmark_multi_seed(
        seeds, config_factory, output_dir=run_dir
    )

    summary = {
        "seeds": list(result.seeds),
        "overall_passed": result.overall_passed,
        "meets_seed_policy": result.meets_seed_policy,
        "mean_router_top1_accuracy": result.mean_router_top1_accuracy,
        "mean_router_topk_accuracy": result.mean_router_topk_accuracy,
        "mean_exact_match": result.mean_exact_match,
        "mean_token_accuracy": result.mean_token_accuracy,
        "active_primitive_parameters": result.active_primitive_parameters,
        "dense_primitive_parameters": result.dense_primitive_parameters,
        "compute_savings_ratio": result.compute_savings_ratio,
        "max_allocated_plastic_parameters": result.max_allocated_plastic_parameters,
        "per_operation_means": result.per_operation_means,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    verdict = "PASS" if result.overall_passed else "FAIL"
    print(f"\nTask A1-B008 Learned Routing & Full Closed Loop Benchmark: {verdict}")
    print(f"Summary written to: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
