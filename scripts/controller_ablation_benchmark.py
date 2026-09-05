#!/usr/bin/env python3
"""Run Controller Ablations and Failure Analysis Benchmark (Phase A.2 Task A2-C011).

Usage:
    python scripts/controller_ablation_benchmark.py \
        --seeds 0 1 2 3 4 --output-dir runs/phase_a2_controller_ablations
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src is on python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from apc.evaluation.controller_ablation_benchmark import (
    ControllerAblationConfig,
    run_controller_ablation_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Task A2-C011: Controller Ablations and Failure Analysis Benchmark"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Random seeds to evaluate (default: 0 1 2 3 4)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/phase_a2_controller_ablations",
        help="Directory to save run artifacts (report.json, BENCHMARK_REPORT.md)",
    )
    parser.add_argument(
        "--num-k",
        type=int,
        default=14,
        help="Number of Known (K) episodes per seed (default: 14)",
    )
    parser.add_argument(
        "--num-c",
        type=int,
        default=12,
        help="Number of Composition (C) episodes per seed (default: 12)",
    )
    parser.add_argument(
        "--num-n",
        type=int,
        default=6,
        help="Number of Novel (N) episodes per seed (default: 6)",
    )
    parser.add_argument(
        "--num-r",
        type=int,
        default=8,
        help="Number of Recurrence (R) episodes per seed (default: 8)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use ('auto', 'cuda', 'cpu')",
    )

    args = parser.parse_args()

    config = ControllerAblationConfig(
        seeds=tuple(args.seeds),
        num_k=args.num_k,
        num_c=args.num_c,
        num_n=args.num_n,
        num_r=args.num_r,
        device_str=args.device,
        output_dir=Path(args.output_dir),
    )

    run_controller_ablation_benchmark(config)
    print("\nBenchmark completed successfully.")
    print(f"Artifacts saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
