#!/usr/bin/env python3
"""Run Full Sequential K/C/N/R Closed-Loop Benchmark (Phase A.2 Task A2-C008 - STOP GATE).

Usage:
    python scripts/sequential_closed_loop_benchmark.py \
        --seeds 0 1 2 3 4 --output-dir runs/phase_a2_sequential_closed_loop
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to python path if run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from apc.evaluation.sequential_closed_loop_benchmark import (
    SequentialClosedLoopConfig,
    run_sequential_closed_loop_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Task A2-C008: Full Sequential K/C/N/R Autonomous Closed Loop Benchmark"
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
        default="runs/phase_a2_sequential_closed_loop",
        help="Directory to save run artifacts (report.json, episodes.jsonl, BENCHMARK_REPORT.md)",
    )
    parser.add_argument(
        "--num-k",
        type=int,
        default=14,
        help="Number of Known (K) episodes per seed (default: 14, min: 10)",
    )
    parser.add_argument(
        "--num-c",
        type=int,
        default=12,
        help="Number of Composition (C) episodes per seed (default: 12, min: 10)",
    )
    parser.add_argument(
        "--num-n",
        type=int,
        default=6,
        help="Number of Novel (N) episodes per seed (default: 6, min: 6)",
    )
    parser.add_argument(
        "--num-r",
        type=int,
        default=8,
        help="Number of Recurrence (R) episodes per seed (default: 8, min: 6)",
    )
    parser.add_argument(
        "--plastic-train-size",
        type=int,
        default=160,
        help="Number of training examples for plastic adaptation (default: 160)",
    )
    parser.add_argument(
        "--compact-steps",
        type=int,
        default=800,
        help="Training steps for compact plastic search (default: 800)",
    )
    parser.add_argument(
        "--fallback-steps",
        type=int,
        default=350,
        help="Training steps for fallback search (default: 350)",
    )
    parser.add_argument(
        "--distillation-steps",
        type=int,
        default=300,
        help="Training steps for distillation (default: 300)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use ('auto', 'cuda', 'cpu')",
    )

    args = parser.parse_args()

    config = SequentialClosedLoopConfig(
        seeds=tuple(args.seeds),
        num_k=args.num_k,
        num_c=args.num_c,
        num_n=args.num_n,
        num_r=args.num_r,
        plastic_train_size=args.plastic_train_size,
        compact_budget_steps=args.compact_steps,
        fallback_budget_steps=args.fallback_steps,
        distillation_steps=args.distillation_steps,
        device_str=args.device,
        output_dir=Path(args.output_dir),
    )

    report = run_sequential_closed_loop_benchmark(config)

    overall_passed = report["overall_passed"]
    print("\n" + "=" * 75)
    if overall_passed:
        print(">>> TASK A2-C008 STOP GATE: PASSED <<<")
        print("All acceptance criteria satisfied across evaluated seeds.")
    else:
        print(">>> TASK A2-C008 STOP GATE: FAILED <<<")
        print("One or more acceptance criteria failed.")
    print("=" * 75)

    sys.exit(0 if overall_passed else 1)


if __name__ == "__main__":
    main()
