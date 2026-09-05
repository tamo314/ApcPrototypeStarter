"""CLI script for Bank Competition and Routing Scaling Benchmark (Task A2-C004).

Usage:
    python scripts/bank_scaling_benchmark.py --seeds 0 1 2 3 4 --sizes 10 16 32 64 128 --device auto
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from apc.evaluation.bank_scaling_benchmark import (
    DEFAULT_BANK_SIZES,
    DEFAULT_SEEDS,
    BankScalingConfig,
    run_bank_scaling_benchmark,
)
from apc.utils.system_info import get_system_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase A.2 Task A2-C004: Bank Competition and Routing Scaling Benchmark"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help=f"Decision seeds to evaluate (default: {list(DEFAULT_SEEDS)})",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_BANK_SIZES),
        help=f"Resident bank sizes to evaluate (default: {list(DEFAULT_BANK_SIZES)})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/phase_a2_bank_scaling_benchmark"),
        help="Directory to save benchmark reports and JSON artifacts",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for computation (default: auto)",
    )
    parser.add_argument(
        "--num-train-examples",
        type=int,
        default=64,
        help="Number of training examples per operation for router calibration",
    )
    parser.add_argument(
        "--num-eval-examples",
        type=int,
        default=200,
        help="Number of evaluation examples per known operation",
    )
    parser.add_argument(
        "--num-recurrence-examples",
        type=int,
        default=50,
        help="Number of recurrence examples per known operation",
    )
    parser.add_argument(
        "--router-steps",
        type=int,
        default=250,
        help="Calibration steps for router",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
        help="Latency profiling warmup iterations",
    )
    parser.add_argument(
        "--profile-steps",
        type=int,
        default=20,
        help="Latency profiling active measurement iterations",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("=" * 70)
    print("Phase A.2 Task A2-C004: Bank Competition and Routing Scaling Benchmark")
    print(f"Seeds: {args.seeds}")
    print(f"Bank Sizes: {args.sizes}")
    print(f"Device: {args.device}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 70)

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        sys_info = get_system_info()
        with open(args.output_dir / "system.json", "w", encoding="utf-8") as f:
            json.dump(sys_info, f, indent=2)

    config = BankScalingConfig(
        bank_sizes=tuple(args.sizes),
        seeds=tuple(args.seeds),
        num_train_examples=args.num_train_examples,
        num_eval_examples=args.num_eval_examples,
        num_recurrence_examples=args.num_recurrence_examples,
        router_steps=args.router_steps,
        warmup_steps=args.warmup_steps,
        active_profile_steps=args.profile_steps,
        device_str=args.device,
        output_dir=args.output_dir,
    )

    t0 = time.perf_counter()
    report = run_bank_scaling_benchmark(config)
    elapsed = time.perf_counter() - t0

    print("\n" + "=" * 70)
    print("Benchmark Completed")
    print(f"Elapsed time: {elapsed:.2f} s")
    print(f"Overall N=128 Gate Passed: {report['overall_passed_n128']}")
    print("=" * 70)

    # Print summary table
    print("\nSummary Results by Bank Size:")
    header = (
        f"{'N':>5} | {'SemOps':>6} | {'Dist':>5} | {'Top-1':>7} | "
        f"{'FalseSel':>8} | {'Recurr':>7} | {'ResParams':>10} | "
        f"{'ActParams':>10} | {'Sparse(ms)':>10} | {'Dense(ms)':>10} | "
        f"{'LatRatio':>8} | {'Status':>6}"
    )
    print(header)
    print("-" * len(header))
    for _size_str, s in report["aggregated_by_size"].items():
        status = "PASS" if s["all_seeds_passed"] else "FAIL"
        print(
            f"{s['bank_size']:>5} | {s['num_semantic_ops']:>6} | {s['num_distractors']:>5} | "
            f"{s['mean_known_task_top1']:>7.4f} | "
            f"{s['mean_distractor_false_selection_rate']:>8.4f} | "
            f"{s['mean_recurrence_top1']:>7.4f} | {s['resident_primitive_params']:>10,} | "
            f"{s['active_primitive_params']:>10,} | {s['mean_sparse_latency_ms']:>10.2f} | "
            f"{s['mean_dense_latency_ms']:>10.2f} | {s['mean_latency_ratio']:>8.2%} | {status:>6}"
        )
    print("-" * len(header))

    return 0 if report["overall_passed_n128"] else 1


if __name__ == "__main__":
    sys.exit(main())
