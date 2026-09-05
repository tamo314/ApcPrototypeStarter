import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from apc.evaluation.compact_lifecycle_benchmark import run_compact_lifecycle_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Compact-First Plastic Lifecycle Benchmark (Task A2-C007)"
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
        default="runs/phase_a2_compact_lifecycle_benchmark",
        help="Directory to save run artifacts",
    )
    parser.add_argument(
        "--compact-steps",
        type=int,
        default=150,
        help="Training steps for compact plastic search",
    )
    parser.add_argument(
        "--fallback-steps",
        type=int,
        default=150,
        help="Training steps for overcomplete fallback",
    )
    parser.add_argument(
        "--distillation-steps",
        type=int,
        default=150,
        help="Training steps for T2 -> T0 distillation",
    )
    args = parser.parse_args()

    summary = run_compact_lifecycle_benchmark(
        seeds=args.seeds,
        output_dir=args.output_dir,
        compact_steps=args.compact_steps,
        fallback_steps=args.fallback_steps,
        distillation_steps=args.distillation_steps,
    )

    if not summary.all_criteria_passed:
        raise SystemExit("Task A2-C007 Compact-First Plastic Lifecycle Benchmark FAILED!")
    print("Task A2-C007 Compact-First Plastic Lifecycle Benchmark PASSED.")


if __name__ == "__main__":
    main()
