"""CLI script for Learned Adequacy and Novelty Controller Benchmark (Task A2-C006).

Usage:
    python scripts/adequacy_controller_benchmark.py --seeds 0 1 2 3 4 --device auto
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from apc.evaluation.adequacy_controller_benchmark import (
    DEFAULT_EPISODES_PER_CAT,
    DEFAULT_SEEDS,
    DEFAULT_SUPPORT_SIZE,
    AdequacyControllerBenchmarkConfig,
    run_adequacy_controller_benchmark,
)
from apc.utils.system_info import get_system_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase A.2 Task A2-C006: Learned Adequacy and Novelty Controller Benchmark"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help=f"Decision seeds to evaluate (default: {list(DEFAULT_SEEDS)})",
    )
    parser.add_argument(
        "--episodes-per-category",
        type=int,
        default=DEFAULT_EPISODES_PER_CAT,
        help=f"Episodes per category (K/C/N/R) per seed (default: {DEFAULT_EPISODES_PER_CAT})",
    )
    parser.add_argument(
        "--support-size",
        type=int,
        default=DEFAULT_SUPPORT_SIZE,
        help=f"Number of support examples per episode (default: {DEFAULT_SUPPORT_SIZE})",
    )
    parser.add_argument(
        "--dev-seed",
        type=int,
        default=42,
        help="Random seed for controller pre-calibration (default: 42)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for computation (default: auto)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/phase_a2_adequacy_controller_benchmark"),
        help="Directory to save benchmark reports and JSON artifacts",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        sys_info = get_system_info()
        with open(args.output_dir / "system.json", "w", encoding="utf-8") as f:
            json.dump(sys_info, f, indent=2)

    config = AdequacyControllerBenchmarkConfig(
        seeds=tuple(args.seeds),
        num_episodes_per_category=args.episodes_per_category,
        support_size=args.support_size,
        dev_seed=args.dev_seed,
        device_str=args.device,
        output_dir=args.output_dir,
    )

    report = run_adequacy_controller_benchmark(config)
    passed = report["overall_passed"]

    print("\n" + "=" * 70)
    print(f"Task A2-C006 STOP GATE Status: {'PASSED' if passed else 'FAILED'}")
    print("=" * 70)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
