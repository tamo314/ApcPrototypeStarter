"""Run Phase A.2 A2-C010 end-to-end compute and latency scaling."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from apc.evaluation.compute_latency_scaling_benchmark import (
    DEFAULT_BANK_SIZES,
    DEFAULT_SEEDS,
    ComputeLatencyScalingConfig,
    run_compute_latency_scaling_benchmark,
)
from apc.utils.system_info import get_system_info


def parse_args() -> argparse.Namespace:
    """Parse the explicit, reproducible C010 measurement configuration."""
    parser = argparse.ArgumentParser(description="A2-C010 end-to-end compute/latency scaling")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_BANK_SIZES))
    parser.add_argument("--profile-batch-size", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--profile-steps", type=int, default=50)
    parser.add_argument("--num-router-train-examples", type=int, default=64)
    parser.add_argument("--num-routing-eval-examples", type=int, default=100)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--bank-source-dir", type=Path, default=Path("runs/phase_a2_bank_scaling_benchmark")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/phase_a2_compute_latency_scaling")
    )
    return parser.parse_args()


def main() -> int:
    """Execute C010 and write reproducible artifacts."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = ComputeLatencyScalingConfig(
        bank_sizes=tuple(args.sizes),
        seeds=tuple(args.seeds),
        num_router_train_examples=args.num_router_train_examples,
        num_routing_eval_examples=args.num_routing_eval_examples,
        profile_batch_size=args.profile_batch_size,
        warmup_steps=args.warmup_steps,
        profile_steps=args.profile_steps,
        device_str=args.device,
        bank_source_dir=args.bank_source_dir,
        output_dir=args.output_dir,
    )
    (args.output_dir / "config.yaml").write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
    )
    (args.output_dir / "system.json").write_text(
        json.dumps(get_system_info(), indent=2), encoding="utf-8"
    )
    started = time.perf_counter()
    report = run_compute_latency_scaling_benchmark(config)
    elapsed = time.perf_counter() - started
    report["elapsed_seconds"] = elapsed
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"A2-C010 hard acceptance: {'PASS' if report['hard_acceptance_passed'] else 'FAIL'}")
    target_label = "N=128" if 128 in config.bank_sizes else f"N={max(config.bank_sizes)}"
    print(f"{target_label} sparse/dense median ratio: {report['n128_median_latency_ratio']:.2%}")
    print(f"Elapsed seconds: {elapsed:.2f}")
    return 0 if report["hard_acceptance_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
