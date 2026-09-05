#!/usr/bin/env python3
"""Run Phase A.2 Task A2-C009 repeated semantic bank-growth STOP gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from apc.evaluation.repeated_bank_growth_benchmark import (
    RepeatedBankGrowthConfig,
    run_repeated_bank_growth_benchmark,
)
from apc.utils.system_info import get_system_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument(
        "--output-dir", default="runs/phase_a2_repeated_bank_growth_stress"
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--support-size", type=int, default=16)
    parser.add_argument("--eval-size", type=int, default=32)
    parser.add_argument("--routing-eval-size", type=int, default=100)
    parser.add_argument("--plastic-train-size", type=int, default=160)
    parser.add_argument("--compact-steps", type=int, default=800)
    parser.add_argument("--fallback-steps", type=int, default=350)
    parser.add_argument("--distillation-steps", type=int, default=300)
    parser.add_argument("--router-steps", type=int, default=250)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    config = RepeatedBankGrowthConfig(
        seeds=tuple(int(seed.strip()) for seed in args.seeds.split(",")),
        support_size=args.support_size,
        eval_size=args.eval_size,
        routing_eval_size=args.routing_eval_size,
        plastic_train_size=args.plastic_train_size,
        compact_budget_steps=args.compact_steps,
        fallback_budget_steps=args.fallback_steps,
        distillation_steps=args.distillation_steps,
        router_steps=args.router_steps,
        device_str=args.device,
        output_dir=output_dir,
    )
    report = run_repeated_bank_growth_benchmark(config)
    (output_dir / "system.json").write_text(
        json.dumps(get_system_info(), indent=2), encoding="utf-8"
    )
    status = "PASSED" if report["overall_passed"] else "FAILED"
    print(f"A2-C009 STOP GATE: {status}")
    raise SystemExit(0 if report["overall_passed"] else 1)


if __name__ == "__main__":
    main()
