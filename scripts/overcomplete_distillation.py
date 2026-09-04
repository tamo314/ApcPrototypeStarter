#!/usr/bin/env python
"""CLI entry point for Task A1-B007X-005 Overcomplete-to-Compact Functional Distillation.

Distills the overcomplete temporary teacher (T2 Overcomplete, 137,482 parameters)
into a compact persistent candidate primitive (T0 Compact, 17,098 parameters <= 25,000)
over a frozen task-blind Shared Core, evaluating exact match, retention, functional agreement,
and compression ratio across 5 seeds.

Usage:
    python scripts/overcomplete_distillation.py \
        --config configs/phase_a1_overcomplete_distillation.yaml \
        --run-dir runs/phase_a1_overcomplete_distillation
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from apc.evaluation.overcomplete_distillation import (
    overcomplete_distillation_config_from_dict,
    run_overcomplete_distillation,
)
from apc.utils.config import load_config, save_config
from apc.utils.system_info import get_system_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to phase_a1_overcomplete_distillation.yaml",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Output directory for distillation run artifacts",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional single seed override (e.g. for testing)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_config = load_config(args.config)
    base_config = overcomplete_distillation_config_from_dict(raw_config)
    if args.seed is not None:
        base_config = dataclasses.replace(base_config, seeds=(args.seed,))

    save_config(base_config.to_dict(), run_dir / "config.yaml")

    system_info = get_system_info(seed=base_config.seeds[0])
    (run_dir / "system.json").write_text(json.dumps(system_info, indent=2), encoding="utf-8")

    summary = run_overcomplete_distillation(base_config, output_dir=run_dir)

    print("\n" + "=" * 115)
    elapsed = summary.elapsed_seconds
    print(f"Task A1-B007X-005 Overcomplete-to-Compact Distillation Summary ({elapsed:.1f}s)")
    print("=" * 115)
    print(f"Teacher Tier: {summary.teacher_tier} | Candidate Tier: {summary.candidate_tier}")
    print(f"All Capacity Compliant (<= 25k, ratio <= 0.25): {summary.all_compliant}")
    print(f"Overall Gate Passed: {summary.overall_passed}")
    print("-" * 115)
    header = (
        f"{'Operation':<16} | {'Cand Prm':<8} | {'Teach Prm':<9} | {'Ratio':<7} | "
        f"{'Cand EM':<8} | {'Teach EM':<8} | {'Retention':<9} | {'Agreement':<9} | "
        f"{'Success':<7} | {'Passed':<6}"
    )
    print(header)
    print("-" * 115)

    for op_name, agg in summary.aggregated.items():
        row = (
            f"{op_name:<16} | {agg.candidate_parameters:<8} | {agg.teacher_parameters:<9} | "
            f"{agg.parameter_ratio:<7.4f} | {agg.mean_candidate_em:<8.4f} | "
            f"{agg.mean_teacher_em:<8.4f} | {agg.mean_retention:<9.4f} | "
            f"{agg.mean_agreement:<9.4f} | {agg.candidate_success_rate:<7.1%} | "
            f"{str(agg.overall_passed):<6}"
        )
        print(row)

    print("\n" + "=" * 115)
    print(f"Artifacts saved to: {run_dir}")
    print("=" * 115 + "\n")


if __name__ == "__main__":
    main()
