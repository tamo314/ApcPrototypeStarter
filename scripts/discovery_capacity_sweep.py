#!/usr/bin/env python
"""CLI entry point for Task A1-B007X-003 Temporary Discovery Capacity Sweep.

Executes a matched discovery capacity sweep across >=5 seeds and novel operations:
evaluates compact (T0), medium (T1), and overcomplete (T2) temporary primitives.

Usage:
    python scripts/discovery_capacity_sweep.py \
        --config configs/phase_a1_discovery_capacity_sweep.yaml \
        --run-dir runs/phase_a1_discovery_capacity_sweep
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from apc.evaluation.discovery_capacity_sweep import (
    discovery_capacity_sweep_config_from_dict,
    run_discovery_capacity_sweep,
)
from apc.utils.config import load_config, save_config
from apc.utils.system_info import get_system_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to phase_a1_discovery_capacity_sweep.yaml",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Output directory for sweep run artifacts",
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
    base_config = discovery_capacity_sweep_config_from_dict(raw_config)
    if args.seed is not None:
        base_config = dataclasses.replace(base_config, seeds=(args.seed,))

    save_config(base_config.to_dict(), run_dir / "config.yaml")

    system_info = get_system_info(seed=base_config.seeds[0])
    (run_dir / "system.json").write_text(json.dumps(system_info, indent=2), encoding="utf-8")

    summary = run_discovery_capacity_sweep(base_config, output_dir=run_dir)

    print("\n" + "=" * 80)
    print(
        f"Task A1-B007X-003 Discovery Capacity Sweep Summary ({summary.elapsed_seconds:.1f}s)"
    )
    print("=" * 80)

    for op_name, t_dict in summary.aggregated.items():
        v = summary.verdicts[op_name]
        print(f"\n--- Operation: {op_name} ---")
        verdict_str = (
            "ADVANTAGE FOUND" if v.advantage_found else "NO ADVANTAGE (Compact Equivalent)"
        )
        print(f"Verdict: {verdict_str}")
        print(f"Reason: {v.summary_reason}")
        print(
            f"{'Tier':<16} | {'Params':<8} | {'Mean EM':<9} | {'Std EM':<8} | "
            f"{'Success':<8} | {'Med Step 95':<12} | {'Mean AUC':<9} | {'Time (s)':<8}"
        )
        print("-" * 96)
        for tier_name, agg in t_dict.items():
            med_step_str = (
                f"{agg.median_step_to_95:.1f}" if agg.median_step_to_95 is not None else "N/A"
            )
            print(
                f"{tier_name:<16} | {agg.parameters:<8} | {agg.mean_em:<9.4f} | "
                f"{agg.std_em:<8.4f} | {agg.success_rate:<8.1%} | {med_step_str:<12} | "
                f"{agg.mean_auc:<9.4f} | {agg.mean_wall_clock:<8.2f}"
            )

    print("\n" + "=" * 80)
    print(f"Overall Discovery-Capacity Advantage Found: {summary.overall_advantage_found}")
    print(f"Artifacts saved to: {run_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
