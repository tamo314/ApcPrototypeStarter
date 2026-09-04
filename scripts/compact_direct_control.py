#!/usr/bin/env python
"""CLI entry point for Task A1-B007X-004 Compact Direct-Learning Control.

Trains the <=25k compact candidate directly from supervised labels under matched
discovery budget across 5 seeds and novel operations.

Usage:
    python scripts/compact_direct_control.py \
        --config configs/phase_a1_compact_direct_control.yaml \
        --run-dir runs/phase_a1_compact_direct_control
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from apc.evaluation.compact_direct_control import (
    compact_direct_control_config_from_dict,
    run_compact_direct_control,
)
from apc.utils.config import load_config, save_config
from apc.utils.system_info import get_system_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to phase_a1_compact_direct_control.yaml",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Output directory for direct control run artifacts",
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
    base_config = compact_direct_control_config_from_dict(raw_config)
    if args.seed is not None:
        base_config = dataclasses.replace(base_config, seeds=(args.seed,))

    save_config(base_config.to_dict(), run_dir / "config.yaml")

    system_info = get_system_info(seed=base_config.seeds[0])
    (run_dir / "system.json").write_text(json.dumps(system_info, indent=2), encoding="utf-8")

    summary = run_compact_direct_control(base_config, output_dir=run_dir)

    print("\n" + "=" * 96)
    elapsed = summary.elapsed_seconds
    print(f"Task A1-B007X-004 Compact Direct-Learning Control Summary ({elapsed:.1f}s)")
    print("=" * 96)
    print(f"Candidate Tier: {summary.candidate_tier}")
    print(f"All Parameters Compliant (<= 25k): {summary.all_compliant}")
    print("-" * 96)
    header = (
        f"{'Operation':<18} | {'Params':<8} | {'Compliant':<9} | {'Mean EM':<9} | "
        f"{'Std EM':<8} | {'Success':<8} | {'Med Step 90':<12} | "
        f"{'Med Step 95':<12} | {'Time (s)':<8}"
    )
    print(header)
    print("-" * 96)

    for op_name, agg in summary.aggregated.items():
        s90_str = f"{agg.median_step_to_90:.1f}" if agg.median_step_to_90 is not None else "N/A"
        s95_str = f"{agg.median_step_to_95:.1f}" if agg.median_step_to_95 is not None else "N/A"
        print(
            f"{op_name:<18} | {agg.parameters:<8} | {str(agg.params_compliant):<9} | "
            f"{agg.mean_em:<9.4f} | {agg.std_em:<8.4f} | {agg.success_rate:<8.1%} | "
            f"{s90_str:<12} | {s95_str:<12} | {agg.mean_wall_clock:<8.2f}"
        )

    print("\n" + "=" * 96)
    print(f"Artifacts saved to: {run_dir}")
    print("=" * 96 + "\n")


if __name__ == "__main__":
    main()
