#!/usr/bin/env python
"""CLI entry point for Task A1-B007X-007 Fresh-Runtime Recurrence After Compression.

Demonstrates that knowledge lives strictly in persistent state:
- Loads Core + Promoted Bank from serialized checkpoints in a fresh runtime
- Verifies 0 temporary parameters and 0 adaptation steps
- Re-presents novel operation (SWAP_PAIRS) with oracle selection
- Verifies acceptance targets:
  * Recurrence EM >= 0.95
  * Same primitive ID reused (ID 8)
  * Adaptation steps = 0
  * Temporary parameters = 0
  * Bank unchanged (weight delta = 0, size = 9)
  * No consolidation / re-distillation triggered

Usage:
    python scripts/fresh_runtime_recurrence.py \
        --config configs/phase_a1_fresh_runtime_recurrence.yaml \
        --run-dir runs/phase_a1_fresh_runtime_recurrence
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from apc.evaluation.fresh_runtime_recurrence import (
    DEFAULT_SEEDS,
    fresh_runtime_recurrence_config_from_dict,
    run_fresh_runtime_recurrence_multi_seed,
)
from apc.utils.config import load_config, save_config
from apc.utils.system_info import get_system_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to phase_a1_fresh_runtime_recurrence.yaml",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Output directory for this benchmark run's artifacts",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seed override, e.g. '0,1,2,3,4'",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_config = load_config(args.config)
    base_config = fresh_runtime_recurrence_config_from_dict(raw_config)
    if args.seeds is not None:
        seeds = tuple(int(s.strip()) for s in args.seeds.split(","))
    else:
        seeds = tuple(raw_config.get("seeds", DEFAULT_SEEDS))

    resolved_config = dataclasses.replace(base_config, seeds=seeds)
    save_config(resolved_config.to_dict(), run_dir / "config.yaml")

    system_info = get_system_info(seed=seeds[0] if seeds else 0)
    (run_dir / "system.json").write_text(
        json.dumps(system_info, indent=2), encoding="utf-8"
    )

    print(
        f"\n=======================================================\n"
        f"Starting Task A1-B007X-007 Fresh-Runtime Recurrence\n"
        f"Operation: {resolved_config.novel_operation}\n"
        f"Seeds: {list(seeds)}\n"
        f"Accuracy Threshold: {resolved_config.recurrence_accuracy_threshold:.2%}\n"
        f"Output Dir: {run_dir}\n"
        f"=======================================================\n"
    )

    summary, passed = run_fresh_runtime_recurrence_multi_seed(
        resolved_config, run_dir=run_dir
    )

    print(json.dumps(summary, indent=2))

    verdict = summary["verdict"]
    print("\n=======================================================")
    print(f"Task A1-B007X-007 Final Verdict: {verdict}")
    print(f"Mean Immediate Recurrence EM: {summary['mean_immediate_exact_match']:.4f}")
    print(f"Std Immediate Recurrence EM: {summary['std_immediate_exact_match']:.4f}")
    print(f"Mean Unconsolidated EM (Control): {summary['mean_unconsolidated_exact_match']:.4f}")
    print(f"Zero Adaptation Steps: {summary['all_zero_adaptation']}")
    print(f"Zero Temporary Parameters: {summary['all_zero_temporary_parameters']}")
    print(f"Core Unchanged: {summary['all_core_unchanged']}")
    print(f"Bank Unchanged: {summary['all_bank_unchanged']}")
    print(f"Reused Same Primitive ID: {summary['all_same_primitive_id']}")
    print(f"Report written to: {run_dir / 'report.json'}")
    print("=======================================================\n")

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()

