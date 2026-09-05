#!/usr/bin/env python
"""CLI entry point for Incremental Router Update Benchmark (Task A2-C003 / Phase A.2 Stop Gate).

Tests class-incremental routing under bank growth from 10 to 16 semantic operations:
10 (initial B008 bank) -> 12 -> 14 -> 16.

Compares conditions R0 (full retrain), R1 (naive new-class update), and R2 (bounded replay).

Usage:
    python scripts/incremental_router_benchmark.py \
        --output-dir runs/phase_a2_incremental_router_gate \
        --seeds 0,1,2,3,4 \
        --conditions R2,R0,R1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from apc.evaluation.incremental_router_benchmark import (
    IncrementalUpdateCondition,
    run_incremental_router_benchmark,
)
from apc.utils.system_info import get_system_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="runs/phase_a2_incremental_router_gate",
        help="Output directory for benchmark artifacts",
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2,3,4",
        help="Comma-separated seeds, e.g. '0,1,2,3,4'",
    )
    parser.add_argument(
        "--conditions",
        default="R2,R0,R1",
        help="Comma-separated conditions, e.g. 'R2,R0,R1'",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = tuple(int(s.strip()) for s in args.seeds.split(","))
    cond_map = {
        "R0": IncrementalUpdateCondition.R0_FULL_RETRAIN,
        "R1": IncrementalUpdateCondition.R1_NAIVE_NEW,
        "R2": IncrementalUpdateCondition.R2_BOUNDED_REPLAY,
    }
    conditions = tuple(cond_map[c.strip()] for c in args.conditions.split(","))

    system_info = get_system_info()
    with open(out_dir / "system.json", "w") as f:
        json.dump(system_info, f, indent=2)

    results = run_incremental_router_benchmark(
        seeds=seeds,
        conditions=conditions,
        output_dir=out_dir,
    )

    r2_res = results.get("R2")
    if r2_res and r2_res.get("all_passed"):
        print("\n==========================================")
        print("A2-C003 STOP GATE: PASSED!")
        print("==========================================")
    else:
        print("\n==========================================")
        print("A2-C003 STOP GATE: EVALUATION COMPLETE (Check results)")
        print("==========================================")


if __name__ == "__main__":
    main()
