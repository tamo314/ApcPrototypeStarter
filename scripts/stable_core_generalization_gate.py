#!/usr/bin/env python
"""CLI entry point for the Stable Core systematic-generalization gate
(Phase A.1 Task A1-006, STOP GATE).

Trains one fresh Stable Core per `(operation, seed)` pair on online-generated
examples of a single deterministic known operation and evaluates exact match
on a large held-out batch of unseen content -- no primitive bank/router/
plastic/consolidation involved, and never a mixed-operation pool (see
`apc.evaluation.stable_core_generalization`'s module docstring for why both
matter). Writes the run artifact layout described in `README.md`:
`config.yaml`, `report.json`, `summary.json`, `system.json`, and one
`<operation>/seed_<n>/metrics.jsonl` per `(operation, seed)` pair.

Two config variants are shipped (see `docs/DECISIONS.md` ADR-0019/ADR-0020):
- `configs/phase_a1_stable_core_gate.yaml` -- primary: all four
  deterministic operations, `permute_symbols=False`.
- `configs/phase_a1_stable_core_gate_permuted_copy_only.yaml` -- secondary:
  `COPY` only, `permute_symbols=True`.

Usage:
    python scripts/stable_core_generalization_gate.py \\
        --config configs/phase_a1_stable_core_gate.yaml \\
        --run-dir runs/phase_a1_stable_core_gate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apc.evaluation.stable_core_generalization import (
    DEFAULT_SEEDS,
    UNSEEN_EXACT_MATCH_THRESHOLD,
    run_stable_core_gate_grid,
    stable_core_gate_config_from_dict,
)
from apc.utils.config import load_config, save_config
from apc.utils.system_info import get_system_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, help="Path to a phase_a1_stable_core_gate.yaml-shaped config"
    )
    parser.add_argument(
        "--run-dir", required=True, help="Output directory for this run's artifacts"
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help=(
            "Comma-separated seed override, e.g. '0,1,2,3,4' "
            "(default: the config's top-level 'seeds' key, or 0-4)"
        ),
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_config = load_config(args.config)
    base_config = stable_core_gate_config_from_dict(raw_config)
    operation_names = base_config.operation_names
    if args.seeds is not None:
        seeds = tuple(int(token) for token in args.seeds.split(","))
    else:
        seeds = tuple(raw_config.get("seeds", DEFAULT_SEEDS))

    resolved_config = base_config.to_dict()
    resolved_config["seeds"] = list(seeds)
    save_config(resolved_config, run_dir / "config.yaml")

    system_info = get_system_info(seed=base_config.seed)
    (run_dir / "system.json").write_text(json.dumps(system_info, indent=2), encoding="utf-8")

    result = run_stable_core_gate_grid(
        base_config,
        operation_names=operation_names,
        seeds=seeds,
        run_dir=run_dir,
        threshold=UNSEEN_EXACT_MATCH_THRESHOLD,
    )
    (run_dir / "report.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    summary = {
        "operation_names": list(result.operation_names),
        "seeds": list(result.seeds),
        "meets_seed_policy": result.meets_seed_policy,
        "threshold": result.threshold,
        "mean_unseen_exact_match": result.mean_unseen_exact_match,
        "stdev_unseen_exact_match": result.stdev_unseen_exact_match,
        "min_unseen_exact_match": result.min_unseen_exact_match,
        "max_unseen_exact_match": result.max_unseen_exact_match,
        "passed": result.passed,
        "per_operation": {
            operation_report.operation_name: {
                "mean_unseen_exact_match": operation_report.multi_seed.mean_unseen_exact_match,
                "stdev_unseen_exact_match": operation_report.multi_seed.stdev_unseen_exact_match,
                "passed": operation_report.multi_seed.passed,
                "per_seed_unseen_exact_match": {
                    str(report.config.seed): report.unseen_exact_match
                    for report in operation_report.multi_seed.per_seed
                },
            }
            for operation_report in result.per_operation
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    verdict = "PASS" if result.passed else "FAIL"
    seed_note = (
        ""
        if result.meets_seed_policy
        else " (WARNING: fewer than 5 seeds -- does not satisfy the gate's seed policy)"
    )
    print(f"A1-006 Stable Core generalization gate: {verdict}{seed_note}")
    print(f"Report written to: {run_dir / 'report.json'}")


if __name__ == "__main__":
    main()
