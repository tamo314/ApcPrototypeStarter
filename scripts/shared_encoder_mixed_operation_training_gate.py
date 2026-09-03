#!/usr/bin/env python
"""CLI entry point for the balanced mixed-operation training gate (Phase A.1
diagnostic Task A1-R005E-S002,
`docs/CODEX_TASKS_A1_R005E_SHARED_ENCODER_GATE.md`).

Per seed: builds the A1-R005E-S001 shared-encoder architecture (`apc.
evaluation.shared_encoder_architecture_gate.build_shared_encoder_
architecture`), jointly trains the whole architecture -- one optimizer over
the shared encoder plus every operator -- on a balanced interleaving of all
four operations' own counterfactual groups, then evaluates the Correct /
effectful Wrong argument / None causal ablation for each operation on a
large unseen batch. See `apc.evaluation.shared_encoder_mixed_operation_
training_gate`'s module docstring for full design rationale. Writes the run
artifact layout described in `README.md`: `config.yaml`, `report.json`,
`summary.json`, `system.json`, and one
`seed_<n>/{report.json, mixed_training_metrics.jsonl}` per seed.

Usage:
    python scripts/shared_encoder_mixed_operation_training_gate.py \\
        --config configs/phase_a1_shared_encoder_mixed_operation_training_gate.yaml \\
        --run-dir runs/phase_a1_shared_encoder_mixed_operation_training_gate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apc.evaluation.shared_encoder_mixed_operation_training_gate import (
    DEFAULT_SEEDS,
    run_shared_mixed_training_gate_multi_seed,
    shared_mixed_training_config_from_dict,
)
from apc.utils.config import load_config, save_config
from apc.utils.system_info import get_system_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a phase_a1_shared_encoder_mixed_operation_training_gate.yaml-shaped config",
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
    base_config = shared_mixed_training_config_from_dict(raw_config)
    if args.seeds is not None:
        seeds = tuple(int(token) for token in args.seeds.split(","))
    else:
        seeds = tuple(raw_config.get("seeds", DEFAULT_SEEDS))

    resolved_config = base_config.to_dict()
    resolved_config["seeds"] = list(seeds)
    save_config(resolved_config, run_dir / "config.yaml")

    system_info = get_system_info(seed=base_config.seed)
    (run_dir / "system.json").write_text(json.dumps(system_info, indent=2), encoding="utf-8")

    result = run_shared_mixed_training_gate_multi_seed(base_config, seeds=seeds, run_dir=run_dir)
    (run_dir / "report.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    summary = {
        "seeds": list(result.seeds),
        "passed": result.passed,
        "meets_seed_policy": result.meets_seed_policy,
        "core_param_count": result.core_param_count,
        "per_operation_summary": {
            name: summary.to_dict() for name, summary in result.per_operation_summary.items()
        },
        "per_seed_operation_realized_sampling_proportions": [
            {
                "seed": report.config.seed,
                "operation_realized_sampling_proportions": (
                    report.operation_realized_sampling_proportions
                ),
                "operation_step_counts": report.operation_step_counts,
                "operation_examples_seen": report.operation_examples_seen,
                "encoder_update_count": report.encoder_update_count,
            }
            for report in result.per_seed
        ],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    verdict = "PASS" if result.passed else "FAIL"
    print(f"A1-R005E-S002 Balanced mixed-operation training gate: {verdict}")
    print(f"Report written to: {run_dir / 'report.json'}")


if __name__ == "__main__":
    main()
