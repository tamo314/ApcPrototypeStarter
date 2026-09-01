#!/usr/bin/env python
"""CLI entry point for the Task/Content representation-probe gate (Phase A.1
Correction Task A1-C005, STOP GATE, H1c).

Freezes one shared Stable Core per seed -- trained the same way Task
A1-C004's explicit-task variant is (`apc.evaluation.shared_core_generalization.
train_shared_core`) -- and fits linear probes on its `encode_split` output:
operation identity and (where applicable) operation arguments from `z_task`,
and per-position content-token identity from `h_content`. Writes the run
artifact layout described in `README.md`: `config.yaml`, `report.json`,
`summary.json`, `system.json`, and one `seed_<n>/metrics.jsonl` (the frozen
core's own training progress) per seed.

Usage:
    python scripts/task_content_probes_gate.py \\
        --config configs/phase_a1_task_content_probes.yaml \\
        --run-dir runs/phase_a1_task_content_probes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apc.evaluation.task_content_probes import (
    ARGUMENT_PROBE_THRESHOLD,
    CONTENT_PROBE_TOKEN_ACCURACY_THRESHOLD,
    DEFAULT_SEEDS,
    OPERATION_ID_PROBE_THRESHOLD,
    run_task_content_probes_multi_seed,
    task_content_probe_config_from_dict,
)
from apc.utils.config import load_config, save_config
from apc.utils.system_info import get_system_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, help="Path to a phase_a1_task_content_probes.yaml-shaped config"
    )
    parser.add_argument(
        "--run-dir", required=True, help="Output directory for this run's artifacts"
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help=(
            "Comma-separated seed override, e.g. '0,1,2' "
            "(default: the config's top-level 'seeds' key, or DEFAULT_SEEDS)"
        ),
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_config = load_config(args.config)
    base_config = task_content_probe_config_from_dict(raw_config)
    if args.seeds is not None:
        seeds = tuple(int(token) for token in args.seeds.split(","))
    else:
        seeds = tuple(raw_config.get("seeds", DEFAULT_SEEDS))

    resolved_config = base_config.to_dict()
    resolved_config["seeds"] = list(seeds)
    save_config(resolved_config, run_dir / "config.yaml")

    system_info = get_system_info(seed=base_config.seed)
    (run_dir / "system.json").write_text(json.dumps(system_info, indent=2), encoding="utf-8")

    result = run_task_content_probes_multi_seed(base_config, seeds=seeds, run_dir=run_dir)
    (run_dir / "report.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    summary = {
        "seeds": list(result.seeds),
        "operation_id_threshold": OPERATION_ID_PROBE_THRESHOLD,
        "argument_threshold": ARGUMENT_PROBE_THRESHOLD,
        "content_threshold": CONTENT_PROBE_TOKEN_ACCURACY_THRESHOLD,
        "mean_operation_id_accuracy": result.mean_operation_id_accuracy,
        "min_operation_id_accuracy": result.min_operation_id_accuracy,
        "mean_content_token_accuracy": result.mean_content_token_accuracy,
        "min_content_token_accuracy": result.min_content_token_accuracy,
        "per_seed": [
            {
                "seed": report.config.seed,
                "shared_core_overall_exact_match": report.shared_core_overall_exact_match,
                "operation_id_accuracy": report.operation_id_accuracy,
                "operation_id_passed": report.operation_id_passed,
                "argument_probes": {
                    name: {"metric_name": r.metric_name, "accuracy": r.accuracy, "passed": r.passed}
                    for name, r in report.argument_probes.items()
                },
                "argument_probes_passed": report.argument_probes_passed,
                "content_token_accuracy": report.content_token_accuracy,
                "content_probe_passed": report.content_probe_passed,
                "operation_from_content_accuracy": report.operation_from_content_accuracy,
                "content_from_task_accuracy": report.content_from_task_accuracy,
                "passed": report.passed,
            }
            for report in result.per_seed
        ],
        "passed": result.passed,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    verdict = "PASS" if result.passed else "FAIL"
    print(f"A1-C005 Task/Content representation-probe gate (H1c): {verdict}")
    print(f"Report written to: {run_dir / 'report.json'}")


if __name__ == "__main__":
    main()
