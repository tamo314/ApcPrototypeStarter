#!/usr/bin/env python
"""CLI entry point for the frozen `h_content` information audit (Phase A.1
diagnostic Task A1-R005E-002).

Per seed, per operation: pretrains and freezes a dedicated, task-blind,
single-operation Stable Core (`apc.evaluation.shared_core_generalization.
train_shared_core`, `include_task_spec=False`), then fits the token
identity/absolute position/full content-sequence reconstruction probes (and,
for BIND, the key/value role and key/paired-value adjacency probes) on its
frozen `h_content`. Diagnostic only -- see `apc.evaluation.
representation_audit` module docstring; there is no STOP GATE verdict.
Writes the run artifact layout described in `README.md`: `config.yaml`,
`report.json`, `summary.json`, `system.json`, and one
`seed_<n>/{report.json, <OPERATION>_core_metrics.jsonl}` per seed.

Usage:
    python scripts/representation_audit.py \\
        --config configs/phase_a1_representation_audit.yaml \\
        --run-dir runs/phase_a1_representation_audit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apc.evaluation.representation_audit import (
    DEFAULT_SEEDS,
    representation_audit_config_from_dict,
    run_representation_audit_multi_seed,
)
from apc.utils.config import load_config, save_config
from apc.utils.system_info import get_system_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a phase_a1_representation_audit.yaml-shaped config",
    )
    parser.add_argument(
        "--run-dir", required=True, help="Output directory for this run's artifacts"
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help=(
            "Comma-separated seed override, e.g. '0,1,2' "
            "(default: the config's top-level 'seeds' key, or 0-2)"
        ),
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_config = load_config(args.config)
    base_config = representation_audit_config_from_dict(raw_config)
    if args.seeds is not None:
        seeds = tuple(int(token) for token in args.seeds.split(","))
    else:
        seeds = tuple(raw_config.get("seeds", DEFAULT_SEEDS))

    resolved_config = base_config.to_dict()
    resolved_config["seeds"] = list(seeds)
    save_config(resolved_config, run_dir / "config.yaml")

    system_info = get_system_info(seed=base_config.seed)
    (run_dir / "system.json").write_text(json.dumps(system_info, indent=2), encoding="utf-8")

    result = run_representation_audit_multi_seed(base_config, seeds=seeds, run_dir=run_dir)
    (run_dir / "report.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    summary = {
        "seeds": list(result.seeds),
        "per_operation_summary": {
            name: summary.to_dict() for name, summary in result.per_operation_summary.items()
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    print("A1-R005E-002 Frozen h_content information audit: diagnostic run complete")
    print(f"Report written to: {run_dir / 'report.json'}")


if __name__ == "__main__":
    main()
