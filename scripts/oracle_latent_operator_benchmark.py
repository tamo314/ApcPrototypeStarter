#!/usr/bin/env python
"""CLI entry point for the oracle latent operator benchmark (Phase A.1
diagnostic Task A1-R005E-003).

Per seed, per operation: pretrains and freezes a dedicated, task-blind,
single-operation Stable Core (`apc.evaluation.shared_core_generalization.
train_shared_core`, `include_task_spec=False`), then evaluates its oracle
latent operator -- for SHIFT/SELECT/BIND, zero-training oracle-addressed
re-routing of the model's own pretrained `decode` head; for COUNT, a small
trained diagnostic readout over oracle-located matched positions. See
`apc.evaluation.oracle_latent_operator_benchmark` module docstring. Writes
the run artifact layout described in `README.md`: `config.yaml`,
`report.json`, `summary.json`, `system.json`, and one
`seed_<n>/{report.json, <OPERATION>_core_metrics.jsonl}` per seed.

Usage:
    python scripts/oracle_latent_operator_benchmark.py \\
        --config configs/phase_a1_oracle_latent_operator_benchmark.yaml \\
        --run-dir runs/phase_a1_oracle_latent_operator_benchmark
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apc.evaluation.oracle_latent_operator_benchmark import (
    DEFAULT_SEEDS,
    oracle_latent_operator_config_from_dict,
    run_oracle_latent_operator_benchmark_multi_seed,
)
from apc.utils.config import load_config, save_config
from apc.utils.system_info import get_system_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a phase_a1_oracle_latent_operator_benchmark.yaml-shaped config",
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
    base_config = oracle_latent_operator_config_from_dict(raw_config)
    if args.seeds is not None:
        seeds = tuple(int(token) for token in args.seeds.split(","))
    else:
        seeds = tuple(raw_config.get("seeds", DEFAULT_SEEDS))

    resolved_config = base_config.to_dict()
    resolved_config["seeds"] = list(seeds)
    save_config(resolved_config, run_dir / "config.yaml")

    system_info = get_system_info(seed=base_config.seed)
    (run_dir / "system.json").write_text(json.dumps(system_info, indent=2), encoding="utf-8")

    result = run_oracle_latent_operator_benchmark_multi_seed(
        base_config, seeds=seeds, run_dir=run_dir
    )
    (run_dir / "report.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    summary = {
        "seeds": list(result.seeds),
        "passed": result.passed,
        "per_operation_summary": {
            name: summary.to_dict() for name, summary in result.per_operation_summary.items()
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    verdict = "PASS" if result.passed else "FAIL"
    print(f"A1-R005E-003 Oracle latent operator benchmark: {verdict}")
    print(f"Report written to: {run_dir / 'report.json'}")


if __name__ == "__main__":
    main()
