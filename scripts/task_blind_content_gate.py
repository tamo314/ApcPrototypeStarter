#!/usr/bin/env python
"""CLI entry point for the Task-blind content encoder path gate (Phase A.1
Post-Correction Task A1-R001, STOP GATE, H2a).

Builds a freshly-initialized shared-core model per seed (see `apc.
evaluation.task_blind_content_gate` module docstring for why training is
not needed to test this architectural invariant), draws procedurally
generated content spanning all eight known operations, and checks:

  - token-level invariance of the content-only encoder input to task spec,
  - representation-level invariance of `content_state` to task spec,
  - absence of any task token from the content-encoder input,
  - invariance of `content_state` to batch padding.

Writes the run artifact layout described in `README.md`: `config.yaml`,
`report.json`, `summary.json`, `system.json`, and one `seed_<n>/report.json`
per seed.

Usage:
    python scripts/task_blind_content_gate.py \\
        --config configs/phase_a1_task_blind_content_gate.yaml \\
        --run-dir runs/phase_a1_task_blind_content_gate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apc.evaluation.task_blind_content_gate import (
    DEFAULT_SEEDS,
    run_task_blind_content_gate_multi_seed,
    task_blind_content_gate_config_from_dict,
)
from apc.utils.config import load_config, save_config
from apc.utils.system_info import get_system_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a phase_a1_task_blind_content_gate.yaml-shaped config",
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
    base_config = task_blind_content_gate_config_from_dict(raw_config)
    if args.seeds is not None:
        seeds = tuple(int(token) for token in args.seeds.split(","))
    else:
        seeds = tuple(raw_config.get("seeds", DEFAULT_SEEDS))

    resolved_config = base_config.to_dict()
    resolved_config["seeds"] = list(seeds)
    save_config(resolved_config, run_dir / "config.yaml")

    system_info = get_system_info(seed=base_config.seed)
    (run_dir / "system.json").write_text(json.dumps(system_info, indent=2), encoding="utf-8")

    result = run_task_blind_content_gate_multi_seed(base_config, seeds=seeds, run_dir=run_dir)
    (run_dir / "report.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    summary = {
        "seeds": list(result.seeds),
        "meets_seed_policy": result.meets_seed_policy,
        "max_representation_absolute_difference": result.max_representation_absolute_difference,
        "max_padding_absolute_difference": result.max_padding_absolute_difference,
        "per_seed": [
            {
                "seed": report.config.seed,
                "num_contents_tested": report.num_contents_tested,
                "all_operations_covered": report.all_operations_covered,
                "token_level_invariant": report.token_level_invariant,
                "representation_invariant": report.representation_invariant,
                "padding_invariant": report.padding_invariant,
                "no_task_token_leak": report.no_task_token_leak,
                "passed": report.passed,
            }
            for report in result.per_seed
        ],
        "passed": result.passed,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    verdict = "PASS" if result.passed else "FAIL"
    print(f"A1-R001 Task-blind content encoder path gate (H2a): {verdict}")
    print(f"Report written to: {run_dir / 'report.json'}")


if __name__ == "__main__":
    main()
