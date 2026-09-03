#!/usr/bin/env python
"""CLI entry point for the Shared queryable representation architecture
wiring gate (Phase A.1 diagnostic Task A1-R005E-S001,
`docs/CODEX_TASKS_A1_R005E_SHARED_ENCODER_GATE.md`).

Builds, per seed, exactly one shared task-blind content encoder plus one
compact operator per configured operation (default SHIFT/SELECT/COUNT/BIND),
and checks:

  - operators hold no encoder submodule of their own (one shared encoder
    object),
  - the shared encoder has no operation-specific module or parameter,
  - every operation's own argument encoder/readout is a distinct object,
  - identical content produces identical encoder states regardless of which
    operation dispatches through it,
  - every operation's own compact operator actually receives a gradient
    through the shared encoder,
  - every operation's own forward pass produces correctly-shaped logits,
  - oracle dispatch rejects an unconfigured operation.

No milestone benchmark is run here (task Acceptance: "No milestone benchmark
yet") -- see `apc.evaluation.shared_encoder_architecture_gate`'s module
docstring.

Writes the run artifact layout described in `README.md`: `config.yaml`,
`report.json`, `summary.json`, `system.json`, and one `seed_<n>/report.json`
per seed.

Usage:
    python scripts/shared_encoder_architecture_gate.py \\
        --config configs/phase_a1_shared_encoder_architecture_gate.yaml \\
        --run-dir runs/phase_a1_shared_encoder_architecture_gate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from apc.evaluation.shared_encoder_architecture_gate import (
    DEFAULT_SEEDS,
    run_shared_encoder_architecture_gate_multi_seed,
    shared_encoder_architecture_config_from_dict,
)
from apc.utils.config import load_config, save_config
from apc.utils.system_info import get_system_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a phase_a1_shared_encoder_architecture_gate.yaml-shaped config",
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
    base_config = shared_encoder_architecture_config_from_dict(raw_config)
    if args.seeds is not None:
        seeds = tuple(int(token) for token in args.seeds.split(","))
    else:
        seeds = tuple(raw_config.get("seeds", DEFAULT_SEEDS))

    resolved_config = base_config.to_dict()
    resolved_config["seeds"] = list(seeds)
    save_config(resolved_config, run_dir / "config.yaml")

    system_info = get_system_info(seed=base_config.seed)
    (run_dir / "system.json").write_text(json.dumps(system_info, indent=2), encoding="utf-8")

    result = run_shared_encoder_architecture_gate_multi_seed(
        base_config, seeds=seeds, run_dir=run_dir
    )
    (run_dir / "report.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    def _seed_summary(report: Any) -> dict[str, Any]:
        return {
            "seed": report.config.seed,
            "operation_names": list(report.operation_names),
            "core_param_count": report.core_param_count,
            "operator_param_counts": report.operator_param_counts,
            "operators_hold_no_encoder_submodule": report.operators_hold_no_encoder_submodule,
            "no_operation_specific_encoder_modules": (
                report.no_operation_specific_encoder_modules
            ),
            "per_operation_argument_encoder_distinct": (
                report.per_operation_argument_encoder_distinct
            ),
            "per_operation_readout_distinct": report.per_operation_readout_distinct,
            "content_state_invariant_across_operations": (
                report.content_state_invariant_across_operations
            ),
            "content_state_max_abs_diff_across_operations": (
                report.content_state_max_abs_diff_across_operations
            ),
            "all_operations_encoder_grad_connected": report.all_operations_encoder_grad_connected,
            "all_output_shapes_valid": report.all_output_shapes_valid,
            "oracle_dispatch_rejects_unknown_operation": (
                report.oracle_dispatch_rejects_unknown_operation
            ),
            "passed": report.passed,
        }

    summary = {
        "seeds": list(result.seeds),
        "meets_seed_policy": result.meets_seed_policy,
        "max_content_state_absolute_difference": result.max_content_state_absolute_difference,
        "per_seed": [_seed_summary(report) for report in result.per_seed],
        "passed": result.passed,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    verdict = "PASS" if result.passed else "FAIL"
    print(f"A1-R005E-S001 Shared encoder architecture wiring gate: {verdict}")
    print(f"Report written to: {run_dir / 'report.json'}")


if __name__ == "__main__":
    main()
