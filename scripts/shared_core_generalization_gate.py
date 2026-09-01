#!/usr/bin/env python
"""CLI entry point for the Shared-Core systematic-generalization gate
(Phase A.1 Correction Task A1-C004, STOP GATE, H1b).

Trains one shared Stable Core per seed on `apc.environments.generator.
build_mixed_operation_generator`'s online-generated mixed-operation stream,
with the model-visible task-specification segment included
(`include_task_spec=True`, Task A1-C003) -- no primitive bank/router/
plastic/consolidation involved -- then evaluates overall and per-operation
exact match on a large held-out batch of unseen content. A negative-control
variant (`include_task_spec=False`) reruns the identical setup with the task
segment stripped from the model input, to verify the gap is attributable to
that segment specifically. Writes the run artifact layout described in
`README.md`: `config.yaml`, `report.json`, `summary.json`, `system.json`,
and one `{explicit,negative_control}/seed_<n>/metrics.jsonl` per variant/seed.

Usage:
    python scripts/shared_core_generalization_gate.py \\
        --config configs/phase_a1_shared_core_gate.yaml \\
        --run-dir runs/phase_a1_shared_core_gate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apc.evaluation.shared_core_generalization import (
    DEFAULT_SEEDS,
    MATERIAL_UNDERPERFORMANCE_MARGIN,
    OVERALL_EXACT_MATCH_THRESHOLD,
    PER_OPERATION_EXACT_MATCH_THRESHOLD,
    run_shared_core_gate_h1b,
    shared_core_gate_config_from_dict,
)
from apc.utils.config import load_config, save_config
from apc.utils.system_info import get_system_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, help="Path to a phase_a1_shared_core_gate.yaml-shaped config"
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
    base_config = shared_core_gate_config_from_dict(raw_config)
    if args.seeds is not None:
        seeds = tuple(int(token) for token in args.seeds.split(","))
    else:
        seeds = tuple(raw_config.get("seeds", DEFAULT_SEEDS))

    resolved_config = base_config.to_dict()
    resolved_config["seeds"] = list(seeds)
    save_config(resolved_config, run_dir / "config.yaml")

    system_info = get_system_info(seed=base_config.seed)
    (run_dir / "system.json").write_text(json.dumps(system_info, indent=2), encoding="utf-8")

    result = run_shared_core_gate_h1b(
        base_config,
        seeds=seeds,
        run_dir=run_dir,
        overall_threshold=OVERALL_EXACT_MATCH_THRESHOLD,
        per_operation_threshold=PER_OPERATION_EXACT_MATCH_THRESHOLD,
        material_underperformance_margin=MATERIAL_UNDERPERFORMANCE_MARGIN,
    )
    (run_dir / "report.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    summary = {
        "operation_names": list(result.explicit.operation_names),
        "seeds": list(result.explicit.seeds),
        "meets_seed_policy": result.meets_seed_policy,
        "overall_threshold": result.explicit.overall_threshold,
        "per_operation_threshold": result.explicit.per_operation_threshold,
        "material_underperformance_margin": result.material_underperformance_margin,
        "explicit": {
            "mean_overall_exact_match": result.explicit.mean_overall_exact_match,
            "stdev_overall_exact_match": result.explicit.stdev_overall_exact_match,
            "passed": result.explicit.passed,
            "per_operation_mean_exact_match": result.explicit.per_operation_mean_exact_match,
            "per_seed_overall_exact_match": {
                str(report.config.seed): report.overall_exact_match
                for report in result.explicit.per_seed
            },
            "low_outlier_threshold": result.explicit.low_outlier_threshold,
            "low_outlier_seed_operations": list(result.explicit.low_outlier_seed_operations),
        },
        "negative_control": {
            "mean_overall_exact_match": result.negative_control.mean_overall_exact_match,
            "stdev_overall_exact_match": result.negative_control.stdev_overall_exact_match,
            "passed": result.negative_control.passed,
            "per_operation_mean_exact_match": (
                result.negative_control.per_operation_mean_exact_match
            ),
            "per_seed_overall_exact_match": {
                str(report.config.seed): report.overall_exact_match
                for report in result.negative_control.per_seed
            },
        },
        "overall_exact_match_gap": result.overall_exact_match_gap,
        "negative_control_materially_underperforms": (
            result.negative_control_materially_underperforms
        ),
        "passed": result.passed,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    verdict = "PASS" if result.passed else "FAIL"
    seed_note = (
        ""
        if result.meets_seed_policy
        else " (WARNING: fewer than 5 seeds -- does not satisfy the gate's seed policy)"
    )
    print(f"A1-C004 Shared-Core generalization gate (H1b): {verdict}{seed_note}")
    print(f"Report written to: {run_dir / 'report.json'}")


if __name__ == "__main__":
    main()
