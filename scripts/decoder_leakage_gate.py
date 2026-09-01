#!/usr/bin/env python
"""CLI entry point for the Decoder leakage control gate (Phase A.1
Post-Correction Task A1-R002, conditional STOP GATE).

Trains one shared Stable Core per seed with no task-specification token ever
visible (`apc.evaluation.shared_core_generalization.train_shared_core`,
`include_task_spec` forced `False`) on `apc.environments.generator.
build_mixed_operation_generator`'s online-generated mixed-operation stream --
no primitive bank/router/plastic/consolidation involved -- then evaluates
overall and per-operation exact match through `apc.core.execution.
evaluate_exact_match_no_primitive`, the causal ablation matrix's "None" arm.
Writes the run artifact layout described in `README.md`: `config.yaml`,
`report.json`, `summary.json`, `system.json`, and one `seed_<n>/{report.json,
metrics.jsonl}` per seed.

Usage:
    python scripts/decoder_leakage_gate.py \\
        --config configs/phase_a1_decoder_leakage_gate.yaml \\
        --run-dir runs/phase_a1_decoder_leakage_gate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apc.evaluation.decoder_leakage_gate import (
    DEFAULT_SEEDS,
    FUTURE_CORRECT_TARGET,
    NO_PRIMITIVE_MATERIAL_CEILING,
    decoder_leakage_gate_config_from_dict,
    run_decoder_leakage_gate_multi_seed,
)
from apc.utils.config import load_config, save_config
from apc.utils.system_info import get_system_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, help="Path to a phase_a1_decoder_leakage_gate.yaml-shaped config"
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
    base_config = decoder_leakage_gate_config_from_dict(raw_config)
    if args.seeds is not None:
        seeds = tuple(int(token) for token in args.seeds.split(","))
    else:
        seeds = tuple(raw_config.get("seeds", DEFAULT_SEEDS))

    resolved_config = base_config.to_dict()
    resolved_config["seeds"] = list(seeds)
    save_config(resolved_config, run_dir / "config.yaml")

    system_info = get_system_info(seed=base_config.seed)
    (run_dir / "system.json").write_text(json.dumps(system_info, indent=2), encoding="utf-8")

    result = run_decoder_leakage_gate_multi_seed(
        base_config,
        seeds=seeds,
        run_dir=run_dir,
        material_ceiling=NO_PRIMITIVE_MATERIAL_CEILING,
        future_correct_target=FUTURE_CORRECT_TARGET,
    )
    (run_dir / "report.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    summary = {
        "operation_names": list(result.operation_names),
        "seeds": list(result.seeds),
        "meets_seed_policy": result.meets_seed_policy,
        "material_ceiling": result.material_ceiling,
        "future_correct_target": result.future_correct_target,
        "mean_overall_exact_match": result.mean_overall_exact_match,
        "stdev_overall_exact_match": result.stdev_overall_exact_match,
        "min_overall_exact_match": result.min_overall_exact_match,
        "max_overall_exact_match": result.max_overall_exact_match,
        "per_operation_mean_exact_match": result.per_operation_mean_exact_match,
        "per_seed_overall_exact_match": {
            str(report.config.seed): report.overall_exact_match for report in result.per_seed
        },
        "causal_mode_matches_plain_generation": result.causal_mode_matches_plain_generation,
        "materially_below_future_correct_target": result.materially_below_future_correct_target,
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
    stop_gate_note = (
        "" if result.passed else " -- STOP GATE: no-primitive path stays near full accuracy"
    )
    print(f"A1-R002 Decoder leakage control gate: {verdict}{seed_note}{stop_gate_note}")
    print(f"Report written to: {run_dir / 'report.json'}")


if __name__ == "__main__":
    main()
