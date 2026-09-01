#!/usr/bin/env python
"""CLI entry point for the Parameterized oracle primitive benchmark (Phase
A.1 Post-Correction Task A1-R005, STOP GATE, H2c).

Per seed: pretrains and freezes a task-blind Stable Core (`apc.evaluation.
shared_core_generalization.train_shared_core`, `include_task_spec=False`,
restricted to `apc.environments.operations.PARAMETERIZED_OPERATION_NAMES`),
trains one argument-conditioned oracle-routed primitive
(`apc.primitives.conditioning.ConditionedPrimitive`) per operation on top of
it, then evaluates the four-arm causal ablation matrix (Correct/Wrong
argument/Wrong family/None) on a large unseen batch. Writes the run artifact
layout described in `README.md`: `config.yaml`, `report.json`,
`summary.json`, `system.json`, and one `seed_<n>/{report.json,
core_metrics.jsonl, primitive_metrics.jsonl}` per seed.

Usage:
    python scripts/parameterized_primitive_gate.py \\
        --config configs/phase_a1_parameterized_primitive_gate.yaml \\
        --run-dir runs/phase_a1_parameterized_primitive_gate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apc.evaluation.parameterized_primitive_gate import (
    CORRECT_THRESHOLD,
    DEFAULT_SEEDS,
    MIN_CAUSAL_GAP,
    NONE_CEILING,
    WRONG_ARGUMENT_CEILING,
    WRONG_FAMILY_CEILING,
    parameterized_primitive_gate_config_from_dict,
    run_parameterized_primitive_gate_multi_seed,
)
from apc.utils.config import load_config, save_config
from apc.utils.system_info import get_system_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a phase_a1_parameterized_primitive_gate.yaml-shaped config",
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
    base_config = parameterized_primitive_gate_config_from_dict(raw_config)
    if args.seeds is not None:
        seeds = tuple(int(token) for token in args.seeds.split(","))
    else:
        seeds = tuple(raw_config.get("seeds", DEFAULT_SEEDS))

    resolved_config = base_config.to_dict()
    resolved_config["seeds"] = list(seeds)
    save_config(resolved_config, run_dir / "config.yaml")

    system_info = get_system_info(seed=base_config.seed)
    (run_dir / "system.json").write_text(json.dumps(system_info, indent=2), encoding="utf-8")

    result = run_parameterized_primitive_gate_multi_seed(
        base_config,
        seeds=seeds,
        run_dir=run_dir,
        correct_threshold=CORRECT_THRESHOLD,
        wrong_argument_ceiling=WRONG_ARGUMENT_CEILING,
        wrong_family_ceiling=WRONG_FAMILY_CEILING,
        none_ceiling=NONE_CEILING,
        min_causal_gap=MIN_CAUSAL_GAP,
    )
    (run_dir / "report.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    summary = {
        "operation_names": list(result.operation_names),
        "seeds": list(result.seeds),
        "meets_seed_policy": result.meets_seed_policy,
        "mean_correct_exact_match": result.mean_correct_exact_match,
        "stdev_correct_exact_match": result.stdev_correct_exact_match,
        "min_correct_exact_match": result.min_correct_exact_match,
        "max_correct_exact_match": result.max_correct_exact_match,
        "mean_wrong_argument_exact_match": result.mean_wrong_argument_exact_match,
        "mean_wrong_family_exact_match": result.mean_wrong_family_exact_match,
        "mean_none_exact_match": result.mean_none_exact_match,
        "mean_causal_gap": result.mean_causal_gap,
        "per_operation_mean_correct_exact_match": result.per_operation_mean_correct_exact_match,
        "correct_threshold": result.correct_threshold,
        "wrong_argument_ceiling": result.wrong_argument_ceiling,
        "wrong_family_ceiling": result.wrong_family_ceiling,
        "none_ceiling": result.none_ceiling,
        "min_causal_gap": result.min_causal_gap,
        "correct_passed": result.correct_passed,
        "wrong_argument_passed": result.wrong_argument_passed,
        "wrong_family_passed": result.wrong_family_passed,
        "none_passed": result.none_passed,
        "causal_gap_passed": result.causal_gap_passed,
        "family_count_passed": result.family_count_passed,
        "per_seed": {
            str(report.config.seed): {
                "correct_exact_match": report.correct_exact_match,
                "wrong_argument_exact_match": report.wrong_argument_exact_match,
                "wrong_family_exact_match": report.wrong_family_exact_match,
                "none_exact_match": report.none_exact_match,
                "causal_gap": report.causal_gap,
                "bank_size": report.bank_size,
            }
            for report in result.per_seed
        },
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
    stop_gate_note = "" if result.passed else " -- STOP GATE: causal ablation matrix not satisfied"
    print(
        f"A1-R005 Parameterized oracle primitive benchmark: {verdict}{seed_note}{stop_gate_note}"
    )
    print(f"Report written to: {run_dir / 'report.json'}")


if __name__ == "__main__":
    main()
