#!/usr/bin/env python
"""CLI entry point for the SHIFT/SELECT sequence counterfactual gate (Phase
A.1 Post-Correction Task A1-R005D-008).

The config file's own `operation` field (`SHIFT` or `SELECT`) selects which
operation this run trains and evaluates -- see `configs/
phase_a1_shift_counterfactual_gate.yaml`/`phase_a1_select_counterfactual_gate.yaml`.
Per seed: pretrains and freezes a task-blind, single-operation Stable Core
(`apc.evaluation.shared_core_generalization.train_shared_core`,
`include_task_spec=False`), trains one argument-conditioned oracle-routed
`ConditionedPrimitive` on counterfactual groups (`apc.evaluation.
sequence_counterfactual_gate.generate_sequence_counterfactual_groups` -- the
same content paired with several distinct argument values whose outputs are
pairwise distinct), then evaluates Correct/effectful Wrong argument/None
(exact match + token accuracy, plus Correct's own length-stratified
breakdown) on a large unseen batch of counterfactual groups. Writes the run
artifact layout described in `README.md`: `config.yaml`, `report.json`,
`summary.json`, `system.json`, and one `seed_<n>/{report.json,
core_metrics.jsonl, primitive_metrics.jsonl}` per seed.

Usage:
    python scripts/sequence_counterfactual_gate.py \\
        --config configs/phase_a1_shift_counterfactual_gate.yaml \\
        --run-dir runs/phase_a1_shift_counterfactual_gate
    python scripts/sequence_counterfactual_gate.py \\
        --config configs/phase_a1_select_counterfactual_gate.yaml \\
        --run-dir runs/phase_a1_select_counterfactual_gate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apc.evaluation.sequence_counterfactual_gate import (
    DEFAULT_SEEDS,
    EXACT_MATCH_THRESHOLD,
    MIN_EXACT_MATCH_CAUSAL_GAP,
    NONE_CEILING,
    TOKEN_ACCURACY_THRESHOLD,
    run_sequence_counterfactual_gate_multi_seed,
    sequence_counterfactual_gate_config_from_dict,
)
from apc.utils.config import load_config, save_config
from apc.utils.system_info import get_system_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a phase_a1_{shift,select}_counterfactual_gate.yaml-shaped config",
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
    base_config = sequence_counterfactual_gate_config_from_dict(raw_config)
    if args.seeds is not None:
        seeds = tuple(int(token) for token in args.seeds.split(","))
    else:
        seeds = tuple(raw_config.get("seeds", DEFAULT_SEEDS))

    resolved_config = base_config.to_dict()
    resolved_config["seeds"] = list(seeds)
    save_config(resolved_config, run_dir / "config.yaml")

    system_info = get_system_info(seed=base_config.seed)
    (run_dir / "system.json").write_text(json.dumps(system_info, indent=2), encoding="utf-8")

    result = run_sequence_counterfactual_gate_multi_seed(
        base_config,
        seeds=seeds,
        run_dir=run_dir,
        exact_match_threshold=EXACT_MATCH_THRESHOLD,
        token_accuracy_threshold=TOKEN_ACCURACY_THRESHOLD,
        none_ceiling=NONE_CEILING,
        min_exact_match_causal_gap=MIN_EXACT_MATCH_CAUSAL_GAP,
    )
    (run_dir / "report.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    summary = {
        "operation": result.operation,
        "seeds": list(result.seeds),
        "meets_seed_policy": result.meets_seed_policy,
        "mean_correct_exact_match": result.mean_correct_exact_match,
        "stdev_correct_exact_match": result.stdev_correct_exact_match,
        "min_correct_exact_match": result.min_correct_exact_match,
        "max_correct_exact_match": result.max_correct_exact_match,
        "mean_correct_token_accuracy": result.mean_correct_token_accuracy,
        "mean_effectful_wrong_argument_exact_match": (
            result.mean_effectful_wrong_argument_exact_match
        ),
        "mean_effectful_wrong_argument_token_accuracy": (
            result.mean_effectful_wrong_argument_token_accuracy
        ),
        "mean_none_exact_match": result.mean_none_exact_match,
        "mean_none_token_accuracy": result.mean_none_token_accuracy,
        "mean_exact_match_causal_gap": result.mean_exact_match_causal_gap,
        "mean_token_accuracy_causal_gap": result.mean_token_accuracy_causal_gap,
        "mean_argument_effect_rate": result.mean_argument_effect_rate,
        "exact_match_threshold": result.exact_match_threshold,
        "token_accuracy_threshold": result.token_accuracy_threshold,
        "none_ceiling": result.none_ceiling,
        "min_exact_match_causal_gap": result.min_exact_match_causal_gap,
        "exact_match_passed": result.exact_match_passed,
        "token_accuracy_passed": result.token_accuracy_passed,
        "none_passed": result.none_passed,
        "causal_gap_passed": result.causal_gap_passed,
        "family_count_passed": result.family_count_passed,
        "per_seed": {
            str(report.config.seed): {
                "correct_exact_match": report.correct_exact_match,
                "correct_token_accuracy": report.correct_token_accuracy,
                "effectful_wrong_argument_exact_match": (
                    report.effectful_wrong_argument_exact_match
                ),
                "effectful_wrong_argument_token_accuracy": (
                    report.effectful_wrong_argument_token_accuracy
                ),
                "none_exact_match": report.none_exact_match,
                "none_token_accuracy": report.none_token_accuracy,
                "exact_match_causal_gap": report.exact_match_causal_gap,
                "token_accuracy_causal_gap": report.token_accuracy_causal_gap,
                "argument_effect_rate": report.argument_effect_rate,
                "bank_size": report.bank_size,
                "length_stratified_correct": report.length_stratified_correct,
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
    print(f"A1-R005D-008 {result.operation} sequence gate: {verdict}{seed_note}{stop_gate_note}")
    print(f"Report written to: {run_dir / 'report.json'}")


if __name__ == "__main__":
    main()
