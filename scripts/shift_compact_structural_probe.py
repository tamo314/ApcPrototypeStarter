#!/usr/bin/env python
"""CLI entry point for the SHIFT compact structural probe (Phase A.1 diagnostic
Task A1-R005E-S006, `docs/CODEX_TASKS_A1_R005E_SHARED_ENCODER_GATE.md`).

Tests whether equipping the primitive-scale compact operator with an explicit
modular relative-position attention bias (`ShiftRelativeCrossPositionOperator`)
allows SHIFT to achieve target accuracy on top of the frozen shared task-blind
representation.

Usage:
    python scripts/shift_compact_structural_probe.py \\
        --config configs/phase_a1_shift_compact_structural_probe.yaml \\
        --run-dir runs/phase_a1_shift_compact_structural_probe
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apc.evaluation.shift_compact_structural_probe import (
    DEFAULT_SEEDS,
    run_shift_compact_structural_probe_multi_seed,
    shift_structural_probe_config_from_dict,
)
from apc.utils.config import load_config, save_config
from apc.utils.system_info import get_system_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to phase_a1_shift_compact_structural_probe.yaml",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Output directory for this run's artifacts",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seed override, e.g. '0,1,2,3,4'",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_config = load_config(args.config)
    base_config = shift_structural_probe_config_from_dict(raw_config)
    if args.seeds is not None:
        seeds = tuple(int(s.strip()) for s in args.seeds.split(","))
    else:
        seeds = tuple(raw_config.get("seeds", DEFAULT_SEEDS))

    resolved_config = base_config.to_dict()
    resolved_config["seeds"] = list(seeds)
    save_config(resolved_config, run_dir / "config.yaml")

    system_info = get_system_info(seed=base_config.seed)
    (run_dir / "system.json").write_text(json.dumps(system_info, indent=2), encoding="utf-8")

    result = run_shift_compact_structural_probe_multi_seed(
        base_config, seeds=seeds, run_dir=run_dir
    )
    (run_dir / "report.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    summary = {
        "seeds": list(result.seeds),
        "passed": result.passed,
        "meets_seed_policy": result.meets_seed_policy,
        "operator_param_count": result.operator_param_count,
        "core_param_count": result.core_param_count,
        "mean_correct_exact_match": result.mean_correct_exact_match,
        "stdev_correct_exact_match": result.stdev_correct_exact_match,
        "min_correct_exact_match": result.min_correct_exact_match,
        "max_correct_exact_match": result.max_correct_exact_match,
        "mean_correct_token_accuracy": result.mean_correct_token_accuracy,
        "mean_wrong_exact_match": result.mean_wrong_exact_match,
        "mean_none_exact_match": result.mean_none_exact_match,
        "mean_exact_match_causal_gap": result.mean_exact_match_causal_gap,
        "mean_argument_effect_rate": result.mean_argument_effect_rate,
        "task_blind_invariant_passed": result.task_blind_invariant_passed,
        "baselines": {
            "s002_shift_correct": result.s002_shift_correct,
            "s002_shift_gap": result.s002_shift_gap,
            "e006a_shift_correct": result.e006a_shift_correct,
            "e006a_shift_gap": result.e006a_shift_gap,
            "e004_shift_correct": result.e004_shift_correct,
            "e004_shift_gap": result.e004_shift_gap,
            "e005_shift_correct": result.e005_shift_correct,
            "e005_shift_gap": result.e005_shift_gap,
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    verdict = "PASS" if result.passed else "FAIL"
    print(f"A1-R005E-S006 SHIFT compact structural probe: {verdict}")
    print(f"Report written to: {run_dir / 'report.json'}")


if __name__ == "__main__":
    main()
