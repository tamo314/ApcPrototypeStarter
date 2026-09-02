#!/usr/bin/env python
"""CLI entry point for the conditioning architecture comparison gate (Phase
A.1 Post-Correction Task A1-R005D-006).

Runs `apc.evaluation.count_conditioning_architecture_gate.
run_conditioning_architecture_comparison` against a
`phase_a1_count_conditioning_architecture_gate.yaml`-shaped baseline config
(A1-R005D-004's own baseline, unmodified): all three conditioning variants
(V0 additive, V1 FiLM, V2 basis modulation) run unconditionally, each
through the full 5-seed gate. Writes the run artifact layout: `config.yaml`,
`comparison_report.json`, `summary.json`, `system.json`, and one
`<variant>/{report.json, seed_<n>/{report.json, core_metrics.jsonl,
primitive_metrics.jsonl}}` per variant.

Usage:
    python scripts/count_conditioning_architecture_gate.py \\
        --config configs/phase_a1_count_conditioning_architecture_gate.yaml \\
        --run-dir runs/phase_a1_count_conditioning_architecture_gate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apc.evaluation.count_conditioning_architecture_gate import (
    run_conditioning_architecture_comparison,
)
from apc.evaluation.count_counterfactual_gate import (
    DEFAULT_SEEDS,
    count_counterfactual_gate_config_from_dict,
)
from apc.primitives.conditioning import DEFAULT_NUM_BASIS_VECTORS
from apc.utils.config import load_config, save_config
from apc.utils.system_info import get_system_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a phase_a1_count_conditioning_architecture_gate.yaml-shaped config",
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
    base_config = count_counterfactual_gate_config_from_dict(raw_config)
    num_basis = raw_config.get("num_basis", DEFAULT_NUM_BASIS_VECTORS)
    if args.seeds is not None:
        seeds = tuple(int(token) for token in args.seeds.split(","))
    else:
        seeds = tuple(raw_config.get("seeds", DEFAULT_SEEDS))

    resolved_config = base_config.to_dict()
    resolved_config["seeds"] = list(seeds)
    resolved_config["num_basis"] = num_basis
    save_config(resolved_config, run_dir / "config.yaml")

    system_info = get_system_info(seed=base_config.seed)
    (run_dir / "system.json").write_text(json.dumps(system_info, indent=2), encoding="utf-8")

    result = run_conditioning_architecture_comparison(
        base_config, seeds=seeds, run_dir=run_dir, num_basis=num_basis
    )
    (run_dir / "comparison_report.json").write_text(
        json.dumps(result.to_dict(), indent=2), encoding="utf-8"
    )

    summary = {
        "seeds": list(seeds),
        "baseline_variant": result.baseline_variant,
        "disqualified_variants": list(result.disqualified_variants),
        "best_by_causal_gap": result.best_by_causal_gap,
        "selected_variant": result.selected_variant,
        "any_variant_passed": result.any_variant_passed,
        "per_variant": {
            variant.variant: {
                "primitive_param_count": variant.primitive_param_count,
                "mean_correct_exact_match": variant.multi_seed.mean_correct_exact_match,
                "mean_effectful_wrong_argument_exact_match": (
                    variant.multi_seed.mean_effectful_wrong_argument_exact_match
                ),
                "mean_none_exact_match": variant.multi_seed.mean_none_exact_match,
                "mean_causal_gap": variant.multi_seed.mean_causal_gap,
                "passed": variant.multi_seed.passed,
            }
            for variant in result.variants
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if result.selected_variant is not None:
        print(
            f"A1-R005D-006 conditioning architecture comparison: PASS with variant "
            f"'{result.selected_variant}' (best by causal gap among passing candidates)"
        )
    else:
        print(
            "A1-R005D-006 conditioning architecture comparison: no variant passed -- "
            f"best by causal gap: {result.best_by_causal_gap!r} -- "
            "per docs/CODEX_TASKS_A1_R005_RETRY.md, STOP (controlled COUNT still cannot pass)"
        )
    print(f"Comparison report written to: {run_dir / 'comparison_report.json'}")


if __name__ == "__main__":
    main()
