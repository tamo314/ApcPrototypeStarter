#!/usr/bin/env python
"""CLI entry point for Task A1-B007X-006 Shadow Validation, Promotion, and Release.

Validates candidate safety on 8 canonical tasks and 6 compositions before promotion:
- canonical forgetting <= 2pp per op
- composition forgetting <= 2pp
- Stable Core strictly unchanged
- existing Bank strictly unchanged

On pass:
- installs candidate into persistent bank (bank size +1)
- candidate status -> STABLE and frozen
- releases temporary discovery capacity to 0
- saves persistent state checkpoint for Task A1-B007X-007

Usage:
    python scripts/shadow_promotion.py \
        --config configs/phase_a1_shadow_promotion.yaml \
        --run-dir runs/phase_a1_shadow_promotion
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from apc.evaluation.shadow_promotion import (
    DEFAULT_SEEDS,
    ShadowPromotionConfig,
    run_shadow_promotion_multi_seed,
    shadow_promotion_config_from_dict,
)
from apc.utils.config import load_config, save_config
from apc.utils.system_info import get_system_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to phase_a1_shadow_promotion.yaml",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Output directory for this benchmark run's artifacts",
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
    base_config = shadow_promotion_config_from_dict(raw_config)
    if args.seeds is not None:
        seeds = tuple(int(s.strip()) for s in args.seeds.split(","))
    else:
        seeds = tuple(raw_config.get("seeds", DEFAULT_SEEDS))

    resolved_config = base_config.to_dict()
    resolved_config["seeds"] = list(seeds)
    save_config(resolved_config, run_dir / "config.yaml")

    system_info = get_system_info(seed=seeds[0] if seeds else 0)
    (run_dir / "system.json").write_text(json.dumps(system_info, indent=2), encoding="utf-8")

    def config_factory(s: int) -> ShadowPromotionConfig:
        return dataclasses.replace(base_config, seeds=(s,))

    print(
        f"\n=======================================================\n"
        f"Starting Task A1-B007X-006 Shadow Validation & Promotion\n"
        f"Operation: {base_config.novel_operation}\n"
        f"Seeds: {list(seeds)}\n"
        f"Max Forgetting Threshold: {base_config.max_canonical_forgetting:.2%}\n"
        f"Output Dir: {run_dir}\n"
        f"=======================================================\n"
    )

    summary = run_shadow_promotion_multi_seed(
        seeds, config_factory, output_dir=run_dir
    )

    summary_dict = summary.to_dict()
    print(json.dumps(summary_dict, indent=2))

    verdict = "PASS" if summary.overall_passed else "FAIL"
    print("\n=======================================================")
    print(f"Task A1-B007X-006 Final Verdict: {verdict}")
    print(f"Meets Seed Policy: {summary.meets_seed_policy}")
    print(f"All Seeds Promoted: {summary.all_seeds_promoted}")
    print(f"Bank Size Transition: {summary.bank_size_transition}")
    print(f"Temporary Capacity Released: {summary.temp_released_completely}")
    print(f"Mean Candidate Novel EM: {summary.mean_candidate_novel_em:.4f}")
    print(f"Max Overall Forgetting: {summary.max_overall_forgetting:.4f}")
    print(f"Core Unchanged: {summary.all_core_unchanged}")
    print(f"Existing Bank Unchanged: {summary.all_existing_bank_unchanged}")
    print(f"Summary written to: {run_dir / 'summary.json'}")
    print("=======================================================\n")

    if not summary.overall_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
