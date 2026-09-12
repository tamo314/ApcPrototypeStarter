"""Run the explicitly preregistered B-C005R3-002R feasibility experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from apc.evaluation.single_family_g1_strict_holdout import (
    SingleFamilyG1Config,
    run_single_family_g1_feasibility,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="B-C005R3-002R fixed-family G1 feasibility")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase_b_b2_single_family_g1_strict_holdout.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config = SingleFamilyG1Config(
        family_id=raw.get("family_id", "sealed_local_neighborhood"),
        oracle_seed=raw.get("oracle_seed", 20260913),
        sequence_length=raw.get("sequence_length", 8),
        verification_examples=raw.get("verification_examples", 32),
        novelty_threshold=raw.get("novelty_threshold", 0.90),
        max_composition_depth=raw.get("max_composition_depth", 2),
        gradient_probe_seed=raw.get("gradient_probe_seed", 10),
        gradient_probe_examples_per_operation=raw.get("gradient_probe_examples_per_operation", 4),
        gradient_probe_lr=raw.get("gradient_probe_lr", 1e-3),
        device=raw.get("device", "cpu"),
        output_dir=args.output_dir or Path(raw["output_dir"]),
    )
    report = run_single_family_g1_feasibility(config)
    print(json.dumps(report["protocol.json"], indent=2))
    return 0 if report["protocol.json"]["result"] == "G1_STRICT_HOLDOUT_FEASIBILITY_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
