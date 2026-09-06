"""Run B-C005D2-002: L3 representation-stage localization audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from apc.evaluation.representation_stage_probe import (
    RepresentationStageProbeConfig,
    run_representation_stage_probe,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase B B-C005D2-002 representation-stage probe")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/phase_b_b2_representation_probe.yaml")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/phase_b_b2_second_diagnostic")
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    data = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config = RepresentationStageProbeConfig(
        original_sealed_seeds=tuple(data["original_sealed_seeds"]),
        development_seeds=tuple(data["development_seeds"]),
        regate_sealed_seeds=tuple(data["regate_sealed_seeds"]),
        bank_size=data["bank_size"],
        target_operations=tuple(data["target_operations"]),
        query_examples=data["query_examples"],
        probe_train_examples=data["probe_train_examples"],
        probe_eval_examples=data["probe_eval_examples"],
        probe_steps=data["probe_steps"],
        probe_lr=data["probe_lr"],
        router_train_examples=data["router_train_examples"],
        router_steps=data["router_steps"],
        router_lr=data["router_lr"],
        top_k=data["top_k"],
        ranking_margin=data["ranking_margin"],
        ranking_beta=data["ranking_beta"],
        arg_lambda=data["arg_lambda"],
        separability_threshold=data["separability_threshold"],
        shuffled_chance_tolerance=data["shuffled_chance_tolerance"],
        deterministic_algorithms=data["deterministic_algorithms"],
        device=args.device if args.device != "auto" else data["device"],
        bank_checkpoint_dir=Path(data["bank_checkpoint_dir"]),
        output_dir=args.output_dir,
    )
    report = run_representation_stage_probe(config)
    print(json.dumps(report["representation_stage_localization"]["verdict"], indent=2))
    print(json.dumps(report["representation_stage_localization"]["per_relation_verdict"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
