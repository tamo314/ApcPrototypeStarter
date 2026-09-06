"""Run B-C005D2-001: development-versus-sealed discrepancy audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from apc.evaluation.hard_negative_second_diagnostic import (
    HardNegativeSecondDiagnosticConfig,
    run_hard_negative_second_diagnostic,
)
from apc.meta.phase_b_protocol import HardNegativeLevel


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase B B-C005D2-001 discrepancy audit")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/phase_b_b2_second_diagnostic.yaml")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/phase_b_b2_second_diagnostic")
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    data = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config = HardNegativeSecondDiagnosticConfig(
        original_sealed_seeds=tuple(data["original_sealed_seeds"]),
        development_seeds=tuple(data["development_seeds"]),
        regate_sealed_seeds=tuple(data["regate_sealed_seeds"]),
        bank_sizes=tuple(data["bank_sizes"]),
        levels=tuple(HardNegativeLevel.from_str(value) for value in data["levels"]),
        target_operations=tuple(data["target_operations"]),
        query_examples=data["query_examples"],
        router_train_examples=data["router_train_examples"],
        router_steps=data["router_steps"],
        router_lr=data["router_lr"],
        top_k=data["top_k"],
        ranking_margin=data["ranking_margin"],
        ranking_beta=data["ranking_beta"],
        arg_lambda=data["arg_lambda"],
        deterministic_algorithms=data["deterministic_algorithms"],
        device=args.device if args.device != "auto" else data["device"],
        bank_checkpoint_dir=Path(data["bank_checkpoint_dir"]),
        output_dir=args.output_dir,
    )
    report = run_hard_negative_second_diagnostic(config)
    print(json.dumps(report["representativeness_verdict"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
