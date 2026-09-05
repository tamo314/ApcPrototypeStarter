"""Run Phase B B-C005D Failure Isolation Diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from apc.evaluation.hard_negative_failure_isolation import (
    HardNegativeFailureIsolationConfig,
    run_hard_negative_failure_isolation,
)
from apc.meta.phase_b_protocol import HardNegativeLevel


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase B B-C005D Failure Isolation Diagnostics")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/phase_b_b2_failure_isolation.yaml")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/phase_b_b2_failure_isolation")
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    if args.config.is_file():
        data = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        levels = tuple(HardNegativeLevel.from_str(lvl) for lvl in data.get("levels", []))
        ckpt_dir = Path(data.get("bank_checkpoint_dir", "runs/phase_a2_bank_scaling_benchmark"))
        config = HardNegativeFailureIsolationConfig(
            seeds=tuple(data.get("seeds", [0])),
            bank_sizes=tuple(data.get("bank_sizes", [16])),
            levels=levels,
            target_operations=tuple(data.get("target_operations", ["SHIFT"])),
            support_examples=data.get("support_examples", 32),
            query_examples=data.get("query_examples", 64),
            router_train_examples=data.get("router_train_examples", 32),
            router_steps=data.get("router_steps", 250),
            top_k=data.get("top_k", 5),
            adequacy_exact_match_threshold=data.get("adequacy_exact_match_threshold", 0.95),
            support_variance_sizes=tuple(data.get("support_variance_sizes", [16, 32, 64, 128])),
            binomial_p_values=tuple(data.get("binomial_p_values", [0.97, 0.98, 0.99, 0.995])),
            deterministic_algorithms=data.get("deterministic_algorithms", True),
            device=args.device if args.device != "auto" else data.get("device", "auto"),
            bank_checkpoint_dir=ckpt_dir,
            output_dir=args.output_dir,
        )
    else:
        config = HardNegativeFailureIsolationConfig(output_dir=args.output_dir, device=args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.yaml").write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
    )

    summary = run_hard_negative_failure_isolation(config)
    print(json.dumps(summary["core_question_answers"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
