# ruff: noqa: E501
"""Run Phase B B-C004 hard-negative construction and diagnostic matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from apc.evaluation.hard_negative_routing_benchmark import (
    DEFAULT_BANK_SIZES,
    HardNegativeBenchmarkConfig,
    run_hard_negative_diagnostic,
)
from apc.utils.system_info import get_system_info


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase B B-C004 hard-negative diagnostic")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--bank-sizes", type=int, nargs="+", default=list(DEFAULT_BANK_SIZES))
    parser.add_argument("--num-eval-examples", type=int, default=32)
    parser.add_argument("--router-steps", type=int, default=250)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/phase_b_hard_negative_diagnostic"))
    args = parser.parse_args()
    config = HardNegativeBenchmarkConfig(
        seeds=tuple(args.seeds), bank_sizes=tuple(args.bank_sizes),
        num_eval_examples=args.num_eval_examples, router_steps=args.router_steps,
        device=args.device, output_dir=args.output_dir,
    )
    report = run_hard_negative_diagnostic(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.yaml").write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
    )
    (args.output_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in report["matrix"]), encoding="utf-8"
    )
    (args.output_dir / "system.json").write_text(
        json.dumps(get_system_info(), indent=2), encoding="utf-8"
    )
    protocol = {
        "phase_task_id": "B-C004",
        "family_split": "DEV_FAMILIES",
        "explicit_task_spec_visible": True,
        "operation_id_visible": True,
        "task_inference_modality": "explicit_task_spec",
        "support_examples": 0,
        "inference_examples": 0,
        "verification_examples": 0,
        "query_examples": args.num_eval_examples,
        "candidate_search_budget": {"top_k": config.top_k},
        "bank_sizes": list(config.bank_sizes),
        "hard_negative_levels": [level.value for level in config.levels],
        "holdout_leak_audit": report["summary"]["leak_audit_passed"],
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["summary"], indent=2))
    return 0 if report["summary"]["infrastructure_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
