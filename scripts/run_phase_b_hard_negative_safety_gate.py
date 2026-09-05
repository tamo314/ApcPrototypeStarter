"""Run Phase B B-C005 hard-negative routing and functional-safety STOP GATE."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from apc.evaluation.hard_negative_routing_benchmark import (
    DEFAULT_BANK_SIZES,
    HardNegativeSafetyConfig,
    run_hard_negative_safety_gate,
)
from apc.meta.phase_b_protocol import HardNegativeLevel
from apc.utils.system_info import get_system_info


def main() -> int:
    """Run the frozen-router B-C005 matrix and save the required artifacts."""
    parser = argparse.ArgumentParser(description="Phase B B-C005 hard-negative safety gate")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--bank-sizes", type=int, nargs="+", default=list(DEFAULT_BANK_SIZES))
    parser.add_argument("--support-examples", type=int, default=32)
    parser.add_argument("--query-examples", type=int, default=64)
    parser.add_argument("--router-steps", type=int, default=250)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--adequacy-exact-match-threshold", type=float, default=0.95)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/phase_b_hard_negative_safety_gate")
    )
    args = parser.parse_args()
    config = HardNegativeSafetyConfig(
        seeds=tuple(args.seeds), bank_sizes=tuple(args.bank_sizes),
        support_examples=args.support_examples, query_examples=args.query_examples,
        router_steps=args.router_steps, top_k=args.top_k,
        adequacy_exact_match_threshold=args.adequacy_exact_match_threshold,
        device=args.device, output_dir=args.output_dir,
    )
    report = run_hard_negative_safety_gate(config)
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
        "phase_task_id": "B-C005",
        "family_split": "DEV_FAMILIES",
        "explicit_task_spec_visible": True,
        "operation_id_visible": True,
        "task_inference_modality": "explicit_task_spec",
        "support_examples": args.support_examples,
        "inference_examples": 0,
        "verification_examples": args.support_examples,
        "query_examples": args.query_examples,
        "candidate_search_budget": {"top_k": args.top_k},
        "adequacy_exact_match_threshold": args.adequacy_exact_match_threshold,
        "bank_sizes": list(args.bank_sizes),
        "hard_negative_levels": [level.value for level in HardNegativeLevel],
        "holdout_leak_audit": report["summary"]["leak_audit_passed"],
        "query_targets_used_for_verification": False,
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    lines = [
        "# B-C005 hard-negative routing and functional safety",
        "",
        f"STOP GATE: {'PASS' if report['summary']['stop_gate_passed'] else 'FAIL'}",
        "",
        "| N / Level | Top-1 | Top-k | Margin | Closed-loop EM |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in report["summary"]["scaling_curves"].items():
        lines.append(
            f"| {name} | {row['top1']:.3f} | {row['topk']:.3f} | "
            f"{row['margin']:.3f} | {row['closed_loop_exact_match']:.3f} |"
        )
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return 0 if report["summary"]["stop_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
