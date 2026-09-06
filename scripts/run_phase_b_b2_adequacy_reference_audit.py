"""Run B-C005D2-005: SHIFT seed-24 adequacy / false-plastic audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from apc.evaluation.adequacy_reference_audit import (
    AdequacyReferenceAuditConfig,
    run_adequacy_reference_audit,
)
from apc.meta.phase_b_protocol import HardNegativeLevel


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase B B-C005D2-005 SHIFT seed-24 adequacy reference audit"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/phase_b_b2_adequacy_reference_audit.yaml")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/phase_b_b2_second_diagnostic")
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    data = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config = AdequacyReferenceAuditConfig(
        sealed_seed=data["sealed_seed"],
        development_seeds=tuple(data["development_seeds"]),
        regate_sealed_seeds=tuple(data["regate_sealed_seeds"]),
        bank_sizes=tuple(data["bank_sizes"]),
        levels=tuple(HardNegativeLevel.from_str(value) for value in data["levels"]),
        target_operation=data["target_operation"],
        support_examples=data["support_examples"],
        query_examples=data["query_examples"],
        router_train_examples=data["router_train_examples"],
        router_steps=data["router_steps"],
        router_lr=data["router_lr"],
        top_k=data["top_k"],
        ranking_margin=data["ranking_margin"],
        ranking_beta=data["ranking_beta"],
        arg_lambda=data["arg_lambda"],
        adequacy_exact_match_threshold=data["adequacy_exact_match_threshold"],
        confidence_level=data["confidence_level"],
        initial_support=data["initial_support"],
        support_increment=data["support_increment"],
        max_support=data["max_support"],
        reference_examples=data["reference_examples"],
        deterministic_algorithms=data["deterministic_algorithms"],
        device=args.device if args.device != "auto" else data["device"],
        bank_checkpoint_dir=Path(data["bank_checkpoint_dir"]),
        output_dir=args.output_dir,
    )
    report = run_adequacy_reference_audit(config)
    print(json.dumps(report["spec_verification"], indent=2))
    print(json.dumps(report["overall_classification"], indent=2))
    for audit in report["bank_size_audits"]:
        print(
            f"bank_size={audit['bank_size']} "
            f"runtime_decision={audit['verifier_trace']['final_decision']} "
            f"reference_em={audit['reference_adequacy']['reference_em']:.4f} "
            f"reference_adequate={audit['reference_adequacy']['reference_adequate']} "
            f"classification={audit['plastic_classification']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
