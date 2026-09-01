#!/usr/bin/env python
"""CLI entry point for Task A1-R005D-002 (`docs/CODEX_TASKS_A1_R005_RETRY.md`,
`docs/EXPERIMENT_PLAN_A1_R005_RETRY.md` D2): audit every
`apc.primitives.conditioning.default_argument_encoder` instance for
structural representation collisions before any retraining.

No model, no training, no GPU -- this only constructs `ArgumentEncoder`
instances and runs them forward on constructed argument values (and, for the
`IntBucketArgumentEncoder` domain audit, Phase A.1's declared legal argument
domains). Writes a JSON report to `--output-dir`.

Usage:
    python scripts/r005d002_audit_argument_encoders.py \\
        --output-dir runs/a1_r005d_002_argument_encoder_audit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from apc.primitives.argument_encoder_audit import audit_phase_a1_default_argument_encoders


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", required=True, help="Directory to write this audit's report to"
    )
    parser.add_argument(
        "--select-seeds",
        default="0,1,2",
        help="Comma-separated seeds to construct independent SELECT encoders for "
        "(default: '0,1,2')",
    )
    args = parser.parse_args()

    select_seeds = tuple(int(token) for token in args.select_seeds.split(","))
    report = audit_phase_a1_default_argument_encoders(select_seeds=select_seeds)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    print(json.dumps(report.to_dict(), indent=2))
    print(f"A1-R005D-002 argument encoder audit written to: {report_path}")

    for operation, audit in report.int_bucket_audits.items():
        status = (
            "PASS (injective)" if audit.injective_on_legal_domain else "FAIL (collisions found)"
        )
        print(f"{operation}.{audit.argument_name}: {status}")

    if report.select_collision_present_in_every_seed:
        print(
            "SELECT.indices: FAIL (order-sensitive collision present in every "
            f"seed checked {report.select_seeds_checked}) -- "
            "STOP GATE: do not retrain SELECT with this encoder "
            "(docs/DECISIONS.md ADR-0031; A1-R005D-003 must supply an "
            "order-preserving replacement first)."
        )
    else:
        print("SELECT.indices: no order-sensitive collision observed.")


if __name__ == "__main__":
    main()
