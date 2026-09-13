#!/usr/bin/env python
"""CLI runner for NRQ-004 Frozen Bundle Compatibility Reconstruction & Depth-2 Control Reproduction.

Usage:
    python scripts/nrq004_bundle_reconstruction.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure src is on pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from apc.evaluation.nrq004_bundle_reconstruction import run_nrq004_full_audit


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    print("================================================================================")
    print("NRQ-004: Frozen Bundle Compatibility Reconstruction & Depth-2 Control Reproduction")
    print("================================================================================")

    report = run_nrq004_full_audit(
        seeds=(0, 1, 2, 3, 4), repo_root=repo_root, num_eval_examples=200
    )

    # Save review record in docs/research/
    record_path = repo_root / "docs" / "research" / "NRQ004_REVIEW_RECORD.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    print(f"Review record written to: {record_path}")

    # Save summary in reconstructed bundle directory
    bundle_summary_path = repo_root / "runs" / "nrq004_reconstructed_bundles" / "summary.json"
    bundle_summary_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_summary_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    print(f"Bundle summary written to: {bundle_summary_path}")

    print("\n--- PROVENANCE AUDIT MATRIX ---")
    for seed, prov in sorted(report.provenance_audit.items()):
        print(
            f"Seed {seed}: Core={prov.core_status} ({prov.core_sha256[:12]}...), "
            f"Bank={prov.bank_status} ({prov.bank_sha256[:12]}...) -> {prov.bundle_status}"
        )

    print("\n--- CONTROL REPRODUCTION RESULTS ---")
    for seed, ev in sorted(report.evaluation_results.items()):
        status_str = (
            "PASS (6/6)"
            if ev.seed_passed
            else f"FAIL ({ev.compositions_passed}/6, {ev.failure_attribution})"
        )
        print(
            f"Seed {seed}: Oracle EM={ev.mean_oracle_exact_match:.4f} | "
            f"Recovered EM={ev.mean_recovered_exact_match:.4f} | "
            f"Agreement={ev.mean_functional_agreement:.4f} | {status_str}"
        )

    print("\n--- DECISION & ATTRIBUTION ---")
    print(f"Decision: {report.resumption_decision}")
    print(f"All Seeds Passed: {report.all_seeds_passed}")
    print(f"Root Cause: {report.attribution_summary['root_cause']}")

    print("\n================================================================================")


if __name__ == "__main__":
    main()
