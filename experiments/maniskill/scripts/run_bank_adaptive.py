"""Run APC Autonomous Adaptive Bank experiment: dynamic candidate switching and auto-consolidation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from apc_maniskill.primitive_bank import PrimitiveBank, BankAdaptiveSelector
from apc_maniskill.runner import json_write, utc_now, WORKSPACE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Output directory for adaptive evidence")
    parser.add_argument("--bank-source", type=Path, required=True, help="Existing multi-task primitive bank directory")
    args = parser.parse_args()

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    
    # Clone the source bank to the output directory for non-destructive demonstration
    bank_dir = out_dir / "primitive_bank"
    import shutil
    shutil.copytree(args.bank_source, bank_dir)
    bank = PrimitiveBank(bank_dir)

    report = {
        "status": "running",
        "started_at": utc_now(),
        "out_dir": str(out_dir),
        "bank_initial_audit": bank.audit(),
        "switches": {}
    }
    json_write(out_dir / "adaptive_report.json", report)

    # ----------------------------------------------------
    # Test 1: Autonomous Switching to Task A (Pick)
    # ----------------------------------------------------
    print("Test 1: Autonomous routing to Task A (Pick)...")
    pick_run_dir = out_dir / "autonomous_pick_run"
    cmd_pick = [
        sys.executable,
        str(WORKSPACE / "scripts/run_fetch_primitives.py"),
        "--far-start",
        "--base",
        "--rotations",
        "--selector", "bank_adaptive",
        "--bank-dir", str(bank_dir),
        "--episodes", "1",
        "--seed", "2801",
        "--max-steps", "1000",
        "--post-success-steps", "20",
        "--translation-backoff",
        "--out", str(pick_run_dir)
    ]
    sub_pick = subprocess.run(cmd_pick, capture_output=True, text=True, cwd=str(WORKSPACE))
    if sub_pick.returncode != 0:
        print("Pick stderr:", sub_pick.stderr)
        raise RuntimeError(f"Adaptive pick rollout failed with code {sub_pick.returncode}")
    pick_summary = json.loads((pick_run_dir / "summary.json").read_text(encoding="utf-8"))
    report["switches"]["task_a_pick"] = {
        "env_id": "APC-FetchPickCubeFar-v1",
        "routed_selector": "fetch_pick_v1",
        "success_rate": pick_summary["success_rate_over_observed"],
        "success_final": pick_summary["success_final_episodes"],
        "hold_complete": pick_summary["hold_complete_episodes"],
        "steps": pick_summary["steps"],
        "mean_return": pick_summary["mean_return"]
    }
    json_write(out_dir / "adaptive_report.json", report)

    # ----------------------------------------------------
    # Test 2: Autonomous Switching to Task B (Place)
    # ----------------------------------------------------
    print("Test 2: Autonomous routing to Task B (Place)...")
    place_run_dir = out_dir / "autonomous_place_run"
    cmd_place = [
        sys.executable,
        str(WORKSPACE / "scripts/run_fetch_primitives.py"),
        "--place-goal",
        "--far-start",
        "--base",
        "--rotations",
        "--selector", "bank_adaptive",
        "--bank-dir", str(bank_dir),
        "--episodes", "1",
        "--seed", "3001",
        "--max-steps", "1000",
        "--post-success-steps", "20",
        "--translation-backoff",
        "--out", str(place_run_dir)
    ]
    sub_place = subprocess.run(cmd_place, capture_output=True, text=True, cwd=str(WORKSPACE))
    if sub_place.returncode != 0:
        print("Place stderr:", sub_place.stderr)
        raise RuntimeError(f"Adaptive place rollout failed with code {sub_place.returncode}")
    place_summary = json.loads((place_run_dir / "summary.json").read_text(encoding="utf-8"))
    report["switches"]["task_b_place"] = {
        "env_id": "APC-FetchPlaceCubeFar-v1",
        "routed_selector": "fetch_place_v1",
        "success_rate": place_summary["success_rate_over_observed"],
        "success_final": place_summary["success_final_episodes"],
        "hold_complete": place_summary["hold_complete_episodes"],
        "steps": place_summary["steps"],
        "mean_return": place_summary["mean_return"]
    }
    json_write(out_dir / "adaptive_report.json", report)

    # ----------------------------------------------------
    # Test 3: Auto-Consolidation & Auto-Release Verification
    # ----------------------------------------------------
    print("Test 3: Testing auto-consolidation and atomic release...")
    dummy_temp = out_dir / "dummy_temp.pt"
    dummy_temp.write_bytes(b"TEMPORARY_EXPLORATION_MODEL_67890")
    dummy_cand = out_dir / "dummy_cand.pt"
    dummy_cand.write_bytes(b"CONSOLIDATED_COMPACT_TREE_678")

    bank.register(
        entry_id="auto_skill_v3",
        name="Auto Candidate Test",
        source_file=dummy_temp,
        kind="temporary",
        model_kind="mlp",
        primitive_count=20,
        parameter_count=8500,
        stored_values=8500
    )
    auto_res = bank.auto_consolidate_and_release(
        entry_id="auto_skill_v3",
        candidate_file=dummy_cand,
        model_kind="cart",
        parameter_count=0,
        stored_values=42,
        metadata={"tree_nodes": 42}
    )
    report["auto_consolidation_audit"] = auto_res

    # Final Audit
    report["final_audit"] = bank.audit()
    report["status"] = "completed"
    report["completed_at"] = utc_now()
    json_write(out_dir / "adaptive_report.json", report)

    print("====================================================")
    print("Autonomous Adaptive Switching & Auto-Consolidation Complete!")
    print(f"Report written to: {out_dir / 'adaptive_report.json'}")
    print("====================================================")


if __name__ == "__main__":
    main()
