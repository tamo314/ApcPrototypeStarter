"""Run multi-task APC primitive bank experiment: cross-task control, acquire, consolidate, release, and multi-task retention."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from apc_maniskill.primitive_bank import PrimitiveBank
from apc_maniskill.runner import RunConfig, collect, json_write, summarize, utc_now, WORKSPACE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New run directory for multi-task evidence")
    parser.add_argument("--pick-candidate", type=Path, required=True, help="Consolidated Pick CART checkpoint")
    parser.add_argument("--place-temp", type=Path, required=True, help="Place temporary MLP checkpoint")
    parser.add_argument("--place-candidate", type=Path, required=True, help="Place consolidated CART checkpoint")
    args = parser.parse_args()

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    bank_dir = out_dir / "primitive_bank"
    bank = PrimitiveBank(bank_dir)

    report = {
        "status": "running",
        "started_at": utc_now(),
        "out_dir": str(out_dir),
        "stages": {}
    }
    json_write(out_dir / "multitask_report.json", report)

    # ----------------------------------------------------
    # Stage 0: Initialize Bank with Task A (Pick) Candidate
    # ----------------------------------------------------
    print("Stage 0: Seeding bank with Task A (Pick) candidate...")
    pick_meta = json.loads((args.pick_candidate.parent / "training.json").read_text(encoding="utf-8"))
    pick_entry = bank.register(
        entry_id="fetch_pick_v1",
        name="Task A: Fetch Pick Cube Far Selector",
        source_file=args.pick_candidate,
        kind="consolidated",
        model_kind="cart",
        primitive_count=20,
        parameter_count=pick_meta["model_parameters"],
        stored_values=pick_meta["model_stored_values"],
        feature_schema=pick_meta["feature_schema"],
        metadata={"tree_nodes": pick_meta.get("tree_nodes"), "task": "APC-FetchPickCubeFar-v1"}
    )
    report["stages"]["stage0_initial_bank"] = {
        "entry_id": pick_entry.id,
        "kind": pick_entry.kind,
        "parameter_count": pick_entry.parameter_count,
        "file_bytes": pick_entry.file_bytes,
        "bank_audit": bank.audit()
    }
    json_write(out_dir / "multitask_report.json", report)

    # ----------------------------------------------------
    # Negative Control: Run Task A Candidate on Task B (Place)
    # ----------------------------------------------------
    print("Negative Control: Evaluating Task A candidate on Task B (PlaceCubeFar)...")
    ctrl_run_dir = out_dir / "control_pick_on_place"
    cmd_ctrl = [
        sys.executable,
        str(WORKSPACE / "scripts/run_fetch_primitives.py"),
        "--place-goal",
        "--far-start",
        "--base",
        "--rotations",
        "--selector", "tree",
        "--checkpoint", str(bank.get_path("fetch_pick_v1")),
        "--episodes", "1",
        "--seed", "3001",
        "--max-steps", "1000",
        "--post-success-steps", "20",
        "--translation-backoff",
        "--allow-task-mismatch",
        "--out", str(ctrl_run_dir)
    ]

    sub_ctrl = subprocess.run(cmd_ctrl, capture_output=True, text=True, cwd=str(WORKSPACE))
    if sub_ctrl.returncode != 0:
        print("Control stderr:", sub_ctrl.stderr)
        raise RuntimeError(f"Control rollout failed with code {sub_ctrl.returncode}")
    ctrl_summary = json.loads((ctrl_run_dir / "summary.json").read_text(encoding="utf-8"))
    report["stages"]["negative_control"] = {
        "hypothesis": "Task A (Pick) candidate cannot solve Task B (Place) due to goal discrepancy",
        "env_id": "APC-FetchPlaceCubeFar-v1",
        "success_rate": ctrl_summary["success_rate_over_observed"],
        "success_final": ctrl_summary["success_final_episodes"],
        "mean_return": ctrl_summary["mean_return"],
        "steps": ctrl_summary["steps"],
        "confirmed": ctrl_summary["success_final_episodes"] == 0
    }
    json_write(out_dir / "multitask_report.json", report)

    # ----------------------------------------------------
    # Stage 1: Register Task B (Place) Temporary Skill
    # ----------------------------------------------------
    print("Stage 1: Registering Task B temporary primitive...")
    place_temp_meta = json.loads((args.place_temp.parent / "training.json").read_text(encoding="utf-8"))
    place_temp_entry = bank.register(
        entry_id="fetch_place_v1",
        name="Task B: Fetch Place Cube Far Selector (temporary)",
        source_file=args.place_temp,
        kind="temporary",
        model_kind="mlp",
        primitive_count=20,
        parameter_count=place_temp_meta["model_parameters"],
        stored_values=place_temp_meta["model_stored_values"],
        feature_schema=place_temp_meta["feature_schema"],
        metadata={"training_source": place_temp_meta["source"], "updates": place_temp_meta["updates"],
                  "task": "APC-FetchPlaceCubeFar-v1"}
    )
    report["stages"]["stage1_place_temporary_acquisition"] = {
        "entry_id": place_temp_entry.id,
        "kind": place_temp_entry.kind,
        "parameter_count": place_temp_entry.parameter_count,
        "file_bytes": place_temp_entry.file_bytes,
        "bank_audit": bank.audit()
    }
    json_write(out_dir / "multitask_report.json", report)

    # ----------------------------------------------------
    # Stage 2: Consolidate Task B into Compact Candidate
    # ----------------------------------------------------
    print("Stage 2: Consolidating Task B into compact candidate...")
    place_cand_meta = json.loads((args.place_candidate.parent / "training.json").read_text(encoding="utf-8"))
    place_cand_entry = bank.consolidate(
        entry_id="fetch_place_v1",
        candidate_file=args.place_candidate,
        model_kind="cart",
        parameter_count=place_cand_meta["model_parameters"],
        stored_values=place_cand_meta["model_stored_values"],
        metadata={"tree_nodes": place_cand_meta["tree_nodes"],
                  "validation_accuracy": place_cand_meta["validation_accuracy"]}
    )
    report["stages"]["stage2_place_consolidation"] = {
        "entry_id": place_cand_entry.id,
        "kind": place_cand_entry.kind,
        "parameter_count": place_cand_entry.parameter_count,
        "file_bytes": place_cand_entry.file_bytes,
        "consolidation_stats": place_cand_entry.metadata["consolidation_stats"],
        "bank_audit": bank.audit()
    }
    json_write(out_dir / "multitask_report.json", report)

    # ----------------------------------------------------
    # Stage 3: Release Task B Temporary Artifacts
    # ----------------------------------------------------
    print("Stage 3: Releasing Task B temporary artifacts...")
    release_info = bank.release_temporary("fetch_place_v1")
    report["stages"]["stage3_place_temporary_release"] = {
        "release_audit": release_info,
        "bank_audit": bank.audit()
    }
    json_write(out_dir / "multitask_report.json", report)

    # ----------------------------------------------------
    # Stage 4: Clean Execution of Task B using Bank Candidate
    # ----------------------------------------------------
    print("Stage 4: Executing Task B with consolidated candidate from bank...")
    place_run_dir = out_dir / "task_b_place_rollout"
    cmd_place = [
        sys.executable,
        str(WORKSPACE / "scripts/run_fetch_primitives.py"),
        "--place-goal",
        "--far-start",
        "--base",
        "--rotations",
        "--selector", "tree",
        "--checkpoint", str(bank.get_path("fetch_place_v1")),
        "--episodes", "1",
        "--seed", "3001",
        "--max-steps", "1000",
        "--post-success-steps", "20",
        "--translation-backoff",
        "--out", str(place_run_dir)
    ]
    sub_place = subprocess.run(cmd_place, capture_output=True, text=True, cwd=str(WORKSPACE))
    if sub_place.returncode != 0:
        print("Task B stderr:", sub_place.stderr)
        raise RuntimeError(f"Task B rollout failed with code {sub_place.returncode}")
    place_summary = json.loads((place_run_dir / "summary.json").read_text(encoding="utf-8"))
    report["stages"]["stage4_task_b_execution"] = {
        "env_id": "APC-FetchPlaceCubeFar-v1",
        "entry_id": "fetch_place_v1",
        "episodes": 1,
        "seed": 3001,
        "success_rate": place_summary["success_rate_over_observed"],
        "success_final": place_summary["success_final_episodes"],
        "hold_complete": place_summary["hold_complete_episodes"],
        "mean_return": place_summary["mean_return"],
        "steps": place_summary["steps"],
        "additional_training_updates": 0
    }
    json_write(out_dir / "multitask_report.json", report)

    # ----------------------------------------------------
    # Stage 5: Non-Destructive Retention of Task A
    # ----------------------------------------------------
    print("Stage 5: Verifying retention of Task A (Pick) from bank...")
    pick_run_dir = out_dir / "task_a_pick_retention_rollout"
    cmd_pick = [
        sys.executable,
        str(WORKSPACE / "scripts/run_fetch_primitives.py"),
        "--far-start",
        "--base",
        "--rotations",
        "--selector", "tree",
        "--checkpoint", str(bank.get_path("fetch_pick_v1")),
        "--episodes", "1",
        "--seed", "2801",
        "--max-steps", "1000",
        "--post-success-steps", "20",
        "--translation-backoff",
        "--out", str(pick_run_dir)
    ]
    sub_pick = subprocess.run(cmd_pick, capture_output=True, text=True, cwd=str(WORKSPACE))
    if sub_pick.returncode != 0:
        print("Task A stderr:", sub_pick.stderr)
        raise RuntimeError(f"Task A retention rollout failed with code {sub_pick.returncode}")
    pick_summary = json.loads((pick_run_dir / "summary.json").read_text(encoding="utf-8"))
    report["stages"]["stage5_task_a_retention"] = {
        "env_id": "APC-FetchPickCubeFar-v1",
        "entry_id": "fetch_pick_v1",
        "episodes": 1,
        "seed": 2801,
        "success_rate": pick_summary["success_rate_over_observed"],
        "success_final": pick_summary["success_final_episodes"],
        "hold_complete": pick_summary["hold_complete_episodes"],
        "mean_return": pick_summary["mean_return"],
        "steps": pick_summary["steps"],
        "retention_confirmed": pick_summary["success_final_episodes"] == 1
    }

    # Final Summary Audit
    report["final_audit"] = bank.audit()
    report["status"] = "completed"
    report["completed_at"] = utc_now()
    json_write(out_dir / "multitask_report.json", report)

    print("====================================================")
    print("Multi-Task APC Primitive Bank Verification Complete!")
    print(f"Report written to: {out_dir / 'multitask_report.json'}")
    print("Bank Entries:")
    for k, v in bank.entries.items():
        print(f"  [{k}] kind={v.kind}, params={v.parameter_count}, bytes={v.file_bytes}")
    print("====================================================")


if __name__ == "__main__":
    main()
