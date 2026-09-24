"""Complete APC Lifecycle: detect deficit, acquire temporary patch, composite rollout, distill to candidate CART patch, release, and verify."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from apc_maniskill.primitive_bank import PrimitiveBank
from apc_maniskill.runner import json_write, utc_now


def run_command(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    print(f"RUNNING: {' '.join(cmd)}")
    t0 = time.time()
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    dt = time.time() - t0
    if res.returncode != 0:
        print(f"STDERR: {res.stderr}")
        raise RuntimeError(f"Command failed ({res.returncode}): {' '.join(cmd)}\n{res.stderr}")
    print(f"COMPLETED in {dt:.1f}s")
    return res


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True,
                        help="Path to frozen Base CART selector.pt")
    parser.add_argument("--target-seed", type=int, default=3001,
                        help="Target TruePlace seed to detect deficit and acquire capability")
    parser.add_argument("--transfer-seed", type=int, default=3002,
                        help="Unseen TruePlace seed to evaluate transfer")
    parser.add_argument("--retention-seed", type=int, default=3201,
                        help="Prior task Pick seed to verify retention")
    parser.add_argument("--max-steps", type=int, default=1200)
    args = parser.parse_args()

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=False)

    py = sys.executable
    repo_dir = Path(__file__).resolve().parents[1]
    bank_dir = out_dir / "primitive_bank"
    bank = PrimitiveBank(bank_dir)

    metrics = {
        "started_at": utc_now(),
        "stages": {},
        "costs": {
            "environment_steps": 0,
            "teacher_calls": 0,
            "gradient_updates": 0,
            "tree_fits": 0,
            "wall_seconds": 0.0,
        },
    }
    t_start = time.time()

    # -------------------------------------------------------------
    # Stage 0: Evaluate Base Model on Target Seed (Deficit Confirmation)
    # -------------------------------------------------------------
    print("\n=== Stage 0: Evaluating Frozen Base Model on Target TruePlace Seed ===")
    stage0_dir = out_dir / "stage0_base_eval"
    cmd0 = [
        py, str(repo_dir / "scripts" / "run_fetch_primitives.py"),
        "--out", str(stage0_dir),
        "--episodes", "1",
        "--seed", str(args.target_seed),
        "--max-steps", str(args.max_steps),
        "--post-success-steps", "20",
        "--selector", "tree",
        "--checkpoint", str(args.base_checkpoint),
        "--true-place-goal",
        "--rotations", "--base", "--far-start",
        "--allow-task-mismatch",
    ]
    run_command(cmd0, cwd=repo_dir)
    ep0 = json.loads((stage0_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()[0])
    metrics["costs"]["environment_steps"] += ep0["steps"]
    
    stage0_consec_20 = ep0.get("consecutive_success_final_20", False)
    stage0_max_consec = ep0.get("max_consecutive_success", 0)
    stage0_success = ep0["success_final"]
    
    metrics["stages"]["stage0_base_eval"] = {
        "task": "APC-FetchTruePlaceFar-v1",
        "seed": args.target_seed,
        "steps": ep0["steps"],
        "success_final": stage0_success,
        "max_consecutive_success": stage0_max_consec,
        "consecutive_success_final_20": stage0_consec_20,
    }
    print(f"Base result: success_final={stage0_success}, max_consec={stage0_max_consec}, consec_20={stage0_consec_20}, steps={ep0['steps']}")
    if stage0_consec_20:
        print("-> Base model already succeeds on this task. Adaptation not needed.")
        metrics["adaptation_needed"] = False
        metrics["costs"]["wall_seconds"] = time.time() - t_start
        metrics["completed_at"] = utc_now()
        json_write(out_dir / "cycle_report.json", metrics)
        md = f"""# APC Autonomous Lifecycle Report\n\n- Base model succeeded without adaptation. Steps: {ep0['steps']}\n"""
        (out_dir / "cycle_summary.md").write_text(md, encoding="utf-8")
        return

    print("-> Deficit confirmed: Base model lacks true place (release/settle) capability.")
    metrics["adaptation_needed"] = True

    # -------------------------------------------------------------
    # Stage 1: Collect Teacher Guidance & Train Local Temporary MLP
    # -------------------------------------------------------------
    print("\n=== Stage 1: Collecting Teacher Guidance on Target TruePlace Seed ===")
    stage1_teacher_dir = out_dir / "stage1_teacher_rollout"
    cmd1_t = [
        py, str(repo_dir / "scripts" / "run_fetch_primitives.py"),
        "--out", str(stage1_teacher_dir),
        "--episodes", "1",
        "--seed", str(args.target_seed),
        "--max-steps", str(args.max_steps),
        "--post-success-steps", "20",
        "--selector", "base_ready_pick",
        "--true-place-goal",
        "--rotations", "--base", "--far-start",
    ]
    run_command(cmd1_t, cwd=repo_dir)
    ep1_t = json.loads((stage1_teacher_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()[0])
    metrics["costs"]["environment_steps"] += ep1_t["steps"]
    metrics["costs"]["teacher_calls"] += ep1_t["steps"]
    metrics["stages"]["stage1_teacher_rollout"] = {
        "task": "APC-FetchTruePlaceFar-v1",
        "seed": args.target_seed,
        "steps": ep1_t["steps"],
        "success_final": ep1_t["success_final"],
        "max_consecutive_success": ep1_t.get("max_consecutive_success", 0),
        "consecutive_success_final_20": ep1_t.get("consecutive_success_final_20", False),
    }

    print("\n=== Stage 1b: Training Local Temporary MLP Selector ===")
    stage1_train_dir = out_dir / "stage1_temp_train"
    cmd1_tr = [
        py, "-m", "apc_maniskill.primitive_learning",
        "--source", str(stage1_teacher_dir),
        "--out", str(stage1_train_dir),
        "--model-kind", "mlp",
        "--updates", "2000",
        "--feature-schema", "fetch_primitive_geometry_features_v5",
        "--train-all",
        "--allow-parameterized-goal-tasks",
    ]
    run_command(cmd1_tr, cwd=repo_dir)
    train1_meta = json.loads((stage1_train_dir / "training.json").read_text(encoding="utf-8"))
    metrics["costs"]["gradient_updates"] += 2000

    temp_entry = bank.register(
        entry_id="fetch_true_place_patch_v1",
        name="Temporary local MLP patch selector for table release and resting",
        source_file=stage1_train_dir / "selector.pt",
        kind="temporary",
        model_kind="mlp",
        primitive_count=20,
        parameter_count=train1_meta["model_parameters"],
        stored_values=train1_meta["model_stored_values"],
        feature_schema="fetch_primitive_geometry_features_v5",
        metadata={
            "target_seed": args.target_seed,
            "training_samples": train1_meta["training_samples"],
            "gradient_updates": 2000,
        },
    )
    metrics["stages"]["stage1_temporary_acquired"] = {
        "entry_id": temp_entry.id,
        "parameters": temp_entry.parameter_count,
        "bytes": temp_entry.file_bytes,
        "sha256": temp_entry.sha256,
    }

    # -------------------------------------------------------------
    # Stage 2: Base + Temporary Physical Rollout (Composite Patch Policy)
    # -------------------------------------------------------------
    print("\n=== Stage 2: Physical Rollout of Base + Temporary Composite Policy ===")
    stage2_comp_dir = out_dir / "stage2_composite_rollout"
    temp_checkpoint = bank.get_path(temp_entry.id)
    cmd2 = [
        py, str(repo_dir / "scripts" / "run_fetch_primitives.py"),
        "--out", str(stage2_comp_dir),
        "--episodes", "1",
        "--seed", str(args.target_seed),
        "--max-steps", str(args.max_steps),
        "--post-success-steps", "20",
        "--selector", "composite_patch",
        "--checkpoint", str(args.base_checkpoint),
        "--patch-checkpoint", str(temp_checkpoint),
        "--patch-mode", "place",
        "--true-place-goal",
        "--rotations", "--base", "--far-start",
        "--allow-task-mismatch",
    ]
    run_command(cmd2, cwd=repo_dir)
    ep2 = json.loads((stage2_comp_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()[0])
    metrics["costs"]["environment_steps"] += ep2["steps"]
    metrics["stages"]["stage2_composite_rollout"] = {
        "task": "APC-FetchTruePlaceFar-v1",
        "seed": args.target_seed,
        "steps": ep2["steps"],
        "success_final": ep2["success_final"],
        "max_consecutive_success": ep2.get("max_consecutive_success", 0),
        "consecutive_success_final_20": ep2.get("consecutive_success_final_20", False),
    }
    print(f"Composite Policy result: success_final={ep2['success_final']}, consec_20={ep2.get('consecutive_success_final_20')}, steps={ep2['steps']}")

    # -------------------------------------------------------------
    # Stage 3: Distill from Composite Rollout into Candidate CART Patch
    # -------------------------------------------------------------
    print("\n=== Stage 3: Distilling Composite Policy Rollout into Candidate CART ===")
    stage3_cand_dir = out_dir / "stage3_cand_train"
    cmd3_c = [
        py, "-m", "apc_maniskill.primitive_learning",
        "--source", str(stage2_comp_dir),
        "--out", str(stage3_cand_dir),
        "--model-kind", "cart",
        "--feature-schema", "fetch_primitive_geometry_features_v5",
        "--partition-grasp",
        "--train-all",
        "--allow-parameterized-goal-tasks",
    ]
    run_command(cmd3_c, cwd=repo_dir)
    train3_meta = json.loads((stage3_cand_dir / "training.json").read_text(encoding="utf-8"))
    metrics["costs"]["tree_fits"] += 1

    cand_entry = bank.consolidate(
        temp_entry.id,
        stage3_cand_dir / "selector.pt",
        model_kind="cart",
        parameter_count=0,
        stored_values=train3_meta["model_stored_values"],
        metadata={
            "tree_nodes": train3_meta["tree_nodes"],
            "tree_fits": 1,
            "target_seed": args.target_seed,
            "distillation_source": "stage2_composite_rollout",
            "temporary_mlp_sha256": metrics["stages"]["stage1_temporary_acquired"]["sha256"],
        },
    )

    # -------------------------------------------------------------
    # Stage 4: Physically Release Temporary Artifacts from Bank
    # -------------------------------------------------------------
    print("\n=== Stage 4: Releasing Temporary Artifacts from Bank ===")
    release_audit = bank.release_temporary(temp_entry.id)
    bank_audit = bank.audit()

    metrics["stages"]["stage4_consolidation_and_release"] = {
        "entry_id": cand_entry.id,
        "parameters": 0,
        "stored_values": cand_entry.stored_values,
        "bytes": cand_entry.file_bytes,
        "reclaimed_parameters": release_audit["reclaimed_parameters"],
        "reclaimed_bytes": release_audit["reclaimed_bytes"],
        "temporary_released": release_audit["status"],
        "bank_audit": bank_audit,
    }

    # -------------------------------------------------------------
    # Stage 5: Evaluate with Candidate CART in Fixed Modular Composite Architecture
    # -------------------------------------------------------------
    cand_checkpoint = bank.get_path(cand_entry.id)

    # 5a: Target TruePlace seed (seed 3001) - Base + Candidate CART Patch
    print("\n=== Stage 5a: Target Seed Evaluation (Base CART + Candidate CART Patch) ===")
    stage5a_dir = out_dir / "stage5a_target_eval"
    cmd5a = [
        py, str(repo_dir / "scripts" / "run_fetch_primitives.py"),
        "--out", str(stage5a_dir),
        "--episodes", "1",
        "--seed", str(args.target_seed),
        "--max-steps", str(args.max_steps),
        "--post-success-steps", "20",
        "--selector", "composite_patch",
        "--checkpoint", str(args.base_checkpoint),
        "--patch-checkpoint", str(cand_checkpoint),
        "--patch-mode", "place",
        "--true-place-goal",
        "--rotations", "--base", "--far-start",
        "--allow-task-mismatch",
    ]
    run_command(cmd5a, cwd=repo_dir)
    ep5a = json.loads((stage5a_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()[0])
    metrics["costs"]["environment_steps"] += ep5a["steps"]
    metrics["stages"]["stage5a_target_eval"] = {
        "task": "APC-FetchTruePlaceFar-v1",
        "seed": args.target_seed,
        "steps": ep5a["steps"],
        "success_final": ep5a["success_final"],
        "max_consecutive_success": ep5a.get("max_consecutive_success", 0),
        "consecutive_success_final_20": ep5a.get("consecutive_success_final_20", False),
    }

    # 5b: Retention on prior Pick seed (seed 3201) - Base + Candidate CART Patch
    print("\n=== Stage 5b: Retention Evaluation on Prior Pick Seed (Base CART + Candidate CART Patch) ===")
    stage5b_dir = out_dir / "stage5b_retention_eval"
    cmd5b = [
        py, str(repo_dir / "scripts" / "run_fetch_primitives.py"),
        "--out", str(stage5b_dir),
        "--episodes", "1",
        "--seed", str(args.retention_seed),
        "--max-steps", str(args.max_steps),
        "--post-success-steps", "20",
        "--selector", "composite_patch",
        "--checkpoint", str(args.base_checkpoint),
        "--patch-checkpoint", str(cand_checkpoint),
        "--patch-mode", "place",
        "--rotations", "--base", "--far-start",
        "--allow-task-mismatch",
    ]
    run_command(cmd5b, cwd=repo_dir)
    ep5b = json.loads((stage5b_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()[0])
    metrics["costs"]["environment_steps"] += ep5b["steps"]
    metrics["stages"]["stage5b_retention_eval"] = {
        "task": "APC-FetchPickCubeFar-v1",
        "seed": args.retention_seed,
        "steps": ep5b["steps"],
        "success_final": ep5b["success_final"],
        "max_consecutive_success": ep5b.get("max_consecutive_success", 0),
        "consecutive_success_final_20": ep5b.get("consecutive_success_final_20", False),
    }

    # 5c: Transfer on unseen TruePlace seed (seed 3002) - Base + Candidate CART Patch
    print("\n=== Stage 5c: Transfer Evaluation on Unseen TruePlace Seed (Base CART + Candidate CART Patch) ===")
    stage5c_dir = out_dir / "stage5c_transfer_eval"
    cmd5c = [
        py, str(repo_dir / "scripts" / "run_fetch_primitives.py"),
        "--out", str(stage5c_dir),
        "--episodes", "1",
        "--seed", str(args.transfer_seed),
        "--max-steps", str(args.max_steps),
        "--post-success-steps", "20",
        "--selector", "composite_patch",
        "--checkpoint", str(args.base_checkpoint),
        "--patch-checkpoint", str(cand_checkpoint),
        "--patch-mode", "place",
        "--true-place-goal",
        "--rotations", "--base", "--far-start",
        "--allow-task-mismatch",
    ]
    run_command(cmd5c, cwd=repo_dir)
    ep5c = json.loads((stage5c_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()[0])
    metrics["costs"]["environment_steps"] += ep5c["steps"]
    metrics["stages"]["stage5c_transfer_eval"] = {
        "task": "APC-FetchTruePlaceFar-v1",
        "seed": args.transfer_seed,
        "steps": ep5c["steps"],
        "success_final": ep5c["success_final"],
        "max_consecutive_success": ep5c.get("max_consecutive_success", 0),
        "consecutive_success_final_20": ep5c.get("consecutive_success_final_20", False),
    }

    # 5d: Baseline Control: Single Candidate CART (No Composite Architecture)
    print("\n=== Stage 5d: Control Baseline: Single Candidate CART Alone ===")
    stage5d_dir = out_dir / "stage5d_control_single_cart_eval"
    cmd5d = [
        py, str(repo_dir / "scripts" / "run_fetch_primitives.py"),
        "--out", str(stage5d_dir),
        "--episodes", "1",
        "--seed", str(args.target_seed),
        "--max-steps", str(args.max_steps),
        "--post-success-steps", "20",
        "--selector", "tree",
        "--checkpoint", str(cand_checkpoint),
        "--true-place-goal",
        "--rotations", "--base", "--far-start",
        "--allow-task-mismatch",
    ]
    run_command(cmd5d, cwd=repo_dir)
    ep5d = json.loads((stage5d_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()[0])
    metrics["costs"]["environment_steps"] += ep5d["steps"]
    metrics["stages"]["stage5d_control_single_cart_eval"] = {
        "task": "APC-FetchTruePlaceFar-v1",
        "seed": args.target_seed,
        "steps": ep5d["steps"],
        "success_final": ep5d["success_final"],
        "max_consecutive_success": ep5d.get("max_consecutive_success", 0),
        "consecutive_success_final_20": ep5d.get("consecutive_success_final_20", False),
    }

    # Wrap up metrics
    metrics["costs"]["wall_seconds"] = time.time() - t_start
    metrics["completed_at"] = utc_now()

    # Save reports
    json_write(out_dir / "cycle_report.json", metrics)

    temp_params = metrics["stages"]["stage1_temporary_acquired"]["parameters"]
    temp_bytes = metrics["stages"]["stage1_temporary_acquired"]["bytes"]

    md = f"""# APC Autonomous Lifecycle Report

- **Run Directory**: `{out_dir}`
- **Target Task**: `APC-FetchTruePlaceFar-v1` (Seed {args.target_seed})
- **Retention Task**: `APC-FetchPickCubeFar-v1` (Seed {args.retention_seed})
- **Transfer Task**: `APC-FetchTruePlaceFar-v1` (Seed {args.transfer_seed})
- **Total Wall Time**: {metrics['costs']['wall_seconds']:.1f} s

## Resource Reclamation & Consolidation
- **Temporary MLP**: {temp_params} params, {temp_bytes} bytes
- **Candidate CART**: 0 gradient params ({cand_entry.stored_values} stored values), {cand_entry.file_bytes} bytes
- **Reclaimed**: {release_audit['reclaimed_parameters']} params (100%), {release_audit['reclaimed_bytes']} bytes released from bank copy
- **Integrity**: Physical files verified by SHA256 in bank ledger

## Cost Accounting
- **Total Environment Steps**: {metrics['costs']['environment_steps']}
- **Teacher Invocations**: {metrics['costs']['teacher_calls']}
- **Gradient Updates**: {metrics['costs']['gradient_updates']}
- **Tree Fits**: {metrics['costs']['tree_fits']}

## Performance Across Stages (Objective Evidence)
| Stage | Execution Subject | Task | Seed | Success Final | 20-Step Continuous Rest | Max Consec | Steps | Notes |
|---|---|---|---|---|---|---|---|---|
| Stage 0 | Frozen Base CART | TruePlaceFar | {args.target_seed} | {metrics['stages']['stage0_base_eval']['success_final']} | {metrics['stages']['stage0_base_eval']['consecutive_success_final_20']} | {metrics['stages']['stage0_base_eval']['max_consecutive_success']} | {metrics['stages']['stage0_base_eval']['steps']} | Deficit confirmed (cannot release/settle) |
| Stage 1 (Teacher) | Hand-crafted Teacher | TruePlaceFar | {args.target_seed} | {metrics['stages']['stage1_teacher_rollout']['success_final']} | {metrics['stages']['stage1_teacher_rollout']['consecutive_success_final_20']} | {metrics['stages']['stage1_teacher_rollout']['max_consecutive_success']} | {metrics['stages']['stage1_teacher_rollout']['steps']} | Teacher demonstration collected |
| Stage 2 (Composite Temp) | Base CART + Temp MLP | TruePlaceFar | {args.target_seed} | {metrics['stages']['stage2_composite_rollout']['success_final']} | {metrics['stages']['stage2_composite_rollout']['consecutive_success_final_20']} | {metrics['stages']['stage2_composite_rollout']['max_consecutive_success']} | {metrics['stages']['stage2_composite_rollout']['steps']} | Hand-designed patch condition rollout |
| Stage 5a (Composite Cand) | Base CART + Candidate CART | TruePlaceFar | {args.target_seed} | {metrics['stages']['stage5a_target_eval']['success_final']} | {metrics['stages']['stage5a_target_eval']['consecutive_success_final_20']} | {metrics['stages']['stage5a_target_eval']['max_consecutive_success']} | {metrics['stages']['stage5a_target_eval']['steps']} | Same composite architecture, Temp replaced by CART patch |
| Stage 5b (Retention) | Base CART + Candidate CART | PickCubeFar | {args.retention_seed} | {metrics['stages']['stage5b_retention_eval']['success_final']} | {metrics['stages']['stage5b_retention_eval']['consecutive_success_final_20']} | {metrics['stages']['stage5b_retention_eval']['max_consecutive_success']} | {metrics['stages']['stage5b_retention_eval']['steps']} | Prior capability retention check in composite architecture |
| Stage 5c (Transfer) | Base CART + Candidate CART | TruePlaceFar | {args.transfer_seed} | {metrics['stages']['stage5c_transfer_eval']['success_final']} | {metrics['stages']['stage5c_transfer_eval']['consecutive_success_final_20']} | {metrics['stages']['stage5c_transfer_eval']['max_consecutive_success']} | {metrics['stages']['stage5c_transfer_eval']['steps']} | Unseen placement transfer check in composite architecture |
| Stage 5d (Control Baseline) | Candidate CART Alone | TruePlaceFar | {args.target_seed} | {metrics['stages']['stage5d_control_single_cart_eval']['success_final']} | {metrics['stages']['stage5d_control_single_cart_eval']['consecutive_success_final_20']} | {metrics['stages']['stage5d_control_single_cart_eval']['max_consecutive_success']} | {metrics['stages']['stage5d_control_single_cart_eval']['steps']} | Control: single monolithic CART without Base composition |
"""
    (out_dir / "cycle_summary.md").write_text(md, encoding="utf-8")
    print(f"\nAPC Cycle Completed! Summary saved to {out_dir / 'cycle_summary.md'}")


if __name__ == "__main__":
    main()
