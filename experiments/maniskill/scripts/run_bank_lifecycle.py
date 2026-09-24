"""Run complete APC primitive bank lifecycle: acquire temporary, consolidate, release, reuse, and retain."""
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
    parser.add_argument("--out", type=Path, required=True, help="New run directory for lifecycle evidence")
    parser.add_argument("--temp-checkpoint", type=Path, required=True, help="Trained temporary MLP model")
    parser.add_argument("--candidate-checkpoint", type=Path, required=True, help="Consolidated compact CART model")
    parser.add_argument("--seed", type=int, default=3201)
    parser.add_argument("--episodes", type=int, default=3)
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
    json_write(out_dir / "lifecycle_report.json", report)

    # ----------------------------------------------------
    # Stage 1: Register Temporary Skill
    # ----------------------------------------------------
    print("Stage 1: Registering temporary primitive...")
    temp_meta = json.loads((args.temp_checkpoint.parent / "training.json").read_text(encoding="utf-8"))
    entry = bank.register(
        entry_id="fetch_pick_autonomous_v1",
        name="Fetch autonomous pick selector (temporary)",
        source_file=args.temp_checkpoint,
        kind="temporary",
        model_kind="mlp",
        primitive_count=20,
        parameter_count=temp_meta["model_parameters"],
        stored_values=temp_meta["model_stored_values"],
        feature_schema=temp_meta["feature_schema"],
        metadata={"training_source": temp_meta["source"], "updates": temp_meta["updates"]}
    )
    stage1 = {
        "action": "register_temporary",
        "entry_id": entry.id,
        "kind": entry.kind,
        "parameter_count": entry.parameter_count,
        "stored_values": entry.stored_values,
        "file_bytes": entry.file_bytes,
        "sha256": entry.sha256,
        "bank_audit": bank.audit()
    }
    report["stages"]["stage1_temporary_acquisition"] = stage1
    json_write(out_dir / "lifecycle_report.json", report)

    # ----------------------------------------------------
    # Stage 2: Consolidate into Compact Candidate
    # ----------------------------------------------------
    print("Stage 2: Consolidating into compact candidate...")
    cand_meta = json.loads((args.candidate_checkpoint.parent / "training.json").read_text(encoding="utf-8"))
    entry = bank.consolidate(
        entry_id="fetch_pick_autonomous_v1",
        candidate_file=args.candidate_checkpoint,
        model_kind="cart",
        parameter_count=cand_meta["model_parameters"],
        stored_values=cand_meta["model_stored_values"],
        metadata={"tree_nodes": cand_meta["tree_nodes"], "validation_accuracy": cand_meta["validation_accuracy"]}
    )
    stage2 = {
        "action": "consolidate",
        "entry_id": entry.id,
        "kind": entry.kind,
        "parameter_count": entry.parameter_count,
        "stored_values": entry.stored_values,
        "file_bytes": entry.file_bytes,
        "sha256": entry.sha256,
        "consolidation_stats": entry.metadata["consolidation_stats"],
        "bank_audit": bank.audit()
    }
    report["stages"]["stage2_consolidation"] = stage2
    json_write(out_dir / "lifecycle_report.json", report)

    # ----------------------------------------------------
    # Stage 3: Release Temporary Artifacts
    # ----------------------------------------------------
    print("Stage 3: Releasing temporary artifacts...")
    release_info = bank.release_temporary("fetch_pick_autonomous_v1")
    stage3 = {
        "action": "release_temporary",
        "release_audit": release_info,
        "bank_audit": bank.audit()
    }
    report["stages"]["stage3_temporary_release"] = stage3
    json_write(out_dir / "lifecycle_report.json", report)

    # ----------------------------------------------------
    # Stage 4: Reuse in Clean Process on Untouched Seeds
    # ----------------------------------------------------
    print("Stage 4: Reusing bank candidate on untouched seeds...")
    consolidated_path = bank.get_path("fetch_pick_autonomous_v1")
    reuse_run_dir = out_dir / "reuse_rollout"
    
    # Run rollout using the bank candidate in a sub-execution
    cmd = [
        sys.executable,
        str(WORKSPACE / "scripts/run_fetch_primitives.py"),
        "--far-start",
        "--base",
        "--rotations",
        "--selector", "tree",
        "--checkpoint", str(consolidated_path),
        "--episodes", str(args.episodes),
        "--seed", str(args.seed),
        "--max-steps", "1200",
        "--post-success-steps", "20",
        "--translation-backoff",
        "--out", str(reuse_run_dir)
    ]
    sub = subprocess.run(cmd, capture_output=True, text=True, cwd=str(WORKSPACE))
    if sub.returncode != 0:
        print("Subprocess stdout:", sub.stdout)
        print("Subprocess stderr:", sub.stderr)
        raise RuntimeError(f"Rollout reuse failed with code {sub.returncode}")

    reuse_summary = json.loads((reuse_run_dir / "summary.json").read_text(encoding="utf-8"))
    stage4 = {
        "action": "clean_reuse_without_retraining",
        "episodes": args.episodes,
        "seed": args.seed,
        "reuse_run": str(reuse_run_dir),
        "completed_episodes": reuse_summary.get("episodes", 0),
        "first_successes": reuse_summary.get("success_rate_over_observed", 0.0) * reuse_summary.get("episodes", 0),
        "final_successes": reuse_summary.get("success_final_episodes", 0),
        "mean_return": reuse_summary.get("mean_return", 0.0),
        "additional_training_updates": 0
    }
    report["stages"]["stage4_clean_reuse"] = stage4
    json_write(out_dir / "lifecycle_report.json", report)

    # ----------------------------------------------------
    # Stage 5: Retention / Non-interference Check
    # ----------------------------------------------------
    print("Stage 5: Verifying retention on past benchmark seed 2801...")
    retention_run_dir = out_dir / "retention_rollout"
    cmd_retention = [
        sys.executable,
        str(WORKSPACE / "scripts/run_fetch_primitives.py"),
        "--far-start",
        "--base",
        "--rotations",
        "--selector", "tree",
        "--checkpoint", str(consolidated_path),
        "--episodes", "1",
        "--seed", "2801",
        "--max-steps", "1200",
        "--post-success-steps", "20",
        "--translation-backoff",
        "--out", str(retention_run_dir)
    ]
    sub_ret = subprocess.run(cmd_retention, capture_output=True, text=True, cwd=str(WORKSPACE))
    if sub_ret.returncode != 0:
        print("Retention stdout:", sub_ret.stdout)
        print("Retention stderr:", sub_ret.stderr)
        raise RuntimeError(f"Retention rollout failed with code {sub_ret.returncode}")

    retention_summary = json.loads((retention_run_dir / "summary.json").read_text(encoding="utf-8"))
    stage5 = {
        "action": "retention_check_benchmark_seed_2801",
        "retention_run": str(retention_run_dir),
        "seed": 2801,
        "completed_episodes": retention_summary.get("episodes", 0),
        "success_final": retention_summary.get("success_final_episodes", 0),
        "non_interference_verified": retention_summary.get("success_final_episodes", 0) > 0
    }
    report["stages"]["stage5_retention_check"] = stage5

    report["status"] = "completed"
    report["completed_at"] = utc_now()
    json_write(out_dir / "lifecycle_report.json", report)
    print("APC Primitive Bank lifecycle completed successfully!")
    print(json.dumps(report["stages"], indent=2))


if __name__ == "__main__":
    main()
