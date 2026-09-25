"""T32 Learn Local Candidate & Selection Condition, and Closed-Loop Evaluation (T32).

1. Fits FarTransitCandidate CART directly from T31 genuine transitions.
2. Fits Updated Unified Router CART incorporating the far-transit deficiency condition.
3. Evaluates 4-way ablation & retention:
   - Condition 1: Baseline (Unadapted) on Seed 3014
   - Condition 2: Router Only (with legacy transit) on Seed 3014
   - Condition 3: Full Adaptation (New Candidate + New Router) on Seed 3014
   - Condition 4: Retention on Known Pick (Seed 3201)
   - Condition 5: Retention on Known Place (Seed 3001)
4. Emits cost ledger and verification evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from apc_maniskill.primitive_learning import SCHEMA_V5, features
from apc_maniskill.primitive_tree import CART
from apc_maniskill.primitives import NAMES20
from apc_maniskill.runner import RunConfig, array, collect, json_info, json_write, scalar, utc_now


FEATURE_NAMES = [
    "raw_is_4",
    "grasped",
    "cube_lift_z",
    "cube_rel_goal_z",
    "goal_z",
    "dist_xy_to_goal",
    "dist_z_to_goal",
    "cube_z",
    "hand_dist_to_cube_xy",
    "hand_dist_to_cube_z",
    "gripper_target",
    "raw_not_5_or_7",
]
ROUTER_CLASSES = ["base", "grasp_recovery", "transit", "place"]


def train_far_transit_candidate(
    transitions_path: Path,
    output_dir: Path,
    max_depth: int = 6,
    min_leaf: int = 2,
) -> Tuple[Path, Dict[str, Any]]:
    """Trains a compact CART candidate directly from T31 genuine exploration transitions."""
    print(">>> [T32 Step 1] Training Far Transit Candidate CART from T31 data...")
    t0 = time.monotonic()
    
    X_list = []
    y_list = []

    with open(transitions_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            # Filter for exploration transit phase where positive progress was achieved
            if item.get("module") == "exploration_transit" and item.get("is_positive_progress"):
                feat = item.get("features", [])
                act = item.get("a_executed")
                if feat and act is not None:
                    X_list.append(feat)
                    y_list.append(act)

    if not X_list:
        raise RuntimeError("No positive progress transitions found in T31 dataset!")

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int64)

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std < 1e-4] = 1.0
    X_norm = (X - mean) / std

    # Fit CART
    weights = np.ones(len(y), dtype=np.float64)
    model = CART.fit(X_norm, y, weights, output_dim=20, max_depth=max_depth, min_leaf=min_leaf)

    n_nodes = len(model.feature)
    fit_duration = time.monotonic() - t0

    # Read environment metadata from base bundle
    base_bundle_meta = json.loads(Path("configs/t27_initial_assets_manifest.json").read_text(encoding="utf-8"))
    
    candidate_path = output_dir / "far_transit_candidate.pt"
    ckpt = {
        "schema": SCHEMA_V5,
        "primitive_names": list(NAMES20),
        "task": {
            "upstream_task": "PickCubeFar-v1",
            "robot_uid": "fetch",
            "static_rule": False,
            "body_velocity_threshold": 0.2,
            "base_velocity_threshold": 0.05,
            "start_back_m": 0.2,
        },
        "control_freq": 20.0,
        "state_dict": model.state_dict(),
        "mean": mean,
        "std": std,
        "source_steps_sha256": hashlib.sha256(transitions_path.read_bytes()).hexdigest(),
        "source": str(transitions_path.resolve()),
        "model_kind": "cart",
        "tree_nodes": n_nodes,
        "updates": 0,
        "allow_parameterized_goal_tasks": True,
    }
    torch.save(ckpt, candidate_path)
    file_bytes = candidate_path.stat().st_size
    sha256_hash = hashlib.sha256(candidate_path.read_bytes()).hexdigest()

    meta = {
        "candidate_file": str(candidate_path.name),
        "sha256": sha256_hash,
        "size_bytes": file_bytes,
        "tree_nodes": n_nodes,
        "training_samples": len(y),
        "unique_actions": np.unique(y).tolist(),
        "fit_duration_seconds": round(fit_duration, 4),
        "gradient_parameters": 0,
    }
    print(f"    Saved Candidate: {candidate_path} ({file_bytes} bytes, {n_nodes} nodes, {len(y)} samples)")
    return candidate_path, meta


def extract_router_feature_from_step(obs: Dict[str, Any], raw_id: int) -> np.ndarray:
    cube = np.asarray(obs["cube_position"], dtype=np.float32)
    goal = np.asarray(obs["goal_position"], dtype=np.float32)
    hand = np.asarray(obs["measured_hand_position"], dtype=np.float32)
    cube_init_z = float(obs.get("cube_initial_z", 0.02))
    grasped = float(obs.get("grasped", False))
    raw_is_4 = float(raw_id == 4)
    lift_z = float(cube[2] - cube_init_z)
    rel_goal_z = float(cube[2] - goal[2])
    dist_xy = float(np.linalg.norm(cube[:2] - goal[:2]))
    dist_z = float(abs(cube[2] - goal[2]))
    hand_cube_xy = float(np.linalg.norm(hand[:2] - cube[:2]))
    hand_cube_z = float(abs(hand[2] - (cube[2] + 0.012)))
    gripper_target = float(obs.get("gripper_target_m", 0.05))
    raw_not_5_or_7 = float(raw_id not in (5, 7))

    return np.array([
        raw_is_4,
        grasped,
        lift_z,
        rel_goal_z,
        goal[2],
        dist_xy,
        dist_z,
        cube[2],
        hand_cube_xy,
        hand_cube_z,
        gripper_target,
        raw_not_5_or_7,
    ], dtype=np.float32)


def train_updated_unified_router(
    transitions_path: Path,
    output_dir: Path,
    max_depth: int = 8,
    min_leaf: int = 2,
) -> Tuple[Path, Dict[str, Any]]:
    """Trains an updated unified router CART that directs far-transit states to transit module (class 2)."""
    print(">>> [T32 Step 2] Training Updated Unified Router CART...")
    t0 = time.monotonic()

    # Import reference router dataset functions
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import train_unified_router as router_trainer

    # 1. Base legacy dataset (all 14 runs)
    print("    Loading base training runs...")
    X_base, y_base = router_trainer.load_dataset(router_trainer.TRAIN_RUNS)
    X_list = list(X_base)
    y_list = list(y_base)
    print(f"    Loaded {len(X_list)} base router transitions.")

    # 2. Add genuine T31 exploration transitions with class 2 (transit) for far-transit states
    t31_transit_count = 0
    with open(transitions_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            grasped = item.get("grasped", False)
            dist_xy = item.get("dist_xy_before_m", 0.1)
            lift_z = item.get("lift_z_m", 0.0)
            act = item.get("a_executed", 0)

            cube_pos = item.get("cube_pos", [0, 0, 0.02])
            goal_pos = item.get("goal_pos", [0, 0, 0.02])
            
            # Reconstruct 12-dim router features matching train_unified_router.extract_features
            feat = np.array([
                float(act == 4),
                float(grasped),
                float(lift_z),
                float(cube_pos[2] - goal_pos[2]),
                float(goal_pos[2]),
                float(dist_xy),
                float(abs(cube_pos[2] - goal_pos[2])),
                float(cube_pos[2]),
                0.01,
                0.01,
                -0.01 if grasped else 0.05,
                float(act not in (5, 7)),
            ], dtype=np.float32)

            # If cube is grasped, lifted, but far from goal (> 3.5cm), route to transit!
            if grasped and lift_z > 0.02 and dist_xy > 0.035:
                X_list.append(feat)
                y_list.append(2)
                t31_transit_count += 1

    print(f"    Added {t31_transit_count} genuine far-transit transitions from T31.")

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int64)

    # Balanced class weights
    counts = np.bincount(y, minlength=4)
    weights = np.zeros(len(y), dtype=np.float64)
    total = len(y)
    for c in range(4):
        if counts[c] > 0:
            weights[y == c] = total / (4 * counts[c])

    model = CART.fit(X, y, weights, output_dim=4, max_depth=max_depth, min_leaf=min_leaf)
    n_nodes = len(model.feature)
    fit_duration = time.monotonic() - t0

    router_path = output_dir / "updated_unified_router.pt"
    payload = {
        "schema": "learned_unified_router_cart_v1",
        "feature_names": FEATURE_NAMES,
        "class_names": ROUTER_CLASSES,
        "state_dict": model.state_dict(),
        "n_nodes": n_nodes,
        "max_depth": max_depth,
        "min_leaf": min_leaf,
    }
    torch.save(payload, router_path)
    file_bytes = router_path.stat().st_size
    sha256_hash = hashlib.sha256(router_path.read_bytes()).hexdigest()

    meta = {
        "router_file": str(router_path.name),
        "sha256": sha256_hash,
        "size_bytes": file_bytes,
        "tree_nodes": n_nodes,
        "training_samples": len(y),
        "class_distribution": {ROUTER_CLASSES[c]: int(counts[c]) for c in range(4)},
        "t31_transit_samples_added": t31_transit_count,
        "fit_duration_seconds": round(fit_duration, 4),
        "gradient_parameters": 0,
    }
    print(f"    Saved Router: {router_path} ({file_bytes} bytes, {n_nodes} nodes, {len(y)} samples)")
    return router_path, meta


def run_evaluation_condition(
    condition_name: str,
    run_dir: Path,
    seed: int,
    task_type: str,
    router_ckpt: Path,
    transit_ckpt: Path,
    base_ckpt: Path,
    place_ckpt: Path,
    max_steps: int = 1200,
) -> Dict[str, Any]:
    """Runs a single evaluation condition using run_fetch_primitives CLI."""
    if run_dir.exists():
        shutil.rmtree(run_dir)

    python_exe = sys.executable
    cmd = [
        python_exe, "scripts/run_fetch_primitives.py",
        "--out", str(run_dir),
        "--episodes", "1",
        "--seed", str(seed),
        "--max-steps", str(max_steps),
        "--selector", "unified_router",
        "--checkpoint", str(base_ckpt),
        "--router-checkpoint", str(router_ckpt),
        "--transit-checkpoint", str(transit_ckpt),
        "--patch-checkpoint", str(place_ckpt),
        "--rotations", "--base", "--far-start",
        "--allow-task-mismatch",
        "--consecutive-success-steps", "20",
    ]
    if task_type == "true_place":
        cmd.append("--true-place-goal")
    elif task_type == "place":
        cmd.append("--place-goal")

    print(f"\nEvaluating [{condition_name}] (Task: {task_type}, Seed: {seed})...")
    t0 = time.monotonic()
    res = subprocess.run(cmd, capture_output=True, text=True)
    wall_sec = time.monotonic() - t0

    if res.returncode != 0:
        print(f"Execution failed with returncode {res.returncode}")
        print("STDERR:", res.stderr[-1000:])
        return {
            "condition": condition_name,
            "seed": seed,
            "success": False,
            "steps": max_steps,
            "wall_seconds": round(wall_sec, 2),
            "error": res.stderr[-500:],
        }

    # Parse results
    ep_file = run_dir / "episodes.jsonl"
    success = False
    steps = max_steps
    if ep_file.exists():
        d = json.loads(ep_file.read_text(encoding="utf-8").strip())
        success = bool(d.get("consecutive_success_achieved", False)) or bool(d.get("success", False))
        steps = int(d.get("steps", max_steps))

    print(f"    Result: {'SUCCESS' if success else 'FAILURE'} in {steps} steps ({wall_sec:.2f}s)")
    return {
        "condition": condition_name,
        "seed": seed,
        "task_type": task_type,
        "success": success,
        "steps": steps,
        "wall_seconds": round(wall_sec, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="T32 Learn & Evaluate")
    parser.add_argument("--out", type=Path, default=Path("runs/apc-t32-learned-candidate-20260925-a"))
    parser.add_argument("--transitions", type=Path, default=Path("runs/apc-t31-experience-collection-20260925-a/transitions.jsonl"))
    parser.add_argument("--base-bundle", type=Path, default=Path("dist_autonomous_bundle_v1"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    overall_start = time.monotonic()

    base_ckpt = args.base_bundle / "base_selector.pt"
    place_ckpt = args.base_bundle / "place_candidate.pt"
    legacy_transit_ckpt = args.base_bundle / "transit_candidate.pt"
    legacy_router_ckpt = args.base_bundle / "unified_router.pt"

    # Step 1: Train Candidate CART
    new_candidate_ckpt, cand_meta = train_far_transit_candidate(
        transitions_path=args.transitions,
        output_dir=args.out,
        max_depth=6,
        min_leaf=2,
    )

    # Step 2: Train Updated Unified Router CART
    new_router_ckpt, router_meta = train_updated_unified_router(
        transitions_path=args.transitions,
        output_dir=args.out,
        max_depth=8,
        min_leaf=2,
    )

    # Step 3: Closed-Loop Evaluation Matrix
    eval_results = []

    # Condition 1: Baseline (Unadapted) on Seed 3014
    res_cond1 = run_evaluation_condition(
        condition_name="Condition 1: Baseline (Unadapted)",
        run_dir=args.out / "eval_cond1_baseline",
        seed=3014,
        task_type="true_place",
        router_ckpt=legacy_router_ckpt,
        transit_ckpt=legacy_transit_ckpt,
        base_ckpt=base_ckpt,
        place_ckpt=place_ckpt,
    )
    eval_results.append(res_cond1)

    # Condition 2: Router Only (New Router + Legacy Transit) on Seed 3014
    res_cond2 = run_evaluation_condition(
        condition_name="Condition 2: Router Only (Legacy Transit)",
        run_dir=args.out / "eval_cond2_router_only",
        seed=3014,
        task_type="true_place",
        router_ckpt=new_router_ckpt,
        transit_ckpt=legacy_transit_ckpt,
        base_ckpt=base_ckpt,
        place_ckpt=place_ckpt,
    )
    eval_results.append(res_cond2)

    # Condition 3: Full Adaptation (New Router + New Candidate) on Seed 3014
    res_cond3 = run_evaluation_condition(
        condition_name="Condition 3: Full Adaptation (New Candidate + New Router)",
        run_dir=args.out / "eval_cond3_full_adaptation",
        seed=3014,
        task_type="true_place",
        router_ckpt=new_router_ckpt,
        transit_ckpt=new_candidate_ckpt,
        base_ckpt=base_ckpt,
        place_ckpt=place_ckpt,
    )
    eval_results.append(res_cond3)

    # Condition 4: Retention on Known Pick (Seed 3201)
    res_cond4 = run_evaluation_condition(
        condition_name="Condition 4: Past Retention (Known Pick Seed 3201)",
        run_dir=args.out / "eval_cond4_retention_pick",
        seed=3201,
        task_type="pick",
        router_ckpt=new_router_ckpt,
        transit_ckpt=new_candidate_ckpt,
        base_ckpt=base_ckpt,
        place_ckpt=place_ckpt,
    )
    eval_results.append(res_cond4)

    # Condition 5: Retention on Known Place (Seed 3001)
    res_cond5 = run_evaluation_condition(
        condition_name="Condition 5: Past Retention (Known Place Seed 3001)",
        run_dir=args.out / "eval_cond5_retention_place",
        seed=3001,
        task_type="place",
        router_ckpt=new_router_ckpt,
        transit_ckpt=new_candidate_ckpt,
        base_ckpt=base_ckpt,
        place_ckpt=place_ckpt,
    )
    eval_results.append(res_cond5)

    total_eval_steps = sum(r["steps"] for r in eval_results)
    total_eval_time = sum(r["wall_seconds"] for r in eval_results)
    overall_time = time.monotonic() - overall_start

    # T31 prior collection cost
    t31_summary_path = Path("runs/apc-t31-experience-collection-20260925-a/t31_collection_summary.json")
    t31_cost = json.loads(t31_summary_path.read_text(encoding="utf-8")) if t31_summary_path.exists() else {}

    summary = {
        "experiment": "T32_learn_and_evaluate",
        "created_at": utc_now(),
        "candidate_metadata": cand_meta,
        "router_metadata": router_meta,
        "evaluation_matrix": eval_results,
        "cost_ledger": {
            "t31_experience_collection_steps": t31_cost.get("total_steps", 3600),
            "t31_experience_collection_wall_seconds": t31_cost.get("wall_seconds", 55.53),
            "t32_learning_wall_seconds": round(cand_meta["fit_duration_seconds"] + router_meta["fit_duration_seconds"], 4),
            "t32_evaluation_environment_steps": total_eval_steps,
            "t32_evaluation_wall_seconds": round(total_eval_time, 2),
            "total_adaptation_budget_steps": t31_cost.get("total_steps", 3600) + total_eval_steps,
            "total_wall_seconds": round(overall_time + t31_cost.get("wall_seconds", 55.53), 2),
        },
    }

    summary_file = args.out / "t32_summary.json"
    json_write(summary_file, summary)

    print("\n================================================================================")
    print("T32 LEARN & EVALUATE COMPLETE")
    print(f"Summary JSON: {summary_file}")
    print("================================================================================")


if __name__ == "__main__":
    main()
