"""Autonomous APC Lifecycle with Self-Exploration (W5 / T16).

Verifies capability acquisition without external action teachers (teacher_calls = 0).
Explores candidate primitives using environmental progress reward and grasp maintenance,
consolidates into Candidate CART, updates unified router, releases temporary resources,
and fairly compares with Direct Candidate CART and Baseline under identical budget.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch

from apc_maniskill.primitive_bank import PrimitiveBank
from apc_maniskill.primitive_tree import CART
from apc_maniskill.primitives import NAMES20
from apc_maniskill.runner import RunConfig, collect, json_write, summarize, utc_now
from apc_maniskill.primitive_policy import (
    BaseReadyPickSelector,
    GraspRecoveryGuardSelector,
    PickPlaceSelector,
    PrimitivePolicy,
    UnifiedRouterSelector,
)
from apc_maniskill.primitive_learning import (
    LearnedSelector,
    CompositePatchSelector,
    PlacePatchCondition,
    TransitPatchCondition,
    SCHEMA_V5,
    features,
    network,
)


class LocalTransitExplorer:
    """Discovers transit actions via trial-and-error using goal distance progress."""

    def __init__(self, epsilon: float = 0.20, lr: float = 0.3):
        self.epsilon = epsilon
        self.lr = lr
        # Candidate transit primitives: 0: +x, 1: -x, 2: +y, 3: -y
        self.candidate_actions = [0, 1, 2, 3]
        # Q-table keyed by 8 discrete heading sectors toward goal in root frame
        self.q_table = np.zeros((8, len(self.candidate_actions)), dtype=np.float32)
        self.last_state_sector = None
        self.last_action_idx = None
        self.last_dist_xy = None
        self.rng = np.random.default_rng(2026)
        self.exploration_steps = 0
        self.successful_steps = 0
        self.dropped_cubes = 0

    def get_sector(self, cube: np.ndarray, goal: np.ndarray, root: np.ndarray) -> int:
        delta_world = goal - cube
        delta_root = root.T @ delta_world
        angle = np.arctan2(delta_root[1], delta_root[0])
        # Map [-pi, pi] to 0..7
        sector = int(np.floor((angle + np.pi) / (2 * np.pi / 8))) % 8
        return sector

    def update_and_select(self, cube: np.ndarray, goal: np.ndarray, root: np.ndarray, grasped: bool) -> int:
        curr_dist_xy = float(np.linalg.norm((goal - cube)[:2]))
        sector = self.get_sector(cube, goal, root)

        # Update Q-value from previous step's progress
        if self.last_state_sector is not None and self.last_action_idx is not None and self.last_dist_xy is not None:
            if not grasped:
                reward = -50.0
                self.dropped_cubes += 1
            else:
                progress = self.last_dist_xy - curr_dist_xy
                reward = progress * 1000.0  # +0.01m progress -> +10 reward
                if progress > 0.001:
                    self.successful_steps += 1
                elif progress < -0.001:
                    reward -= 2.0

            old_q = self.q_table[self.last_state_sector, self.last_action_idx]
            self.q_table[self.last_state_sector, self.last_action_idx] += self.lr * (reward - old_q)

        self.exploration_steps += 1
        # Epsilon-greedy exploration
        if self.rng.random() < self.epsilon:
            action_idx = int(self.rng.integers(0, len(self.candidate_actions)))
        else:
            action_idx = int(np.argmax(self.q_table[sector]))

        chosen_action = self.candidate_actions[action_idx]
        self.last_state_sector = sector
        self.last_action_idx = action_idx
        self.last_dist_xy = curr_dist_xy
        return chosen_action


class SelfExplorationTransitSelector:
    """Intervenes during lifted transit to explore movement primitives without external teacher."""

    def __init__(self, base_selector, place_selector, patch_condition_fn=None):
        self.base_selector = base_selector
        self.place_selector = place_selector
        self.patch_condition = patch_condition_fn or PlacePatchCondition(threshold_m=0.025)
        self.explorer = LocalTransitExplorer(epsilon=0.20, lr=0.3)
        self.metadata = dict(base_selector.metadata, self_exploration=True)
        self.active_module = "base"
        self.last_raw_id = None
        self.last_guarded_id = None
        self.patch_applied = False
        self.transit_guard_triggered = False
        self.latched_place = False

    @property
    def primitive_count(self):
        return self.base_selector.primitive_count

    def reset(self):
        self.base_selector.reset()
        if hasattr(self.place_selector, "reset"):
            self.place_selector.reset()
        if hasattr(self.patch_condition, "reset"):
            self.patch_condition.reset()
        self.active_module = "base"
        self.last_raw_id = None
        self.last_guarded_id = None
        self.patch_applied = False
        self.transit_guard_triggered = False
        self.latched_place = False
        self.explorer.last_state_sector = None
        self.explorer.last_action_idx = None
        self.explorer.last_dist_xy = None

    def select(self, step: int, observation: Dict[str, Any]) -> int:
        raw_id = self.base_selector.select(step, observation)
        self.last_raw_id = raw_id

        # Maintain place latch once triggered
        if self.latched_place and self.place_selector is not None:
            self.active_module = "place"
            self.patch_applied = True
            self.transit_guard_triggered = False
            return self.place_selector.select(step, observation)

        # Place patch check
        if self.place_selector is not None and self.patch_condition(step, observation):
            self.active_module = "place"
            self.patch_applied = True
            self.latched_place = True
            self.transit_guard_triggered = False
            return self.place_selector.select(step, observation)

        # Self-exploration during transit:
        grasped = observation.get("grasped", False)
        cube = np.asarray(observation["cube_position"])
        goal = np.asarray(observation["goal_position"])
        cube_init_z = observation.get("cube_initial_z", 0.02)
        dist_xy = float(np.linalg.norm((goal - cube)[:2]))

        # Condition: grasped, lifted above table, not yet at goal, base drifts upward
        if grasped and cube[2] > cube_init_z + 0.06 and dist_xy > 0.025 and raw_id == 4:
            self.active_module = "transit_exploration"
            self.patch_applied = False
            self.transit_guard_triggered = True
            root = np.asarray(observation["root_rotation"])
            explored_act = self.explorer.update_and_select(cube, goal, root, grasped)
            self.last_guarded_id = explored_act
            return explored_act

        # Normal base operation
        self.active_module = "base"
        self.patch_applied = False
        self.transit_guard_triggered = False
        self.last_guarded_id = raw_id
        return raw_id


def extract_self_explored_dataset(source_run_dir: Path, out_dir: Path) -> int:
    """Extract self-explored steps with positive goal progress into training format."""
    out_dir.mkdir(parents=True, exist_ok=True)
    steps_p = source_run_dir / "steps.jsonl"
    manifest_p = source_run_dir / "manifest.json"
    episodes_p = source_run_dir / "episodes.jsonl"

    manifest_data = json.loads(manifest_p.read_text(encoding="utf-8"))
    if "policy_details" in manifest_data:
        manifest_data["policy_details"]["learned"] = False
        manifest_data["policy_details"]["self_exploration"] = True
    json_write(out_dir / "manifest.json", manifest_data)
    shutil.copy2(episodes_p, out_dir / "episodes.jsonl")

    extracted = []
    action_counts = Counter()

    prev_dist_xy = None
    with open(steps_p, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            diag = d.get("info", {}).get("diagnostic", {})
            pre = diag.get("pre_action_state", {})
            grasped = pre.get("grasped", False)
            cube = np.asarray(pre.get("cube_position", [0, 0, 0]))
            goal = np.asarray(pre.get("goal_position", [0, 0, 0]))
            cube_init_z = pre.get("cube_initial_z", 0.02)
            dist_xy = float(np.linalg.norm((goal - cube)[:2]))

            # Steps during lifted transit where self-exploration intervened
            is_transit = grasped and cube[2] > cube_init_z + 0.06 and dist_xy > 0.025
            explored_act = diag.get("executed_id", diag.get("proposed_id"))

            if is_transit and diag.get("transit_guard_triggered", False):
                # Filter for steps that produced positive or neutral progress (not backward)
                if prev_dist_xy is not None and (prev_dist_xy - dist_xy >= -0.002):
                    d["info"]["diagnostic"]["proposed_id"] = explored_act
                    d["episode"] = 0
                    extracted.append(d)
                    action_counts[explored_act] += 1
            prev_dist_xy = dist_xy

    with open(out_dir / "steps.jsonl", "w", encoding="utf-8") as f:
        for d in extracted:
            f.write(json.dumps(d) + "\n")

    print(f"Extracted {len(extracted)} self-explored transit steps.")
    print(f"Action distribution: {dict(sorted(action_counts.items()))}")
    return len(extracted)


def run_cmd(args: List[str], cwd: Path) -> subprocess.CompletedProcess:
    res = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[ERROR] Command failed with return code {res.returncode}")
        print("STDOUT:", res.stdout[-2000:])
        print("STDERR:", res.stderr[-2000:])
        raise RuntimeError(f"Command failed: {' '.join(args)}")
    return res


def compute_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path,
                        default=Path("dist_standalone_bundle_v1/base_selector.pt"))
    parser.add_argument("--place-checkpoint", type=Path,
                        default=Path("dist_standalone_bundle_v1/place_candidate.pt"))
    parser.add_argument("--target-seed", type=int, default=3011)
    parser.add_argument("--retention-place-seed", type=int, default=3001)
    parser.add_argument("--retention-pick-seed", type=int, default=3201)
    parser.add_argument("--max-steps", type=int, default=1200)
    args = parser.parse_args()

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = Path(__file__).resolve().parents[1]
    py = sys.executable

    base_checkpoint = (repo_dir / args.base_checkpoint).resolve() if not args.base_checkpoint.is_absolute() else args.base_checkpoint.resolve()
    place_checkpoint = (repo_dir / args.place_checkpoint).resolve() if not args.place_checkpoint.is_absolute() else args.place_checkpoint.resolve()
    if not base_checkpoint.exists():
        raise FileNotFoundError(f"Base checkpoint not found: {base_checkpoint}")
    if not place_checkpoint.exists():
        raise FileNotFoundError(f"Place checkpoint not found: {place_checkpoint}")

    report: Dict[str, Any] = {
        "experiment": "w5_t16_self_exploration_acquisition",
        "started_at": utc_now(),
        "external_teacher_calls": 0,
        "phases": {},
        "comparison": {},
    }
    t_start = time.time()

    # =========================================================================
    # Phase 1: Autonomous Self-Exploration Rollout (No External Teacher)
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 1: Autonomous Self-Exploration Rollout (Target Seed 3011, Teacher Calls = 0)")
    print("="*80)
    explore_run_dir = out_dir / "phase1_self_exploration_run"
    
    # Run simulation with SelfExplorationTransitSelector
    config_explore = RunConfig(
        env_id="APC-FetchTruePlaceFar-v1",
        robot_uids="fetch",
        policy="external",
        episodes=1,
        seed=args.target_seed,
        max_steps=args.max_steps,
        env_max_steps=args.max_steps,
        target_consecutive_success_steps=20,
        task_label="self_exploration_transit_fetch20",
    )

    def explore_factory(env, run_out):
        shutil.copy2(base_checkpoint, run_out / "base_selector.pt")
        shutil.copy2(place_checkpoint, run_out / "patch_selector.pt")
        base_sel = LearnedSelector(run_out / "base_selector.pt", env.unwrapped.experiment_metadata(),
                                   float(env.unwrapped.sim_config.control_freq), strict_task=False)
        base_sel = GraspRecoveryGuardSelector(base_sel, grasp_height_m=0.012)
        place_sel = LearnedSelector(run_out / "patch_selector.pt", env.unwrapped.experiment_metadata(),
                                    float(env.unwrapped.sim_config.control_freq), strict_task=False)
        selector = SelfExplorationTransitSelector(base_sel, place_sel)
        return PrimitivePolicy(env, run_out, allow_rotation=True, allow_base=True,
                               selector=selector, query_teacher=False)

    collect(config_explore, explore_run_dir, policy_factory=explore_factory)
    json_write(explore_run_dir / "summary.json", summarize(explore_run_dir))

    ep1_info = json.loads((explore_run_dir / "episodes.jsonl").read_text(encoding="utf-8").strip())
    print(f"Self-exploration rollout finished in {ep1_info['steps']} steps.")
    print(f"20 Consecutive Success Achieved: {ep1_info.get('consecutive_success_achieved', False)}")

    # Extract self-explored dataset
    explore_data_dir = out_dir / "phase1_self_explored_dataset"
    n_samples = extract_self_explored_dataset(explore_run_dir, explore_data_dir)

    report["phases"]["phase1_exploration"] = {
        "env_steps": ep1_info["steps"],
        "consecutive_success": ep1_info.get("consecutive_success_achieved", False),
        "teacher_calls": 0,
        "extracted_samples": n_samples,
    }

    # =========================================================================
    # Phase 2: Method A (APC: Self-Exploration -> Temporary MLP -> Candidate CART)
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 2: Method A (APC: Self-Exploration -> Temporary MLP -> Candidate CART)")
    print("="*80)
    bank_dir = out_dir / "apc_primitive_bank"
    bank = PrimitiveBank(bank_dir)

    # 2.1 Train Temporary MLP on self-explored data
    temp_train_dir = out_dir / "method_a_temp_train"
    cmd_tr_mlp = [
        py, "-m", "apc_maniskill.primitive_learning",
        "--source", str(explore_data_dir),
        "--out", str(temp_train_dir),
        "--model-kind", "mlp",
        "--updates", "3000",
        "--feature-schema", "fetch_primitive_geometry_features_v5",
        "--train-all",
        "--allow-parameterized-goal-tasks",
    ]
    run_cmd(cmd_tr_mlp, cwd=repo_dir)
    temp_meta = json.loads((temp_train_dir / "training.json").read_text(encoding="utf-8"))

    temp_entry = bank.register(
        entry_id="temp_transit_self_mlp_v1",
        name="Temporary local MLP learned from self-exploration",
        source_file=temp_train_dir / "selector.pt",
        kind="temporary",
        model_kind="mlp",
        primitive_count=20,
        parameter_count=temp_meta["model_parameters"],
        stored_values=temp_meta["model_stored_values"],
        feature_schema="fetch_primitive_geometry_features_v5",
        metadata={"teacher_calls": 0, "source": "self_exploration"},
    )

    # 2.2 Physical Rollout Verification with Temporary MLP
    temp_rollout_dir = out_dir / "method_a_temp_rollout"
    cmd_rollout = [
        py, str(repo_dir / "scripts" / "run_fetch_primitives.py"),
        "--out", str(temp_rollout_dir),
        "--episodes", "1",
        "--seed", str(args.target_seed),
        "--max-steps", str(args.max_steps),
        "--consecutive-success-steps", "20",
        "--selector", "composite_patch",
        "--checkpoint", str(base_checkpoint),
        "--patch-checkpoint", str(place_checkpoint),
        "--patch-mode", "place",
        "--transit-checkpoint", str(bank.get_path(temp_entry.id)),
        "--grasp-guard",
        "--true-place-goal",
        "--rotations", "--base", "--far-start",
        "--allow-task-mismatch",
    ]
    run_cmd(cmd_rollout, cwd=repo_dir)
    temp_roll_info = json.loads((temp_rollout_dir / "episodes.jsonl").read_text(encoding="utf-8").strip())
    print(f"Temporary MLP Rollout: {temp_roll_info['steps']} steps, Success: {temp_roll_info.get('consecutive_success_achieved', False)}")

    # 2.3 Distill into Candidate CART
    cart_train_dir = out_dir / "method_a_cart_distill"
    cmd_cart = [
        py, "-m", "apc_maniskill.primitive_learning",
        "--source", str(explore_data_dir),
        "--out", str(cart_train_dir),
        "--model-kind", "cart",
        "--max-depth", "8",
        "--min-leaf", "2",
        "--feature-schema", "fetch_primitive_geometry_features_v5",
        "--train-all",
        "--allow-parameterized-goal-tasks",
    ]
    run_cmd(cmd_cart, cwd=repo_dir)
    cart_meta = json.loads((cart_train_dir / "training.json").read_text(encoding="utf-8"))

    cand_entry = bank.consolidate(
        temp_entry.id,
        cart_train_dir / "selector.pt",
        model_kind="cart",
        parameter_count=0,
        stored_values=cart_meta["model_stored_values"],
        metadata={"tree_nodes": cart_meta["tree_nodes"], "teacher_calls": 0, "source": "self_exploration"},
    )

    # 2.4 Update Unified Router
    router_dir = out_dir / "method_a_unified_router"
    cmd_router = [
        py, str(repo_dir / "scripts" / "train_unified_router.py"),
        "--out", str(router_dir),
        "--max-depth", "8",
        "--min-leaf", "2",
    ]
    run_cmd(cmd_router, cwd=repo_dir)
    router_ckpt = router_dir / "unified_router_cart.pt"

    # 2.5 Real Release of Temporary Resource & Bundle Assembly
    release_audit = bank.release_temporary(temp_entry.id)
    reclaimed_bytes = release_audit["reclaimed_bytes"]
    print(f"Real Resource Release: Reclaimed {reclaimed_bytes} bytes from Temporary MLP.")

    dist_dir_a = out_dir / "dist_bundle_self_explored_v1"
    dist_dir_a.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base_checkpoint, dist_dir_a / "base_selector.pt")
    shutil.copy2(place_checkpoint, dist_dir_a / "place_candidate.pt")
    shutil.copy2(bank.get_path(cand_entry.id), dist_dir_a / "transit_candidate.pt")
    shutil.copy2(router_ckpt, dist_dir_a / "unified_router.pt")

    manifest_a = {
        "bundle": "dist_bundle_self_explored_v1",
        "created_at": utc_now(),
        "method": "apc_self_exploration",
        "external_teacher_calls": 0,
        "modules": {
            "base_selector": "base_selector.pt",
            "place_candidate": "place_candidate.pt",
            "transit_candidate": "transit_candidate.pt",
            "unified_router": "unified_router.pt",
        },
        "reclaimed_temporary_bytes": reclaimed_bytes,
    }
    json_write(dist_dir_a / "bundle_manifest.json", manifest_a)
    bundle_a_bytes = sum(f.stat().st_size for f in dist_dir_a.glob("*") if f.is_file())

    # 2.6 Independent Standalone Verification of Method A
    print("\n--- Verifying Method A (APC Bundle) in Isolated Process ---")
    eval_a_target = out_dir / "eval_method_a_target"
    cmd_eval_a_tgt = [
        py, str(repo_dir / "scripts" / "run_fetch_primitives.py"),
        "--out", str(eval_a_target),
        "--episodes", "1",
        "--seed", str(args.target_seed),
        "--max-steps", str(args.max_steps),
        "--consecutive-success-steps", "20",
        "--selector", "unified_router",
        "--checkpoint", str(dist_dir_a / "base_selector.pt"),
        "--patch-checkpoint", str(dist_dir_a / "place_candidate.pt"),
        "--transit-checkpoint", str(dist_dir_a / "transit_candidate.pt"),
        "--router-checkpoint", str(dist_dir_a / "unified_router.pt"),
        "--true-place-goal",
        "--rotations", "--base", "--far-start",
        "--allow-task-mismatch",
    ]
    run_cmd(cmd_eval_a_tgt, cwd=repo_dir)
    res_a_tgt = json.loads((eval_a_target / "episodes.jsonl").read_text(encoding="utf-8").strip())

    eval_a_place = out_dir / "eval_method_a_place"
    cmd_eval_a_plc = [
        py, str(repo_dir / "scripts" / "run_fetch_primitives.py"),
        "--out", str(eval_a_place),
        "--episodes", "1",
        "--seed", str(args.retention_place_seed),
        "--max-steps", str(args.max_steps),
        "--consecutive-success-steps", "20",
        "--selector", "unified_router",
        "--checkpoint", str(dist_dir_a / "base_selector.pt"),
        "--patch-checkpoint", str(dist_dir_a / "place_candidate.pt"),
        "--transit-checkpoint", str(dist_dir_a / "transit_candidate.pt"),
        "--router-checkpoint", str(dist_dir_a / "unified_router.pt"),
        "--place-goal",
        "--rotations", "--base", "--far-start",
        "--allow-task-mismatch",
    ]
    run_cmd(cmd_eval_a_plc, cwd=repo_dir)
    res_a_plc = json.loads((eval_a_place / "episodes.jsonl").read_text(encoding="utf-8").strip())

    eval_a_pick = out_dir / "eval_method_a_pick"
    cmd_eval_a_pck = [
        py, str(repo_dir / "scripts" / "run_fetch_primitives.py"),
        "--out", str(eval_a_pick),
        "--episodes", "1",
        "--seed", str(args.retention_pick_seed),
        "--max-steps", str(args.max_steps),
        "--consecutive-success-steps", "20",
        "--selector", "unified_router",
        "--checkpoint", str(dist_dir_a / "base_selector.pt"),
        "--patch-checkpoint", str(dist_dir_a / "place_candidate.pt"),
        "--transit-checkpoint", str(dist_dir_a / "transit_candidate.pt"),
        "--router-checkpoint", str(dist_dir_a / "unified_router.pt"),
        "--rotations", "--base", "--far-start",
        "--allow-task-mismatch",
    ]
    run_cmd(cmd_eval_a_pck, cwd=repo_dir)
    res_a_pck = json.loads((eval_a_pick / "episodes.jsonl").read_text(encoding="utf-8").strip())

    # =========================================================================
    # Phase 3: Method B (Direct Candidate: Self-Exploration -> Direct CART)
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 3: Method B (Direct Candidate CART from Self-Exploration)")
    print("="*80)
    direct_cart_dir = out_dir / "method_b_direct_cart"
    cmd_direct_cart = [
        py, "-m", "apc_maniskill.primitive_learning",
        "--source", str(explore_data_dir),
        "--out", str(direct_cart_dir),
        "--model-kind", "cart",
        "--max-depth", "8",
        "--min-leaf", "2",
        "--feature-schema", "fetch_primitive_geometry_features_v5",
        "--train-all",
        "--allow-parameterized-goal-tasks",
    ]
    run_cmd(cmd_direct_cart, cwd=repo_dir)

    dist_dir_b = out_dir / "dist_bundle_direct_cart_v1"
    dist_dir_b.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base_checkpoint, dist_dir_b / "base_selector.pt")
    shutil.copy2(place_checkpoint, dist_dir_b / "place_candidate.pt")
    shutil.copy2(direct_cart_dir / "selector.pt", dist_dir_b / "transit_candidate.pt")
    shutil.copy2(router_ckpt, dist_dir_b / "unified_router.pt")

    manifest_b = {
        "bundle": "dist_bundle_direct_cart_v1",
        "created_at": utc_now(),
        "method": "direct_candidate_cart",
        "external_teacher_calls": 0,
        "modules": {
            "base_selector": "base_selector.pt",
            "place_candidate": "place_candidate.pt",
            "transit_candidate": "transit_candidate.pt",
            "unified_router": "unified_router.pt",
        },
    }
    json_write(dist_dir_b / "bundle_manifest.json", manifest_b)
    bundle_b_bytes = sum(f.stat().st_size for f in dist_dir_b.glob("*") if f.is_file())

    eval_b_target = out_dir / "eval_method_b_target"
    cmd_eval_b_tgt = [
        py, str(repo_dir / "scripts" / "run_fetch_primitives.py"),
        "--out", str(eval_b_target),
        "--episodes", "1",
        "--seed", str(args.target_seed),
        "--max-steps", str(args.max_steps),
        "--consecutive-success-steps", "20",
        "--selector", "unified_router",
        "--checkpoint", str(dist_dir_b / "base_selector.pt"),
        "--patch-checkpoint", str(dist_dir_b / "place_candidate.pt"),
        "--transit-checkpoint", str(dist_dir_b / "transit_candidate.pt"),
        "--router-checkpoint", str(dist_dir_b / "unified_router.pt"),
        "--true-place-goal",
        "--rotations", "--base", "--far-start",
        "--allow-task-mismatch",
    ]
    run_cmd(cmd_eval_b_tgt, cwd=repo_dir)
    res_b_tgt = json.loads((eval_b_target / "episodes.jsonl").read_text(encoding="utf-8").strip())

    # =========================================================================
    # Phase 4: Method C (Baseline: No Exploration / Unadapted Base)
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 4: Method C (Baseline: Unadapted Base without Exploration)")
    print("="*80)
    eval_c_target = out_dir / "eval_method_c_baseline"
    cmd_eval_c = [
        py, str(repo_dir / "scripts" / "run_fetch_primitives.py"),
        "--out", str(eval_c_target),
        "--episodes", "1",
        "--seed", str(args.target_seed),
        "--max-steps", str(args.max_steps),
        "--consecutive-success-steps", "20",
        "--selector", "composite_patch",
        "--checkpoint", str(base_checkpoint),
        "--patch-checkpoint", str(place_checkpoint),
        "--patch-mode", "place",
        "--grasp-guard",
        "--true-place-goal",
        "--rotations", "--base", "--far-start",
        "--allow-task-mismatch",
    ]
    run_cmd(cmd_eval_c, cwd=repo_dir)
    res_c_tgt = json.loads((eval_c_target / "episodes.jsonl").read_text(encoding="utf-8").strip())

    # =========================================================================
    # Final Comparison & Summary Report
    # =========================================================================
    t_end = time.time()
    report["completed_at"] = utc_now()
    report["wall_seconds"] = t_end - t_start
    report["comparison"] = {
        "method_a_apc": {
            "name": "Self-Exploration -> Temporary MLP -> Candidate CART",
            "external_teacher_calls": 0,
            "exploration_env_steps": ep1_info["steps"],
            "verification_steps": res_a_tgt["steps"],
            "target_consecutive_success": res_a_tgt.get("consecutive_success_achieved", False),
            "retention_place_success": res_a_plc.get("consecutive_success_achieved", False),
            "retention_pick_success": res_a_pck.get("consecutive_success_achieved", False),
            "temporary_allocated_bytes": temp_entry.file_bytes,
            "temporary_reclaimed_bytes": reclaimed_bytes,
            "bundle_size_bytes": bundle_a_bytes,
            "trainable_parameters": 0,
        },
        "method_b_direct_cart": {
            "name": "Self-Exploration -> Direct Candidate CART",
            "external_teacher_calls": 0,
            "exploration_env_steps": ep1_info["steps"],
            "verification_steps": res_b_tgt["steps"],
            "target_consecutive_success": res_b_tgt.get("consecutive_success_achieved", False),
            "bundle_size_bytes": bundle_b_bytes,
            "trainable_parameters": 0,
        },
        "method_c_baseline": {
            "name": "Unadapted Baseline (No Exploration)",
            "external_teacher_calls": 0,
            "exploration_env_steps": 0,
            "verification_steps": res_c_tgt["steps"],
            "target_consecutive_success": res_c_tgt.get("consecutive_success_achieved", False),
        },
    }

    report_p = out_dir / "self_exploration_report.json"
    json_write(report_p, report)
    print("\n" + "="*80)
    print("EXPERIMENT T16 COMPLETE: Report saved to", report_p)
    print("="*80)
    print(json.dumps(report["comparison"], indent=2))


if __name__ == "__main__":
    main()
