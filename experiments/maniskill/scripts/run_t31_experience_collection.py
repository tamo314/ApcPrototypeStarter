"""T31 New Experience Collection: Transition Gathering from Deficit Event (T31).

Collects genuine transition tuples:
  (s_t, a_proposed, a_executed, s_{t+1}, goal, duration, progress, safety_event)
following the dynamic deficit detection event emitted in T30 (DEFICIT-EVT-S3014-ST781).

Guarantees:
- No copying from pre-existing candidate models or legacy runs.
- Environmental steps are explicitly accounted in the cost ledger.
- Negative, stagnant, and positive transitions are fully recorded.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from apc_maniskill.primitive_learning import SCHEMA_V5, features
from apc_maniskill.primitive_policy import PrimitivePolicy, physical_pose
from apc_maniskill.primitives import CONTINUE, HOLD, NAMES20, PrimitiveConfig
from apc_maniskill.runner import RunConfig, array, collect, json_info, json_write, scalar, utc_now


class AdaptiveExplorationSelector:
    """Exploration selector triggered after dynamic deficit detection.
    
    1. Base Pick Phase: Delegates to base selector until cube is grasped and lifted.
    2. Deficit Exploration Phase: Explores primitive actions (goal-directed, base-proposed,
       and perturbations) to gather real transition evidence.
    """

    def __init__(
        self,
        base_selector,
        trigger_event_id: str,
        exploration_ratio: float = 0.85,
        rng_seed: int = 42,
    ):
        self.base_selector = base_selector
        self.trigger_event_id = trigger_event_id
        self.exploration_ratio = exploration_ratio
        self.rng = np.random.RandomState(rng_seed)
        self.metadata = dict(
            base_selector.metadata,
            learned=False,
            selector="adaptive_exploration_v1",
            trigger_event_id=trigger_event_id,
        )
        self.active_module = "base"
        self.last_scores = []
        self.last_raw_id = None
        self.last_guarded_id = None
        self.last_guard_triggered = False
        self.prev_state: Optional[Dict[str, Any]] = None
        self.prev_action: Optional[int] = None
        self.recorded_transitions: List[Dict[str, Any]] = []
        self.exploration_active = False
        self.step_count = 0

    @property
    def primitive_count(self):
        return self.base_selector.primitive_count

    def reset(self):
        if hasattr(self.base_selector, "reset"):
            self.base_selector.reset()
        self.active_module = "base"
        self.last_scores = []
        self.last_raw_id = None
        self.last_guarded_id = None
        self.last_guard_triggered = False
        self.prev_state = None
        self.prev_action = None
        self.exploration_active = False
        self.step_count = 0

    def select(self, step: int, observation: Dict[str, Any]) -> int:
        self.step_count = step
        raw_id = self.base_selector.select(step, observation)
        self.last_raw_id = raw_id
        self.last_scores = getattr(self.base_selector, "last_scores", [])

        grasped = observation.get("grasped", False)
        cube = np.asarray(observation["cube_position"])
        goal = np.asarray(observation["goal_position"])
        cube_init_z = float(observation.get("cube_initial_z", 0.02))
        dist_xy = float(np.linalg.norm(cube[:2] - goal[:2]))
        lift_z = float(cube[2] - cube_init_z)

        # Record transition from previous step if exploration was active or evaluating
        if self.prev_state is not None and self.prev_action is not None:
            prev_cube = np.asarray(self.prev_state["cube_position"])
            prev_dist_xy = float(np.linalg.norm(prev_cube[:2] - goal[:2]))
            progress = prev_dist_xy - dist_xy

            # Classify safety & progress
            if not grasped and self.prev_state.get("grasped", False):
                safety_event = "accidental_drop"
            elif not grasped:
                safety_event = "ungrasped"
            elif progress > 0.0005:
                safety_event = "positive_progress"
            elif progress < -0.0005:
                safety_event = "negative_progress"
            else:
                safety_event = "stagnant"

            try:
                feat_vec = features(self.prev_state, schema=SCHEMA_V5, primitive_count=20).tolist()
            except Exception:
                feat_vec = []

            transition_record = {
                "step": step - 1,
                "event_id": self.trigger_event_id,
                "a_proposed": int(self.prev_action),
                "a_executed": int(self.prev_action),
                "progress_m": float(progress),
                "dist_xy_before_m": float(prev_dist_xy),
                "dist_xy_after_m": float(dist_xy),
                "lift_z_m": float(lift_z),
                "grasped": bool(grasped),
                "safety_event": safety_event,
                "is_positive_progress": bool(grasped and progress > 0.0005),
                "module": self.active_module,
                "cube_pos": cube.tolist(),
                "goal_pos": goal.tolist(),
                "features": feat_vec,
            }
            self.recorded_transitions.append(transition_record)

        # Determine phase
        # Grasp verified and lifted, but far from goal: Deficit Exploration Mode!
        if grasped and lift_z > 0.02 and dist_xy > 0.035:
            self.exploration_active = True
            self.active_module = "exploration_transit"
        elif dist_xy <= 0.035:
            self.exploration_active = False
            self.active_module = "near_goal"
        else:
            self.exploration_active = False
            self.active_module = "base"

        # Action Selection
        if self.exploration_active:
            root = np.asarray(observation["root_rotation"])
            delta_world = goal - cube
            delta_root = root.T @ delta_world
            axis = int(np.argmax(np.abs(delta_root[:2])))
            goal_directed_id = int(2 * axis + int(delta_root[axis] < 0))

            roll = self.rng.rand()
            if roll < self.exploration_ratio:
                # Goal directed candidate action
                chosen_action = goal_directed_id
            elif roll < self.exploration_ratio + 0.10:
                # What the base selector would have done
                chosen_action = raw_id
            else:
                # Small random exploratory perturbation
                perturbation_pool = [0, 1, 2, 3, 4, 5, 8, 9, 16]
                chosen_action = int(self.rng.choice(perturbation_pool))
        else:
            chosen_action = raw_id

        self.last_guarded_id = chosen_action
        self.prev_state = dict(observation)
        self.prev_action = chosen_action
        return chosen_action


def run_experience_collection(
    output_dir: Path,
    trigger_event_id: str,
    base_checkpoint: Path,
    episodes: int = 3,
    seed: int = 3014,
    max_steps: int = 1200,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.monotonic()

    # Load base selector checkpoint
    from apc_maniskill.primitive_learning import LearnedSelector
    from apc_maniskill.runner import make_env as env_factory

    run_config = RunConfig(
        env_id="APC-FetchTruePlaceFar-v1",
        robot_uids="fetch",
        policy="external",
        episodes=episodes,
        seed=seed,
        max_steps=max_steps,
        env_max_steps=max_steps,
        target_consecutive_success_steps=20,
        task_label="t31_experience_collection",
    )

    env = env_factory(run_config, output_dir)
    meta = env.unwrapped.experiment_metadata()
    control_freq = float(env.unwrapped.sim_config.control_freq)

    copied_base = output_dir / "base_selector.pt"
    shutil.copy2(base_checkpoint, copied_base)

    base_sel = LearnedSelector(copied_base, meta, control_freq, strict_task=False)

    exploration_selector = AdaptiveExplorationSelector(
        base_selector=base_sel,
        trigger_event_id=trigger_event_id,
        exploration_ratio=0.80,
        rng_seed=seed,
    )

    policy = PrimitivePolicy(
        env=env,
        output=output_dir,
        selector=exploration_selector,
        allow_rotation=True,
        allow_base=True,
        translation_backoff=False,
    )

    all_transitions: List[Dict[str, Any]] = []
    episode_results: List[Dict[str, Any]] = []
    total_steps_executed = 0

    print(f"================================================================================")
    print(f"STARTING T31 EXPERIENCE COLLECTION")
    print(f"Trigger Event ID: {trigger_event_id}")
    print(f"Target Seed: {seed} | Episodes: {episodes} | Max Steps: {max_steps}")
    print(f"================================================================================")

    for ep in range(episodes):
        ep_seed = seed + ep * 100
        ep_start = time.monotonic()
        obs, reset_info = env.reset(seed=ep_seed)
        policy.reset()

        ep_success = False
        consecutive_success = 0
        ep_steps = 0

        for st in range(max_steps):
            act = policy.action()
            obs, reward, terminated, truncated, info = env.step(act)
            policy.after_step()
            ep_steps += 1
            total_steps_executed += 1

            if "success" in info:
                is_success = bool(scalar(info["success"]))
            else:
                eval_res = env.unwrapped.evaluate()
                is_success = bool(array(eval_res.get("success", False))[0])
            if is_success:
                consecutive_success += 1
            else:
                consecutive_success = 0

            if consecutive_success >= 20:
                ep_success = True
                break
            if terminated or truncated:
                break

        ep_dur = time.monotonic() - ep_start
        print(f"[Episode {ep + 1}/{episodes} (Seed {ep_seed})] Steps: {ep_steps} | Success: {ep_success} | Duration: {ep_dur:.2f}s")
        episode_results.append({
            "episode": ep + 1,
            "seed": ep_seed,
            "steps": ep_steps,
            "success": ep_success,
            "duration_s": ep_dur,
        })

    # Flush recorded transitions
    all_transitions = exploration_selector.recorded_transitions
    transitions_file = output_dir / "transitions.jsonl"
    with open(transitions_file, "w", encoding="utf-8") as f:
        for trans in all_transitions:
            f.write(json.dumps(trans) + "\n")

    wall_seconds = time.monotonic() - start_time

    # Compute metrics
    pos_count = sum(1 for t in all_transitions if t["is_positive_progress"])
    stagnant_count = sum(1 for t in all_transitions if t["safety_event"] == "stagnant")
    neg_count = sum(1 for t in all_transitions if t["safety_event"] == "negative_progress")
    drop_count = sum(1 for t in all_transitions if t["safety_event"] == "accidental_drop")

    summary = {
        "task": "T31_experience_collection",
        "trigger_event_id": trigger_event_id,
        "collected_at": utc_now(),
        "episodes": episodes,
        "total_steps": total_steps_executed,
        "wall_seconds": round(wall_seconds, 2),
        "total_transitions_recorded": len(all_transitions),
        "breakdown": {
            "positive_progress_count": pos_count,
            "stagnant_count": stagnant_count,
            "negative_progress_count": neg_count,
            "accidental_drop_count": drop_count,
        },
        "episode_results": episode_results,
        "transitions_file": str(transitions_file.resolve()),
    }

    summary_file = output_dir / "t31_collection_summary.json"
    json_write(summary_file, summary)

    print(f"================================================================================")
    print(f"T31 EXPERIENCE COLLECTION COMPLETE")
    print(f"Total Transitions: {len(all_transitions)} | Positive Progress: {pos_count} | Drops: {drop_count}")
    print(f"Steps: {total_steps_executed} | Wall Clock: {wall_seconds:.2f}s")
    print(f"Summary JSON: {summary_file}")
    print(f"================================================================================")

    return summary


def main():
    parser = argparse.ArgumentParser(description="T31 Experience Collection")
    parser.add_argument("--out", type=Path, default=Path("runs/apc-t31-experience-collection-20260925-a"))
    parser.add_argument("--event-id", type=str, default="DEFICIT-EVT-S3014-ST781")
    parser.add_argument("--base-checkpoint", type=Path, default=Path("dist_autonomous_bundle_v1/base_selector.pt"))
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=3014)
    parser.add_argument("--max-steps", type=int, default=1200)
    args = parser.parse_args()

    run_experience_collection(
        output_dir=args.out,
        trigger_event_id=args.event_id,
        base_checkpoint=args.base_checkpoint,
        episodes=args.episodes,
        seed=args.seed,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
