"""Diagnose divergence of transit selectors on seed 3007 trajectory."""
import json
import numpy as np
import torch
from pathlib import Path

from apc_maniskill.primitive_learning import LearnedSelector, TransitPatchCondition
from apc_maniskill.primitive_policy import TransitGuardSelector

GUARD_RUN = Path("runs/apc-matrix-config1_guard-seed3007-20260925-a")
BASE_CKPT = "runs/single-goal-conditioned-cart-20260925-a/selector.pt"
TEMP_CKPT = "runs/apc-transit-temp-train-20260925-a/selector.pt"
CAND_CKPT = "runs/apc-transit-cand-cart-20260925-a/selector.pt"

# Mock metadata for selector loading
METADATA = {
    "robot_type": "fetch",
    "control_mode": "pd_joint_delta_pos",
    "feature_schema": "fetch_state_diff_v1",
}
CONTROL_FREQ = 20.0

def main():
    steps_path = GUARD_RUN / "steps.jsonl"
    if not steps_path.exists():
        print(f"Error: {steps_path} not found")
        return

    # Load selectors
    base_sel = LearnedSelector(Path(BASE_CKPT), METADATA, CONTROL_FREQ, strict_task=False)
    guard_sel = TransitGuardSelector(base_sel)
    temp_sel = LearnedSelector(Path(TEMP_CKPT), METADATA, CONTROL_FREQ, strict_task=False)
    cand_sel = LearnedSelector(Path(CAND_CKPT), METADATA, CONTROL_FREQ, strict_task=False)
    cond_fn = TransitPatchCondition()

    print(f"{'Step':<5} | {'BaseRaw':<7} | {'GuardAct':<8} | {'GuardTrig':<9} | {'CondFn':<6} | {'MLPAct':<6} | {'CARTAct':<7} | {'DistXY':<7} | {'CubeZ':<6}")
    print("-" * 80)

    divergence_count = 0
    with open(steps_path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            sdata = json.loads(line)
            step = sdata["step"]
            diag = sdata.get("info", {}).get("diagnostic", {})
            obs = diag.get("pre_action_state")
            if not obs:
                continue

            # Check if this is within the transit / grasp region
            grasped = obs.get("grasped", False)
            cube_z = obs.get("cube_position", [0, 0, 0])[2]
            cube_init_z = obs.get("cube_initial_z", 0.02)
            dist_xy = sdata.get("info", {}).get("cube_goal_dist_xy_m", [None])[0]

            # Only inspect steps where cube is lifted and grasped
            if grasped and cube_z > cube_init_z + 0.06:
                # Query all 3 models on EXACT same observation
                base_act = base_sel.select(step, obs)
                guard_act = guard_sel.select(step, obs)
                guard_trig = guard_sel.last_guard_triggered
                cond_active = cond_fn(step, obs)
                mlp_act = temp_sel.select(step, obs)
                cart_act = cand_sel.select(step, obs)

                is_divergent = (guard_act != mlp_act) or (guard_act != cart_act) or (mlp_act != cart_act)
                if guard_trig or is_divergent:
                    divergence_count += 1
                    dist_str = f"{dist_xy:.4f}" if dist_xy is not None else "-"
                    print(f"{step:<5} | {base_act:<7} | {guard_act:<8} | {str(guard_trig):<9} | {str(cond_active):<6} | {mlp_act:<6} | {cart_act:<7} | {dist_str:<7} | {cube_z:.4f}")

    print(f"\nTotal inspected divergence steps: {divergence_count}")

if __name__ == "__main__":
    main()
