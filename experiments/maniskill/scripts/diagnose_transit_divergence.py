"""Diagnose divergence of transit selectors across multiple seeds on exact saved trajectories."""
import argparse
import json
from pathlib import Path

from apc_maniskill.primitive_learning import LearnedSelector, TransitPatchCondition
from apc_maniskill.primitive_policy import TransitGuardSelector

BASE_CKPT = "runs/single-goal-conditioned-cart-20260925-a/selector.pt"
DEFAULT_TEMP_CKPT = "runs/apc-transit-temp-train-20260925-b/selector.pt"
DEFAULT_CAND_CKPT = "runs/apc-transit-cand-cart-20260925-b/selector.pt"

METADATA = {
    "robot_type": "fetch",
    "control_mode": "pd_joint_delta_pos",
    "feature_schema": "fetch_state_diff_v1",
}
CONTROL_FREQ = 20.0


def diagnose_seed(guard_run_path, temp_ckpt, cand_ckpt, seed):
    guard_run = Path(guard_run_path)
    steps_path = guard_run / "steps.jsonl"
    if not steps_path.exists():
        print(f"Error: {steps_path} not found")
        return

    base_sel = LearnedSelector(Path(BASE_CKPT), METADATA, CONTROL_FREQ, strict_task=False)
    guard_sel = TransitGuardSelector(base_sel)
    temp_sel = LearnedSelector(Path(temp_ckpt), METADATA, CONTROL_FREQ, strict_task=False)
    cand_sel = LearnedSelector(Path(cand_ckpt), METADATA, CONTROL_FREQ, strict_task=False)
    cond_fn = TransitPatchCondition()

    print(f"\n=== DIAGNOSING SEED {seed} ({guard_run.name}) ===")
    print(f"{'Step':<5} | {'BaseRaw':<7} | {'GuardAct':<8} | {'GuardTrig':<9} | {'CondFn':<6} | {'MLPAct':<6} | {'CARTAct':<7} | {'DistXY':<7} | {'CubeZ':<6}")
    print("-" * 85)

    guard_triggered_steps = 0
    divergence_on_guard_trig = 0
    divergence_on_wide_cond = 0
    total_wide_steps = 0

    with open(steps_path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            sdata = json.loads(line)
            step = sdata["step"]
            diag = sdata.get("info", {}).get("diagnostic", {})
            obs = diag.get("pre_action_state")
            if not obs:
                continue

            grasped = obs.get("grasped", False)
            cube_z = obs.get("cube_position", [0, 0, 0])[2]
            cube_init_z = obs.get("cube_initial_z", 0.02)
            dist_xy = sdata.get("info", {}).get("cube_goal_dist_xy_m", [None])[0]

            if grasped and cube_z > cube_init_z + 0.06:
                total_wide_steps += 1
                base_act = base_sel.select(step, obs)
                guard_act = guard_sel.select(step, obs)
                guard_trig = guard_sel.last_guard_triggered
                cond_active = cond_fn(step, obs)
                mlp_act = temp_sel.select(step, obs)
                cart_act = cand_sel.select(step, obs)

                if guard_trig:
                    guard_triggered_steps += 1
                    if (mlp_act != guard_act) or (cart_act != guard_act):
                        divergence_on_guard_trig += 1

                is_divergent = (guard_act != mlp_act) or (guard_act != cart_act) or (mlp_act != cart_act)
                if is_divergent:
                    divergence_on_wide_cond += 1

                # Print if guard triggered OR if there's divergence during guard trigger or near goal
                if guard_trig or (is_divergent and guard_triggered_steps > 0 and dist_xy is not None and dist_xy < 0.12):
                    dist_str = f"{dist_xy:.4f}" if dist_xy is not None else "-"
                    print(f"{step:<5} | {base_act:<7} | {guard_act:<8} | {str(guard_trig):<9} | {str(cond_active):<6} | {mlp_act:<6} | {cart_act:<7} | {dist_str:<7} | {cube_z:.4f}")

    print(f"\nSummary for Seed {seed}:")
    print(f"  Total wide transit condition steps: {total_wide_steps}")
    print(f"  Guard triggered steps: {guard_triggered_steps}")
    print(f"  Divergence on guard trigger: {divergence_on_guard_trig} / {guard_triggered_steps}")
    print(f"  Total wide condition divergences: {divergence_on_wide_cond} / {total_wide_steps}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[3007, 3011, 3012])
    parser.add_argument("--temp-ckpt", type=str, default=DEFAULT_TEMP_CKPT)
    parser.add_argument("--cand-ckpt", type=str, default=DEFAULT_CAND_CKPT)
    args = parser.parse_args()

    for seed in args.seeds:
        # Locate guard run
        guard_runs = list(Path("runs").glob(f"*-config1_guard-seed{seed}-20260925-a"))
        if not guard_runs:
            guard_runs = list(Path("runs").glob(f"*-guard-seed{seed}-*"))
        if not guard_runs:
            print(f"No guard run found for seed {seed}")
            continue
        diagnose_seed(guard_runs[0], args.temp_ckpt, args.cand_ckpt, seed)


if __name__ == "__main__":
    main()
