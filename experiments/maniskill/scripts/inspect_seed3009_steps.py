import json
import numpy as np

def main():
    p = "runs/apc-t18-t19-compositional-transfer-20260925-a/eval_seed3009_full_router/steps.jsonl"
    with open(p) as f:
        for i, line in enumerate(f):
            if 635 <= i <= 670:
                d = json.loads(line)
                diag = d.get("info", {}).get("diagnostic", {})
                pre = diag.get("pre_action_state", {})
                tra = diag.get("transit_applied", False) or diag.get("transit_guard_triggered", False)
                act = diag.get("executed_id")
                prop = diag.get("proposed_id")
                raw = diag.get("base_raw_id", diag.get("raw_model_id"))
                cube = pre.get("cube_position", [0,0,0])
                goal = pre.get("goal_position", [0,0,0])
                dist_xy = float(np.linalg.norm(np.array(cube[:2]) - np.array(goal[:2])))
                print(f"Step {i:4d}: tra={int(tra)} raw={raw} prop={prop} act={act} cube_z={cube[2]:.3f} dist_xy={dist_xy:.3f}")

if __name__ == "__main__":
    main()
