import json
from pathlib import Path
from collections import Counter

run_dir = Path("runs/apc-t32-learned-candidate-20260925-a/eval_cond3_full_adaptation")
steps_file = run_dir / "steps.jsonl"
ep_file = run_dir / "episodes.jsonl"

ep_data = json.loads(ep_file.read_text(encoding="utf-8"))
print("Final info:", ep_data.get("final_info"))

modules = []
executed_acts = []
dists = []
cube_zs = []
overrides = []

with open(steps_file, encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        diag = d.get("info", {}).get("diagnostic", {})
        pre = diag.get("pre_action_state", {})
        modules.append(diag.get("active_module"))
        executed_acts.append(diag.get("executed_id"))
        overrides.append(diag.get("override_reason_code"))
        cube = pre.get("cube_position")
        goal = pre.get("goal_position")
        if cube and goal:
            import numpy as np
            dist = np.linalg.norm(np.array(cube[:2]) - np.array(goal[:2]))
            dists.append(dist)
            cube_zs.append(cube[2])

print(f"Total steps: {len(modules)}")
print(f"Executed actions: {Counter(executed_acts)}")
print(f"Overrides: {Counter(overrides)}")
if dists:
    print(f"Start dist_xy: {dists[0]:.4f}m, Final dist_xy: {dists[-1]:.4f}m, Min dist_xy: {min(dists):.4f}m")
    print(f"Start cube_z: {cube_zs[0]:.4f}m, Final cube_z: {cube_zs[-1]:.4f}m, Max cube_z: {max(cube_zs):.4f}m")
