import json
from pathlib import Path

from glob import glob

for path in sorted(glob("runs/*/steps.jsonl")):
    triggered = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            diag = d.get("info", {}).get("diagnostic", {})
            if diag.get("transit_guard_triggered"):
                pre = diag.get("pre_action_state", {})
                cube = pre.get("cube_position")
                goal = pre.get("goal_position")
                triggered.append((d["step"], diag.get("base_raw_id"), diag.get("transit_id"), cube, goal))
    if triggered:
        print(f"{path}: {len(triggered)} triggers")
        for s in triggered[:5]:
            print(f"  Step {s[0]}: base={s[1]} -> transit={s[2]}, cube_z={s[3][2]:.3f}, goal_z={s[4][2]:.3f}")
        if len(triggered) > 5:
            print(f"  ... and {len(triggered) - 5} more")


