import json
from pathlib import Path
import numpy as np

RUNS = [
    ("runs/apc-t11-gate-eval-seed3011-20260925-a", "seed 3011 (True Place, Learned Gate)"),
    ("runs/apc-t10-retention-place-seed3001-20260925-b", "seed 3001 (Place Retention)"),
    ("runs/apc-t10-retention-pick-seed3201-20260925-a", "seed 3201 (Pick Retention)"),
    ("runs/apc-t06-secure-seed3009-20260925-a", "seed 3009 (Missed Grasp Recovery)"),
    ("runs/apc-t06-secure-seed3008-20260925-b", "seed 3008 (Lost Grasp Recovery)"),
]

def analyze_handovers(run_dir, label):
    path = Path(run_dir) / "steps.jsonl"
    if not path.exists():
        print(f"Skipping {label}: {path} not found")
        return

    module_sequence = []
    transitions = []
    last_mod = None
    step_records = []

    with open(path) as f:
        for line in f:
            d = json.loads(line)
            step = d["step"]
            diag = d.get("info", {}).get("diagnostic", {})
            pre = diag.get("pre_action_state", {})
            if not pre:
                continue

            active_mod = diag.get("active_module", "base")
            if diag.get("transit_guard_triggered", False):
                active_mod = "transit"
            elif diag.get("patch_applied", False):
                active_mod = "place"
            elif diag.get("recovery_active", False) or diag.get("last_override_reason_code") == 7:
                active_mod = "grasp_recovery"

            if active_mod != last_mod:
                if last_mod is not None:
                    transitions.append((last_mod, active_mod, step, pre))
                last_mod = active_mod
            module_sequence.append(active_mod)
            step_records.append((step, active_mod, pre))

    mod_counts = {}
    for m in module_sequence:
        mod_counts[m] = mod_counts.get(m, 0) + 1

    print(f"\n==========================================")
    print(f"Run: {label}")
    print(f"Total Steps: {len(module_sequence)}")
    print(f"Module Step Distribution: {mod_counts}")
    print(f"Number of Module Transitions: {len(transitions)}")
    print(f"Transitions details:")
    for src, dst, step, pre in transitions[:10]:
        cube = pre.get("cube_position", [0, 0, 0])
        goal = pre.get("goal_position", [0, 0, 0])
        grasped = pre.get("grasped", False)
        d_xy = np.linalg.norm(np.array(cube[:2]) - np.array(goal[:2]))
        print(f"  Step {step:4d}: {src:15s} -> {dst:15s} | grasped={str(grasped):5s} | cube_z={cube[2]:.3f} | d_xy_to_goal={d_xy:.3f}")
    if len(transitions) > 10:
        print(f"  ... and {len(transitions) - 10} more transitions")

if __name__ == "__main__":
    for r, l in RUNS:
        analyze_handovers(r, l)
