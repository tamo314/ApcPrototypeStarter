import json
from pathlib import Path

runs = [
    "runs/apc-t10-standalone-eval-seed3011-20260925-a",
    "runs/apc-t10-retention-place-seed3001-20260925-b",
    "runs/apc-t10-retention-pick-seed3201-20260925-a",
    "runs/apc-transit-cand-eval-seed3012-20260925-b",
    "runs/apc-ablation-grasp-guard-seed3009-20260925-a",
    "runs/apc-ablation-grasp-guard-seed3008-20260925-a"
]

total_steps = 0
guard_triggered = 0
raw_id_4_count = 0

for r in runs:
    path = Path(r) / "steps.jsonl"
    if not path.exists():
        continue
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            total_steps += 1
            info = data.get("info", {})
            diag = info.get("diagnostic", {})
            pre = diag.get("pre_action_state", {})
            trig = diag.get("transit_guard_triggered", False)
            raw_id = diag.get("base_raw_id", pre.get("selected_id", None))
            if trig:
                guard_triggered += 1
            if raw_id == 4:
                raw_id_4_count += 1

ratio = (guard_triggered / total_steps * 100) if total_steps > 0 else 0
print(f"Total steps: {total_steps}")
print(f"Guard triggered: {guard_triggered} ({ratio:.2f}%)")
print(f"raw_id == 4 count: {raw_id_4_count}")
