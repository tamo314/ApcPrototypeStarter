import json
from pathlib import Path

path = "runs/apc-t06-grasp_guard-seed3009-20260925-a/steps.jsonl"
with open(path) as f:
    for line in f:
        d = json.loads(line)
        diag = d.get("info", {}).get("diagnostic", {})
        # Check all keys that might indicate grasp guard
        for k, v in diag.items():
            if "guard" in k.lower() or "recovery" in k.lower() or "grasp" in k.lower() or "module" in k.lower():
                if v:
                    print(f"Step {d['step']}: {k} = {v}")
                    break
        else:
            continue
        break
