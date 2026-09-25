import json
from pathlib import Path
from collections import Counter

pos_actions = []
with open("runs/apc-t31-experience-collection-20260925-a/transitions.jsonl") as f:
    for line in f:
        d = json.loads(line)
        if d.get("module") == "exploration_transit" and d.get("is_positive_progress"):
            pos_actions.append(d.get("a_executed"))

print("Positive progress actions count:", len(pos_actions))
print("Action counts:", Counter(pos_actions))
