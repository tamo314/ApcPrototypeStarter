import sys
sys.path.append(".")
import json, torch
import numpy as np
from pathlib import Path
from scripts.train_transit_gate import extract_features, VAL_RUNS, FEATURE_NAMES
from apc_maniskill.primitive_tree import CART

ckpt = torch.load("runs/learned_transit_gate_v1/transit_gate_cart.pt", weights_only=False)
model = CART(ckpt["n_nodes"], 2)
model.load_state_dict(ckpt["state_dict"])

for r in VAL_RUNS:
    fps = []
    with open(f"{r}/steps.jsonl") as f:
        for line in f:
            d = json.loads(line)
            diag = d.get("info", {}).get("diagnostic", {})
            pre = diag.get("pre_action_state", {})
            if not pre: continue
            raw_id = diag.get("base_raw_id", diag.get("proposed_id", pre.get("selected_id", None)))
            trig = bool(diag.get("transit_guard_triggered", False))
            feat = extract_features(pre, raw_id)
            pred = int(model(torch.from_numpy(feat)).argmax())
            if pred == 1 and not trig:
                fps.append((d["step"], raw_id, pre["cube_position"][2], pre["goal_position"][2]))
    print(f"{r}: {len(fps)} False Positives")
    for step, raw, cz, gz in fps[:5]:
        print(f"  Step {step}: raw_id={raw}, cube_z={cz:.3f}, goal_z={gz:.3f}")
