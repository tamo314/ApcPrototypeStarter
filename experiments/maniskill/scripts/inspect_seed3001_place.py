import json

path = "runs/apc-regression-place-config3_cand_cart-seed3001-20260925-a/steps.jsonl"
with open(path) as f:
    for line in f:
        d = json.loads(line)
        diag = d["info"]["diagnostic"]
        exec_id = diag.get("executed_id")
        prop_id = diag.get("proposed_id")
        if exec_id == 6 or prop_id == 6:
            pre = diag["pre_action_state"]
            c = pre["cube_position"]
            g = pre["goal_position"]
            dxy = ((c[0]-g[0])**2 + (c[1]-g[1])**2)**0.5
            print(f"step {d['step']}: dxy={dxy:.4f}, cube_z={c[2]:.4f}, grasped={pre['grasped']}, patch={diag.get('patch_applied')}")
