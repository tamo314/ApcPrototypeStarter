import json

f1 = "runs/apc-t04-cart_guarded-data_d-seed3011-20260925-a/steps.jsonl"
f2 = "runs/apc-t06-regression-seed3011-20260925-c/steps.jsonl"

with open(f1) as h1, open(f2) as h2:
    step = 0
    for l1, l2 in zip(h1, h2):
        d1 = json.loads(l1)
        d2 = json.loads(l2)
        diag1 = d1["info"]["diagnostic"]
        diag2 = d2["info"]["diagnostic"]
        act1 = diag1.get("executed_id")
        act2 = diag2.get("executed_id")
        prop1 = diag1.get("proposed_id")
        prop2 = diag2.get("proposed_id")
        mod1 = diag1.get("active_module")
        mod2 = diag2.get("active_module")
        grd1 = diag1.get("transit_guard_triggered")
        grd2 = diag2.get("transit_guard_triggered")
        grsp_grd = diag2.get("last_guard_triggered") or (mod2 == "grasp_recovery")
        
        obs1 = d1["info"]["diagnostic"]["pre_action_state"]
        obs2 = d2["info"]["diagnostic"]["pre_action_state"]
        sel1 = obs1.get("selected_id")
        sel2 = obs2.get("selected_id")
        if sel1 != sel2 or act1 != act2:
            print(f"Diff at step {step}:")
            print(f"  Run 1: sel={sel1}, prop={prop1}, exec={act1}")
            print(f"  Run 2: sel={sel2}, prop={prop2}, exec={act2}")
            print(f"  Run 2 base_raw_id: {diag2.get('base_raw_id')}")
            print(f"  Run 2 last_guarded_id: {diag2.get('transit_id')}")
            print(f"  Run 2 active_module: {diag2.get('active_module')}")
            break
        step += 1
