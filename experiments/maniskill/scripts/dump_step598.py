import json

with open("runs/apc-t04-cart_guarded-data_d-seed3011-20260925-a/steps.jsonl") as f:
    for line in f:
        d = json.loads(line)
        if d["step"] == 598:
            diag = d["info"]["diagnostic"]
            print("Run 1 Step 598:")
            for k, v in diag.items():
                if k not in ("pre_action_qpos", "qpos", "pre_action_state", "post_action_state", "table_contact_force_by_link_n"):
                    print(f"  {k}: {v}")
            break

with open("runs/apc-t06-regression-seed3011-20260925-c/steps.jsonl") as f:
    for line in f:
        d = json.loads(line)
        if d["step"] == 598:
            diag = d["info"]["diagnostic"]
            print("\nRun 2 Step 598:")
            for k, v in diag.items():
                if k not in ("pre_action_qpos", "qpos", "pre_action_state", "post_action_state", "table_contact_force_by_link_n"):
                    print(f"  {k}: {v}")
            break
