import json

f1 = "runs/apc-t04-cart_guarded-data_d-seed3011-20260925-a/steps.jsonl"
f2 = "runs/apc-t06-regression-seed3011-20260925-c/steps.jsonl"

with open(f1) as h1, open(f2) as h2:
    for step, (l1, l2) in enumerate(zip(h1, h2)):
        if step == 598:
            d1 = json.loads(l1)
            d2 = json.loads(l2)
            obs1 = d1["info"]["diagnostic"]["pre_action_state"]
            obs2 = d2["info"]["diagnostic"]["pre_action_state"]
            print(f"Comparing step {step} pre_action_state:")
            for k in obs1:
                v1 = obs1[k]
                v2 = obs2.get(k)
                if v1 != v2:
                    print(f"  Field '{k}' differs:")
                    print(f"    Run 1: {v1}")
                    print(f"    Run 2: {v2}")
            break
