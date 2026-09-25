import json

def main():
    p = "runs/apc-t18-t19-compositional-transfer-20260925-a/eval_seed3008_full_router/steps.jsonl"
    with open(p) as f:
        first = json.loads(f.readline())
        diag = first.get("info", {}).get("diagnostic", {})
        print("diag keys:", list(diag.keys()))
        print("pre keys:", list(diag.get("pre_action_state", {}).keys()))

    with open(p) as f:
        for i, line in enumerate(f):
            if i in [500, 550, 560, 570, 580, 590, 600, 650, 700, 800, 1000]:
                d = json.loads(line)
                diag = d.get("info", {}).get("diagnostic", {})
                pre = diag.get("pre_action_state", {})
                mod = diag.get("active_module", "NONE")
                act = diag.get("executed_id")
                prop = diag.get("proposed_id")
                grasped = pre.get("grasped")
                cube = pre.get("cube_position", [0, 0, 0])
                hand = pre.get("measured_hand_position", [0, 0, 0])
                print(f"Step {i:4d}: mod={str(mod)} prop={prop} act={act} grasped={grasped} hand_z={hand[2]:.3f} cube_z={cube[2]:.3f}")

if __name__ == "__main__":
    main()
