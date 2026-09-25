import json

with open("runs/apc-t13-unified-router-eval-seed3011-20260925-a/steps.jsonl") as f:
    for line in f:
        d = json.loads(line)
        step = d["step"]
        if 350 <= step <= 380:
            diag = d.get("info", {}).get("diagnostic", {})
            pre = diag.get("pre_action_state", {})
            hand = pre.get("measured_hand_position", [0,0,0])
            cube = pre.get("cube_position", [0,0,0])
            mode = pre.get("mode_code")
            act = diag.get("executed_id")
            base_p = pre.get("base_pose", [0,0,0])
            base_v = pre.get("base_velocity", [0,0,0])
            print(f"Step {step}: mode={mode}, act={act}, base_x={base_p[0]:.3f}, base_vx={base_v[0]:.4f}, hand_z={hand[2]:.3f}, cube_z={cube[2]:.3f}")
