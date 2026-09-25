import json
import sys

def inspect_range(path, start, end):
    print(f"=== Inspecting {path} (steps {start} to {end}) ===")
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            step = d["step"]
            if step < start:
                continue
            if step > end:
                break
            info = d.get("info", {})
            diag = info.get("diagnostic", {})
            pre = diag.get("pre_action_state", {})
            grasped_val = info.get("is_grasped", False)
            if isinstance(grasped_val, list):
                grasped_val = grasped_val[0]
            grasped = bool(grasped_val or pre.get("grasped", False))
            cube_pos = pre.get("cube_position") or [0, 0, 0]
            hand_pos = pre.get("measured_hand_position") or [0, 0, 0]
            grip = pre.get("gripper_target_m", 0.0)
            sel_id = diag.get("proposed_id")
            exec_id = diag.get("executed_id")
            override = diag.get("override_reason_code")
            cube_z = cube_pos[2]
            hand_z = hand_pos[2]
            dz = hand_z - cube_z
            dxy = ((hand_pos[0]-cube_pos[0])**2 + (hand_pos[1]-cube_pos[1])**2)**0.5
            act_mod = diag.get("active_module", "")
            patch_applied = diag.get("patch_applied", False)
            dg = info.get('cube_goal_dist_xy_m', 0.0)
            if isinstance(dg, list): dg = dg[0]
            print(f"step {step:4d}: grasped={grasped} grip={grip:+.2f} hand_z={hand_z:.3f} cube_z={cube_z:.3f} (dz={dz:+.3f}, dxy={dxy:.3f}) dist_goal={float(dg):.3f} sel={sel_id} exec={exec_id} mod={act_mod} patch={patch_applied} over={override}")

if __name__ == "__main__":
    path = sys.argv[1]
    start = int(sys.argv[2])
    end = int(sys.argv[3])
    inspect_range(path, start, end)
