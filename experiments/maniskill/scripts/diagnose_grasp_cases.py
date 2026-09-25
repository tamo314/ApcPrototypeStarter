import json
import sys

def analyze(path):
    print(f"=== Analyzing {path} ===")
    grasped_steps = []
    first_grasp = None
    last_grasp = None
    lost_grasp_count = 0
    was_grasped = False
    with open(path) as f:
        first = json.loads(f.readline())
        info = first.get("info", {})
        diag = info.get("diagnostic", {})
        pre = diag.get("pre_action_state", {})
        print(f"  info keys: {list(info.keys())}")
        print(f"  diagnostic keys: {list(diag.keys())}")
        print(f"  pre_action_state keys: {list(pre.keys())}")
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            step = d["step"]
            info = d.get("info", {})
            diag = info.get("diagnostic", {})
            pre = diag.get("pre_action_state", {})
            grasped_val = info.get("is_grasped", False)
            if isinstance(grasped_val, list):
                grasped_val = grasped_val[0]
            grasped = bool(grasped_val or pre.get("grasped", False))
            
            cube_pos = pre.get("cube_position")
            if cube_pos is None:
                cube_h = info.get("cube_height_m", 0.0)
                if isinstance(cube_h, list):
                    cube_h = cube_h[0]
                cube_pos = [0, 0, cube_h]
            cube_z = float(cube_pos[2]) if not isinstance(cube_pos[2], list) else float(cube_pos[2][0])
            cube_xy = cube_pos[:2]
            
            dist_xy_val = info.get("cube_goal_dist_xy_m", 0.0)
            if isinstance(dist_xy_val, list):
                dist_xy_val = dist_xy_val[0]
            dist_xy = float(dist_xy_val)
            
            sel_id = diag.get("proposed_id", None)
            exec_id = diag.get("executed_id", None)
            
            if grasped and not was_grasped:
                print(f"  [Grasp acquired] Step {step}: cube_z={cube_z:.3f}, dist_xy={dist_xy:.3f}")
                if first_grasp is None:
                    first_grasp = step
            elif not grasped and was_grasped:
                print(f"  [Grasp LOST] Step {step}: cube_z={cube_z:.3f}, dist_xy={dist_xy:.3f}")
                lost_grasp_count += 1
            was_grasped = grasped

            if step % 200 == 0:
                print(f"  Step {step:4d}: grasped={grasped}, cube_z={cube_z:.3f}, dist_xy={dist_xy:.3f}, act={sel_id}/{exec_id}")

    print(f"Summary for {path}: first_grasp={first_grasp}, lost_count={lost_grasp_count}, final_grasped={was_grasped}\n")

if __name__ == "__main__":
    for p in sys.argv[1:]:
        analyze(p)
