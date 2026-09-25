import torch
import numpy as np
import json
from apc_maniskill.primitive_learning import features

path = "runs/apc-cycle-true-place-20260925-c/primitive_bank/primitives/fetch_true_place_patch_v1_consolidated_selector.pt"
ckpt = torch.load(path, map_location="cpu", weights_only=False)
sd = ckpt["state_dict"]
mean = ckpt["mean"]
std = ckpt["std"]
schema = ckpt["schema"]

feature_indices = sd["feature"]
thresholds = sd["threshold"]
lefts = sd["left"]
rights = sd["right"]
scores = sd["scores"]
# Inspect feature decomposition
obs_dummy = {
    "measured_hand_position": [0,0,0], "measured_hand_quaternion": [1,0,0,0],
    "cube_position": [0,0,0], "goal_position": [0,0,0], "hand_target_position": [0,0,0],
    "hand_target_quaternion": [1,0,0,0], "root_rotation": [[1,0,0],[0,1,0],[0,0,1]],
    "cube_initial_z": 0.02, "base_pose": [0,0,0], "base_target": [0,0,0],
    "base_target_age_steps": 0, "mode_code": 0, "hand_target_valid": 1,
    "position_tolerance_m": 0.003, "gripper_target_m": 0.05, "target_age_steps": 0,
    "grasped": False, "selected_id": 0
}
f_total = features(obs_dummy, schema, 20)
print(f"Total features in schema {schema}: {len(f_total)}")

# Determine schema and print offsets
print("Schema is:", schema)
# In SCHEMA_V5:
# values:
# 0:3 cube - hand (3)
# 3:6 goal - cube (3)
# 6:9 target - hand (3)
# 9:13 hand_quat (4)
# 13:17 target_quat (4)
# 17:19 cube_z_rel, hand_cube_z (2)
# 19:22 base_pose (3)
# 22:25 base_target - base (3)
# 25:28 base_age/40, mode, valid (3)
# 28:30 ready, base_ge_195 (2)
# 30:37 yaw, angle, dist_target, dist_cube_xy, dist_offset0, dist_goal_cube, pos_tol (7)
# 37:46 offset0 (9)
# 46:55 offset1 (9)
# 55:64 offset2 (9)
# 64:67 grip, age, grasped (3)
# 67:87 one-hot selected_id (20)
# Total: 87


# Read step 980 from seed 3008 run c
step_file = "runs/apc-t06-grasp_guard-seed3008-20260925-c/steps.jsonl"
with open(step_file) as f:
    for line in f:
        d = json.loads(line)
        if d["step"] == 980:
            obs = d["info"]["diagnostic"]["pre_action_state"]
            x = (features(obs, schema, 20) - mean) / std
            # Trace tree
            curr = 0
            path_trace = []
            while feature_indices[curr] >= 0:
                f_idx = feature_indices[curr]
                th = thresholds[curr]
                val = x[f_idx]
                if val <= th:
                    path_trace.append((curr, f_idx, val, th, "<=", lefts[curr]))
                    curr = lefts[curr]
                else:
                    path_trace.append((curr, f_idx, val, th, ">", rights[curr]))
                    curr = rights[curr]
actions = scores.argmax(dim=-1)

# Find paths to all nodes where action == 6
def find_paths_to_leaf(target_node):
    paths = []
    def dfs(node, current_path):
        if node == target_node:
            paths.append(list(current_path))
            return
        if feature_indices[node] < 0:
            return
        left = lefts[node].item()
        right = rights[node].item()
        f_idx = feature_indices[node].item()
        th = thresholds[node].item()
        if left >= 0:
            current_path.append((node, f_idx, "<=", th, left))
            dfs(left, current_path)
            current_path.pop()
        if right >= 0:
            current_path.append((node, f_idx, ">", th, right))
            dfs(right, current_path)
            current_path.pop()
    dfs(0, [])
    return paths

for target in [5, 33, 52]:
    print(f"\nPaths to Node {target} (action={actions[target]}):")
    for p in find_paths_to_leaf(target):
        for step_n in p:
            print(f"  Node {step_n[0]}: feat[{step_n[1]}] {step_n[2]} {step_n[3]:.3f} -> Node {step_n[4]}")

