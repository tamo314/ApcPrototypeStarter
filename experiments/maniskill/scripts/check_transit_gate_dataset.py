import json
from pathlib import Path
import numpy as np

# Training runs (strictly excluding evaluation seeds: 3011, 3001, 3201)
TRAIN_RUNS = [
    "runs/dataset-transit-teacher-20260925-d",
    "runs/apc-ablation-scope-cart_guarded-seed3007-20260925-a",
    "runs/apc-ablation-scope-cart_guarded-seed3012-20260925-a",
    "runs/apc-ablation-scope-hand_guard-seed3007-20260925-a",
    "runs/apc-ablation-scope-hand_guard-seed3012-20260925-a",
    "runs/apc-matrix-v2-config1_guard-seed3010-20260925-a",
    "runs/apc-matrix-v2-config1_guard-seed3012-20260925-a",
    "runs/apc-matrix-v2-config1_guard-seed3013-20260925-a",
]

# Validation / Evaluation runs
VAL_RUNS = [
    "runs/apc-t10-standalone-eval-seed3011-20260925-a",
    "runs/apc-t10-retention-place-seed3001-20260925-b",
    "runs/apc-t10-retention-pick-seed3201-20260925-a",
]

FEATURE_NAMES = [
    "raw_is_4",           # 1 if base proposed hand_z_plus (4), else 0
    "grasped",            # 1 if grasped, else 0
    "cube_lift_z",        # cube_z - cube_initial_z
    "cube_rel_goal_z",    # cube_z - goal_z
    "goal_z",             # goal_z
    "dist_xy_to_goal",    # horizontal distance from cube to goal
    "dist_z_to_goal",     # abs(cube_z - goal_z)
    "cube_z",             # absolute cube_z
]

def extract_features(state, raw_id):
    cube = np.asarray(state["cube_position"], dtype=np.float32)
    goal = np.asarray(state["goal_position"], dtype=np.float32)
    cube_init_z = float(state.get("cube_initial_z", 0.02))
    grasped = float(state.get("grasped", False))
    raw_is_4 = float(raw_id == 4)
    lift_z = float(cube[2] - cube_init_z)
    rel_goal_z = float(cube[2] - goal[2])
    dist_xy = float(np.linalg.norm(cube[:2] - goal[:2]))
    dist_z = float(abs(cube[2] - goal[2]))
    
    return np.array([
        raw_is_4,
        grasped,
        lift_z,
        rel_goal_z,
        goal[2],
        dist_xy,
        dist_z,
        cube[2]
    ], dtype=np.float32)

def load_dataset(run_dirs):
    X, y = [], []
    for r in run_dirs:
        path = Path(r) / "steps.jsonl"
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                info = d.get("info", {})
                diag = info.get("diagnostic", {})
                pre = diag.get("pre_action_state", {})
                if not pre or "cube_position" not in pre:
                    continue
                raw_id = diag.get("base_raw_id", diag.get("proposed_id", pre.get("selected_id", None)))
                trig = bool(diag.get("transit_guard_triggered", False))
                
                feat = extract_features(pre, raw_id)
                X.append(feat)
                y.append(1 if trig else 0)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

X_train, y_train = load_dataset(TRAIN_RUNS)
X_val, y_val = load_dataset(VAL_RUNS)

print(f"Train samples: {len(X_train)}, Positives: {int(np.sum(y_train))} ({np.mean(y_train)*100:.2f}%)")
print(f"Val samples:   {len(X_val)}, Positives: {int(np.sum(y_val))} ({np.mean(y_val)*100:.2f}%)")
