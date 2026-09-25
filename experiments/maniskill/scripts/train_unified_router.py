import argparse
import json
from pathlib import Path
import numpy as np
import torch

from apc_maniskill.primitive_tree import CART

TRAIN_RUNS = [
    "runs/dataset-transit-teacher-20260925-d",
    "runs/apc-ablation-scope-cart_guarded-seed3007-20260925-a",
    "runs/apc-ablation-scope-cart_guarded-seed3012-20260925-a",
    "runs/apc-ablation-scope-hand_guard-seed3007-20260925-a",
    "runs/apc-ablation-scope-hand_guard-seed3012-20260925-a",
    "runs/apc-matrix-v2-config1_guard-seed3010-20260925-a",
    "runs/apc-matrix-v2-config1_guard-seed3012-20260925-a",
    "runs/apc-matrix-v2-config1_guard-seed3013-20260925-a",
    "runs/apc-regression-place-config1_guard-seed3002-20260925-a",
    "runs/primitive-base-recover-pick1900-20260923-a",
    "runs/primitive-base-recover-pick1901-20260923-a",
    "runs/apc-t06-secure-seed3009-20260925-a",
    "runs/apc-t06-secure-seed3008-20260925-b",
    "runs/place-teacher-3001",
]

VAL_RUNS = [
    "runs/apc-t10-standalone-eval-seed3011-20260925-a",
    "runs/apc-t10-retention-place-seed3001-20260925-b",
    "runs/apc-t10-retention-pick-seed3201-20260925-a",
]

# Classes:
# 0: base
# 1: grasp_recovery
# 2: transit
# 3: place

CLASS_NAMES = ["base", "grasp_recovery", "transit", "place"]

FEATURE_NAMES = [
    "raw_is_4",
    "grasped",
    "cube_lift_z",
    "cube_rel_goal_z",
    "goal_z",
    "dist_xy_to_goal",
    "dist_z_to_goal",
    "cube_z",
    "hand_dist_to_cube_xy",
    "hand_dist_to_cube_z",
    "gripper_target",
    "raw_not_5_or_7",
]

def extract_features(state, raw_id):
    cube = np.asarray(state["cube_position"], dtype=np.float32)
    goal = np.asarray(state["goal_position"], dtype=np.float32)
    hand = np.asarray(state["measured_hand_position"], dtype=np.float32)
    cube_init_z = float(state.get("cube_initial_z", 0.02))
    grasped = float(state.get("grasped", False))
    raw_is_4 = float(raw_id == 4)
    lift_z = float(cube[2] - cube_init_z)
    rel_goal_z = float(cube[2] - goal[2])
    dist_xy = float(np.linalg.norm(cube[:2] - goal[:2]))
    dist_z = float(abs(cube[2] - goal[2]))
    hand_cube_xy = float(np.linalg.norm(hand[:2] - cube[:2]))
    hand_cube_z = float(abs(hand[2] - (cube[2] + 0.012)))
    gripper_target = float(state.get("gripper_target_m", 0.05))
    raw_not_5_or_7 = float(raw_id not in (5, 7))

    return np.array([
        raw_is_4,
        grasped,
        lift_z,
        rel_goal_z,
        goal[2],
        dist_xy,
        dist_z,
        cube[2],
        hand_cube_xy,
        hand_cube_z,
        gripper_target,
        raw_not_5_or_7,
    ], dtype=np.float32)

def determine_label(diag, pre, raw_id):
    # Check place
    if diag.get("patch_applied", False):
        return 3
    # Check transit
    if diag.get("transit_guard_triggered", False):
        return 2
    # Check grasp recovery
    if diag.get("recovery_active", False) or diag.get("last_override_reason_code") == 7:
        return 1

    # In case the log didn't record active_module, use well-defined non-conflicting rules:
    grasped = pre.get("grasped", False)
    cube = np.asarray(pre["cube_position"])
    goal = np.asarray(pre["goal_position"])
    hand = np.asarray(pre["measured_hand_position"])
    dist_xy = np.linalg.norm(cube[:2] - goal[:2])
    dist_z = abs(cube[2] - goal[2])

    # Place condition
    if grasped and dist_xy < 0.025 and dist_z < 0.015 and goal[2] <= 0.05:
        return 3
    # Transit condition
    if grasped and raw_id == 4 and (cube[2] - goal[2]) > 0.08:
        return 2
    # Grasp recovery condition (missed grasp)
    d_xy = np.linalg.norm(hand[:2] - cube[:2])
    d_z = abs(hand[2] - (cube[2] + 0.012))
    gripper_target = pre.get("gripper_target_m", 0.05)
    if not grasped and gripper_target >= 0 and raw_id not in (5, 7) and d_xy < 0.012 and d_z < 0.010 and dist_xy > 0.05:
        return 1

    return 0

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
                label = determine_label(diag, pre, raw_id)
                feat = extract_features(pre, raw_id)
                X.append(feat)
                y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

def main():
    parser = argparse.ArgumentParser(description="Train Unified CART Router for APC modules")
    parser.add_argument("--out", type=Path, default=Path("runs/learned_unified_router_v1"))
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-leaf", type=int, default=2)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    X_train, y_train = load_dataset(TRAIN_RUNS)
    X_val, y_val = load_dataset(VAL_RUNS)

    print(f"Train size: {len(X_train)}")
    for c, cname in enumerate(CLASS_NAMES):
        count = int(np.sum(y_train == c))
        print(f"  Class {c} ({cname}): {count} ({count/len(y_train)*100:.2f}%)")

    print(f"\nVal size: {len(X_val)}")
    for c, cname in enumerate(CLASS_NAMES):
        count = int(np.sum(y_val == c))
        print(f"  Class {c} ({cname}): {count} ({count/len(y_val)*100:.2f}%)")

    # Balanced class weights for multi-class
    counts = np.bincount(y_train, minlength=len(CLASS_NAMES))
    weights = np.zeros(len(y_train), dtype=np.float64)
    total = len(y_train)
    n_classes = len(CLASS_NAMES)
    for c in range(n_classes):
        if counts[c] > 0:
            w_c = total / (n_classes * counts[c])
            weights[y_train == c] = w_c

    # Fit CART
    model = CART.fit(
        X_train, y_train, weights, output_dim=len(CLASS_NAMES),
        max_depth=args.max_depth, min_leaf=args.min_leaf
    )

    # Predictions
    with torch.no_grad():
        train_logits = model(torch.from_numpy(X_train))
        train_preds = train_logits.argmax(1).numpy()
        val_logits = model(torch.from_numpy(X_val))
        val_preds = val_logits.argmax(1).numpy()

    def eval_multiclass(y_true, y_pred, label_name):
        print(f"\n--- {label_name} Performance ---")
        acc = np.mean(y_true == y_pred)
        print(f"Overall Accuracy: {acc*100:.2f}%")
        print("Per-class Metrics:")
        for c, cname in enumerate(CLASS_NAMES):
            tp = int(np.sum((y_true == c) & (y_pred == c)))
            fp = int(np.sum((y_true != c) & (y_pred == c)))
            fn = int(np.sum((y_true == c) & (y_pred != c)))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            print(f"  {cname:15s} (c={c}): Precision = {prec:.4f}, Recall = {rec:.4f} (TP={tp}, FP={fp}, FN={fn})")

    eval_multiclass(y_train, train_preds, "Train Set")
    eval_multiclass(y_val, val_preds, "Validation Set (Seeds 3011, 3001, 3201)")

    n_nodes = len(model.feature)
    ckpt_path = args.out / "unified_router_cart.pt"
    payload = {
        "schema": "learned_unified_router_cart_v1",
        "feature_names": FEATURE_NAMES,
        "class_names": CLASS_NAMES,
        "state_dict": model.state_dict(),
        "n_nodes": n_nodes,
        "max_depth": args.max_depth,
        "min_leaf": args.min_leaf,
    }
    torch.save(payload, ckpt_path)
    file_bytes = ckpt_path.stat().st_size
    print(f"\nSaved Unified Router checkpoint to {ckpt_path} ({file_bytes} bytes, {n_nodes} nodes)")

    with open(args.out / "summary.json", "w") as f:
        json.dump(dict(schema=payload["schema"], n_nodes=n_nodes, file_bytes=file_bytes), f, indent=2)

if __name__ == "__main__":
    main()
