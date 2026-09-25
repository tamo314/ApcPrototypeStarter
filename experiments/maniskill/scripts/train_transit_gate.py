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
]

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

def print_tree_text(model, feature_names):
    features = model.feature.numpy()
    thresholds = model.threshold.numpy()
    lefts = model.left.numpy()
    rights = model.right.numpy()
    scores = model.scores.numpy()

    def recurse(node, depth=0):
        indent = "  " * depth
        feat = features[node]
        if feat < 0:
            pred = int(np.argmax(scores[node]))
            prob = np.exp(scores[node])
            prob /= prob.sum()
            print(f"{indent}-> Leaf: predict class {pred} (probs: [0: {prob[0]:.3f}, 1: {prob[1]:.3f}])")
            return
        fname = feature_names[feat]
        thresh = thresholds[node]
        print(f"{indent}if {fname} <= {thresh:.4f}:")
        recurse(lefts[node], depth + 1)
        print(f"{indent}else: ({fname} > {thresh:.4f})")
        recurse(rights[node], depth + 1)

    print("\n--- Learned CART Gate Decision Rules ---")
    recurse(0)

def main():
    parser = argparse.ArgumentParser(description="Train learned CART gate for transit module activation")
    parser.add_argument("--out", type=Path, default=Path("runs/learned_transit_gate_v1"))
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--min-leaf", type=int, default=2)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    X_train, y_train = load_dataset(TRAIN_RUNS)
    X_val, y_val = load_dataset(VAL_RUNS)

    print(f"Train size: {len(X_train)} (positives: {int(np.sum(y_train))})")
    print(f"Val size:   {len(X_val)} (positives: {int(np.sum(y_val))})")

    # Balanced class weights
    n_pos = int(np.sum(y_train))
    n_neg = len(y_train) - n_pos
    w_pos = len(y_train) / (2.0 * n_pos)
    w_neg = len(y_train) / (2.0 * n_neg)
    weights = np.where(y_train == 1, w_pos, w_neg).astype(np.float64)

    # Fit CART
    model = CART.fit(
        X_train, y_train, weights, output_dim=2,
        max_depth=args.max_depth, min_leaf=args.min_leaf
    )

    # Predictions
    with torch.no_grad():
        train_logits = model(torch.from_numpy(X_train))
        train_preds = train_logits.argmax(1).numpy()
        val_logits = model(torch.from_numpy(X_val))
        val_preds = val_logits.argmax(1).numpy()

    def calc_metrics(y_true, y_pred):
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
        acc = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
        return dict(tp=tp, fp=fp, tn=tn, fn=fn, precision=precision, recall=recall, fpr=fpr, fnr=fnr, accuracy=acc)

    train_m = calc_metrics(y_train, train_preds)
    val_m = calc_metrics(y_val, val_preds)

    print("\n--- Train Metrics ---")
    for k, v in train_m.items():
        print(f"  {k}: {v if isinstance(v, int) else f'{v:.4f}'}")

    print("\n--- Val Metrics (Independent Seeds: 3011, 3001, 3201) ---")
    for k, v in val_m.items():
        print(f"  {k}: {v if isinstance(v, int) else f'{v:.4f}'}")

    print_tree_text(model, FEATURE_NAMES)

    n_nodes = len(model.feature)
    ckpt_path = args.out / "transit_gate_cart.pt"
    payload = {
        "schema": "learned_transit_gate_cart_v1",
        "feature_names": FEATURE_NAMES,
        "state_dict": model.state_dict(),
        "n_nodes": n_nodes,
        "max_depth": args.max_depth,
        "min_leaf": args.min_leaf,
        "train_metrics": train_m,
        "val_metrics": val_m,
    }
    torch.save(payload, ckpt_path)
    file_bytes = ckpt_path.stat().st_size
    print(f"\nSaved learned gate checkpoint to {ckpt_path} ({file_bytes} bytes, {n_nodes} nodes)")

    with open(args.out / "summary.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)

if __name__ == "__main__":
    main()
