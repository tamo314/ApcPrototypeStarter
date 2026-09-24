"""Small supervised selector for recorded one-step primitive choices."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from torch import nn

from .primitives import NAMES, NAMES20
from .runner import json_write


SCHEMA_V1 = "fetch_primitive_features_v1"
SCHEMA_V2 = "fetch_primitive_relative_features_v2"
SCHEMA_V3 = "fetch_primitive_relative_base_features_v3"
SCHEMA_V4 = "fetch_primitive_base_ready_features_v4"
SCHEMA_V5 = "fetch_primitive_geometry_features_v5"
SCHEMA_V6 = "fetch_primitive_geometry_no_history_v6"


def features(state, schema=SCHEMA_V1, primitive_count=16):
    if schema == SCHEMA_V1:
        fields = ("measured_hand_position", "measured_hand_quaternion", "cube_position",
                  "goal_position", "hand_target_position", "hand_target_quaternion",
                  "root_rotation")
        values = [np.asarray(state[name], dtype=np.float32).reshape(-1) for name in fields]
    elif schema in (SCHEMA_V2, SCHEMA_V3, SCHEMA_V4, SCHEMA_V5, SCHEMA_V6):
        hand = np.asarray(state["measured_hand_position"], dtype=np.float32)
        cube = np.asarray(state["cube_position"], dtype=np.float32)
        goal = np.asarray(state["goal_position"], dtype=np.float32)
        target = np.asarray(state["hand_target_position"], dtype=np.float32)
        root = np.asarray(state["root_rotation"], dtype=np.float32)
        values = [root.T @ (cube - hand), root.T @ (goal - cube), root.T @ (target - hand),
                  np.asarray(state["measured_hand_quaternion"], dtype=np.float32),
                  np.asarray(state["hand_target_quaternion"], dtype=np.float32),
                  np.asarray([cube[2] - state["cube_initial_z"], hand[2] - cube[2]], dtype=np.float32)]
    else:
        raise ValueError("Unknown primitive feature schema")
    if schema in (SCHEMA_V3, SCHEMA_V4, SCHEMA_V5, SCHEMA_V6):
        if primitive_count != 20:
            raise ValueError("Base feature schema requires the 20-ID bank")
        base = np.asarray(state["base_pose"], dtype=np.float32)
        base_target = np.asarray(state["base_target"], dtype=np.float32)
        values += [base, base_target - base,
                   np.asarray([state["base_target_age_steps"] / 40,
                               state["mode_code"], state["hand_target_valid"]], dtype=np.float32)]
        if schema in (SCHEMA_V4, SCHEMA_V5, SCHEMA_V6):
            ready = (state["mode_code"] == 1 and state["base_target_age_steps"] >= 15
                     and np.linalg.norm(base_target - base) < 0.001)
            values += [np.asarray([float(ready), float(base[0] >= 0.195)], dtype=np.float32)]
    if schema in (SCHEMA_V5, SCHEMA_V6):
        # Observable geometry only: no teacher call, stage, desired ID or action mask.
        # The 12 mm grasp offset and 12 cm approach height are task priors shared
        # with the demonstration policy, recorded by this versioned schema.
        q = np.asarray(state["measured_hand_quaternion"], dtype=np.float32)
        target_q = np.asarray(state["hand_target_quaternion"], dtype=np.float32)
        pending_angle = 2 * np.arccos(np.clip(abs(np.dot(q, target_q)), 0, 1))
        offsets = [cube - hand + [0, 0, .012], cube - hand + [0, 0, .12], goal - cube]
        values += [np.asarray([2 * np.arctan2(q[2], q[0]), pending_angle,
                               np.linalg.norm(target - hand),
                               np.linalg.norm((cube - hand)[:2]),
                               np.linalg.norm(offsets[0]), np.linalg.norm(goal - cube),
                               state["position_tolerance_m"]], dtype=np.float32)]
        for delta in offsets:
            delta_root = root.T @ delta
            magnitude = np.abs(delta_root)
            values += [delta_root, magnitude,
                       magnitude[[0, 0, 1]] - magnitude[[1, 2, 2]]]
    values += [np.asarray([state["gripper_target_m"], min(state["target_age_steps"], 40) / 40,
                           float(state["grasped"])], dtype=np.float32)]
    if schema != SCHEMA_V6:
        values += [np.eye(primitive_count, dtype=np.float32)[state["selected_id"]]]
    return np.concatenate(values).astype(np.float32)


def network(input_dim, output_dim=16):
    return nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(),
                         nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, output_dim))


def train(source: Path, output: Path, *, updates=3000, seed=0,
          dagger_source: Path | list[Path] | None = None,
          sampling="uniform", std_floor=1e-4, feature_schema=SCHEMA_V1,
          train_all=False, extra_teacher_sources: list[Path] | None = None,
          dagger_strides: list[int] | None = None, model_kind="mlp",
          max_depth=12, min_leaf=2, partition_grasp=False):
    if updates < 1:
        raise ValueError("updates must be positive")
    if not np.isfinite(std_floor) or std_floor <= 0:
        raise ValueError("std_floor must be finite and positive")
    if model_kind not in ("mlp", "cart"):
        raise ValueError("Unknown selector model kind")
    if partition_grasp and model_kind != "cart":
        raise ValueError("Physical grasp partition requires CART")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / "primitive_learning.py")
    shutil.copy2(source / "manifest.json", output / "source_manifest.json")
    steps = source / "steps.jsonl"
    digest = hashlib.sha256(steps.read_bytes()).hexdigest()
    rows = [json.loads(line) for line in steps.read_text(encoding="utf-8").splitlines()]
    primary_rows = len(rows)
    teacher_paths = [source] + list(extra_teacher_sources or [])
    teacher_digests = [digest]
    dagger_rows = []
    dagger_sources = ([] if dagger_source is None else
                      [dagger_source] if isinstance(dagger_source, Path) else list(dagger_source))
    dagger_strides = [1] * len(dagger_sources) if dagger_strides is None else dagger_strides
    if len(dagger_strides) != len(dagger_sources) or any(stride < 1 for stride in dagger_strides):
        raise ValueError("Specify one positive stride per DAgger source")
    dagger_digests = []
    source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    names = source_manifest["policy_details"]["primitive_names"]
    if names not in (list(NAMES), list(NAMES20)) or source_manifest["status"] != "completed":
        raise ValueError("Training requires a completed fetch16 or fetch20 teacher run")
    if len(names) == 20 and feature_schema not in (SCHEMA_V3, SCHEMA_V4, SCHEMA_V5, SCHEMA_V6):
        raise ValueError("The 20-ID bank requires base-state features")
    if len(names) == 16 and feature_schema in (SCHEMA_V3, SCHEMA_V4, SCHEMA_V5, SCHEMA_V6):
        raise ValueError("Base-state features require the 20-ID bank")
    for index, teacher_path in enumerate(teacher_paths[1:], start=1):
        teacher_manifest = json.loads((teacher_path / "manifest.json").read_text(encoding="utf-8"))
        if (teacher_manifest["status"] != "completed" or teacher_manifest["task"] != source_manifest["task"]
                or teacher_manifest["control_freq"] != source_manifest["control_freq"]
                or teacher_manifest["policy_details"]["primitive_names"] != names
                or teacher_manifest["policy_details"]["learned"]):
            raise ValueError("Additional teacher source task or primitive schema mismatch")
        shutil.copy2(teacher_path / "manifest.json", output / f"teacher_manifest_{index:02d}.json")
        teacher_steps = teacher_path / "steps.jsonl"
        teacher_digests.append(hashlib.sha256(teacher_steps.read_bytes()).hexdigest())
        rows.extend(json.loads(line) for line in teacher_steps.read_text(encoding="utf-8").splitlines())
    for dagger_path, stride in zip(dagger_sources, dagger_strides):
        dagger_manifest = json.loads((dagger_path / "manifest.json").read_text(encoding="utf-8"))
        if (dagger_manifest["status"] != "completed" or dagger_manifest["task"] != source_manifest["task"]
                or dagger_manifest["policy_details"]["primitive_names"] != names):
            raise ValueError("DAgger source task or primitive schema mismatch")
        dagger_steps = dagger_path / "steps.jsonl"
        dagger_digests.append(hashlib.sha256(dagger_steps.read_bytes()).hexdigest())
        new_rows = [json.loads(line) for line in dagger_steps.read_text(encoding="utf-8").splitlines()]
        if not all(row["info"]["diagnostic"].get("teacher_label_valid") for row in new_rows):
            raise ValueError("DAgger rows require explicit valid teacher labels")
        dagger_rows.extend(row for row in new_rows if row["step"] % stride == 0)
    episodes = sorted({row["episode"] for row in rows[:primary_rows]})
    if len(episodes) < 2:
        raise ValueError("Need at least two episodes for episode-level validation")
    x = np.stack([features(row["info"]["diagnostic"]["pre_action_state"], feature_schema, len(names))
                  for row in rows + dagger_rows])
    y = np.asarray([row["info"]["diagnostic"]["proposed_id"] for row in rows]
                   + [row["info"]["diagnostic"]["teacher_id"] for row in dagger_rows], dtype=np.int64)
    train_mask = np.asarray([train_all or row["episode"] != episodes[-1]
                             for row in rows[:primary_rows]]
                            + [True] * (len(rows) - primary_rows + len(dagger_rows)))
    mean = x[train_mask].mean(0)
    std = np.maximum(x[train_mask].std(0), std_floor)
    x = (x - mean) / std
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    xt = torch.from_numpy(x[train_mask])
    yt = torch.from_numpy(y[train_mask])
    xv = torch.from_numpy(x[~train_mask])
    yv = torch.from_numpy(y[~train_mask])
    rng = np.random.default_rng(seed)
    if sampling not in ("uniform", "sqrt_inverse", "fourth_root_inverse"):
        raise ValueError("Unknown sampling rule")
    probabilities = None
    if sampling in ("sqrt_inverse", "fourth_root_inverse"):
        counts = np.bincount(y[train_mask], minlength=len(names))
        exponent = 0.5 if sampling == "sqrt_inverse" else 0.25
        probabilities = 1 / np.power(counts[y[train_mask]], exponent)
        probabilities = probabilities / probabilities.sum()
    if model_kind == "cart":
        from .primitive_tree import CART
        shutil.copy2(Path(__file__).with_name("primitive_tree.py"), output / "primitive_tree.py")
        weights = np.ones(len(xt)) if probabilities is None else probabilities * len(xt)
        model = CART.fit(xt.numpy(), yt.numpy(), weights, len(names),
                         max_depth=max_depth, min_leaf=min_leaf,
                         root_feature=(x.shape[1] - (1 if feature_schema == SCHEMA_V6 else len(names) + 1))
                         if partition_grasp else None)
        updates = 0  # One tree fit; no gradient optimizer updates.
    else:
        model = network(x.shape[1], len(names))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(updates):
            sampled = (rng.integers(0, len(xt), 128) if probabilities is None
                       else rng.choice(len(xt), 128, replace=True, p=probabilities))
            indices = torch.from_numpy(sampled)
            loss = nn.functional.cross_entropy(model(xt[indices]), yt[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        training_accuracy = float((model(xt).argmax(1) == yt).float().mean())
        validation_loss = None
        validation_accuracy = None
        if len(xv):
            validation_logits = model(xv)
            validation_loss = float(nn.functional.cross_entropy(validation_logits, yv))
            validation_accuracy = float((validation_logits.argmax(1) == yv).float().mean())
    checkpoint = dict(schema=feature_schema, primitive_names=names,
                      task=source_manifest["task"], control_freq=source_manifest["control_freq"],
                      state_dict=model.state_dict(), mean=mean, std=std,
                      source_steps_sha256=digest, source=str(source.resolve()),
                      teacher_steps_sha256=teacher_digests,
                      dagger_steps_sha256=dagger_digests,
                      dagger_source_strides=dagger_strides,
                      train_episodes=episodes if train_all else episodes[:-1],
                      validation_episode=None if train_all else episodes[-1],
                      updates=updates, seed=seed)
    checkpoint["sampling"] = sampling
    checkpoint["std_floor"] = std_floor
    model_details = dict(model_kind=model_kind,
                         partition_grasp=partition_grasp,
                         tree_nodes=len(model.feature) if model_kind == "cart" else None,
                         tree_max_depth=max_depth if model_kind == "cart" else None,
                         tree_min_leaf=min_leaf if model_kind == "cart" else None,
                         tree_fits=int(model_kind == "cart"))
    checkpoint.update(model_details)
    torch.save(checkpoint, output / "selector.pt")
    json_write(output / "training.json", dict(source=str(source.resolve()),
               source_steps_sha256=digest, dagger_steps_sha256=dagger_digests,
               teacher_sources=[str(path.resolve()) for path in teacher_paths],
               teacher_steps_sha256=teacher_digests,
               dagger_sources=[str(path.resolve()) for path in dagger_sources],
               dagger_source_strides=dagger_strides,
               source_environment_steps=sum(row["steps"] for path in teacher_paths for row in
                   [json.loads(line) for line in (path / "episodes.jsonl").read_text().splitlines()]),
               dagger_environment_steps=sum(row["steps"] for path in dagger_sources for row in
                   [json.loads(line) for line in (path / "episodes.jsonl").read_text().splitlines()]),
               dagger_samples=len(dagger_rows),
               training_samples=int(train_mask.sum()), validation_samples=int((~train_mask).sum()),
               train_episodes=episodes if train_all else episodes[:-1],
               validation_episode=None if train_all else episodes[-1],
               updates=updates, seed=seed, training_accuracy=training_accuracy,
               validation_loss=validation_loss,
               sampling=sampling, std_floor=std_floor, feature_schema=feature_schema,
               validation_accuracy=validation_accuracy,
               model_parameters=sum(p.numel() for p in model.parameters()),
               model_stored_values=sum(t.numel() for t in model.state_dict().values()),
               **model_details,
               label_counts=np.bincount(y, minlength=len(names)).tolist()))


class LearnedSelector:
    def __init__(self, checkpoint: Path, task, control_freq, strict_task: bool = True):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (saved["schema"] not in (SCHEMA_V1, SCHEMA_V2, SCHEMA_V3, SCHEMA_V4, SCHEMA_V5, SCHEMA_V6)
                or saved["primitive_names"] not in (list(NAMES), list(NAMES20))
                or (strict_task and saved["task"] != task) or saved["control_freq"] != control_freq):
            raise ValueError("Primitive selector checkpoint schema or task mismatch")

        self.primitive_count = len(saved["primitive_names"])
        if (saved["schema"] in (SCHEMA_V3, SCHEMA_V4, SCHEMA_V5, SCHEMA_V6)) != (self.primitive_count == 20):
            raise ValueError("Primitive bank and feature schema mismatch")
        self.model_kind = saved.get("model_kind", "mlp")
        if self.model_kind == "cart":
            from .primitive_tree import CART
            self.model = CART(saved["tree_nodes"], self.primitive_count)
        elif self.model_kind == "mlp":
            self.model = network(len(saved["mean"]), self.primitive_count)
        else:
            raise ValueError("Unknown checkpoint model kind")
        self.model.load_state_dict(saved["state_dict"])
        self.model.eval()
        torch.set_num_threads(1)
        self.schema = saved["schema"]
        self.mean = saved["mean"]
        self.std = saved["std"]
        self.metadata = dict(learned=True, selector=("cart_v1" if self.model_kind == "cart" else "mlp64x2_v1"),
                             selector_schema=self.schema, selector_model_kind=self.model_kind,
                             selector_parameters=sum(p.numel() for p in self.model.parameters()),
                             selector_stored_values=sum(t.numel() for t in self.model.state_dict().values()),
                             tree_nodes=saved.get("tree_nodes"), tree_fits=saved.get("tree_fits", 0),
                             partition_grasp=saved.get("partition_grasp", False),
                             checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                             source_steps_sha256=saved["source_steps_sha256"],
                             training_updates=saved["updates"])
        self.last_scores = []

    def reset(self):
        self.last_scores = []

    def select(self, step, observation):
        x = torch.from_numpy(((features(observation, self.schema, self.primitive_count)
                               - self.mean) / self.std).astype(np.float32))
        with torch.no_grad():
            logits = self.model(x)
        self.last_scores = logits.tolist()
        return int(logits.argmax())


# Preserve the existing public import used by earlier scripts and saved sources.
MLPSelector = LearnedSelector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dagger-source", type=Path, action="append")
    parser.add_argument("--dagger-stride", type=int, action="append")
    parser.add_argument("--extra-teacher-source", type=Path, action="append")
    parser.add_argument("--sampling", choices=["uniform", "sqrt_inverse", "fourth_root_inverse"], default="uniform")
    parser.add_argument("--std-floor", type=float, default=1e-4)
    parser.add_argument("--feature-schema", choices=[SCHEMA_V1, SCHEMA_V2, SCHEMA_V3, SCHEMA_V4, SCHEMA_V5, SCHEMA_V6], default=SCHEMA_V1)
    parser.add_argument("--train-all", action="store_true")
    parser.add_argument("--model-kind", choices=["mlp", "cart"], default="mlp")
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--min-leaf", type=int, default=2)
    parser.add_argument("--partition-grasp", action="store_true",
                        help="CART structural prior: split measured grasp state before fitting decisions")
    args = parser.parse_args()
    train(args.source, args.out, updates=args.updates, seed=args.seed,
          dagger_source=args.dagger_source, sampling=args.sampling,
          std_floor=args.std_floor, feature_schema=args.feature_schema,
          train_all=args.train_all, extra_teacher_sources=args.extra_teacher_source,
          dagger_strides=args.dagger_stride, model_kind=args.model_kind,
          max_depth=args.max_depth, min_leaf=args.min_leaf, partition_grasp=args.partition_grasp)


if __name__ == "__main__":
    main()
