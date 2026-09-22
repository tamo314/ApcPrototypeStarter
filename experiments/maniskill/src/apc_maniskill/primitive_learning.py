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

from .primitives import NAMES
from .runner import json_write


SCHEMA_V1 = "fetch_primitive_features_v1"
SCHEMA_V2 = "fetch_primitive_relative_features_v2"


def features(state, schema=SCHEMA_V1):
    if schema == SCHEMA_V1:
        fields = ("measured_hand_position", "measured_hand_quaternion", "cube_position",
                  "goal_position", "hand_target_position", "hand_target_quaternion",
                  "root_rotation")
        values = [np.asarray(state[name], dtype=np.float32).reshape(-1) for name in fields]
    elif schema == SCHEMA_V2:
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
    values += [np.asarray([state["gripper_target_m"], min(state["target_age_steps"], 40) / 40,
                           float(state["grasped"])], dtype=np.float32),
               np.eye(16, dtype=np.float32)[state["selected_id"]]]
    return np.concatenate(values)


def network(input_dim):
    return nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(),
                         nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 16))


def train(source: Path, output: Path, *, updates=3000, seed=0,
          dagger_source: Path | list[Path] | None = None,
          sampling="uniform", std_floor=1e-4, feature_schema=SCHEMA_V1,
          train_all=False, extra_teacher_sources: list[Path] | None = None):
    if updates < 1:
        raise ValueError("updates must be positive")
    if not np.isfinite(std_floor) or std_floor <= 0:
        raise ValueError("std_floor must be finite and positive")
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
    dagger_digests = []
    source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    names = source_manifest["policy_details"]["primitive_names"]
    if names != list(NAMES) or source_manifest["status"] != "completed":
        raise ValueError("Training requires a completed fetch16 teacher run")
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
    for dagger_path in dagger_sources:
        dagger_manifest = json.loads((dagger_path / "manifest.json").read_text(encoding="utf-8"))
        if (dagger_manifest["status"] != "completed" or dagger_manifest["task"] != source_manifest["task"]
                or dagger_manifest["policy_details"]["primitive_names"] != names):
            raise ValueError("DAgger source task or primitive schema mismatch")
        dagger_steps = dagger_path / "steps.jsonl"
        dagger_digests.append(hashlib.sha256(dagger_steps.read_bytes()).hexdigest())
        new_rows = [json.loads(line) for line in dagger_steps.read_text(encoding="utf-8").splitlines()]
        if not all(row["info"]["diagnostic"].get("teacher_label_valid") for row in new_rows):
            raise ValueError("DAgger rows require explicit valid teacher labels")
        dagger_rows.extend(new_rows)
    episodes = sorted({row["episode"] for row in rows[:primary_rows]})
    if len(episodes) < 2:
        raise ValueError("Need at least two episodes for episode-level validation")
    x = np.stack([features(row["info"]["diagnostic"]["pre_action_state"], feature_schema)
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
    model = network(x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    xt = torch.from_numpy(x[train_mask])
    yt = torch.from_numpy(y[train_mask])
    xv = torch.from_numpy(x[~train_mask])
    yv = torch.from_numpy(y[~train_mask])
    rng = np.random.default_rng(seed)
    if sampling not in ("uniform", "sqrt_inverse"):
        raise ValueError("Unknown sampling rule")
    probabilities = None
    if sampling == "sqrt_inverse":
        counts = np.bincount(y[train_mask], minlength=16)
        probabilities = 1 / np.sqrt(counts[y[train_mask]])
        probabilities = probabilities / probabilities.sum()
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
    checkpoint = dict(schema=feature_schema, primitive_names=list(NAMES),
                      task=source_manifest["task"], control_freq=source_manifest["control_freq"],
                      state_dict=model.state_dict(), mean=mean, std=std,
                      source_steps_sha256=digest, source=str(source.resolve()),
                      teacher_steps_sha256=teacher_digests,
                      dagger_steps_sha256=dagger_digests,
                      train_episodes=episodes if train_all else episodes[:-1],
                      validation_episode=None if train_all else episodes[-1],
                      updates=updates, seed=seed)
    checkpoint["sampling"] = sampling
    checkpoint["std_floor"] = std_floor
    torch.save(checkpoint, output / "selector.pt")
    json_write(output / "training.json", dict(source=str(source.resolve()),
               source_steps_sha256=digest, dagger_steps_sha256=dagger_digests,
               teacher_sources=[str(path.resolve()) for path in teacher_paths],
               teacher_steps_sha256=teacher_digests,
               dagger_sources=[str(path.resolve()) for path in dagger_sources],
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
               label_counts=np.bincount(y, minlength=16).tolist()))


class MLPSelector:
    def __init__(self, checkpoint: Path, task, control_freq):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (saved["schema"] not in (SCHEMA_V1, SCHEMA_V2) or saved["primitive_names"] != list(NAMES)
                or saved["task"] != task or saved["control_freq"] != control_freq):
            raise ValueError("Primitive selector checkpoint schema or task mismatch")
        self.model = network(len(saved["mean"]))
        self.model.load_state_dict(saved["state_dict"])
        self.model.eval()
        torch.set_num_threads(1)
        self.schema = saved["schema"]
        self.mean = saved["mean"]
        self.std = saved["std"]
        self.metadata = dict(learned=True, selector="mlp64x2_v1", selector_schema=self.schema,
                             selector_parameters=sum(p.numel() for p in self.model.parameters()),
                             checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                             source_steps_sha256=saved["source_steps_sha256"],
                             training_updates=saved["updates"])
        self.last_scores = []

    def reset(self):
        self.last_scores = []

    def select(self, step, observation):
        x = torch.from_numpy(((features(observation, self.schema) - self.mean) / self.std).astype(np.float32))
        with torch.no_grad():
            logits = self.model(x)
        self.last_scores = logits.tolist()
        return int(logits.argmax())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dagger-source", type=Path, action="append")
    parser.add_argument("--extra-teacher-source", type=Path, action="append")
    parser.add_argument("--sampling", choices=["uniform", "sqrt_inverse"], default="uniform")
    parser.add_argument("--std-floor", type=float, default=1e-4)
    parser.add_argument("--feature-schema", choices=[SCHEMA_V1, SCHEMA_V2], default=SCHEMA_V1)
    parser.add_argument("--train-all", action="store_true")
    args = parser.parse_args()
    train(args.source, args.out, updates=args.updates, seed=args.seed,
          dagger_source=args.dagger_source, sampling=args.sampling,
          std_floor=args.std_floor, feature_schema=args.feature_schema,
          train_all=args.train_all, extra_teacher_sources=args.extra_teacher_source)


if __name__ == "__main__":
    main()
