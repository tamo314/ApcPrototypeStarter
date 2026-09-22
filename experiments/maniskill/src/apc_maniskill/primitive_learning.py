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


SCHEMA = "fetch_primitive_features_v1"


def features(state):
    fields = ("measured_hand_position", "measured_hand_quaternion", "cube_position",
              "goal_position", "hand_target_position", "hand_target_quaternion",
              "root_rotation")
    values = [np.asarray(state[name], dtype=np.float32).reshape(-1) for name in fields]
    values += [np.asarray([state["gripper_target_m"], min(state["target_age_steps"], 40) / 40,
                           float(state["grasped"])], dtype=np.float32),
               np.eye(16, dtype=np.float32)[state["selected_id"]]]
    return np.concatenate(values)


def network(input_dim):
    return nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(),
                         nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 16))


def train(source: Path, output: Path, *, updates=3000, seed=0, dagger_source: Path | None = None,
          sampling="uniform"):
    if updates < 1:
        raise ValueError("updates must be positive")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / "primitive_learning.py")
    shutil.copy2(source / "manifest.json", output / "source_manifest.json")
    steps = source / "steps.jsonl"
    digest = hashlib.sha256(steps.read_bytes()).hexdigest()
    rows = [json.loads(line) for line in steps.read_text(encoding="utf-8").splitlines()]
    dagger_rows = []
    dagger_digest = None
    source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    names = source_manifest["policy_details"]["primitive_names"]
    if names != list(NAMES) or source_manifest["status"] != "completed":
        raise ValueError("Training requires a completed fetch16 teacher run")
    if dagger_source is not None:
        dagger_manifest = json.loads((dagger_source / "manifest.json").read_text(encoding="utf-8"))
        if (dagger_manifest["status"] != "completed" or dagger_manifest["task"] != source_manifest["task"]
                or dagger_manifest["policy_details"]["primitive_names"] != names):
            raise ValueError("DAgger source task or primitive schema mismatch")
        dagger_steps = dagger_source / "steps.jsonl"
        dagger_digest = hashlib.sha256(dagger_steps.read_bytes()).hexdigest()
        dagger_rows = [json.loads(line) for line in dagger_steps.read_text(encoding="utf-8").splitlines()]
        if not all(row["info"]["diagnostic"].get("teacher_label_valid") for row in dagger_rows):
            raise ValueError("DAgger rows require explicit valid teacher labels")
    episodes = sorted({row["episode"] for row in rows})
    if len(episodes) < 2:
        raise ValueError("Need at least two episodes for episode-level validation")
    x = np.stack([features(row["info"]["diagnostic"]["pre_action_state"])
                  for row in rows + dagger_rows])
    y = np.asarray([row["info"]["diagnostic"]["proposed_id"] for row in rows]
                   + [row["info"]["diagnostic"]["teacher_id"] for row in dagger_rows], dtype=np.int64)
    train_mask = np.asarray([row["episode"] != episodes[-1] for row in rows]
                            + [True] * len(dagger_rows))
    mean = x[train_mask].mean(0)
    std = np.maximum(x[train_mask].std(0), 1e-4)
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
        validation_logits = model(xv)
        validation_loss = float(nn.functional.cross_entropy(validation_logits, yv))
        validation_accuracy = float((validation_logits.argmax(1) == yv).float().mean())
    checkpoint = dict(schema=SCHEMA, primitive_names=list(NAMES),
                      task=source_manifest["task"], control_freq=source_manifest["control_freq"],
                      state_dict=model.state_dict(), mean=mean, std=std,
                      source_steps_sha256=digest, source=str(source.resolve()),
                      dagger_steps_sha256=dagger_digest,
                      train_episodes=episodes[:-1], validation_episode=episodes[-1],
                      updates=updates, seed=seed)
    checkpoint["sampling"] = sampling
    torch.save(checkpoint, output / "selector.pt")
    json_write(output / "training.json", dict(source=str(source.resolve()),
               source_steps_sha256=digest, dagger_steps_sha256=dagger_digest,
               source_environment_steps=sum(row["steps"] for row in
                   [json.loads(line) for line in (source / "episodes.jsonl").read_text().splitlines()]),
               dagger_environment_steps=sum(row["steps"] for row in
                   [json.loads(line) for line in (dagger_source / "episodes.jsonl").read_text().splitlines()])
                   if dagger_source is not None else 0,
               dagger_samples=len(dagger_rows),
               training_samples=int(train_mask.sum()), validation_samples=int((~train_mask).sum()),
               train_episodes=episodes[:-1], validation_episode=episodes[-1],
               updates=updates, seed=seed, validation_loss=validation_loss,
               sampling=sampling,
               validation_accuracy=validation_accuracy,
               model_parameters=sum(p.numel() for p in model.parameters()),
               label_counts=np.bincount(y, minlength=16).tolist()))


class MLPSelector:
    def __init__(self, checkpoint: Path, task, control_freq):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (saved["schema"] != SCHEMA or saved["primitive_names"] != list(NAMES)
                or saved["task"] != task or saved["control_freq"] != control_freq):
            raise ValueError("Primitive selector checkpoint schema or task mismatch")
        self.model = network(len(saved["mean"]))
        self.model.load_state_dict(saved["state_dict"])
        self.model.eval()
        torch.set_num_threads(1)
        self.mean = saved["mean"]
        self.std = saved["std"]
        self.metadata = dict(learned=True, selector="mlp64x2_v1", selector_schema=SCHEMA,
                             selector_parameters=sum(p.numel() for p in self.model.parameters()),
                             checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                             source_steps_sha256=saved["source_steps_sha256"],
                             training_updates=saved["updates"])
        self.last_scores = []

    def reset(self):
        self.last_scores = []

    def select(self, step, observation):
        x = torch.from_numpy(((features(observation) - self.mean) / self.std).astype(np.float32))
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
    parser.add_argument("--dagger-source", type=Path)
    parser.add_argument("--sampling", choices=["uniform", "sqrt_inverse"], default="uniform")
    args = parser.parse_args()
    train(args.source, args.out, updates=args.updates, seed=args.seed,
          dagger_source=args.dagger_source, sampling=args.sampling)


if __name__ == "__main__":
    main()
