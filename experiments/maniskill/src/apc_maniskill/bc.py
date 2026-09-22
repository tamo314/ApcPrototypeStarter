"""Small CPU behavior-cloning baseline; hand-designed teacher, learned base only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from torch import nn

FEATURE_SCHEMA = "fetch-state40-relative-goal-body-velocity-v1"
ENV_ID = "APC-FetchReachGoal-v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def features(observations: np.ndarray) -> np.ndarray:
    """State40: qpos15, qvel15, goal2, relative2, base_pose3, base_velocity3."""
    obs = np.asarray(observations, dtype=np.float32)
    if obs.shape[-1] != 40 or not np.isfinite(obs).all():
        raise ValueError("BC requires finite FetchReach state40 observations")
    yaw = obs[..., 36]
    c, s = np.cos(yaw), np.sin(yaw)
    vx, vy = obs[..., 37], obs[..., 38]
    return np.stack((obs[..., 32], obs[..., 33], c * vx + s * vy,
                     -s * vx + c * vy, obs[..., 39]), axis=-1)


def make_model() -> nn.Module:
    return nn.Sequential(nn.Linear(5, 32), nn.Tanh(), nn.Linear(32, 32),
                         nn.Tanh(), nn.Linear(32, 2), nn.Tanh())


def load_demonstrations(path: Path):
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    if (manifest["status"] != "completed" or config["env_id"] != ENV_ID
            or config["policy"] != "fetch_goal" or config["robot_uids"] != "fetch"
            or config["control_mode"] != "pd_joint_delta_pos"
            or manifest["task"]["task_version"] != 1
            or manifest["action_shape"] != [13]
            or manifest["task"]["action_mapping"]["base"] != [11, 13]):
        raise ValueError("Expected a completed v1 FetchReach scripted teacher run")
    rows = [json.loads(line) for line in (path / "episodes.jsonl").read_text().splitlines()]
    if not rows or len(rows) != manifest["completed_episodes"] or len(rows) != config["episodes"]:
        raise ValueError("Incomplete teacher episode accounting")
    xs, ys, hashes = [], [], {}
    for row in rows:
        trajectory = (path / row["trajectory"]).resolve()
        if trajectory.parent != path.resolve():
            raise ValueError("Trajectory must be inside the teacher run")
        with np.load(trajectory, allow_pickle=False) as data:
            obs, actions = data["observations"], data["actions"]
            count = row["steps"]
            if obs.shape != (count + 1, 1, 40) or actions.shape != (count, 13) or count < 1:
                raise ValueError("Teacher observation/action alignment or shape is invalid")
            if not all(np.isfinite(data[key]).all() for key in data.files):
                raise ValueError("Nonfinite teacher trajectory")
            if (np.abs(actions) > 1).any():
                raise ValueError("Teacher action outside normalized bounds")
            # Pair the PRE-action observation with the action actually applied.
            xs.append(features(obs[:-1, 0]))
            ys.append(actions[:, 11:13].astype(np.float32))
        hashes[trajectory.name] = sha256(trajectory)
    source = dict(run=str(path.resolve()), config=config, seeds=[r["seed"] for r in rows],
                  manifest_sha256=sha256(path / "manifest.json"), trajectory_sha256=hashes,
                  control_freq=manifest["control_freq"], task=manifest["task"])
    return np.concatenate(xs), np.concatenate(ys), source


def train(demo_run: Path, output: Path, *, updates=1000, seed=0) -> Path:
    from .runner import json_write, provenance, snapshot_source, utc_now

    if type(updates) is not int or updates < 1 or not 0 <= seed < 2**32:
        raise ValueError("Positive updates and a uint32 seed are required")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(status="running", started_at=utc_now(), provenance=provenance(),
                  algorithm="behavior_cloning", teacher="hand-designed fetch_goal",
                  feature_schema=FEATURE_SCHEMA, device="cpu", torch_threads=1,
                  seed=seed, requested_updates=updates, completed_updates=0,
                  batch_size=64, learning_rate=0.001, loss="uniform action MSE",
                  architecture=[5, 32, 32, 2], environment_steps_during_training=0)
    json_write(output / "training.json", report)
    try:
        snapshot_source(output)
        x, y, source = load_demonstrations(demo_run)
        report.update(source=source, samples=len(x), source_environment_steps=len(x))
        torch.set_num_threads(1)
        torch.manual_seed(seed)
        model = make_model()
        report["parameter_count"] = sum(p.numel() for p in model.parameters())
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        mean, scale = x.mean(dim=0), x.std(dim=0, unbiased=False).clamp_min(0.05)
        x = (x - mean) / scale
        optimizer = torch.optim.Adam(model.parameters(), lr=report["learning_rate"])
        with torch.no_grad():
            report["initial_train_mse"] = float(nn.functional.mse_loss(model(x), y))
        json_write(output / "training.json", report)
        with (output / "losses.jsonl").open("x", encoding="utf-8") as log:
            for update in range(1, updates + 1):
                indices = torch.randint(len(x), (report["batch_size"],))
                loss = nn.functional.mse_loss(model(x[indices]), y[indices])
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                report["completed_updates"] = update
                if update == 1 or update % 100 == 0 or update == updates:
                    log.write(json.dumps(dict(update=update, minibatch_mse=loss.item())) + "\n")
                    log.flush()
        model.eval()
        with torch.no_grad():
            report["final_train_mse"] = float(nn.functional.mse_loss(model(x), y))
        if not np.isfinite(report["final_train_mse"]):
            raise ValueError("Nonfinite final training loss")
        checkpoint = dict(format_version=1, feature_schema=FEATURE_SCHEMA,
                          model_state=model.state_dict(), mean=mean, scale=scale,
                          env_id=ENV_ID, control_mode="pd_joint_delta_pos",
                          control_freq=source["control_freq"], task=source["task"],
                          training_seeds=source["seeds"], training_source=source["run"],
                          training_seed=seed, updates=updates)
        torch.save(checkpoint, output / "policy.pt")
        report.update(status="completed", checkpoint="policy.pt",
                      checkpoint_sha256=sha256(output / "policy.pt"))
    except BaseException as exc:
        report.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      error=f"{type(exc).__name__}: {exc}")
        (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        report.update(finished_at=utc_now(), wall_seconds=time.monotonic() - started)
        json_write(output / "training.json", report)
    return output


class BCPolicy:
    def __init__(self, path: Path, task: dict, control_freq: float):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if (checkpoint["format_version"] != 1 or checkpoint["feature_schema"] != FEATURE_SCHEMA
                or checkpoint["env_id"] != ENV_ID
                or checkpoint["control_mode"] != "pd_joint_delta_pos"
                or checkpoint["control_freq"] != control_freq
                or checkpoint["task"] != json.loads(json.dumps(task))):
            raise ValueError("Checkpoint does not match the current FetchReach task")
        torch.set_num_threads(1)
        self.model = make_model()
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.mean, self.scale = checkpoint["mean"], checkpoint["scale"]
        if (self.mean.shape != (5,) or self.scale.shape != (5,)
                or not torch.isfinite(self.mean).all() or not torch.isfinite(self.scale).all()
                or not (self.scale > 0).all()
                or not all(torch.isfinite(p).all() for p in self.model.parameters())):
            raise ValueError("Invalid checkpoint normalization or parameters")
        self.metadata = dict(source="learned behavior cloning; shared hand-designed posture servo",
                             checkpoint_path=str(path.resolve()), checkpoint_sha256=sha256(path),
                             feature_schema=FEATURE_SCHEMA,
                             training_source=checkpoint["training_source"],
                             training_seeds=checkpoint["training_seeds"],
                             updates=checkpoint["updates"],
                             parameter_count=sum(p.numel() for p in self.model.parameters()))

    def predict(self, observations: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(features(observations))
        with torch.inference_mode():
            return self.model((x - self.mean) / self.scale).numpy()
