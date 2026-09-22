"""Small CPU behavior-cloning baseline for scripted Fetch pick-and-place."""
from __future__ import annotations

import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from torch import nn

from .bc import sha256

FEATURE_SCHEMA = "fetch-pick-state54-v1"
ENV_ID = "APC-FetchPickCube-v1"
LEARNED_ACTION_SLICE = (0, 11)


def features(observations: np.ndarray) -> np.ndarray:
    """Use the upstream state54 observation without teacher-only stage labels."""
    obs = np.asarray(observations, dtype=np.float32)
    if obs.shape[-1] != 54 or not np.isfinite(obs).all():
        raise ValueError("Operation BC requires finite Fetch/PickCube state54 observations")
    return obs


def make_model(hidden_width=64) -> nn.Module:
    if type(hidden_width) is not int or hidden_width < 1:
        raise ValueError("hidden_width must be a positive integer")
    return nn.Sequential(nn.Linear(54, hidden_width), nn.Tanh(),
                         nn.Linear(hidden_width, hidden_width), nn.Tanh(),
                         nn.Linear(hidden_width, 11), nn.Tanh())


def load_demonstrations(path: Path, *, successful_only=False, exclude_grasp_open_conflicts=False,
                        exclude_invalid_ik_labels=False):
    if type(exclude_grasp_open_conflicts) is not bool or type(exclude_invalid_ik_labels) is not bool:
        raise ValueError("Operation label filters must be boolean")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    policy = manifest.get("policy_details", {})
    scripted = (config["task_label"] == "scripted_fetch_arm_pick_place"
                and policy.get("learned") is False and policy.get("protocol") == "pick_place")
    relabeled = (config["task_label"] == "operation_dagger_relabel"
                 and policy.get("relabel") == "dagger"
                 and policy.get("label_source") == "hand-designed upstream IK joint tracking")
    if (manifest["status"] != "completed" or config["env_id"] != ENV_ID
            or config["policy"] != "external" or config["robot_uids"] != "fetch"
            or config["control_mode"] != "pd_joint_delta_pos"
            or not (scripted or relabeled)
            or policy.get("action_mapping", {}).get("base") != [11, 13]
            or manifest["action_shape"] != [13]):
        raise ValueError("Expected a compatible scripted or DAgger-labeled Fetch pick-place rollout")
    rows = [json.loads(line) for line in (path / "episodes.jsonl").read_text().splitlines()]
    if not rows or len(rows) != manifest["completed_episodes"] or len(rows) != config["episodes"]:
        raise ValueError("Incomplete teacher episode accounting")
    available_rows = rows
    episode_success = {}
    for row in available_rows:
        trajectory = (path / row["trajectory"]).resolve()
        if trajectory.parent != path.resolve():
            raise ValueError("Trajectory must be inside the teacher run")
        with np.load(trajectory, allow_pickle=False) as data:
            success = data["success"]
            if success.shape != (row["steps"],) or not np.isin(success, (-1, 0, 1)).all():
                raise ValueError("Teacher success trajectory is not aligned")
            known = success[success >= 0]
            success_ever = bool(known.max()) if len(known) else None
            success_final = bool(success[-1]) if success[-1] >= 0 else None
        if row["success_ever"] != success_ever or row["success_final"] != success_final:
            raise ValueError("Teacher episode success does not match the saved trajectory")
        episode_success[row["episode"]] = success_ever
    if successful_only and scripted:
        rows = [row for row in rows if episode_success[row["episode"]]]
        if not rows:
            raise ValueError("No successful teacher trajectories are available")
    step_rows = ([json.loads(line) for line in (path / "steps.jsonl").read_text().splitlines()]
                 if relabeled else [])
    xs, ys, hashes = [], [], {}
    selected_samples_before_filter = 0
    excluded_grasp_open_conflicts = 0
    excluded_invalid_ik_labels = 0
    for row in rows:
        trajectory = (path / row["trajectory"]).resolve()
        if trajectory.parent != path.resolve():
            raise ValueError("Trajectory must be inside the teacher run")
        with np.load(trajectory, allow_pickle=False) as data:
            obs, actions = data["observations"], data["actions"]
            count = row["steps"]
            if obs.shape != (count + 1, 1, 54) or actions.shape != (count, 13) or count < 1:
                raise ValueError("Teacher observation/action alignment or shape is invalid")
            if not all(np.isfinite(data[key]).all() for key in data.files):
                raise ValueError("Nonfinite teacher trajectory")
            if (np.abs(actions) > 1).any():
                raise ValueError("Teacher action outside normalized bounds")
            labels = actions
            valid_ik = np.ones(count, dtype=bool)
            if relabeled:
                episode_steps = [step for step in step_rows if step["episode"] == row["episode"]]
                if (len(episode_steps) != count
                        or [step["step"] for step in episode_steps] != list(range(count))):
                    raise ValueError("DAgger step labels do not match the saved trajectory")
                labels = np.asarray([step["info"]["diagnostic"]["teacher_action"]
                                     for step in episode_steps], dtype=np.float32)
                behavior = np.asarray([step["info"]["diagnostic"]["behavior_action"]
                                       for step in episode_steps], dtype=np.float32)
                if labels.shape != (count, 13) or not np.isfinite(labels).all():
                    raise ValueError("DAgger teacher labels are invalid")
                if not np.array_equal(behavior, actions):
                    raise ValueError("Logged DAgger behavior does not match the trajectory action")
                valid_ik = np.asarray([
                    step["info"]["diagnostic"]["ik_success"]
                    and step["info"]["diagnostic"]["ik_within_limits"]
                    and step["info"]["diagnostic"]["ik_table_clear"]
                    for step in episode_steps], dtype=bool)
            if (np.abs(labels) > 1).any():
                raise ValueError("Operation teacher label outside normalized bounds")
            if (np.abs(labels[:, 11:13]) > 1e-7).any():
                raise ValueError("Operation teacher must keep the mobile base command at zero")
            sample_features = features(obs[:-1, 0])
            sample_labels = labels[:, 0:11].astype(np.float32)
            selected_samples_before_filter += count
            keep = np.ones(count, dtype=bool)
            if relabeled and exclude_grasp_open_conflicts:
                conflict = (sample_features[:, 30] > 0.5) & (sample_labels[:, 7] > 0)
                excluded_grasp_open_conflicts += int(conflict.sum())
                keep &= ~conflict
            if relabeled and exclude_invalid_ik_labels:
                excluded_invalid_ik_labels += int((~valid_ik).sum())
                keep &= valid_ik
            xs.append(sample_features[keep])
            ys.append(sample_labels[keep])
        hashes[trajectory.name] = sha256(trajectory)
    source = dict(run=str(path.resolve()), config=config, seeds=[row["seed"] for row in rows],
                  available_seeds=[row["seed"] for row in available_rows],
                  excluded_seeds=[row["seed"] for row in available_rows if row not in rows],
                  successful_only=successful_only and scripted,
                  data_kind="dagger_relabel" if relabeled else "scripted_demonstration",
                  manifest_sha256=sha256(path / "manifest.json"), trajectory_sha256=hashes,
                  episodes_sha256=sha256(path / "episodes.jsonl"),
                  steps_sha256=sha256(path / "steps.jsonl") if relabeled else None,
                  selected_samples_before_filter=selected_samples_before_filter,
                  available_environment_steps=sum(row["steps"] for row in available_rows),
                  retained_samples=sum(len(item) for item in xs),
                  excluded_grasp_open_conflicts=excluded_grasp_open_conflicts,
                  excluded_invalid_ik_labels=excluded_invalid_ik_labels,
                  excluded_samples_total=selected_samples_before_filter - sum(len(item) for item in xs),
                  control_freq=manifest["control_freq"], task=manifest["task"],
                  teacher_policy=policy)
    return np.concatenate(xs), np.concatenate(ys), source


def train(demo_run: Path, output: Path, *, updates=1000, seed=0, hidden_width=64,
          successful_only=False, extra_runs=(), exclude_grasp_open_conflicts=False,
          exclude_invalid_ik_labels=False) -> Path:
    from .runner import json_write, provenance, snapshot_source, utc_now

    if type(updates) is not int or updates < 1 or not 0 <= seed < 2**32:
        raise ValueError("Positive updates and a uint32 seed are required")
    if type(hidden_width) is not int or hidden_width < 1:
        raise ValueError("hidden_width must be a positive integer")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(status="running", started_at=utc_now(), provenance=provenance(),
                  algorithm="behavior_cloning", teacher="hand-designed Fetch pick-place IK",
                  feature_schema=FEATURE_SCHEMA, device="cpu", torch_threads=1, seed=seed,
                  requested_updates=updates, completed_updates=0, batch_size=64,
                  learning_rate=0.001, loss="minibatch action MSE", sampling="uniform",
                   architecture=[54, hidden_width, hidden_width, 11],
                   learned_action_slice=list(LEARNED_ACTION_SLICE),
                   fixed_base_action=[0.0, 0.0], successful_only=successful_only,
                   exclude_grasp_open_conflicts=exclude_grasp_open_conflicts,
                   exclude_invalid_ik_labels=exclude_invalid_ik_labels,
                   environment_steps_during_training=0)
    json_write(output / "training.json", report)
    try:
        snapshot_source(output)
        datasets = [load_demonstrations(
            path, successful_only=successful_only,
            exclude_grasp_open_conflicts=exclude_grasp_open_conflicts,
            exclude_invalid_ik_labels=exclude_invalid_ik_labels)
                    for path in (demo_run, *extra_runs)]
        source = datasets[0][2]
        if any(item[2]["task"] != source["task"] or item[2]["control_freq"] != source["control_freq"]
               for item in datasets):
            raise ValueError("All operation sources must share task and control metadata")
        x = np.concatenate([item[0] for item in datasets])
        y = np.concatenate([item[1] for item in datasets])
        report.update(source=source, sources=[item[2] for item in datasets], samples=len(x),
                      source_environment_steps=sum(
                          item[2]["available_environment_steps"] for item in datasets),
                      selected_source_samples_before_filter=sum(
                          item[2]["selected_samples_before_filter"] for item in datasets),
                      excluded_grasp_open_conflicts=sum(
                          item[2]["excluded_grasp_open_conflicts"] for item in datasets),
                      excluded_invalid_ik_labels=sum(
                          item[2]["excluded_invalid_ik_labels"] for item in datasets),
                      excluded_samples_total=sum(
                          item[2]["excluded_samples_total"] for item in datasets))
        torch.set_num_threads(1)
        torch.manual_seed(seed)
        model = make_model(hidden_width)
        report["parameter_count"] = sum(p.numel() for p in model.parameters())
        report["parameter_bytes"] = sum(p.numel() * p.element_size() for p in model.parameters())
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
            prediction = model(x)
            report["final_train_mse"] = float(nn.functional.mse_loss(prediction, y))
            report["final_channel_mse"] = (prediction - y).square().mean(dim=0).tolist()
        if not np.isfinite(report["final_train_mse"]):
            raise ValueError("Nonfinite final training loss")
        checkpoint_sources = [dict(run=item[2]["run"], data_kind=item[2]["data_kind"],
                                   seeds=item[2]["seeds"], manifest_sha256=item[2]["manifest_sha256"],
                                   episodes_sha256=item[2]["episodes_sha256"],
                                   trajectory_sha256=item[2]["trajectory_sha256"],
                                   steps_sha256=item[2]["steps_sha256"],
                                   available_environment_steps=item[2]["available_environment_steps"],
                                   selected_samples_before_filter=item[2]["selected_samples_before_filter"],
                                   retained_samples=item[2]["retained_samples"],
                                   excluded_grasp_open_conflicts=item[2]["excluded_grasp_open_conflicts"],
                                   excluded_invalid_ik_labels=item[2]["excluded_invalid_ik_labels"],
                                   excluded_samples_total=item[2]["excluded_samples_total"])
                              for item in datasets]
        checkpoint = dict(format_version=1, feature_schema=FEATURE_SCHEMA,
                          model_state=model.state_dict(), mean=mean, scale=scale,
                          env_id=ENV_ID, control_mode="pd_joint_delta_pos",
                          control_freq=source["control_freq"], task=source["task"],
                          training_seeds=sorted({value for _, _, item in datasets
                                                 for value in item["seeds"]}),
                           training_source=source["run"], training_sources=checkpoint_sources,
                           training_seed=seed, updates=updates, hidden_width=hidden_width,
                           algorithm="behavior_cloning", learned_action_slice=LEARNED_ACTION_SLICE,
                           fixed_base_action=(0.0, 0.0), successful_only=successful_only,
                           exclude_grasp_open_conflicts=exclude_grasp_open_conflicts,
                           exclude_invalid_ik_labels=exclude_invalid_ik_labels)
        torch.save(checkpoint, output / "policy.pt")
        report.update(status="completed", checkpoint="policy.pt",
                      checkpoint_sha256=sha256(output / "policy.pt"),
                      checkpoint_bytes=(output / "policy.pt").stat().st_size)
    except BaseException as exc:
        report.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      error=f"{type(exc).__name__}: {exc}")
        (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        report.update(finished_at=utc_now(), wall_seconds=time.monotonic() - started)
        json_write(output / "training.json", report)
    return output


class OperationBCPolicy:
    def __init__(self, path: Path, task: dict, control_freq: float):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if (checkpoint["format_version"] != 1 or checkpoint["feature_schema"] != FEATURE_SCHEMA
                or checkpoint["env_id"] != ENV_ID
                or checkpoint["control_mode"] != "pd_joint_delta_pos"
                or checkpoint["control_freq"] != control_freq
                or checkpoint["task"] != json.loads(json.dumps(task))
                or tuple(checkpoint["learned_action_slice"]) != LEARNED_ACTION_SLICE
                or tuple(checkpoint["fixed_base_action"]) != (0.0, 0.0)):
            raise ValueError("Checkpoint does not match the current Fetch pick-place task")
        torch.set_num_threads(1)
        self.model = make_model(checkpoint.get("hidden_width", 64))
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.mean, self.scale = checkpoint["mean"], checkpoint["scale"]
        if (self.mean.shape != (54,) or self.scale.shape != (54,)
                or not torch.isfinite(self.mean).all() or not torch.isfinite(self.scale).all()
                or not (self.scale > 0).all()
                or not all(torch.isfinite(parameter).all() for parameter in self.model.parameters())):
            raise ValueError("Invalid checkpoint normalization or parameters")
        self.metadata = dict(source="learned operation behavior cloning; no IK/stage fallback",
                             checkpoint_path=str(path.resolve()), checkpoint_sha256=sha256(path),
                             feature_schema=FEATURE_SCHEMA,
                             training_source=checkpoint["training_source"],
                             training_sources=checkpoint.get("training_sources", [dict(
                                 run=checkpoint["training_source"], data_kind="legacy")]),
                             training_seeds=checkpoint["training_seeds"],
                             updates=checkpoint["updates"], algorithm=checkpoint["algorithm"],
                              hidden_width=checkpoint.get("hidden_width", 64),
                              successful_only=checkpoint.get("successful_only", False),
                              exclude_grasp_open_conflicts=checkpoint.get(
                                  "exclude_grasp_open_conflicts", False),
                              exclude_invalid_ik_labels=checkpoint.get(
                                  "exclude_invalid_ik_labels", False),
                             learned_action_slice=list(LEARNED_ACTION_SLICE),
                             fixed_base_action=[0.0, 0.0],
                             parameter_bytes=sum(p.numel() * p.element_size() for p in self.model.parameters()),
                             parameter_count=sum(p.numel() for p in self.model.parameters()))

    def predict(self, observations: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(features(observations))
        with torch.inference_mode():
            return self.model((x - self.mean) / self.scale).numpy()
