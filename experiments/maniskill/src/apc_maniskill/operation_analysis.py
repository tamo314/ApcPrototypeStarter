"""Offline diagnostics for scripted and DAgger-labeled Fetch operation data."""
from __future__ import annotations

import json
from pathlib import Path
import time
import traceback

import numpy as np

from .bc import sha256
from .operation_bc import load_demonstrations


STAGE_NAMES = ("approach", "descend", "close", "lift", "transport", "hold")


def _quantiles(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return dict(count=0, p05=None, median=None, p95=None)
    return dict(count=int(len(values)), p05=float(np.quantile(values, 0.05)),
                median=float(np.median(values)), p95=float(np.quantile(values, 0.95)))


def _load_source(path: Path, *, successful_only: bool, source_index: int) -> dict:
    x, y, provenance = load_demonstrations(path, successful_only=successful_only)
    episode_rows = [json.loads(line) for line in (path / "episodes.jsonl").read_text().splitlines()]
    if successful_only and provenance["data_kind"] == "scripted_demonstration":
        episode_rows = [row for row in episode_rows if row["success_ever"]]
    step_rows = [json.loads(line) for line in (path / "steps.jsonl").read_text().splitlines()]
    steps_by_episode = {}
    for step in step_rows:
        steps_by_episode.setdefault(step["episode"], []).append(step)

    episodes, steps, stages, grasped, lifts = [], [], [], [], []
    ik_valid, joint_scales, reached, labels_by_episode = [], [], [], []
    behavior_errors = []
    cursor = 0
    episode_summaries = []
    for row in episode_rows:
        count = row["steps"]
        episode_steps = steps_by_episode.get(row["episode"], [])
        if ([item["step"] for item in episode_steps] != list(range(count))
                or len(episode_steps) != count):
            raise ValueError("Step log is not aligned with the selected episode")
        trajectory = (path / row["trajectory"]).resolve()
        if trajectory.parent != path.resolve():
            raise ValueError("Trajectory must be inside the operation data run")
        with np.load(trajectory, allow_pickle=False) as data:
            observations = data["observations"][:-1, 0]
            behavior = data["actions"][:, 0:11]
        episode_labels = y[cursor:cursor + count]
        if observations.shape != (count, 54) or episode_labels.shape != (count, 11):
            raise ValueError("Analysis data is not aligned with validated demonstrations")
        diagnostics = [item["info"]["diagnostic"] for item in episode_steps]
        episode_stages = np.asarray([item["stage"] for item in diagnostics], dtype=np.int16)
        if ((episode_stages < 0) | (episode_stages >= len(STAGE_NAMES))).any():
            raise ValueError("Unknown operation teacher stage")
        episode_grasped = observations[:, 30] > 0.5
        episode_lifts = observations[:, 43] - observations[0, 43]
        episode_ik = np.asarray([item["ik_success"] and item["ik_within_limits"]
                                 and item["ik_table_clear"] for item in diagnostics], dtype=bool)
        episode_scales = np.asarray([item["joint_action_scale"] for item in diagnostics], dtype=np.float32)
        episode_reached = np.asarray([item["reached"] for item in diagnostics], dtype=bool)

        episodes.extend([row["episode"]] * count)
        steps.extend(range(count))
        stages.append(episode_stages)
        grasped.append(episode_grasped)
        lifts.append(episode_lifts)
        ik_valid.append(episode_ik)
        joint_scales.append(episode_scales)
        reached.append(episode_reached)
        labels_by_episode.append(episode_labels)
        if provenance["data_kind"] == "dagger_relabel":
            behavior_errors.append(np.square(behavior - episode_labels))
        episode_summaries.append(dict(
            episode=row["episode"], seed=row["seed"], samples=count,
            stage_counts={STAGE_NAMES[index]: int(np.sum(episode_stages == index))
                          for index in range(len(STAGE_NAMES))},
            grasped_samples=int(episode_grasped.sum()),
            lift_above_2cm_samples=int((episode_lifts > 0.02).sum())))
        cursor += count
    if cursor != len(x):
        raise ValueError("Selected episode count does not match validated data")

    stages = np.concatenate(stages)
    grasped = np.concatenate(grasped)
    lifts = np.concatenate(lifts)
    ik_valid = np.concatenate(ik_valid)
    joint_scales = np.concatenate(joint_scales)
    reached = np.concatenate(reached)
    labels = np.concatenate(labels_by_episode)
    conflict = grasped & (stages == 0) & (labels[:, 7] > 0)
    summary = dict(
        source_index=source_index, run=str(path.resolve()), data_kind=provenance["data_kind"],
        samples=len(x), episodes=len(episode_rows), seeds=[row["seed"] for row in episode_rows],
        stage_counts={STAGE_NAMES[index]: int(np.sum(stages == index))
                      for index in range(len(STAGE_NAMES))},
        grasped_samples=int(grasped.sum()), lift_above_2cm_samples=int((lifts > 0.02).sum()),
        max_lift_m=float(lifts.max()), open_gripper_labels=int((labels[:, 7] > 0).sum()),
        closed_gripper_labels=int((labels[:, 7] < 0).sum()),
        grasped_open_approach_conflicts=int(conflict.sum()),
        invalid_ik_samples=int((~ik_valid).sum()),
        invalid_ik_zero_arm_samples=int((~ik_valid & np.all(np.abs(labels[:, 0:7]) <= 1e-7, axis=1)).sum()),
        scaled_joint_action_samples=int((joint_scales < 1).sum()),
        teacher_target_reached_samples=int(reached.sum()),
        action_mean=labels.mean(axis=0).tolist(), action_std=labels.std(axis=0).tolist(),
        manifest_sha256=provenance["manifest_sha256"],
        episodes_sha256=provenance["episodes_sha256"],
        steps_sha256=sha256(path / "steps.jsonl"),
        trajectory_sha256=provenance["trajectory_sha256"], episodes_detail=episode_summaries)
    if behavior_errors:
        summary["behavior_teacher_mse"] = float(np.concatenate(behavior_errors).mean())
    return dict(x=x, y=y, source=np.full(len(x), source_index, dtype=np.int16),
                episode=np.asarray(episodes, dtype=np.int16), step=np.asarray(steps, dtype=np.int16),
                stage=stages, grasped=grasped, lift=lifts, conflict=conflict,
                task=provenance["task"], control_freq=provenance["control_freq"], summary=summary)


def _nearest_neighbors(x: np.ndarray, y: np.ndarray, source: np.ndarray, episode: np.ndarray,
                       stage: np.ndarray, grasped: np.ndarray, k: int) -> tuple[dict, dict]:
    if type(k) is not int or k < 1:
        raise ValueError("nearest-neighbor count must be positive")
    scale = np.maximum(x.std(axis=0), 0.05)
    standardized = (x - x.mean(axis=0)) / scale
    query_indices, neighbor_indices, distances = [], [], []
    queries_without_neighbors = 0
    for query in range(len(x)):
        eligible = (grasped == grasped[query]) & ~(
            (source == source[query]) & (episode == episode[query]))
        candidates = np.flatnonzero(eligible)
        if len(candidates) < k:
            queries_without_neighbors += 1
            continue
        candidate_distances = np.sqrt(np.mean(
            np.square(standardized[candidates] - standardized[query]), axis=1))
        nearest = np.argpartition(candidate_distances, k - 1)[:k]
        ordered = nearest[np.argsort(candidate_distances[nearest])]
        query_indices.extend([query] * k)
        neighbor_indices.extend(candidates[ordered])
        distances.extend(candidate_distances[ordered])
    queries = np.asarray(query_indices, dtype=np.int32)
    neighbors = np.asarray(neighbor_indices, dtype=np.int32)
    distances = np.asarray(distances, dtype=np.float32)
    action_distance = np.sqrt(np.mean(np.square(y[queries] - y[neighbors]), axis=1))
    same_source = source[queries] == source[neighbors]
    same_stage = stage[queries] == stage[neighbors]
    gripper_mismatch = np.signbit(y[queries, 7]) != np.signbit(y[neighbors, 7])

    groups = {
        "all": np.ones(len(queries), dtype=bool),
        "same_stage_same_source": same_stage & same_source,
        "same_stage_cross_source": same_stage & ~same_source,
        "different_stage_same_source": ~same_stage & same_source,
        "different_stage_cross_source": ~same_stage & ~same_source,
    }
    report = {}
    for name, mask in groups.items():
        report[name] = dict(state_rms_distance=_quantiles(distances[mask]),
                            action_rms_difference=_quantiles(action_distance[mask]),
                            gripper_sign_mismatch_count=int(gripper_mismatch[mask].sum()),
                            gripper_sign_mismatch_fraction=(float(gripper_mismatch[mask].mean())
                                                            if mask.any() else None))
    report["queries_without_matching_neighbors"] = queries_without_neighbors
    details = dict(query_index=queries, neighbor_index=neighbors, state_rms_distance=distances,
                   action_rms_difference=action_distance.astype(np.float32),
                   same_source=same_source, same_stage=same_stage,
                   gripper_sign_mismatch=gripper_mismatch, normalization_scale=scale)
    return report, details


def _checkpoint_errors(paths: list[Path], x: np.ndarray, y: np.ndarray, source: np.ndarray,
                       stage: np.ndarray, task: dict, control_freq: float) -> list[dict]:
    from .operation_bc import OperationBCPolicy

    results = []
    for path in paths:
        policy = OperationBCPolicy(path, task, control_freq)
        squared_error = np.square(policy.predict(x) - y)
        by_source = {}
        for source_index in np.unique(source):
            mask = source == source_index
            by_source[str(int(source_index))] = dict(
                samples=int(mask.sum()), mse=float(squared_error[mask].mean()),
                channel_mse=squared_error[mask].mean(axis=0).tolist())
        by_stage = {}
        for stage_index, name in enumerate(STAGE_NAMES):
            mask = stage == stage_index
            by_stage[name] = dict(samples=int(mask.sum()),
                                  mse=float(squared_error[mask].mean()) if mask.any() else None,
                                  channel_mse=(squared_error[mask].mean(axis=0).tolist()
                                               if mask.any() else None))
        results.append(dict(checkpoint=str(path.resolve()), sha256=sha256(path),
                            metadata=policy.metadata, mse=float(squared_error.mean()),
                            channel_mse=squared_error.mean(axis=0).tolist(),
                            by_source=by_source, by_stage=by_stage))
    return results


def analyze(demo_run: Path, relabel_runs: list[Path], output: Path, *, neighbors=5,
            checkpoints=(), successful_only=True) -> Path:
    from .runner import json_write, provenance, snapshot_source, utc_now

    if not relabel_runs:
        raise ValueError("At least one DAgger relabel run is required")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(status="running", started_at=utc_now(), provenance=provenance(),
                  analysis="operation_dataset_conflicts_v1", environment_steps=0,
                  learning_updates=0, nearest_neighbors=neighbors,
                  nearest_neighbor_rules=dict(state="state54 standardized with scale floor 0.05",
                                              distance="root mean square over 54 features",
                                              exclude="same source and episode",
                                              discrete_match="pre-action corrected grasp flag"))
    json_write(output / "analysis.json", report)
    try:
        snapshot_source(output)
        datasets = [_load_source(demo_run, successful_only=successful_only, source_index=0)]
        datasets.extend(_load_source(path, successful_only=False, source_index=index)
                        for index, path in enumerate(relabel_runs, start=1))
        first_source = datasets[0]
        if any(item["task"] != first_source["task"]
               or item["control_freq"] != first_source["control_freq"] for item in datasets):
            raise ValueError("All operation analysis sources must share task and control metadata")
        x = np.concatenate([item["x"] for item in datasets])
        y = np.concatenate([item["y"] for item in datasets])
        source = np.concatenate([item["source"] for item in datasets])
        episode = np.concatenate([item["episode"] for item in datasets])
        step = np.concatenate([item["step"] for item in datasets])
        stage = np.concatenate([item["stage"] for item in datasets])
        grasped = np.concatenate([item["grasped"] for item in datasets])
        lift = np.concatenate([item["lift"] for item in datasets])
        conflict = np.concatenate([item["conflict"] for item in datasets])
        nearest, details = _nearest_neighbors(x, y, source, episode, stage, grasped, neighbors)
        np.savez_compressed(output / "nearest_pairs.npz", source=source, episode=episode,
                            step=step, stage=stage, grasped=grasped, lift_m=lift,
                            grasped_open_approach_conflict=conflict, **details)
        report.update(status="completed", samples=len(x), sources=[item["summary"] for item in datasets],
                      aggregate=dict(
                          stage_counts={STAGE_NAMES[index]: int(np.sum(stage == index))
                                        for index in range(len(STAGE_NAMES))},
                          grasped_samples=int(grasped.sum()),
                          open_gripper_labels=int((y[:, 7] > 0).sum()),
                          closed_gripper_labels=int((y[:, 7] < 0).sum()),
                          grasped_open_approach_conflicts=int(conflict.sum())),
                      nearest_neighbor_summary=nearest,
                      checkpoint_errors=_checkpoint_errors(
                           list(checkpoints), x, y, source, stage,
                           first_source["task"], first_source["control_freq"]),
                      nearest_pairs="nearest_pairs.npz",
                      nearest_pairs_sha256=sha256(output / "nearest_pairs.npz"))
    except BaseException as exc:
        report.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      error=f"{type(exc).__name__}: {exc}")
        (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        report.update(finished_at=utc_now(), wall_seconds=time.monotonic() - started)
        json_write(output / "analysis.json", report)
    return output


def _rollout_summary(path: Path) -> dict:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    if (manifest["status"] != "completed" or config["env_id"] != "APC-FetchPickCube-v1"
            or config["policy"] != "fetch_pick_bc" or manifest["action_shape"] != [13]):
        raise ValueError("Expected a completed learned Fetch pick-place rollout")
    rows = [json.loads(line) for line in (path / "episodes.jsonl").read_text().splitlines()]
    if len(rows) != manifest["completed_episodes"] or len(rows) != config["episodes"]:
        raise ValueError("Incomplete learned rollout episode accounting")
    checkpoint = (path / "policy.pt").resolve()
    if (checkpoint.parent != path.resolve()
            or sha256(checkpoint) != manifest["policy_details"]["checkpoint_sha256"]):
        raise ValueError("Learned rollout checkpoint does not match its manifest")
    episodes, hashes = [], {}
    for row in rows:
        trajectory = (path / row["trajectory"]).resolve()
        if trajectory.parent != path.resolve():
            raise ValueError("Trajectory must be inside the learned rollout")
        with np.load(trajectory, allow_pickle=False) as data:
            observations, actions, success = (
                data["observations"][:, 0], data["actions"], data["success"])
            if (observations.shape != (row["steps"] + 1, 54)
                    or actions.shape != (row["steps"], 13)
                    or success.shape != (row["steps"],)
                    or not np.isin(success, (-1, 0, 1)).all()
                    or not np.isfinite(observations).all() or not np.isfinite(actions).all()):
                raise ValueError("Invalid learned rollout trajectory")
            known = success[success >= 0]
            success_ever = bool(known.max()) if len(known) else None
            success_final = bool(success[-1]) if success[-1] >= 0 else None
            if row["success_ever"] != success_ever or row["success_final"] != success_final:
                raise ValueError("Learned episode success does not match the saved trajectory")
            successful_steps = np.flatnonzero(success == 1)
            first_success = int(successful_steps[0]) if len(successful_steps) else None
            success_after_first = success[first_success:] if first_success is not None else np.empty(0)
            first_success_step = first_success + 1 if first_success is not None else -1
            post_success_steps = row["steps"] - first_success_step if first_success is not None else 0
            hold_complete = (first_success is not None
                             and post_success_steps >= config["post_success_steps"])
            if config["post_success_steps"] and (
                    row.get("first_success_step") != first_success_step
                    or row.get("post_success_steps") != post_success_steps
                    or row.get("hold_complete") != hold_complete):
                raise ValueError("Learned episode hold accounting does not match the trajectory")
            grasp_indices = np.flatnonzero(observations[:, 30] > 0.5)
            lift = observations[:, 43] - observations[0, 43]
            cube_goal_distance = np.linalg.norm(observations[:, 51:54], axis=1)
            episodes.append(dict(
                episode=row["episode"], seed=row["seed"], steps=row["steps"],
                success_ever=row["success_ever"], success_final=row["success_final"],
                first_success_step=first_success_step,
                hold_complete=hold_complete if config["post_success_steps"] else None,
                post_success_steps=post_success_steps,
                successful_steps_after_first=int((success_after_first == 1).sum()),
                observed_steps_after_first=int(len(success_after_first)),
                success_continuous_after_first=(bool(np.all(success_after_first == 1))
                                                if len(success_after_first) else None),
                min_tcp_cube_distance_m=float(np.linalg.norm(
                    observations[:, 48:51], axis=1).min()),
                min_cube_goal_distance_m=float(cube_goal_distance.min()),
                final_cube_goal_distance_m=float(cube_goal_distance[-1]),
                max_cube_lift_m=float(lift.max()),
                grasped_observations=int((observations[:, 30] > 0.5).sum()),
                first_grasp_observation=(int(grasp_indices[0]) if len(grasp_indices) else None),
                gripper_action_min=float(actions[:, 7].min()),
                gripper_action_max=float(actions[:, 7].max()),
                base_action_abs_max=float(np.abs(actions[:, 11:13]).max())))
        hashes[trajectory.name] = sha256(trajectory)
    return dict(
        run=str(path.resolve()), manifest_sha256=sha256(path / "manifest.json"),
        episodes_sha256=sha256(path / "episodes.jsonl"),
        checkpoint_sha256=sha256(checkpoint),
        protocol=dict(max_steps=config["max_steps"], env_max_steps=config["env_max_steps"],
                      post_success_steps=config["post_success_steps"]),
        trajectory_sha256=hashes, episodes=len(rows), steps=sum(row["steps"] for row in rows),
        successful_episodes=sum(row["success_ever"] for row in rows),
        final_successful_episodes=sum(row["success_final"] for row in rows),
        hold_complete_episodes=sum(item["hold_complete"] is True for item in episodes),
        continuous_success_after_first_episodes=sum(
            item["success_continuous_after_first"] is True for item in episodes),
        grasped_episodes=sum(item["grasped_observations"] > 0 for item in episodes),
        lifted_above_2cm_episodes=sum(item["max_cube_lift_m"] > 0.02 for item in episodes),
        episode_metrics=episodes)


def analyze_rollouts(run_dirs: list[Path], output: Path) -> Path:
    from .runner import json_write, provenance, snapshot_source, utc_now

    if not run_dirs:
        raise ValueError("At least one learned operation rollout is required")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(status="running", started_at=utc_now(), provenance=provenance(),
                  analysis="operation_rollout_comparison_v2", environment_steps=0,
                  learning_updates=0)
    json_write(output / "analysis.json", report)
    try:
        snapshot_source(output)
        report.update(status="completed", rollouts=[_rollout_summary(path) for path in run_dirs])
    except BaseException as exc:
        report.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      error=f"{type(exc).__name__}: {exc}")
        (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        report.update(finished_at=utc_now(), wall_seconds=time.monotonic() - started)
        json_write(output / "analysis.json", report)
    return output
