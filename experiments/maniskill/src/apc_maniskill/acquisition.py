"""T32-R3: event-driven acquisition by outcome-evaluated local option search.

Given a deficit event (JSON written by the online detector) and the bank that was
active, the acquirer:

1. starts at the onset of the detected stall (event step - streak) on the logged
   trajectory, reproduced by replaying its proposed primitive IDs from reset;
2. at each decision point, evaluates every option in a disclosed option set
   (one existing primitive ID followed by CONTINUE holds) plus the unchanged bank
   policy as baseline, by branching the option and then letting the unchanged bank
   policy run for a short horizon.  Scores use reachability over the horizon,
   rejections, grasp retention and success, not one-step distance;
3. greedily extends the prefix with the best option until the bank baseline is
   best (handoff) or the budget ends;
4. replays the resulting trajectory once and returns its records.

Learning (candidate CART and router labels) uses only these records and the
option outcomes.  The option set and hold lengths are a human prior shared with
every compared method; there is no external teacher or scripted solution.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from .experience_loop import (HOLDS, Bank, EpisodeLog, build_policy, make_task_env, option_ids,
                              read_jsonl, run_episode)

# Disclosed option prior: (existing ID, repeats) with executor-settling holds (HOLDS).
# v2 adds 3x repeats of the hand translations so multi-step hand motions are visible
# to a one-decision lookahead (v1 single options plateaued on 3014).
DEFAULT_OPTIONS = tuple([(i, 1) for i in (0, 1, 2, 3, 4, 5, 13, 16, 17, 18, 19)]
                        + [(i, 3) for i in (0, 1, 2, 3, 4, 5)])


@dataclass
class EvalJob:
    task: str
    seed: int
    bank: dict
    prefix: list
    option: tuple | None  # None: unchanged bank policy (baseline)
    horizon: int
    out: str


_CACHE: dict = {}


def _policy_for(task: str, bank: Bank, out: Path):
    key = (task, json.dumps({k: str(v) for k, v in bank.files().items()}, sort_keys=True))
    if key not in _CACHE:
        for env, _ in _CACHE.values():
            env.close()
        _CACHE.clear()
        env = make_task_env(task, out / f"worker-{os.getpid()}")
        _CACHE[key] = (env, build_policy(env, out / f"worker-{os.getpid()}", bank))
    return _CACHE[key]


def evaluate_option(job: EvalJob) -> dict:
    """Replay prefix, apply option, run the unchanged bank policy for the horizon."""
    bank = Bank.from_roles(Path("."), job.bank)
    env, policy = _policy_for(job.task, bank, Path(job.out))
    script = list(job.prefix) + ([] if job.option is None else option_ids(job.option))
    start = len(job.prefix)
    stats = dict(min_dist=np.inf, rejections=0, rejection_kinds={}, grasp_lost=False,
                 min_lift=np.inf, success=False, first_place_step=None, end_dist=None)

    def on_step(step, r, pol):
        if step < start:
            return None
        rc = r["reward_components"]
        stats["min_dist"] = min(stats["min_dist"], rc["cube_goal_dist_xy_after_m"])
        stats["min_lift"] = min(stats["min_lift"], rc["cube_lift_after_m"])
        if r["override_reason_code"]:
            stats["rejections"] += 1
            kind = r["override_reason"]
            stats["rejection_kinds"][kind] = stats["rejection_kinds"].get(kind, 0) + 1
        # Grasp loss = held at some point in the branch, then released away from goal
        # (v1 flagged every ungrasped step, penalising pre-grasp branches).
        stats["held"] = stats.get("held", False) or rc["grasped_after"]
        if stats["held"] and not rc["grasped_after"] and rc["cube_goal_dist_xy_after_m"] > 0.04:
            stats["grasp_lost"] = True
        if r["selected_module"] == "place" and stats["first_place_step"] is None:
            stats["first_place_step"] = step
        stats["end_dist"] = rc["cube_goal_dist_xy_after_m"]
        stats["end_grasped"] = rc["grasped_after"]
        stats["end_lift"] = rc["cube_lift_after_m"]
        if r["terminated"]:
            stats["success"] = True
        return None

    t0 = time.monotonic()
    result = run_episode(env, policy, seed=job.seed, max_steps=len(script) + job.horizon,
                         initial_script=script, on_step=on_step)
    stats.update(option=job.option, option_steps=len(script) - start, env_steps=result.steps,
                 prefix_steps=start, wall_seconds=time.monotonic() - t0)
    return {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in stats.items()}


INVERSE = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4, 10: 11, 11: 10, 12: 13, 13: 12, 14: 15, 15: 14,
           16: 17, 17: 16, 18: 19, 19: 18}


def score(stats: dict) -> float:
    """Outcome score at the end of the horizon (not the transient minimum).

    v1 used the minimum distance within the horizon; left/right turns that swept
    the cube past the goal then alternated without progress
    (runs/t32r3-acquire-20260926-a-failed-oscillation).  v2 scores where the
    unchanged bank leaves the cube, with rejection, grasp-loss and success terms.
    """
    end = stats["end_dist"] if stats["end_dist"] is not None else 1.0
    value = -end - 0.0005 * stats["rejections"]
    if stats["grasp_lost"]:
        value -= 0.2
    if stats["success"]:
        value += 0.2
    return value - 1e-5 * stats["option_steps"]


def acquire(event: dict, bank: Bank, out: Path, *, options=DEFAULT_OPTIONS, horizon=250,
            max_decisions=10, margin=0.003, workers=4, start="onset") -> dict:
    """Run the local option search for one event. Writes option outcomes to ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    history = [r for r in read_jsonl(Path(event["observation_history"]))
               if r["environment_seed"] == event["environment_seed"] and r["task"] == event["task"]]
    # "onset": stall onset (event step - streak); "event": first step after detection.
    onset = (int(event["detected_at_step"]) + 1 if start == "event"
             else max(0, int(event["detected_at_step"]) - int(event["streak_steps"])))
    prefix = [int(r["proposed_id"]) for r in history if r["control_step"] < onset]
    if len(prefix) != onset:
        raise ValueError("Observation history does not cover the stall onset")
    bank_paths = {k: str(v) for k, v in bank.files().items()}
    decisions, env_steps, wall = [], 0, 0.0
    chosen: list[tuple] = []
    decision_steps: list[int] = []
    outcomes_path = out / "option_outcomes.jsonl"
    stop_reason = "budget"
    t0 = time.monotonic()
    with get_context("spawn").Pool(workers) as pool:
        for j in range(max_decisions):
            # An option that directly undoes the previous choice is not re-evaluated.
            allowed = [o for o in options if not chosen or INVERSE.get(o[0]) != chosen[-1][0]]
            jobs = [EvalJob(event["task"], event["environment_seed"], bank_paths, prefix, o, horizon,
                            str(out)) for o in (None, *allowed)]
            results = pool.map(evaluate_option, jobs)
            env_steps += sum(r["env_steps"] for r in results)
            for r in results:
                r["score"] = score(r)
                r["decision"] = j
                r["decision_step"] = len(prefix)
            with outcomes_path.open("a", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")
            baseline = results[0]
            best = max(results[1:], key=lambda r: r["score"])
            decisions.append(dict(decision=j, step=len(prefix), baseline_score=baseline["score"],
                                  best_option=best["option"], best_score=best["score"],
                                  baseline_min_dist=baseline["min_dist"], best_min_dist=best["min_dist"],
                                  baseline_end_dist=baseline["end_dist"], best_end_dist=best["end_dist"],
                                  baseline_success=baseline["success"], best_success=best["success"],
                                  scores={json.dumps(r["option"]): r["score"] for r in results}))
            print(f"  decision {j} step {len(prefix)}: baseline {baseline['score']:.4f} "
                  f"(end {baseline['end_dist']:.3f} succ {baseline['success']}) best option {best['option']} "
                  f"{best['score']:.4f} (end {best['end_dist']:.3f} succ {best['success']})", flush=True)
            if best["score"] < baseline["score"] + margin:
                stop_reason = "bank_baseline_best"
                break
            chosen.append(tuple(best["option"]))
            decision_steps.append(len(prefix))
            prefix = prefix + option_ids(best["option"])
    wall = time.monotonic() - t0
    return dict(event_id=event["event_id"], onset_step=onset, start_mode=start, decisions=decisions,
                chosen_options=[list(c) for c in chosen], decision_steps=decision_steps, window_end_step=len(prefix), stop_reason=stop_reason,
                search_env_steps=env_steps, search_wall_seconds=wall, prefix=prefix,
                option_prior=dict(options=[list(o) for o in options], holds={str(k): v for k, v in HOLDS.items()},
                                  no_immediate_inverse=True), score_version="end_dist_v2",
                horizon=horizon, margin=margin)


def replay_acquired(event: dict, bank: Bank, acquisition: dict, out: Path, max_steps=1200) -> dict:
    """Run the acquired trajectory once (prefix script, then the bank) and log it."""
    env = make_task_env(event["task"], out / "replay_env")
    policy = build_policy(env, out / "replay_env", bank)
    log = EpisodeLog(out / "acquired_trajectory.jsonl", run_id=out.name,
                     episode_id=f"acquired-{event['event_id']}", seed=event["environment_seed"],
                     task=event["task"], bank_hash=bank.manifest()["bank_hash"],
                     event_id=event["event_id"], source_event_hash=event["event_hash"])
    result = run_episode(env, policy, seed=event["environment_seed"], max_steps=max_steps, log=log,
                         initial_script=acquisition["prefix"])
    log.close()
    env.close()
    return result.as_dict()
