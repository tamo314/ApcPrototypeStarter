"""T32-R2: compare interventions from the same reproduced state (prefix replay).

For each representative control step k and each branch, a worker resets the task,
re-runs the unchanged policy for k steps (deterministic on PhysX CPU; counted as
prefix cost), applies the branch for a short window, then lets the unchanged
policy continue to the episode limit.  All branches keep the executor's IK,
joint-limit, table and base-path checks.  Scripted branches are hand-designed
reachability references, not learning results.

Example:
  python scripts/run_t32r2_branch_diagnosis.py --out runs/t32r2-branch-20260926-a \
      --task true_place --seed 3014 --state 660 --state 779 --state 900
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from apc_maniskill.experience_loop import (CONTINUE, Bank, EpisodeLog, build_policy, make_task_env,
                                           op, read_jsonl, run_episode)
from apc_maniskill.runner import json_write, provenance, utc_now


class GoalHandSelector:
    """Hand-designed diagnostic: move the held cube toward goal (+4 cm carry height).

    One measured-position update at a time, CONTINUE while the previous target
    is pending (same convention as the physical teacher).  Not a learned module.
    """

    metadata = dict(selector="goal_hand_diagnostic_v1", learned=False)

    def __init__(self, carry_height=0.04):
        self.carry_height = carry_height

    def reset(self):
        pass

    def select(self, step, s):
        cube = np.asarray(s["cube_position"])
        goal = np.asarray(s["goal_position"])
        hand = np.asarray(s["measured_hand_position"])
        pending = np.linalg.norm(np.asarray(s["hand_target_position"]) - hand)
        if s["mode_code"] == 0 and pending > s["position_tolerance_m"] and s["target_age_steps"] < 12:
            return CONTINUE
        desired = goal + ([0, 0, self.carry_height] if np.linalg.norm((goal - cube)[:2]) > 0.02 else 0)
        delta = np.asarray(s["root_rotation"]).T @ (desired - cube)
        if np.max(np.abs(delta)) < 0.004:
            return CONTINUE
        axis = int(np.argmax(np.abs(delta)))
        return 2 * axis + int(delta[axis] < 0)


def base_settle(primitive, count, hold=25):
    seq = []
    for _ in range(count):
        seq += op(primitive, hold)
    return seq


def hand_ops(primitive, count, hold=8):
    seq = []
    for _ in range(count):
        seq += op(primitive, hold)
    return seq


# name -> (script before window module, module forced for the rest of the window or None)
BRANCHES = {
    "continue_policy": ([], None),
    "force_old_transit": ([], "transit"),
    "force_goal_hand": ([], "goal_hand"),
    "lift_then_goal_hand": (hand_ops(4, 5), "goal_hand"),
    "pitch_up_then_goal_hand": (hand_ops(13, 3), "goal_hand"),
    "pitch_down_then_goal_hand": (hand_ops(12, 3), "goal_hand"),
    "retreat_lift_then_goal_hand": (hand_ops(1, 3) + hand_ops(4, 5), "goal_hand"),
    "hold_continue_then_policy": (op(9, 1) + [CONTINUE] * 20, None),
    "base_back_then_goal_hand": (base_settle(17, 2), "goal_hand"),
    "base_turn_left_then_goal_hand": (base_settle(18, 4), "goal_hand"),
    "base_turn_left_fwd_then_goal_hand": (base_settle(18, 4) + base_settle(16, 2), "goal_hand"),
    "base_turn_left_then_policy": (base_settle(18, 4), None),
}


def worker(job):
    out = Path(job["out"])
    task, seed, k, name, window, max_steps = (job["task"], job["seed"], job["state"], job["branch"],
                                              job["window"], job["max_steps"])
    script, module = BRANCHES[name]
    bank = Bank.initial(Path(job["bundle"]))
    manifest = bank.manifest()
    env = make_task_env(task, out / "env" / f"{name}-{k}")
    branch_end = k + len(script) + (window if module is not None else 0)

    def force(step, s):
        start = k + len(script)
        return module if module is not None and start <= step < branch_end else None

    policy = build_policy(env, out, bank, force=force, transit_selector=None)
    policy.selector.modules["goal_hand"] = GoalHandSelector()

    def on_step(step, record, pol):
        if step == k - 1 and script:
            pol.selector.script = list(script)

    log = EpisodeLog(out / "branches" / f"{name}-k{k}.jsonl", run_id=out.name,
                     episode_id=f"{task}-{seed}-k{k}-{name}", seed=seed, task=task,
                     bank_hash=manifest["bank_hash"], store_states=False)
    t0 = time.monotonic()
    result = run_episode(env, policy, seed=seed, max_steps=max_steps, log=log, on_step=on_step)
    log.close()
    env.close()
    rows = read_jsonl(out / "branches" / f"{name}-k{k}.jsonl")
    pre = rows[k - 1] if k > 0 else None
    window_rows = [r for r in rows if k <= r["control_step"] < max(branch_end, k + window)]
    end = window_rows[-1] if window_rows else rows[-1]
    rej = {}
    for r in window_rows:
        if r["override_reason_code"]:
            key = r["override_reason"] + (":ik_not_converged" if r["override_reason_code"] == 2
                                          and not r["ik_success"] else "")
            rej[key] = rej.get(key, 0) + 1
    disp = np.sum([r["actual_displacements"]["base_xy_m"] for r in window_rows], axis=0).tolist() if window_rows else None
    return dict(
        branch=name, state=k, seed=seed, task=task, script_len=len(script), forced_module=module,
        window_steps=len(window_rows), prefix_steps=k, total_env_steps=result.steps,
        forced_steps=sum(r["forced"] for r in window_rows),
        scripted_steps=sum(r["scripted"] for r in window_rows),
        proposed_counts=_count(r["proposed_id"] for r in window_rows),
        executed_counts=_count(r["executed_id"] for r in window_rows),
        rejections=rej,
        dist_before=pre["reward_components"]["cube_goal_dist_xy_after_m"] if pre else None,
        dist_after_window=end["reward_components"]["cube_goal_dist_xy_after_m"],
        min_dist_window=min(r["reward_components"]["cube_goal_dist_xy_after_m"] for r in window_rows) if window_rows else None,
        grasped_after_window=end["reward_components"]["grasped_after"],
        lift_after_window=end["reward_components"]["cube_lift_after_m"],
        base_xy_displacement_window=disp,
        terminal=dict(success=result.success, steps=result.steps,
                      first_success_step=result.first_success_step,
                      max_consecutive=result.max_consecutive, final=result.final,
                      module_counts=result.module_counts),
        wall_seconds=time.monotonic() - t0)


def _count(values):
    out = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task", default="true_place")
    parser.add_argument("--seed", type=int, default=3014)
    parser.add_argument("--state", type=int, action="append", required=True)
    parser.add_argument("--branch", action="append", default=None)
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--bundle", default="dist_autonomous_bundle_v1")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    branches = args.branch or list(BRANCHES)
    json_write(args.out / "manifest.json", dict(
        run_id=args.out.name, created_at=utc_now(), command=sys.argv, task=args.task, seed=args.seed,
        states=args.state, window=args.window, branches={b: dict(script=BRANCHES[b][0],
                                                                  module=BRANCHES[b][1]) for b in branches},
        restore_method="reset + deterministic replay of the unchanged policy for k steps",
        safety="executor IK/joint/table/base-path checks unchanged in every branch",
        provenance=provenance()))
    jobs = [dict(out=str(args.out), task=args.task, seed=args.seed, state=k, branch=b,
                 window=args.window, max_steps=args.max_steps, bundle=args.bundle)
            for k in args.state for b in branches]
    t0 = time.monotonic()
    with get_context("spawn").Pool(args.workers) as pool:
        results = []
        for r in pool.imap_unordered(worker, jobs):
            results.append(r)
            print(f"k={r['state']:4d} {r['branch']:36s} d0={r['dist_before']:.3f} "
                  f"dW={r['dist_after_window']:.3f} min={r['min_dist_window']:.3f} "
                  f"grasp={r['grasped_after_window']} rej={r['rejections']} "
                  f"final={r['terminal']['final']['cube_goal_dist_xy_m']:.3f} "
                  f"success={r['terminal']['success']}", flush=True)
    results.sort(key=lambda r: (r["state"], branches.index(r["branch"])))
    json_write(args.out / "summary.json", dict(
        run_id=args.out.name, finished_at=utc_now(), wall_seconds=time.monotonic() - t0,
        env_steps_total=sum(r["total_env_steps"] for r in results),
        prefix_steps_total=sum(r["prefix_steps"] for r in results), results=results))


if __name__ == "__main__":
    main()
