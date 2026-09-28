"""T39: stronger shared-model baseline via DAgger on the teacher bank's decisions.

The shared model (one CART or MLP over schema-v5 features, with the same shared
grasp-recovery guard and executor safety as APC) is first fit on replay rows of the
teacher bank (as in T29).  Each DAgger round deploys the shared model on the
development conditions while the teacher bank runs in shadow on the same states
(its internal state follows the student's trajectory) and records its choice; the
aggregated rows are refit.  The teacher is the bank APC itself starts from or has
grown into, so both methods receive the same knowledge; every deployment step is
counted as cost.

Example:
  python scripts/run_t39_shared_dagger.py --out runs/t39-dagger-20260926-a \
     --teacher runs/t33c-acquire-20260926-b/bank_A_full --replay runs/t34-replay-bankAC-20260926-a \
     --dev true_place:3011 --rounds 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_t29_shared_baselines import constant_router, train_mlp  # noqa: E402
from run_t32r3_acquire import load_bank  # noqa: E402

from apc_maniskill.experience_loop import (Bank, EpisodeLog, RoutedSelector, build_policy,  # noqa: E402
                                           fit_cart, load_module, load_router, make_task_env,
                                           read_jsonl, run_episode, save_candidate)
from apc_maniskill.primitive_learning import SCHEMA_V5  # noqa: E402
from apc_maniskill.primitive_policy import GraspRecoveryGuardSelector  # noqa: E402
from apc_maniskill.primitives import NAMES20  # noqa: E402
from apc_maniskill.runner import json_write, provenance, utc_now  # noqa: E402


class Shadow:
    """Execute the student; query the teacher on the same state (label only)."""

    def __init__(self, student: RoutedSelector, teacher: RoutedSelector):
        self.student, self.teacher = student, teacher
        self.metadata = dict(student.metadata, selector="dagger_shadow_v1")

    @property
    def primitive_count(self):
        return 20

    def reset(self):
        self.student.reset()
        self.teacher.reset()
        self.decision = {}
        self.active_module = "shared"

    def select(self, step, observation):
        teacher_id = int(self.teacher.select(step, observation))
        action = int(self.student.select(step, observation))
        self.decision = dict(self.student.decision, teacher_id=teacher_id)
        self.active_module = self.student.active_module
        self.last_scores = self.student.last_scores
        return action


def routed(bank: Bank, env) -> RoutedSelector:
    meta = env.unwrapped.experiment_metadata()
    freq = float(env.unwrapped.sim_config.control_freq)
    base = load_module(bank.base, meta, freq)
    modules = {"transit": load_module(bank.transit, meta, freq) if bank.transit else None,
               "place": load_module(bank.place, meta, freq) if bank.place else None,
               **{k: load_module(v, meta, freq) for k, v in bank.candidate_roles().items()}}
    return RoutedSelector(base, GraspRecoveryGuardSelector(base), modules, load_router(bank.router))


def deploy(job):
    out, student_dir, teacher_dir, task, seed = job
    student, teacher = load_bank(Path(student_dir)), load_bank(Path(teacher_dir))
    cell = Path(out) / f"{task}_{seed}"
    env = make_task_env(task, cell)
    policy = build_policy(env, cell, student)
    policy.selector = Shadow(routed(student, env), routed(teacher, env))
    log = EpisodeLog(cell / "transitions.jsonl", run_id=Path(out).name, episode_id=f"{task}-{seed}",
                     seed=seed, task=task, bank_hash=student.manifest()["bank_hash"], store_states=False)
    res = run_episode(env, policy, seed=seed, max_steps=1200, log=log)
    log.close()
    env.close()
    return dict(task=task, seed=seed, success=res.success, steps=res.steps, log=str(cell / "transitions.jsonl"))


def write_shared_bank(dst: Path, model_path: Path) -> Path:
    dst.mkdir(parents=True, exist_ok=False)
    constant_router(dst / "router.pt")
    (dst / "base.pt").write_bytes(model_path.read_bytes())
    bank = Bank(base=dst / "base.pt", router=dst / "router.pt")
    json_write(dst / "bank_manifest.json", dict(schema="apc_bank_version_v1", created_at=utc_now(),
                                                roles=dict(base="base.pt", router="router.pt"),
                                                **bank.manifest(), method="shared_dagger"))
    return dst


def fit(kind, x, y, out: Path, task_meta, depth, updates):
    if kind == "cart":
        tree, mean, std = fit_cart(x, y, 20, max_depth=depth, min_leaf=2)
        save_candidate(out, tree, mean, std, task_meta=task_meta, source=dict(rows=len(y), sha256=""))
        return dict(nodes=int(tree.feature.shape[0]))
    model, mean, std, acc = train_mlp(x, y, updates=updates)
    torch.save({"schema": SCHEMA_V5, "primitive_names": list(NAMES20), "task": task_meta,
                "control_freq": 20.0, "state_dict": model.state_dict(), "mean": mean, "std": std,
                "source_steps_sha256": "", "model_kind": "mlp", "updates": updates,
                "allow_parameterized_goal_tasks": True}, out)
    return dict(parameters=sum(p.numel() for p in model.parameters()), train_accuracy=acc)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--replay", type=Path, action="append", required=True,
                   help="run dirs (or matrix runs with cells/) of the teacher bank")
    p.add_argument("--dev", action="append", required=True)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--kind", choices=["cart", "mlp"], default="cart")
    p.add_argument("--depth", type=int, default=16)
    p.add_argument("--updates", type=int, default=4000)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    teacher = load_bank(args.teacher)
    task_meta = json.loads(json.dumps(torch.load(teacher.base, weights_only=False)["task"]))
    x, y, replay_steps = [], [], 0
    for run in args.replay:
        files = [run / "transitions.jsonl"] if (run / "transitions.jsonl").exists() else \
            sorted((run / "cells").glob("*/transitions.jsonl"))
        for f in files:
            for r in read_jsonl(f):
                replay_steps += 1
                x.append(r["candidate_x"])
                y.append(r["proposed_id"])
    rounds, deploy_steps = [], 0
    for k in range(args.rounds + 1):
        model_path = args.out / f"shared_{args.kind}_r{k}.pt"
        info = fit(args.kind, x, y, model_path, task_meta, args.depth, args.updates)
        bank_dir = write_shared_bank(args.out / f"bank_r{k}", model_path)
        entry = dict(round=k, rows=len(y), model=info, bank=str(bank_dir))
        if k < args.rounds:
            jobs = [(str(args.out / f"deploy_r{k}"), str(bank_dir), str(args.teacher), *d.split(":"))
                    for d in args.dev]
            jobs = [(a, b, c, t, int(s)) for a, b, c, t, s in jobs]
            with get_context("spawn").Pool(args.workers) as pool:
                results = pool.map(deploy, jobs)
            added = 0
            for res in results:
                for r in read_jsonl(Path(res["log"])):
                    x.append(r["candidate_x"])
                    y.append(r["teacher_id"])
                    added += 1
            deploy_steps += sum(r["steps"] for r in results)
            entry.update(deploy_success=sum(r["success"] for r in results), deploy_n=len(results),
                         added_rows=added, deploy_env_steps=sum(r["steps"] for r in results))
        rounds.append(entry)
        print(json.dumps(entry), flush=True)
    json_write(args.out / "summary.json", dict(
        run_id=args.out.name, created_at=utc_now(), command=sys.argv, provenance=provenance(),
        teacher_bank_hash=teacher.manifest()["bank_hash"], kind=args.kind, rounds=rounds,
        replay_source_env_steps=replay_steps, deploy_env_steps=deploy_steps,
        env_steps=deploy_steps, wall_seconds=time.monotonic() - t0))


if __name__ == "__main__":
    main()
