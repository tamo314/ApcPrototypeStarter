"""T33-B: learn when to hand over to the fixed Place module (routing adaptation).

Base, Transit, Place and the router stay fixed.  For each accepted place-stall event:

1. run the task with place entry suppressed ("continue existing control") to see
   which boundary states the unchanged Base reaches;
2. from boundary states on that path (prefix replay), branch "enter Place now"
   and record the terminal outcome (release + settle + 20-step hold);
3. fit a small CART gate on the shared schema-v5 features: enter / not yet.

Positive/negative place entries already observed in replayed development runs
(parent-bank logs) are added with their recorded outcomes.  The T28 0.015 m
threshold is only a diagnostic comparison, not a label.

Example:
  python scripts/run_t33b_place_entry.py --out runs/t33b-place-entry-20260926-a \
     --event runs/t32r1-transitions-20260926-a/events/EVT-5739812cb384.json \
     --replay runs/t32r1-transitions-20260926-a
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

from apc_maniskill.experience_loop import (Bank, EpisodeLog, build_policy, candidate_features,
                                           fit_cart, make_task_env, read_jsonl, run_episode)
from apc_maniskill.primitive_tree import CART
from apc_maniskill.runner import json_write, provenance, utc_now


class PlaceGate:
    """Learned place-entry gate on schema-v5 features (enter now / not yet)."""

    def __init__(self, path: Path, boundary_m: float = 0.05):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.kind = ck.get("kind", "classifier")
        if self.kind == "value":
            self.grid, self.enter = np.asarray(ck["grid_m"]), np.asarray(ck["enter_on_grid"])
            return
        self.tree = CART(ck["n_nodes"], 2)
        self.tree.load_state_dict(ck["state_dict"])
        self.mean, self.std = ck["mean"], ck["std"]
        self.boundary_m = ck.get("boundary_m", boundary_m)

    def __call__(self, step, s, router_x):
        if self.kind == "value":
            return bool(self.enter[min(np.searchsorted(self.grid, router_x[5]), len(self.grid) - 1)])
        x = (np.asarray(router_x, np.float32) - self.mean) / self.std
        with torch.no_grad():
            return bool(int(self.tree(torch.from_numpy(x.astype(np.float32))).argmax()) == 1)


def threshold_gate(threshold):
    def gate(step, s, router_x):
        return bool(router_x[5] <= threshold)
    return gate


def _suppress_until(k):
    """Veto the router's Place proposals before step k, accept from k on."""
    def gate(step, s, router_x):
        return step >= k
    return gate


LEVELS = (0.03, 0.025, 0.02, 0.017, 0.015, 0.013, 0.011, 0.009, 0.007, 0.005)


def branch(job):
    out, task, seed, k, bank_dir, max_steps = job
    bank = Bank.initial(Path(bank_dir))
    env = make_task_env(task, Path(out) / "env" / f"{seed}-{k}")
    gate = _suppress_until(k if k is not None else 10 ** 9)
    policy = build_policy(env, Path(out), bank, place_gate=gate)
    name = f"suppressed-{seed}" if k is None else f"enter-{seed}-k{k}"
    log = EpisodeLog(Path(out) / "branches" / f"{name}.jsonl", run_id=Path(out).name, episode_id=name,
                     seed=seed, task=task, bank_hash=bank.manifest()["bank_hash"], store_states=True)
    t0 = time.monotonic()
    res = run_episode(env, policy, seed=seed, max_steps=max_steps, log=log)
    log.close()
    env.close()
    return dict(seed=seed, task=task, k=k, success=res.success, steps=res.steps,
                final=res.final, module_counts=res.module_counts, wall=time.monotonic() - t0,
                log=str(Path(out) / "branches" / f"{name}.jsonl"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--event", type=Path, action="append", default=[])
    p.add_argument("--replay", type=Path, action="append", default=[])
    p.add_argument("--bank", default="dist_autonomous_bundle_v1")
    p.add_argument("--known", action="append", default=[],
                   help="task:seed development conditions also branched (early vs late entry)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--gate-kind", choices=["classifier", "value"], default="value",
                   help="value: enter iff the estimated success of entering now is not below "
                        "the best estimate reachable by waiting (closer distances)")
    p.add_argument("--value-eps", type=float, default=0.02)
    p.add_argument("--from-run", type=Path, default=None,
                   help="reuse suppressed/branch rollouts of a previous run (no new env steps)")
    p.add_argument("--max-steps", type=int, default=1200)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    events = [json.loads(e.read_text()) for e in args.event]
    events = [e for e in events if e["reason"].startswith("Near-goal placement stall")]
    log = dict(run_id=args.out.name, created_at=utc_now(), command=sys.argv, provenance=provenance(),
               events=[e["event_id"] for e in events], known=args.known)
    if not events:
        log["result"] = "no_event_no_learning"
        json_write(args.out / "summary.json", log)
        return log
    t0 = time.monotonic()
    ctx = get_context("spawn")
    if args.from_run is not None:
        prev = json.loads((args.from_run / "summary.json").read_text())
        suppressed, branches = prev["suppressed"], prev["branches"]
        log["reused_rollouts_from"] = str(args.from_run)
    else:
        sources = [(e["task"], e["environment_seed"], e["observation_history"]) for e in events]
        sources += [(k.split(":")[0], int(k.split(":")[1]), None) for k in args.known]
        with ctx.Pool(args.workers) as pool:
            suppressed = pool.map(branch, [(str(args.out), task, seed, None, args.bank, args.max_steps)
                                           for task, seed, _ in sources])
            jobs = []
            for (task, seed, history), sup in zip(sources, suppressed):
                rows = read_jsonl(Path(sup["log"]))
                hist = ([r for r in read_jsonl(Path(history)) if r["environment_seed"] == seed]
                        if history else [])
                orig_entry = next((r["control_step"] for r in hist if r["selected_module"] == "place"), None)
                # States where the router proposes Place; first crossing of each distance level.
                proposing = [r for r in rows if r["router_class"] == 3 and r["s_t"]["grasped"]]
                ks = sorted({next(r["control_step"] for r in proposing if r["router_x"][5] <= level)
                             for level in LEVELS if any(r["router_x"][5] <= level for r in proposing)})
                if orig_entry is not None and orig_entry < len(rows):
                    ks = sorted(set(ks + [orig_entry]))
                jobs += [(str(args.out), task, seed, k, args.bank, args.max_steps) for k in ks]
            branches = pool.map(branch, jobs)
    xs, ys, meta = [], [], []
    for b in branches:
        rows = read_jsonl(Path(b["log"]))
        r = rows[b["k"]]
        xs.append(r["router_x"])
        ys.append(int(b["success"]))
        meta.append(dict(source="branch", seed=b["seed"], k=b["k"], dist=r["router_x"][5],
                         success=b["success"]))
    # Recorded place entries from parent-bank development logs, with their outcomes.
    for run in args.replay:
        rows = read_jsonl(run / "transitions.jsonl")
        for ep in sorted({r["episode_id"] for r in rows}):
            er = [r for r in rows if r["episode_id"] == ep]
            entry = next((r for r in er if r["selected_module"] == "place"), None)
            if entry is None:
                continue
            ok = any(r["terminated"] for r in er)
            xs.append(entry["router_x"])
            ys.append(int(ok))
            meta.append(dict(source="replay", seed=entry["environment_seed"], k=entry["control_step"],
                             dist=entry["router_x"][5], success=ok))
    gate_path = args.out / "place_gate.pt"
    if args.gate_kind == "value":
        # P(success | enter at distance d) from branch/replay outcomes (1-D CART),
        # then enter iff no closer distance has a higher estimated success.
        d = np.asarray(xs, np.float32)[:, 5:6]
        tree, mean, std = fit_cart(d, ys, 2, max_depth=3, min_leaf=4)
        grid = np.linspace(0.0, 0.05, 201, dtype=np.float32)
        with torch.no_grad():
            prob = tree(torch.from_numpy(((grid[:, None] - mean) / std).astype(np.float32))).exp()[:, 1].numpy()
        best_closer = np.maximum.accumulate(prob)
        enter = prob >= best_closer - args.value_eps
        torch.save(dict(schema="place_entry_gate_v2", kind="value", grid_m=grid.tolist(),
                        success_estimate=prob.tolist(), enter_on_grid=enter.tolist(),
                        eps=args.value_eps, rows=len(ys), state_dict=tree.state_dict(),
                        n_nodes=int(tree.feature.shape[0]),
                        semantics="veto: consulted only when the router proposes Place"), gate_path)
        acc = None
        first_wait = float(grid[np.argmax(~enter)]) if (~enter).any() else None
        log["value_gate"] = dict(enter_up_to_m=first_wait,
                                 estimate_by_cm={f"{g*100:.1f}": round(float(p), 3)
                                                 for g, p in zip(grid[::10], prob[::10])})
    else:
        tree, mean, std = fit_cart(xs, ys, 2, max_depth=3, min_leaf=2)
        torch.save(dict(schema="place_entry_gate_v1", feature_schema="router_v1",
                        semantics="veto: consulted only when the router proposes Place",
                        state_dict=tree.state_dict(), n_nodes=int(tree.feature.shape[0]), mean=mean,
                        std=std, boundary_m=0.05, rows=len(ys)), gate_path)
        acc = float((tree(torch.from_numpy((np.asarray(xs, np.float32) - mean) / std)).argmax(1).numpy()
                     == np.asarray(ys)).mean())
    log.update(result="learned", gate=str(gate_path), rows=len(ys), positives=int(sum(ys)),
               train_accuracy=acc, tree_nodes=int(tree.feature.shape[0]),
               suppressed=suppressed, branches=branches, labels=meta,
               env_steps=0 if args.from_run else sum(b["steps"] for b in suppressed + branches),
               wall_seconds=time.monotonic() - t0)
    json_write(args.out / "summary.json", log)
    for m in meta:
        print(m)
    print({k: log[k] for k in ("rows", "positives", "train_accuracy", "tree_nodes", "env_steps")})
    return log


if __name__ == "__main__":
    main()
