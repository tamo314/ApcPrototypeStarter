"""T47: counterfactual applicability labels at Candidate decisions, and the gate fit.

For each cell of a bank-matrix run where the bank's router handed control to a
candidate_* module (non-committed decision at control step k), two branches are
re-run from reset with deterministic prefix replay of the logged proposed IDs:

* ``continue``: the unchanged bank from step k (the router proposes the candidate again);
* ``veto``: the same bank, but every later candidate proposal keeps Base control.

Label "allow" when continue beats veto (success vs failure, or both failed and the
final cube-goal xy distance is at least ``--dist-margin`` closer); otherwise "veto".
``continue`` should reproduce the logged episode; the agreement is reported as the
replay fidelity.  The gate is a shallow CART over the router-v2 features at k,
saved as the bank role ``gate`` (the candidates and the router are not changed).

  python scripts/run_t47_counterfactual.py --out runs/t47-cf-20260926-a \
     --run runs/t47-tune-v3-20260926-a --bank-name v3 --bank <bank dir>
  python scripts/run_t47_counterfactual.py --fit-only --out runs/t47-gate-even \
     --labels runs/t47-cf-20260926-a/labels.jsonl --bank <bank dir> --seed-parity even
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_bank_matrix import load_bank  # noqa: E402
from run_t32r3_acquire import write_bank  # noqa: E402

from apc_maniskill.experience_loop import (CART, build_policy, make_task_env, read_jsonl,  # noqa: E402
                                           run_episode, save_gate)
from apc_maniskill.runner import json_write, provenance, utc_now  # noqa: E402


def branch(job):
    bank_dir, task, seed, prefix, mode, max_steps, work = job
    bank = load_bank(Path(bank_dir))
    env = make_task_env(task, Path(work))
    policy = build_policy(env, Path(work), bank,
                          candidate_gate=(lambda x2: False) if mode == "veto" else None)
    try:
        res = run_episode(env, policy, seed=seed, max_steps=max_steps, initial_script=list(prefix))
        return dict(mode=mode, success=res.success, steps=res.steps,
                    final_dist=res.final.get("cube_goal_dist_xy_m"),
                    candidate_steps=sum(v for k, v in res.module_counts.items() if k.startswith("candidate")),
                    run_error=None)
    except Exception as exc:  # recorded, never a label
        return dict(mode=mode, run_error=f"{type(exc).__name__}: {exc}", steps=0)
    finally:
        env.close()


def label(cont, veto, margin):
    if cont["success"] != veto["success"]:
        return int(cont["success"])
    if not cont["success"] and cont["final_dist"] is not None and veto["final_dist"] is not None:
        return int(cont["final_dist"] <= veto["final_dist"] - margin)
    return 0  # both succeed: the candidate is not needed there


def fit_gate(rows, depth, min_leaf):
    x = np.asarray([r["router_x2"] for r in rows], np.float32)
    y = np.asarray([r["label"] for r in rows], np.int64)
    w = np.ones(len(y), np.float32)
    if len(np.unique(y)) == 1:  # constant gate
        tree = CART.fit(np.vstack([x, x[:1]]), np.append(y, 1 - y[0]), np.append(w, 1e-6), 2,
                        max_depth=1, min_leaf=len(y) + 1)
    else:
        tree = CART.fit(x, y, w, 2, max_depth=depth, min_leaf=min_leaf)
    with np.errstate(all="ignore"):
        import torch
        pred = tree(torch.from_numpy(x)).argmax(-1).numpy()
    return tree, float((pred == y).mean())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bank", type=Path, required=True, help="bank that was run (gate is added to it)")
    p.add_argument("--run", type=Path, action="append", default=[], help="bank-matrix runs")
    p.add_argument("--bank-name", default="v3")
    p.add_argument("--labels", type=Path, action="append", default=[], help="existing labels.jsonl")
    p.add_argument("--fit-only", action="store_true")
    p.add_argument("--seed-parity", choices=["all", "even", "odd"], default="all",
                   help="fit on labels of these seeds only (cross-validation folds)")
    p.add_argument("--max-decisions", type=int, default=6, help="per cell")
    p.add_argument("--only-seed", type=int, action="append", default=[], help="smoke tests")
    p.add_argument("--dist-margin", type=float, default=0.01)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--min-leaf", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=1200)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    log = dict(run_id=args.out.name, created_at=utc_now(), command=sys.argv, provenance=provenance(),
               bank=str(args.bank), bank_hash=load_bank(args.bank).manifest()["bank_hash"])
    rows = [r for f in args.labels for r in read_jsonl(f)]
    if not args.fit_only:
        points = []
        for run in args.run:
            s = json.loads((run / "summary.json").read_text())
            for c in s["cells"]:
                if c["bank"] != args.bank_name or c["run_error"] or (args.only_seed and c["seed"] not in args.only_seed):
                    continue
                cell = run / "cells" / f"{c['bank']}__{c['task']}_{c['seed']}"
                trans = sorted(read_jsonl(cell / "transitions.jsonl"), key=lambda r: r["control_step"])
                dec = [r for r in trans if str(r["selected_module"]).startswith("candidate")
                       and not r.get("committed")][:args.max_decisions]
                for r in dec:
                    k = r["control_step"]
                    points.append(dict(task=c["task"], seed=c["seed"], step=k, module=r["selected_module"],
                                       router_x2=r["router_x2"], actual_success=c["success"],
                                       actual_final_dist=c["final"]["cube_goal_dist_xy_m"],
                                       prefix=[t["proposed_id"] for t in trans[:k]]))
        jobs = []
        for i, pt in enumerate(points):
            for mode in ("continue", "veto"):
                jobs.append((str(args.bank), pt["task"], pt["seed"], pt["prefix"], mode, args.max_steps,
                             str(args.out / "work" / f"{i:04d}_{mode}")))
        print(f"{len(points)} decision points, {len(jobs)} branches", flush=True)
        with get_context("spawn").Pool(args.workers) as pool:
            results = pool.map(branch, jobs, chunksize=1)
        with (args.out / "labels.jsonl").open("w") as fh:
            for i, pt in enumerate(points):
                cont, veto = results[2 * i], results[2 * i + 1]
                row = {k: v for k, v in pt.items() if k != "prefix"} | dict(
                    prefix_steps=len(pt["prefix"]), cont=cont, veto=veto)
                if cont["run_error"] or veto["run_error"]:
                    row["label"] = None
                else:
                    row["label"] = label(cont, veto, args.dist_margin)
                    row["replay_faithful"] = (cont["success"] == pt["actual_success"]
                                              and abs(cont["final_dist"] - pt["actual_final_dist"]) < 1e-3)
                fh.write(json.dumps(row) + "\n")
                rows.append(row)
        log.update(decision_points=len(points), branch_env_steps=sum(r["steps"] for r in results),
                   run_errors=sum(bool(r["run_error"]) for r in results))
    valid = [r for r in rows if r.get("label") is not None]
    faithful = [r for r in valid if r.get("replay_faithful")]
    fold = [r for r in faithful if args.seed_parity == "all" or (r["seed"] % 2 == 0) == (args.seed_parity == "even")]
    log.update(labels=len(valid), replay_faithful=len(faithful),
               allow=sum(r["label"] for r in valid), fit_rows=len(fold), seed_parity=args.seed_parity,
               outcome_pairs={f"cont={a},veto={b}": sum(1 for r in valid if (r["cont"]["success"], r["veto"]["success"]) == (a, b))
                              for a in (True, False) for b in (True, False)})
    if fold:
        tree, acc = fit_gate(fold, args.depth, args.min_leaf)
        gate_path = args.out / "gate.pt"
        save_gate(gate_path, tree, source=dict(rows=len(fold), allow=sum(r["label"] for r in fold),
                                               seed_parity=args.seed_parity, dist_margin=args.dist_margin,
                                               depth=args.depth, min_leaf=args.min_leaf))
        bank = load_bank(args.bank)
        roles = dict(base=bank.base, router=bank.router, transit=bank.transit, place=bank.place,
                     parent_router=bank.parent_router, **bank.candidate_roles(), gate=gate_path)
        manifest = write_bank(args.out / "bank_gated", {k: v for k, v in roles.items() if v is not None},
                              bank.manifest(), dict(design="extend", gate=args.seed_parity))
        log.update(gate=dict(nodes=int(tree.feature.shape[0]), train_accuracy=acc,
                             bank_dir=str(args.out / "bank_gated"), bank_hash=manifest["bank_hash"]))
    log.update(finished_at=utc_now(), wall_seconds=time.monotonic() - t0)
    json_write(args.out / "summary.json", log)
    print(json.dumps({k: v for k, v in log.items() if k not in ("command", "provenance")}, indent=1))
    return log


if __name__ == "__main__":
    main()
