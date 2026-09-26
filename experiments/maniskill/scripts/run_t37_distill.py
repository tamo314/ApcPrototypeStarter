"""T37: Temporary -> Candidate distillation by querying the option-search planner.

Temporary = the outcome-evaluated option search (expensive look-ahead planner, not
kept at run time).  Direct Candidate = CART fitted only on the acquired trajectories'
decision rows.  Distilled Candidate = the same rows plus labels obtained by querying
the Temporary at the decision states the deployed Candidate itself visits
(one DAgger round).  The router is left unchanged, so only the behaviour differs.

Query label at a visited decision state: the best option if it beats "let the current
bank continue" (which re-applies the Candidate) by the margin, otherwise the
Candidate's own option at that state (agreement).  Candidate-visited states are
reproduced by prefix replay of the logged proposed IDs.

Example:
  python scripts/run_t37_distill.py --out runs/t37-distill-20260926-a \
     --bank runs/t33c-acquire-20260926-b/bank_A_full \
     --direct runs/t32r3-acquire-20260926-d --query true_place:3014 --query true_place:3050
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_bank_matrix  # noqa: E402
from run_t32r3_acquire import load_bank, write_bank  # noqa: E402

from apc_maniskill.acquisition import acquire  # noqa: E402
from apc_maniskill.experience_loop import fit_cart, read_jsonl, save_option_candidate  # noqa: E402
from apc_maniskill.runner import json_write, provenance, utc_now  # noqa: E402


def direct_rows(acq_dirs, options, name):
    xs, ys = [], []
    for d in acq_dirs:
        s = json.loads((d / "acquisition_summary.json").read_text())
        if Path(s.get("candidate", {}).get("path", "")).stem != name:
            continue
        for e in s["events"]:
            a = e.get("acquisition")
            if not a or not a.get("chosen_options"):
                continue
            rows = {r["control_step"]: r for r in read_jsonl(d / e["event_id"] / "acquired_trajectory.jsonl")}
            for step, choice in zip(a["decision_steps"], a["chosen_options"]):
                xs.append(rows[step]["candidate_x"])
                ys.append(options.index(tuple(choice)))
    return xs, ys


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--candidate-name", default="candidate_A")
    p.add_argument("--direct", type=Path, action="append", required=True,
                   help="acquisition runs holding the direct decision rows")
    p.add_argument("--query", action="append", required=True, help="task:seed to deploy and query")
    p.add_argument("--max-queries-per-episode", type=int, default=3)
    p.add_argument("--horizon", type=int, default=250)
    p.add_argument("--margin", type=float, default=0.003)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    bank = load_bank(args.bank)
    ck = torch.load(bank.candidate_roles()[args.candidate_name], map_location="cpu", weights_only=False)
    options = [tuple(o) for o in ck["options"]]
    from apc_maniskill.primitive_tree import CART
    tree = CART(ck["tree_nodes"], len(options))
    tree.load_state_dict(ck["state_dict"])

    def own_option(x):
        z = (np.asarray(x, np.float32) - ck["mean"]) / ck["std"]
        return int(tree(torch.from_numpy(z.astype(np.float32))).argmax())

    # 1. deploy the current bank (with the direct Candidate) on the query conditions
    argv_m = ["--out", str(args.out / "deploy"), "--bank", f"current={args.bank}",
              "--workers", str(args.workers)]
    for q in args.query:
        argv_m += ["--episode", q]
    run_bank_matrix.main(argv_m)
    deploy = json.loads((args.out / "deploy" / "summary.json").read_text())
    queries, q_steps = [], 0
    for cell_dir in sorted((args.out / "deploy" / "cells").glob("*")):
        rows = read_jsonl(cell_dir / "transitions.jsonl")
        visits = [r for r in rows if r["selected_module"] == args.candidate_name and not r["committed"]]
        for r in visits[:args.max_queries_per_episode]:
            k = r["control_step"]
            event = dict(event_id=f"Q-{r['environment_seed']}-{k}", task=r["task"],
                         environment_seed=r["environment_seed"], detected_at_step=k - 1, streak_steps=0,
                         observation_history=str(cell_dir / "transitions.jsonl"))
            res = acquire(event, bank, args.out / "queries" / event["event_id"], horizon=args.horizon,
                          max_decisions=1, margin=args.margin, workers=args.workers, start="event")
            q_steps += res["search_env_steps"]
            dec = res["decisions"][0]
            own = own_option(r["candidate_x"])
            improved = dec["best_score"] >= dec["baseline_score"] + args.margin
            label = options.index(tuple(dec["best_option"])) if improved else own
            queries.append(dict(seed=r["environment_seed"], task=r["task"], step=k,
                                own_option=list(options[own]), best_option=list(dec["best_option"]),
                                baseline_score=dec["baseline_score"], best_score=dec["best_score"],
                                label=list(options[label]), relabelled=label != own,
                                candidate_x=r["candidate_x"]))
            print(f"query {r['environment_seed']}@{k}: own {options[own]} best {dec['best_option']} "
                  f"improved={improved}", flush=True)
    xs_d, ys_d = direct_rows(args.direct, options, args.candidate_name)
    xs_q = [q["candidate_x"] for q in queries]
    ys_q = [options.index(tuple(q["label"])) for q in queries]
    if not xs_q:
        log = dict(result="no_candidate_visits", deploy=deploy["table"])
        json_write(args.out / "summary.json", log)
        return log
    t_fit = time.monotonic()
    tree_d, mean, std = fit_cart(xs_d + xs_q, ys_d + ys_q, len(options), max_depth=8, min_leaf=1)
    cand_path = args.out / f"{args.candidate_name}_distilled.pt"
    meta = save_option_candidate(cand_path, tree_d, mean, std, options,
                                 source=dict(direct_rows=len(ys_d), query_rows=len(ys_q)))
    roles = dict(base=bank.base, router=bank.router, transit=bank.transit, place=bank.place,
                 **bank.candidate_roles())
    roles[args.candidate_name] = cand_path
    parent = bank.manifest()
    new = write_bank(args.out / "bank_distilled", roles, parent,
                     dict(design="extend", distilled_candidate=args.candidate_name))
    log = dict(run_id=args.out.name, created_at=utc_now(), command=sys.argv, provenance=provenance(),
               result="distilled", parent_bank_hash=parent["bank_hash"], new_bank_hash=new["bank_hash"],
               bank_dir=str(args.out / "bank_distilled"), candidate=meta,
               direct_rows=len(ys_d), query_rows=len(ys_q),
               relabelled=sum(q["relabelled"] for q in queries),
               queries=[{k: v for k, v in q.items() if k != "candidate_x"} for q in queries],
               deploy=deploy["table"],
               cost=dict(deploy_env_steps=deploy["env_steps"], query_env_steps=q_steps,
                         fit_seconds=time.monotonic() - t_fit, wall_seconds=time.monotonic() - t0))
    json_write(args.out / "summary.json", log)
    print(json.dumps({k: log[k] for k in ("direct_rows", "query_rows", "relabelled", "cost")}, indent=1))
    return log


if __name__ == "__main__":
    main()
