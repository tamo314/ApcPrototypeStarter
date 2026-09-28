"""T61: refit an option Candidate from the decision rows of acquisition runs.

Rows are (candidate_x at a decision step, outcome-selected option) exactly as
run_t32r3_acquire collects them.  ``--success-only`` keeps only events whose
acquired trajectory reached task success (the others still carry the search's
best option, but the trajectory they lead to failed).

  python scripts/refit_candidate.py --acquisition runs/t57-acquire-20260927-a \
     --acquisition runs/t34-stream-AC-20260926-d/acquire_t02_candidate_A --success-only \
     --reference runs/t57-acquire-20260927-a/candidate_A.pt --out runs/t61-cand-succ/candidate_A.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from apc_maniskill.experience_loop import fit_cart, read_jsonl, save_option_candidate
from apc_maniskill.runner import json_write, provenance, utc_now


def rows(acq_dirs, name, options, success_only):
    xs, ys, used = [], [], []
    for d in acq_dirs:
        s = json.loads((d / "acquisition_summary.json").read_text())
        if Path(s.get("candidate", {}).get("path", "")).stem != name:
            continue
        for e in s["events"]:
            a = e.get("acquisition")
            if not a or not a.get("chosen_options"):
                continue
            ok = bool(e.get("acquired_replay", {}).get("success"))
            if success_only and not ok:
                continue
            by_step = {r["control_step"]: r for r in read_jsonl(d / e["event_id"] / "acquired_trajectory.jsonl")}
            for step, choice in zip(a["decision_steps"], a["chosen_options"]):
                if tuple(choice) in options:
                    xs.append(by_step[step]["candidate_x"])
                    ys.append(options.index(tuple(choice)))
            used.append(dict(run=str(d), seed=e["seed"], acquired_success=ok, rows=len(a["decision_steps"])))
    return xs, ys, used


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--acquisition", type=Path, action="append", required=True)
    p.add_argument("--reference", type=Path, required=True, help="candidate checkpoint giving the option list")
    p.add_argument("--candidate-name", default="candidate_A")
    p.add_argument("--success-only", action="store_true")
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    ck = torch.load(args.reference, map_location="cpu", weights_only=False)
    options = [tuple(o) for o in ck["options"]]
    xs, ys, used = rows(args.acquisition, args.candidate_name, options, args.success_only)
    if not xs:
        raise SystemExit("no candidate rows")
    tree, mean, std = fit_cart(xs, ys, len(options), max_depth=args.depth, min_leaf=1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta = save_option_candidate(args.out, tree, mean, std, options,
                                 source=dict(rows=len(ys), success_only=args.success_only, events=used))
    log = dict(created_at=utc_now(), command=sys.argv, provenance=provenance(), candidate=meta, rows=len(ys),
               events=used, success_only=args.success_only)
    json_write(args.out.with_suffix(".json"), log)
    print(json.dumps(dict(rows=len(ys), events=len(used), nodes=meta["tree_nodes"])))
    return log


if __name__ == "__main__":
    main()
