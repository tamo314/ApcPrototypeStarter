"""T46: where do Candidate interventions help or hurt? (tuning conditions only)

For every cell of the given bank-matrix runs where a candidate_* module was
selected (non-committed decision), compare the outcome with the reference bank on
the same condition (gain / loss / both_fail / both_success) and record, at each
candidate decision, the router's candidate probability and the router-v2 context
(stall, xy distance, grasped).  Output: per-cell rows plus threshold tables
"if candidate decisions with p < tau (or stall < s) had been vetoed, which cells
would have had no candidate decision".  This is descriptive only: vetoing a
decision changes the trajectory, so the closed-loop effect is measured in T47.

  python scripts/analyze_t46_interventions.py --out docs/T46_INTERVENTIONS.json \
     --run runs/t42-final-20260926-a:initial --run runs/t34-matrix-20260926-b:old
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from apc_maniskill.experience_loop import load_router, read_jsonl

STALL, DIST, GRASPED = 12, 5, 1  # router_x2 indices


def cell_rows(run: Path, ref: str, exclude: set[int]):
    s = json.loads((run / "summary.json").read_text())
    man = json.loads((run / "manifest.json").read_text())
    banks = man.get("banks", man)
    cells = {(c["bank"], c["seed"]): c for c in s["cells"]}
    out = []
    for (bank, seed), c in sorted(cells.items()):
        if bank == ref or seed in exclude or c["run_error"]:
            continue
        if not any(k.startswith("candidate") for k in (c.get("module_counts") or {})):
            continue
        router = load_router(Path(banks[bank]["files"]["router"]["path"]))
        names = router.class_names
        cand_idx = [i for i, n in enumerate(names) if n.startswith("candidate")]
        dec = []
        for r in read_jsonl(run / "cells" / f"{bank}__{c['task']}_{seed}" / "transitions.jsonl"):
            if not str(r["selected_module"]).startswith("candidate") or r.get("committed"):
                continue
            x2 = np.asarray(r["router_x2"], np.float32)
            with torch.no_grad():
                p = torch.exp(router(torch.from_numpy(x2))).numpy()
            dec.append(dict(step=r["control_step"], module=r["selected_module"],
                            p=float(p[cand_idx].max()), stall=float(x2[STALL] * 100),
                            dist=float(x2[DIST]), grasped=bool(x2[GRASPED] > 0.5)))
        ref_ok = cells[(ref, seed)]["success"]
        outcome = {(False, True): "gain", (True, False): "loss", (False, False): "both_fail",
                   (True, True): "both_success"}[(ref_ok, c["success"])]
        out.append(dict(run=run.name, bank=bank, seed=seed, outcome=outcome, decisions=dec,
                        min_p=min(d["p"] for d in dec) if dec else None,
                        max_p=max(d["p"] for d in dec) if dec else None,
                        max_stall=max(d["stall"] for d in dec) if dec else None))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--run", action="append", required=True, help="run_dir:reference_bank")
    p.add_argument("--exclude-seed", type=int, action="append", default=[])
    args = p.parse_args()
    rows = []
    for spec in args.run:
        run, ref = spec.rsplit(":", 1)
        rows += cell_rows(Path(run), ref, set(args.exclude_seed))
    tables = {}
    for key, grid in (("p", [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]), ("stall", [0, 5, 10, 20, 40, 80])):
        t = []
        for th in grid:
            kept = [r for r in rows if any(d[key] >= th for d in r["decisions"])]
            cnt = {o: sum(r["outcome"] == o for r in kept) for o in ("gain", "loss", "both_fail", "both_success")}
            t.append(dict(threshold=th, cells_with_candidate=len(kept), **cnt))
        tables[key] = t
    by = {}
    for r in rows:
        by.setdefault(r["outcome"], []).append(r)
    summary = {o: dict(n=len(v), median_max_p=float(np.median([r["max_p"] for r in v])),
                       median_max_stall=float(np.median([r["max_stall"] for r in v])),
                       median_decisions=float(np.median([len(r["decisions"]) for r in v])))
               for o, v in by.items()}
    out = dict(note="descriptive; open-loop veto counts, closed-loop effect in T47", runs=args.run,
               excluded_seeds=args.exclude_seed, summary=summary, tables=tables,
               cells=[{k: v for k, v in r.items() if k != "decisions"} | dict(n_decisions=len(r["decisions"]))
                      for r in rows])
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(dict(summary=summary, tables=tables), indent=1))


if __name__ == "__main__":
    main()
