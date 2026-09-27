"""T51: tuning success per candidate-use budget, from counterfactual labels (no new simulation).

Vetoing candidates from the k-th non-committed candidate decision on equals a budget of
k-1 candidate decisions per episode; cells with at most b decisions keep their
logged outcome.  Decisions beyond --max-decisions (6 in T47/T48) were not branched,
so budgets are reported up to that cap.

  python scripts/analyze_t51_budget.py --labels runs/t48-cf-defer-20260926-a/labels.jsonl \
     --matrix runs/t48-tune-defer-20260926-a --out docs/T51_BUDGET.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from apc_maniskill.experience_loop import read_jsonl


def outcomes(labels, matrix: Path, budget: int):
    cells = {c["seed"]: c for c in json.loads((matrix / "summary.json").read_text())["cells"]}
    by = {}
    for r in labels:
        by.setdefault(r["seed"], []).append(r)
    out = {}
    for seed, c in cells.items():
        dec = sorted(by.get(seed, []), key=lambda r: r["step"])
        out[seed] = dec[budget]["veto"]["success"] if budget < len(dec) else c["success"]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--matrix", type=Path, required=True)
    p.add_argument("--reference", action="append", default=[], help="run:bank of the parent bank (for gains/losses)")
    p.add_argument("--max-budget", type=int, default=6)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    labels = read_jsonl(args.labels)
    ref = {}
    for spec in args.reference:
        run, bank = spec.rsplit(":", 1)
        ref.update({c["seed"]: c["success"] for c in json.loads((Path(run) / "summary.json").read_text())["cells"]
                    if c["bank"] == bank})
    rows = []
    for b in range(args.max_budget + 1):
        o = outcomes(labels, args.matrix, b)
        row = dict(budget=b, successes=sum(o.values()), n=len(o))
        if ref:
            row.update(gained=sorted(s for s in o if o[s] and not ref[s]),
                       lost=sorted(s for s in o if ref[s] and not o[s]))
        rows.append(row)
    best = max(rows, key=lambda r: (r["successes"], -r["budget"]))  # pre-registered: max, tie -> smaller
    out = dict(labels=str(args.labels), matrix=str(args.matrix), rows=rows, selected_budget=best["budget"])
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    for r in rows:
        print(r)
    print("selected", best["budget"])


if __name__ == "__main__":
    main()
