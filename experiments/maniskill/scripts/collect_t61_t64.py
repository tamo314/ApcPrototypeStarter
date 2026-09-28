"""T64: collect the router x candidate factor analysis (T62), the final check (T63) and the ledger.

Cells not re-run in T62 are identical to the reference bank with the same router
(trajectories coincide until the first candidate decision; checked on sample cells).

  python scripts/collect_t61_t64.py --out docs/T61_T64_RESULTS.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_t34_t44 import final_table, wilson  # noqa: E402
from collect_t51_t55 import sign_test  # noqa: E402

from apc_maniskill.experience_loop import read_jsonl  # noqa: E402

RUNS = Path("runs")
GROUPS = [("t49-final-20260926-a", range(3130, 3190)), ("t54-final-20260927-a", range(3400, 3480)),
          ("t59-final-20260927-a", range(3480, 3580))]


def cells(run, bank):
    return {c["seed"]: c for c in json.loads((RUNS / run / "summary.json").read_text())["cells"] if c["bank"] == bank}


def paired(x, ref, seeds):
    g = [s for s in seeds if x[s]["success"] and not ref[s]["success"]]
    lo = [s for s in seeds if ref[s]["success"] and not x[s]["success"]]
    return dict(successes=sum(x[s]["success"] for s in seeds), n=len(seeds),
                wilson95=wilson(sum(x[s]["success"] for s in seeds), len(seeds)),
                gained=g, lost=lo, sign_test_p=sign_test(len(g), len(lo)))


def option_counts(run, bank):
    c = Counter()
    for f in (RUNS / run / "cells").glob(f"{bank}__*/transitions.jsonl"):
        for r in read_jsonl(f):
            if r["selected_module"] == "candidate_A" and not r["committed"]:
                c[r["proposed_id"]] += 1
    return dict(sorted(c.items()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    init, old = {}, {}
    for run, _ in GROUPS:
        init.update(cells(run, "initial"))
        old.update(cells(run, "defer"))
    new = {**cells("t58-precheck-20260927-a", "Aprime"), **cells("t59-final-20260927-a", "Aprime")}
    seeds = sorted(init)
    over = lambda base, run, bank: {s: cells(run, bank).get(s, base[s]) for s in seeds}  # noqa: E731
    banks = dict(oldR_oldC=old, oldR_newC=over(old, "t62-oldR-20260927-a", "oldR_newC"),
                 oldR_succC=over(old, "t62-oldR-20260927-a", "oldR_succC"),
                 newR_oldC=over(new, "t62-newR-20260927-a", "newR_oldC"), newR_newC=new)
    t62 = {k: dict(vs_initial=paired(v, init, seeds), vs_oldR_oldC=paired(v, old, seeds)) for k, v in banks.items()}
    t62["initial"] = dict(successes=sum(init[s]["success"] for s in seeds), n=len(seeds))
    ledger = [dict(run=r, env_steps=json.loads((RUNS / r / "summary.json").read_text())["env_steps"])
              for r in ("t62-oldR-20260927-a", "t62-newR-20260927-a")]
    cands = {}
    for name, path in (("C_old", "runs/t48-deferring-20260926-a/candidate_A.json"),
                       ("C_succ", "runs/t61-cand-succ-20260927-a/candidate_A.json")):
        f = Path(path)
        cands[name] = json.loads(f.read_text())["candidate"] if f.exists() else None
    out = dict(note="Collected by scripts/collect_t61_t64.py; success = 20 consecutive steps.",
               t61_candidates=cands, t62_factor=t62,
               t62_option_counts=dict(oldR_newC=option_counts("t62-oldR-20260927-a", "oldR_newC"),
                                      oldR_succC=option_counts("t62-oldR-20260927-a", "oldR_succC"),
                                      oldR_oldC_t59=option_counts("t59-final-20260927-a", "defer")))
    if (RUNS / "t63-final-20260927-a" / "summary.json").exists():
        t = final_table("t63-final-20260927-a")
        s = json.loads((RUNS / "t63-final-20260927-a" / "summary.json").read_text())
        c = {(x["bank"], x["seed"]): x for x in s["cells"]}
        fs = sorted({x["seed"] for x in s["cells"]})
        for bank, row in t.items():
            row["sign_test_p_vs_initial"] = sign_test(len(row["gained_vs_initial"]), len(row["lost_vs_initial"]))
            row["vs_oldR_oldC"] = paired({x: c[(bank, x)] for x in fs}, {x: c[("oldR_oldC", x)] for x in fs}, fs)
        out["t63_final"] = t
        ledger.append(dict(run="t63-final-20260927-a", env_steps=s["env_steps"]))
        allinit = {**init, **{x: c[("initial", x)] for x in fs}}
        allold = {**old, **{x: c[("oldR_oldC", x)] for x in fs}}
        out["oldR_oldC_pooled_independent"] = paired(allold, allinit, sorted(allinit))
    out["ledger"] = ledger
    out["env_steps_total"] = sum(x["env_steps"] for x in ledger)
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("env_steps_total", out["env_steps_total"])
    for k, v in t62.items():
        print("T62", k, {kk: (vv if not isinstance(vv, dict) else {x: vv[x] for x in ("successes", "gained", "lost", "sign_test_p")}) for kk, vv in v.items()})
    for bank, row in out.get("t63_final", {}).items():
        print("T63", bank, {k: row[k] for k in ("successes", "n", "wilson95", "gained_vs_initial", "lost_vs_initial",
                                                "candidate_invoked", "candidate_invoked_success", "run_errors",
                                                "sign_test_p_vs_initial")},
              {k: row["vs_oldR_oldC"][k] for k in ("gained", "lost", "sign_test_p")})
    print("pooled", out.get("oldR_oldC_pooled_independent"))


if __name__ == "__main__":
    main()
