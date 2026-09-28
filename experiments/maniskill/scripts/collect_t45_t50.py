"""T50: collect T45-T49 results, intervals, gains/losses and the cost ledger.

Reads only run summaries/labels; missing runs are listed as missing.

  python scripts/collect_t45_t50.py --out docs/T45_T50_RESULTS.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_t34_t44 import final_table, wilson  # noqa: E402

RUNS = Path("runs")
TUNE = [*range(3066, 3114), *range(3115, 3130), *range(3300, 3360)]
MATRIX = ["t47-tune-v3-20260926-a", "t47-tune-initial-20260926-a", "t48-cv-odd-20260926-a",
          "t48-cv-even-20260926-a", "t48-veto-20260926-a", "t48-tune-defer-20260926-a",
          "t48-cv-defer-odd-20260926-a", "t48-cv-defer-even-20260926-a", "t49-final-20260926-a"]
CF = ["t47-cf-20260926-a", "t48-cf-defer-20260926-a"]


def cells(run, bank=None):
    d = json.loads((RUNS / run / "summary.json").read_text())
    return {c["seed"]: c for c in d["cells"] if bank is None or c["bank"] == bank}


def initial_tune():
    out = {}
    for run, bank in (("t34-matrix-20260926-b", "old"), ("t42-final-20260926-a", "initial"),
                      ("t47-tune-initial-20260926-a", "initial")):
        out.update(cells(run, bank))
    return {s: out[s] for s in TUNE}


def tune_view(full: dict, override: dict, ref: dict):
    """Cells not re-run in `override` are identical to `full` by construction (checked)."""
    get = {s: override.get(s, full[s]) for s in TUNE}
    ok = [s for s in TUNE if get[s]["success"]]
    ref_ok = {s for s in TUNE if ref[s]["success"]}
    return dict(successes=len(ok), n=len(TUNE), wilson95=wilson(len(ok), len(TUNE)),
                gained_vs_initial=sorted(set(ok) - ref_ok), lost_vs_initial=sorted(ref_ok - set(ok)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    ledger, total = [], 0
    for run in MATRIX:
        if not (RUNS / run / "summary.json").exists():
            ledger.append(dict(run=run, status="missing"))
            continue
        s = json.loads((RUNS / run / "summary.json").read_text())["env_steps"]
        ledger.append(dict(run=run, env_steps=s))
        total += s
    for run in CF:
        s = json.loads((RUNS / run / "summary.json").read_text())["branch_env_steps"]
        ledger.append(dict(run=run, env_steps=s))
        total += s
    init = initial_tune()
    v3 = cells("t47-tune-v3-20260926-a")
    defer = cells("t48-tune-defer-20260926-a")
    t42 = json.loads((RUNS / "t42-final-20260926-a" / "summary.json").read_text())
    t42c = {(c["bank"], c["seed"]): c for c in t42["cells"]}
    clean = [s for s in range(3090, 3130) if s != 3114]
    t45 = {b: dict(successes=sum(t42c[(b, s)]["success"] for s in clean), n=len(clean),
                   wilson95=wilson(sum(t42c[(b, s)]["success"] for s in clean), len(clean)))
           for b in t42["table"]}
    cf = {}
    for run in CF:
        d = json.loads((RUNS / run / "summary.json").read_text())
        cf[run] = {k: d.get(k) for k in ("decision_points", "labels", "replay_faithful", "allow",
                                         "outcome_pairs", "branch_env_steps", "gate")}
    t48 = dict(
        initial=tune_view(init, {}, init),
        v3=tune_view(v3, {}, init),
        v3_cv_gate=tune_view(v3, {**cells("t48-cv-odd-20260926-a"), **cells("t48-cv-even-20260926-a")}, init),
        v3_veto_candidates=tune_view(v3, cells("t48-veto-20260926-a"), init),
        defer=tune_view(defer, {}, init),
        defer_cv_gate=tune_view(defer, {**cells("t48-cv-defer-odd-20260926-a"),
                                        **cells("t48-cv-defer-even-20260926-a")}, init),
    )
    cand = lambda d: sorted(s for s in d if any(k.startswith("candidate") for k in d[s]["module_counts"]))
    identical = dict(
        defer_noncandidate_cells_equal_initial=all(
            defer[s]["steps"] == init[s]["steps"] and defer[s]["success"] == init[s]["success"]
            for s in TUNE if s not in cand(defer)),
        candidate_cells_v3=len(cand(v3)), candidate_cells_defer=len(cand(defer)))
    out = dict(note="Collected by scripts/collect_t45_t50.py; success = 20 consecutive steps.",
               ledger=ledger, env_steps_total=total, t45_t42_without_3114=t45,
               t47_t48_counterfactual=cf, t48_tuning=t48, t48_checks=identical)
    if (RUNS / "t49-final-20260926-a" / "summary.json").exists():
        out["t49_final"] = final_table("t49-final-20260926-a")
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("env_steps_total", total)
    for k, v in t48.items():
        print("T48", k, v["successes"], v["n"], "gain", v["gained_vs_initial"], "loss", v["lost_vs_initial"])
    for bank, row in out.get("t49_final", {}).items():
        print("T49", bank, {k: row[k] for k in ("successes", "n", "wilson95", "gained_vs_initial", "lost_vs_initial",
                                                "candidate_invoked", "candidate_invoked_success", "run_errors")})


if __name__ == "__main__":
    main()
