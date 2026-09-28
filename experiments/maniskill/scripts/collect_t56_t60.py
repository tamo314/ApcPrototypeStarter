"""T60: collect T56-T59 results (harvest, bulk acquisition, pre-check, final) and the cost ledger.

  python scripts/collect_t56_t60.py --out docs/T56_T60_RESULTS.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_t34_t44 import final_table, wilson  # noqa: E402
from collect_t51_t55 import sign_test  # noqa: E402

RUNS = Path("runs")


def cells(run, bank):
    return {c["seed"]: c for c in json.loads((RUNS / run / "summary.json").read_text())["cells"] if c["bank"] == bank}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    harvest_steps, kinds, n_ep = 0, Counter(), 0
    for f in sorted(glob.glob(str(RUNS / "t56-harvest-20260927-p*" / "summary.json"))):
        d = json.loads(Path(f).read_text())
        harvest_steps += sum(e["steps"] for e in d["episodes"])
        n_ep += len(d["episodes"])
        for ev in glob.glob(str(Path(f).parent / "events" / "*.json")):
            kinds[json.loads(Path(ev).read_text())["reason"].split(" (")[0]] += 1
    acq = json.loads((RUNS / "t57-acquire-20260927-a" / "acquisition_summary.json").read_text())
    acq_steps = acq["cost"]["search_env_steps"] + acq["cost"]["replay_env_steps"]
    pre = json.loads((RUNS / "t58-precheck-20260927-a" / "summary.json").read_text())
    ledger = [dict(run="t56-harvest-20260927-p1..p4", env_steps=harvest_steps),
              dict(run="t57-acquire-20260927-a", env_steps=acq_steps,
                   parent_replay_source_steps_reused=acq["cost"]["parent_replay_source_steps"]),
              dict(run="t58-precheck-20260927-a", env_steps=pre["env_steps"])]
    out = dict(note="Collected by scripts/collect_t56_t60.py; success = 20 consecutive steps.",
               t56_harvest=dict(episodes=n_ep, event_kinds=dict(kinds)),
               t57_acquisition=dict(
                   events=[dict(seed=e["seed"], status=e["status"], chosen_options=e["acquisition"]["chosen_options"],
                                acquired_success=e["acquired_replay"]["success"]) for e in acq["events"]],
                   candidate={k: acq["candidate"][k] for k in ("tree_nodes", "rows", "train_accuracy", "label_counts")},
                   router={k: acq["router"][k] for k in ("tree_nodes", "rows", "outcome_rows", "class_counts")},
                   cost=acq["cost"]))
    seeds = [*range(3130, 3190), *range(3400, 3480)]
    init = {**cells("t49-final-20260926-a", "initial"), **cells("t54-final-20260927-a", "initial")}
    views = dict(initial=init, defer={**cells("t49-final-20260926-a", "defer"), **cells("t54-final-20260927-a", "defer")},
                 Aprime=cells("t58-precheck-20260927-a", "Aprime"))
    out["t58_precheck"] = {}
    for name, x in views.items():
        ok = [s for s in seeds if x[s]["success"]]
        g = [s for s in ok if not init[s]["success"]]
        lo = [s for s in seeds if init[s]["success"] and not x[s]["success"]]
        out["t58_precheck"][name] = dict(successes=len(ok), n=len(seeds), wilson95=wilson(len(ok), len(seeds)),
                                         gained_vs_initial=g, lost_vs_initial=lo, sign_test_p=sign_test(len(g), len(lo)))
    if (RUNS / "t59-final-20260927-a" / "summary.json").exists():
        t = final_table("t59-final-20260927-a")
        c = {(x["bank"], x["seed"]): x["success"] for x in
             json.loads((RUNS / "t59-final-20260927-a" / "summary.json").read_text())["cells"]}
        fs = sorted({s for _, s in c})
        for row in t.values():
            row["sign_test_p_vs_initial"] = sign_test(len(row["gained_vs_initial"]), len(row["lost_vs_initial"]))
        g = [s for s in fs if c[("Aprime", s)] and not c[("defer", s)]]
        lo = [s for s in fs if c[("defer", s)] and not c[("Aprime", s)]]
        out["t59_final"] = t
        out["t59_Aprime_vs_defer"] = dict(gained=g, lost=lo, sign_test_p=sign_test(len(g), len(lo)))
        ledger.append(dict(run="t59-final-20260927-a",
                           env_steps=json.loads((RUNS / "t59-final-20260927-a" / "summary.json").read_text())["env_steps"]))
    out["ledger"] = ledger
    out["env_steps_total"] = sum(x["env_steps"] for x in ledger)
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("env_steps_total", out["env_steps_total"])
    for k, v in out["t58_precheck"].items():
        print("T58", k, v)
    for bank, row in out.get("t59_final", {}).items():
        print("T59", bank, {k: row[k] for k in ("successes", "n", "wilson95", "gained_vs_initial", "lost_vs_initial",
                                                "candidate_invoked", "candidate_invoked_success", "run_errors",
                                                "sign_test_p_vs_initial")})
    print(out.get("t59_Aprime_vs_defer"))


if __name__ == "__main__":
    main()
