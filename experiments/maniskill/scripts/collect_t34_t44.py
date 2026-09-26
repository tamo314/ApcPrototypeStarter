"""T43: collect T34-T44 results, intervals, gains/losses and the cost ledger.

Reads only run summaries/transition logs; missing runs are listed as missing.

  python scripts/collect_t34_t44.py --out docs/T34_T44_RESULTS.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

RUNS = Path("runs")

LEDGER = [
    ("t36-heldout-20260926-a", "T36 held-out 3040-3042", "matrix"),
    ("t34-sweep-dev-20260926-a", "Router regularization sweep (dev)", "matrix"),
    ("t34-stream-AC-20260926-a-failed-router-defaults", "T34 stream a (failed: router defaults, stopped)", "stream_partial"),
    ("t34-stream-AC-20260926-b", "T34 stream b (A->C, no acquired-item retention)", "stream"),
    ("t34-stream-AC-20260926-c", "T34 stream c (+retention growth)", "stream"),
    ("t34-stream-AC-20260926-d", "T34 stream d (+acquired-item replay)", "stream"),
    ("t35-stream-CA-20260926-a", "T35 reversed stream (C->A)", "stream"),
    ("t37-distill-20260926-a", "T37 distillation (deploy + Temporary queries)", "t37"),
    ("t39-dagger-20260926-a", "T39 shared DAgger", "t39"),
    ("t42-final-20260926-a", "T42 final independent (3090-3129)", "matrix"),
]


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def steps(run: Path, kind: str):
    if kind == "matrix":
        return json.loads((run / "summary.json").read_text())["env_steps"]
    if kind == "stream":
        d = json.loads((run / "stream_summary.json").read_text())
        return d.get("env_steps_items", 0) + d.get("env_steps_s_matrix", 0)
    if kind == "stream_partial":
        total = 0
        for f in run.rglob("summary.json"):
            d = json.loads(f.read_text())
            total += d.get("env_steps", 0) if "table" in d else d.get("recomputed_from_transitions", {}).get("env_steps", 0)
        for f in run.rglob("acquisition_summary.json"):
            c = json.loads(f.read_text()).get("cost", {})
            total += c.get("search_env_steps", 0) + c.get("replay_env_steps", 0)
        return total
    if kind == "t37":
        c = json.loads((run / "summary.json").read_text())["cost"]
        return c["deploy_env_steps"] + c["query_env_steps"]
    if kind == "t39":
        return json.loads((run / "summary.json").read_text())["env_steps"]
    raise ValueError(kind)


def final_table(run: str, reference="initial"):
    d = json.loads((RUNS / run / "summary.json").read_text())
    cells = {(c["bank"], c["seed"]): c for c in d["cells"]}
    seeds = sorted({c["seed"] for c in d["cells"]})
    out = {}
    for bank in d["table"]:
        ok = [s for s in seeds if cells[(bank, s)]["success"]]
        errors = [s for s in seeds if cells[(bank, s)]["run_error"]]
        ref_ok = {s for s in seeds if cells[(reference, s)]["success"]}
        cand = [s for s in seeds if any(k.startswith("candidate") for k in (cells[(bank, s)].get("module_counts") or {}))]
        out[bank] = dict(
            successes=len(ok), n=len(seeds), wilson95=wilson(len(ok), len(seeds)),
            gained_vs_initial=sorted(set(ok) - ref_ok), lost_vs_initial=sorted(ref_ok - set(ok)),
            candidate_invoked=len(cand),
            candidate_invoked_success=sum(cells[(bank, s)]["success"] for s in cand),
            candidate_invoked_failed=[s for s in cand if not cells[(bank, s)]["success"]],
            run_errors=errors,
            ik_or_table_rejections=sum(sum(v for k, v in (cells[(bank, s)].get("rejection_counts") or {}).items()
                                           if k.startswith("ik_or")) for s in seeds))
    return out


def stream_view(run: str):
    d = json.loads((RUNS / run / "stream_summary.json").read_text())
    return dict(
        items=[dict(t=i["t"], item=i["item"], success=i["success"], bank_version=i["bank_version"],
                    events=[{k: e.get(k) for k in ("kind", "decision", "history", "candidate", "adoption",
                                                   "acquisition_env_steps", "replay_env_steps")}
                            for e in i["events"]], env_steps=i["env_steps"]) for i in d["items"]],
        s_matrix=[dict(version=s["bank_version"], after_item=s["after_item"], row=s["row"],
                       total=sum(s["row"].values())) for s in d["s_matrix"]],
        adopted=d.get("adopted_updates"), rejected=d.get("rejected_updates"),
        missed=d.get("missed_failures"), grown_retention=d.get("grown_retention"),
        wall_seconds=d.get("wall_seconds"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    ledger, total = [], 0
    for run, phase, kind in LEDGER:
        path = RUNS / run
        if not path.exists():
            ledger.append(dict(run=run, phase=phase, status="missing"))
            continue
        s = steps(path, kind)
        total += s
        ledger.append(dict(run=run, phase=phase, env_steps=s))
    sweep = json.loads((RUNS / "t34-sweep-dev-20260926-a" / "summary.json").read_text())
    t37 = json.loads((RUNS / "t37-distill-20260926-a" / "summary.json").read_text())
    t39 = json.loads((RUNS / "t39-dagger-20260926-a" / "summary.json").read_text())
    held = json.loads((RUNS / "t36-heldout-20260926-a" / "summary.json").read_text())
    out = dict(
        note="Collected by scripts/collect_t34_t44.py from run summaries; success = 20 consecutive steps.",
        ledger=ledger, env_steps_total=total,
        t36_heldout=held["table"],
        sweep_dev=dict(table=sweep["table"], successes=sweep["successes"]),
        streams={r: stream_view(r) for r in ("t34-stream-AC-20260926-b", "t34-stream-AC-20260926-c",
                                             "t34-stream-AC-20260926-d", "t35-stream-CA-20260926-a")},
        t37={k: t37[k] for k in ("direct_rows", "query_rows", "relabelled", "queries", "cost", "candidate")},
        t39=dict(rounds=t39["rounds"], env_steps=t39["env_steps"]),
        t42_final=final_table("t42-final-20260926-a"),
    )
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("env_steps_total", total)
    for bank, row in out["t42_final"].items():
        print(bank, {k: row[k] for k in ("successes", "n", "wilson95", "gained_vs_initial", "lost_vs_initial",
                                         "candidate_invoked", "candidate_invoked_success", "run_errors")})


if __name__ == "__main__":
    main()
