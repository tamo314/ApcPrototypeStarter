"""Collect the T32-R..T38 cost ledger and result tables from run summaries.

Every number is read from the run directories' own summary files (recomputed from
transition logs where the run wrote them).  Runs that are missing are listed as
missing; nothing is filled in.  Output: a small JSON meant to be committed.

  python scripts/collect_t32r_ledger.py --out docs/T32R_T38_RESULTS.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RUNS = Path("runs")

LEDGER = [
    # (run, phase, kind)
    ("t32r-repro-20260926-s3014", "T32-R1 legacy reproduction", "legacy_runner"),
    ("t32r-repro-20260926-s3013", "T32-R1 legacy reproduction", "legacy_runner"),
    ("t32r1-transitions-20260926-a", "T32-R1 executed transitions", "r1"),
    ("t32r1-parity-20260926-a", "T32-R1 router-v2 parity", "r1"),
    ("t32r2-branch-20260926-b", "T32-R2 branch diagnosis", "r2"),
    ("t32r3-screen-20260926-g3018", "T32-R3 event screening", "r1"),
    ("t32r3-screen-20260926-g3021", "T32-R3 event screening", "r1"),
    ("t32r3-screen-20260926-g3024", "T32-R3 event screening", "r1"),
    ("t32r3-screen-20260926-g3027", "T32-R3 event screening", "r1"),
    ("t32r3-acquire-20260926-a-failed-oscillation", "T32-R3 acquisition v1 (failed: min-dist score)", "acq_partial"),
    ("t32r3-acquire-20260926-b", "T32-R3 acquisition v2 (per-step candidate, superseded)", "acq"),
    ("t32r3-acquire-20260926-c", "T32-R3 acquisition v3 (option candidate, replace design)", "acq"),
    ("t32r3-acquire-20260926-d", "T32-R3 acquisition v4 (extend design, event start)", "acq"),
    ("t32r3-acquire-20260926-e", "T32-R3 router refit with Pick/Place replay (reuse d)", "acq"),
    ("t32r1-replay-pickplace-20260926-a", "T32-R3 parent replay (Pick dev)", "r1"),
    ("t32r1-replay-pickplace-20260926-b", "T32-R3 parent replay (Place dev)", "r1"),
    ("t29-shared-20260926-v0", "T29 shared baselines (replay only)", "fit"),
    ("t29-shared-20260926-vA", "T29 shared baselines (+ A window data)", "fit"),
    ("t32r4-matrix-20260926-a-failed-keyerror", "T32-R4 matrix (run_error: script bug)", "matrix"),
    ("t32r4-matrix-20260926-b", "T32-R4 matrix replace design + T29", "matrix"),
    ("t32r4-matrix-20260926-c", "T32-R4 matrix extend design", "matrix"),
    ("t32r4-matrix-20260926-d", "T32-R4 matrix extend + refit router + new seeds", "matrix"),
    ("t33b-place-entry-20260926-a", "T33-B gate v1 (forcing gate, coarse states)", "t33b"),
    ("t33b-place-entry-20260926-b", "T33-B gate v2 (veto, level states)", "t33b"),
    ("t33b-place-entry-20260926-c", "T33-B gate v3 (+ known-success branching)", "t33b"),
    ("t33b-place-entry-20260926-d", "T33-B value gate (reuses c rollouts)", "t33b"),
    ("t33b-matrix-20260926-a", "T33-B matrix v1 gates", "matrix"),
    ("t33b-matrix-20260926-b", "T33-B matrix v2 gates", "matrix"),
    ("t33b-matrix-20260926-c", "T33-B matrix classifier gate v3", "matrix"),
    ("t33b-matrix-20260926-d", "T33-B matrix value gate", "matrix"),
    ("t33c-detect-20260926-g3018", "T33-C re-detection under bank A", "r1"),
    ("t33c-detect-20260926-g3026", "T33-C re-detection under bank A", "r1"),
    ("t33c-detect-20260926-g3065", "T33-C re-detection under bank A", "r1"),
    ("t34-replay-bankA-20260926-a", "T33-C parent (bank A) replay", "matrix"),
    ("t34-regress-20260926-a", "Regression after multi-candidate refactor", "matrix"),
    ("t33c-acquire-20260926-a-failed-grasploss", "T33-C acquisition (failed: grasp-loss score bug)", "acq"),
    ("t33c-acquire-20260926-b", "T33-C acquisition (candidate_C)", "acq"),
    ("t34-matrix-20260926-a", "T34 bank sequence matrix old/A/AC/AC+gate", "matrix"),
    ("t38-release-20260926-a", "T38 fresh-process release check", "t38"),
]


def env_steps(run: Path, kind: str):
    s = run / "summary.json"
    if kind == "legacy_runner":
        rows = [json.loads(l) for l in (run / "episodes.jsonl").read_text().splitlines() if l.strip()]
        return sum(r["steps"] for r in rows), None
    if kind == "r1":
        d = json.loads(s.read_text())
        return d["recomputed_from_transitions"]["env_steps"], d.get("wall_seconds")
    if kind == "r2":
        d = json.loads(s.read_text())
        return d["env_steps_total"], d["wall_seconds"]
    if kind == "matrix":
        d = json.loads(s.read_text())
        return d["env_steps"], d["wall_seconds"]
    if kind == "t33b":
        d = json.loads(s.read_text())
        return d.get("env_steps"), d.get("wall_seconds")
    if kind == "acq":
        d = json.loads((run / "acquisition_summary.json").read_text())
        c = d.get("cost", {})
        return (c.get("search_env_steps", 0) + c.get("replay_env_steps", 0)), c.get("total_wall_seconds")
    if kind == "acq_partial":
        rows = [json.loads(l) for p in run.glob("*/option_outcomes.jsonl")
                for l in p.read_text().splitlines() if l.strip()]
        return sum(r["env_steps"] for r in rows), None
    if kind == "t38":
        d = json.loads(s.read_text())
        return sum(e["steps"] for e in d["results"]["bank"].get("episodes", [])), None
    if kind == "fit":
        return 0, None
    raise ValueError(kind)


def table(run: str):
    d = json.loads((RUNS / run / "summary.json").read_text())
    return dict(table=d["table"], successes=d["successes"])


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
        steps, wall = env_steps(path, kind)
        total += steps or 0
        ledger.append(dict(run=run, phase=phase, env_steps=steps,
                           wall_seconds=None if wall is None else round(wall, 1)))
    acq = {}
    for run in ("t32r3-acquire-20260926-c", "t32r3-acquire-20260926-d", "t32r3-acquire-20260926-e",
                "t33c-acquire-20260926-b"):
        d = json.loads((RUNS / run / "acquisition_summary.json").read_text())
        acq[run] = dict(
            result=d["result"], cost=d.get("cost"),
            events=[dict(event_id=e["event_id"], seed=e["seed"], status=e["status"],
                         history=e.get("history"),
                         chosen_options=e.get("acquisition", {}).get("chosen_options"),
                         decision_steps=e.get("acquisition", {}).get("decision_steps"),
                         acquired_replay_success=e.get("acquired_replay", {}).get("success"),
                         acquired_replay_steps=e.get("acquired_replay", {}).get("steps"))
                    for e in d["events"]],
            candidate={k: v for k, v in d.get("candidate", {}).items() if k != "path"},
            router={k: v for k, v in d.get("router", {}).items() if k != "path"})
    gates = {}
    for run in ("t33b-place-entry-20260926-b", "t33b-place-entry-20260926-c", "t33b-place-entry-20260926-d"):
        d = json.loads((RUNS / run / "summary.json").read_text())
        gates[run] = dict(rows=d.get("rows"), positives=d.get("positives"), tree_nodes=d.get("tree_nodes"),
                          value_gate=d.get("value_gate"),
                          labels=d.get("labels"))
    t38 = json.loads((RUNS / "t38-release-20260926-a" / "summary.json").read_text())
    out = dict(
        note=("Collected from run summaries by scripts/collect_t32r_ledger.py. Success = 20 consecutive "
              "task-success steps. runs/ is not committed; paths are relative to experiments/maniskill."),
        ledger=ledger, env_steps_total=total,
        matrices={r: table(r) for r in ("t32r4-matrix-20260926-b", "t32r4-matrix-20260926-c",
                                        "t32r4-matrix-20260926-d", "t33b-matrix-20260926-b",
                                        "t33b-matrix-20260926-c", "t33b-matrix-20260926-d",
                                        "t34-matrix-20260926-a")},
        r2=[{k: r[k] for k in ("branch", "state", "dist_before", "dist_after_window", "min_dist_window",
                               "grasped_after_window", "rejections", "forced_steps", "scripted_steps")}
            | dict(terminal_success=r["terminal"]["success"],
                   terminal_dist=r["terminal"]["final"]["cube_goal_dist_xy_m"])
            for r in json.loads((RUNS / "t32r2-branch-20260926-b" / "summary.json").read_text())["results"]],
        acquisitions=acq, place_gates=gates,
        t38=dict(model_bytes=t38["model_bytes"], model_bytes_total=t38["model_bytes_total"],
                 hash_ok=t38["results"]["bank"].get("hash_ok"),
                 files=t38["results"]["bank"].get("files_present"),
                 episodes=t38["results"]["bank"].get("episodes"),
                 peak_rss_mb_bank=t38["results"]["bank"].get("peak_rss_mb"),
                 peak_rss_mb_runtime_only=t38["results"]["runtime_only"].get("peak_rss_mb")),
    )
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("env_steps_total", total, "bytes", args.out.stat().st_size)
    for row in ledger:
        print(row)


if __name__ == "__main__":
    main()
