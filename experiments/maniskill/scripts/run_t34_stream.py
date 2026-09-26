"""T34/T35: continual acquisition over an ordered task stream with one evolving bank.

For each stream item (task:seed) in order:

1. run it once with the current bank and the online detector (T32-R1 recorder);
2. if a detector event of an acquirable kind is raised and not already handled:
   a. collect parent-router replay with the current bank on the replay set,
   b. run the same event-driven option search and fit (T32-R3, extend design) with
      the candidate name mapped from the event kind (earlier acquisitions of that
      candidate are refit together),
   c. adoption check (local, not a research gate): the new bank is adopted if it
      does not lose any retention-set success of the current bank and the event
      condition's final distance does not get worse; otherwise ``rejected_update``;
3. after every adopted update (and at the start) evaluate S[t, j] over the fixed
   evaluation conditions.

The stream order and condition names live only here; the policy and the acquirer
only see observations, goals and the event JSON.  Physical episodes are reset
between items; the bank is never reset.

Example:
  python scripts/run_t34_stream.py --out runs/t34-stream-20260926-a \
     --stream true_place:3014 --stream true_place:3026 --eval true_place:3014 \
     --retention pick:3202 --replay-set true_place:3011
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_bank_matrix  # noqa: E402
import run_t32r1_transitions  # noqa: E402
import run_t32r3_acquire  # noqa: E402

from apc_maniskill.runner import json_write, provenance, utc_now  # noqa: E402

KIND_TO_CANDIDATE = {
    "Carrying cube but failing to make horizontal progress toward goal": "candidate_A",
    "Free-space kinodynamic rejection streak": "candidate_A",
    "Pre-grasp approach time budget exceeded without achieving grasp": "candidate_C",
}


def kind_of(reason: str) -> str:
    return reason.split(" (")[0]


def matrix(out: Path, banks: dict, episodes: list, workers: int, max_steps: int) -> dict:
    argv = ["--out", str(out), "--workers", str(workers), "--max-steps", str(max_steps)]
    for name, path in banks.items():
        argv += ["--bank", f"{name}={path}"]
    for ep in episodes:
        argv += ["--episode", ep]
    run_bank_matrix.main(argv)
    return json.loads((out / "summary.json").read_text())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bank", type=Path, default=Path("dist_autonomous_bundle_v1"))
    p.add_argument("--stream", action="append", required=True, help="ordered task:seed items")
    p.add_argument("--eval", action="append", default=[], help="S[t,j] columns (task:seed)")
    p.add_argument("--retention", action="append", default=[], help="adoption-check conditions")
    p.add_argument("--replay-set", action="append", default=[],
                   help="conditions replayed with the current bank for router self-labels")
    p.add_argument("--max-steps", type=int, default=1200)
    p.add_argument("--no-grow-retention", action="store_true",
                   help="do not add items solved by adopted updates to the retention check")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--acq-args", default="--outcome-weight 3 --router-depth 6",
                   help="extra args for run_t32r3_acquire (default: dev-selected regularization)")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    registry = args.out / "registry.json"
    bank = args.bank
    versions = [dict(version=0, bank=str(bank), adopted_at_item=None)]
    prior = {}  # candidate name -> list of acquisition dirs
    acquired_solved = []  # event items solved by adopted updates (added to retention)
    log = dict(run_id=args.out.name, created_at=utc_now(), command=sys.argv, provenance=provenance(),
               stream=args.stream, eval=args.eval, retention=args.retention, items=[], s_matrix=[],
               versions=versions)
    evals = sorted(set(args.eval))

    def record_s(tag, t):
        if not evals:
            return
        d = matrix(args.out / f"S_{tag}", {"current": bank}, evals, args.workers, args.max_steps)
        log["s_matrix"].append(dict(after_item=t, bank_version=len(versions) - 1, bank=str(bank),
                                    row=d["table"]["current"], env_steps=d["env_steps"]))
        json_write(args.out / "stream_summary.json", log)

    record_s("v0", -1)
    for t, item in enumerate(args.stream):
        task, seed = item.split(":")
        item_dir = args.out / f"item{t:02d}_{task}_{seed}"
        run_t32r1_transitions.main(["--out", str(item_dir), "--episode", item, "--bundle", str(bank),
                                    "--max-steps", str(args.max_steps), "--no-states"])
        same_version_items = [args.out / f"item{i['t']:02d}_{i['item'].replace(':', '_')}"
                              for i in log["items"] if i["bank_version"] == len(versions) - 1]
        summary = json.loads((item_dir / "summary.json").read_text())
        ep = summary["episodes"][0]
        entry = dict(t=t, item=item, bank_version=len(versions) - 1, success=ep["success"],
                     steps=ep["steps"], modules=ep["module_counts"], events=[], env_steps=ep["steps"])
        for event_file in ep.get("event_files", []):
            event = json.loads(Path(event_file).read_text())
            kind = kind_of(event["reason"])
            cand = KIND_TO_CANDIDATE.get(kind)
            ev = dict(event_id=event["event_id"], kind=kind, step=event["detected_at_step"])
            entry["events"].append(ev)
            if cand is None:
                ev["decision"] = "no_acquirer_for_kind"
                continue
            if ep["success"]:
                ev["decision"] = "episode_succeeded_no_update"  # detector alarm on a solved item
                continue
            # a. parent-router replay with the current bank
            replay_dir = args.out / f"replay_t{t:02d}"
            rep = matrix(replay_dir, {"parent": bank}, args.replay_set, args.workers,
                         args.max_steps) if args.replay_set else {"env_steps": 0}
            # b. acquisition + fit
            acq_dir = args.out / f"acquire_t{t:02d}_{cand}"
            acq_argv = ["--out", str(acq_dir), "--bank", str(bank), "--event", event_file,
                        "--registry", str(registry), "--design", "extend", "--start", "event",
                        "--candidate-name", cand, "--workers", str(args.workers)]
            for cell in sorted((replay_dir / "cells").glob("*")) if args.replay_set else []:
                acq_argv += ["--replay", str(cell)]
            # the event item's own log (parent labels before the event; later rows are
            # excluded by the acquirer) and earlier items run with this bank version
            for d in [item_dir, *same_version_items]:
                acq_argv += ["--replay", str(d)]
            for d in prior.get(cand, []):
                acq_argv += ["--prior-acquisition", str(d)]
            acq_argv += args.acq_args.split()
            acq = run_t32r3_acquire.main(acq_argv)
            ev.update(candidate=cand, acquisition_result=acq["result"],
                      replay_env_steps=rep["env_steps"],
                      acquisition_env_steps=(acq.get("cost", {}).get("search_env_steps", 0)
                                             + acq.get("cost", {}).get("replay_env_steps", 0)))
            entry["env_steps"] += ev["replay_env_steps"] + ev["acquisition_env_steps"]
            status = [e for e in acq["events"] if e["event_id"] == event["event_id"]]
            ev["registry_status"] = status[0]["status"] if status else None
            ev["history"] = status[0].get("history") if status else None
            if acq["result"] != "learned":
                ev["decision"] = acq["result"]
                continue
            new_bank = Path(acq["banks"]["bank_A_full"]["dir"])
            # c. adoption check on retention set + the event item
            retention = sorted(set(args.retention + acquired_solved))
            check = matrix(args.out / f"adopt_t{t:02d}", {"current": bank, "new": new_bank},
                           sorted(set(retention + [item])), args.workers, args.max_steps)
            entry["env_steps"] += check["env_steps"]
            cells = {(c["bank"], f"{c['task']}:{c['seed']}"): c for c in check["cells"]}
            lost = [e for e in retention if cells[("current", e)]["success"]
                    and not cells[("new", e)]["success"]]
            d_cur = cells[("current", item)]["final"]["cube_goal_dist_xy_m"]
            d_new = cells[("new", item)]["final"]["cube_goal_dist_xy_m"]
            ev.update(adoption=dict(lost_retention=lost, event_item_success_new=cells[("new", item)]["success"],
                                    event_item_final_dist_current=d_cur, event_item_final_dist_new=d_new))
            if lost or (d_new > d_cur + 1e-3 and not cells[("new", item)]["success"]):
                ev["decision"] = "rejected_update"
                continue
            ev["decision"] = "adopted"
            if cells[("new", item)]["success"] and not args.no_grow_retention:
                acquired_solved.append(item)  # keep what this update newly solved
            prior.setdefault(cand, []).append(acq_dir)
            bank = new_bank
            versions.append(dict(version=len(versions), bank=str(bank), adopted_at_item=t,
                                 candidate=cand, event_id=event["event_id"]))
            record_s(f"v{len(versions) - 1}", t)
            break  # one update per item
        log["items"].append(entry)
        json_write(args.out / "stream_summary.json", log)
        print(f"[{t}] {item} success={entry['success']} events="
              f"{[(e['kind'][:20], e.get('decision')) for e in entry['events']]} "
              f"bank_v={len(versions) - 1}", flush=True)
    log.update(finished_at=utc_now(), wall_seconds=time.monotonic() - t0,
               env_steps_items=sum(i["env_steps"] for i in log["items"]),
               env_steps_s_matrix=sum(s["env_steps"] for s in log["s_matrix"]),
               adopted_updates=len(versions) - 1, grown_retention=acquired_solved,
               rejected_updates=sum(e.get("decision") == "rejected_update"
                                    for i in log["items"] for e in i["events"]),
               missed_failures=[i["item"] for i in log["items"] if not i["success"] and not i["events"]])
    json_write(args.out / "stream_summary.json", log)
    return log


if __name__ == "__main__":
    main()
