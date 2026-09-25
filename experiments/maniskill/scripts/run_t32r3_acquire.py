"""T32-R3: event JSON -> local option search -> Candidate/Router fit -> new bank version.

No event file, or only duplicate/stale events: nothing is learned (recorded as such).
Router training rows are the parent router's own decisions on replayed development
trajectories (their cost is listed) plus outcome-derived labels from the search.

Example:
  python scripts/run_t32r3_acquire.py --out runs/t32r3-acquire-20260926-a \
     --event runs/t32r1-transitions-20260926-a/events/EVT-2eeae25b1ff4.json \
     --replay runs/t32r1-transitions-20260926-a --registry runs/t32r-registry.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

from apc_maniskill.acquisition import DEFAULT_OPTIONS, acquire, replay_acquired
from apc_maniskill.experience_loop import (MODULES, MODULES_EXT, Bank, fit_cart, read_jsonl,
                                           router_x2_from_rows, save_option_candidate, save_router,
                                           sha256_file)
from apc_maniskill.runner import json_write, provenance, utc_now

TRANSIT = MODULES.index("transit")
CANDIDATE = MODULES_EXT.index("candidate_A")


def load_bank(path: Path) -> Bank:
    if (path / "bank_manifest.json").exists():
        m = json.loads((path / "bank_manifest.json").read_text())
        return Bank.from_roles(path, m["roles"])
    return Bank.initial(path)


def write_bank(dst: Path, roles: dict, parent: dict, notes: dict) -> dict:
    dst.mkdir(parents=True, exist_ok=False)
    rel = {}
    for role, src in roles.items():
        name = f"{role}.pt"
        shutil.copy2(src, dst / name)
        rel[role] = name
    bank = Bank.from_roles(dst, rel)
    manifest = dict(schema="apc_bank_version_v1", created_at=utc_now(), roles=rel,
                    parent_bank_hash=parent["bank_hash"], **bank.manifest(), **notes)
    json_write(dst / "bank_manifest.json", manifest)
    return manifest


def router_rows_from_replays(runs, exclude, schema="router_v1"):
    """(router input, router_class) of the parent router on replayed trajectories."""
    xs, ys, steps = [], [], 0
    for run in runs:
        rows = read_jsonl(Path(run) / "transitions.jsonl")
        steps += len(rows)
        for ep in sorted({r["episode_id"] for r in rows}):
            er = sorted((r for r in rows if r["episode_id"] == ep), key=lambda r: r["control_step"])
            x2 = router_x2_from_rows(er) if schema == "router_v2" else None
            for i, r in enumerate(er):
                if r.get("router_x") is None or r["selected_module"] == "script":
                    continue
                key = (r["task"], r["environment_seed"])
                if key in exclude and r["control_step"] >= exclude[key]:
                    continue  # outcome evidence below replaces these labels
                xs.append(x2[i] if x2 is not None else r["router_x"])
                ys.append(r["router_class"])
    return xs, ys, steps


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--event", type=Path, action="append", default=[])
    p.add_argument("--bank", type=Path, default=Path("dist_autonomous_bundle_v1"))
    p.add_argument("--replay", type=Path, action="append", default=[],
                   help="run dirs whose transitions provide parent-router replay rows")
    p.add_argument("--registry", type=Path, required=True)
    p.add_argument("--horizon", type=int, default=250)
    p.add_argument("--max-decisions", type=int, default=10)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--options", nargs="+", default=[f"{i}:{r}" for i, r in DEFAULT_OPTIONS],
                   help="option prior as ID:repeats")
    p.add_argument("--candidate-depth", type=int, default=8)
    p.add_argument("--router-depth", type=int, default=10)
    p.add_argument("--design", choices=["extend", "replace"], default="extend",
                   help="extend: candidate_A is a 5th router class next to the old Transit; "
                        "replace: candidate_A takes the Transit slot (T32/R3-c design)")
    p.add_argument("--start", choices=["event", "onset"], default="event")
    p.add_argument("--candidate-name", default="candidate_A",
                   help="router class / bank role of the new candidate (extend design)")
    p.add_argument("--outcome-weight", type=float, default=50.0)
    p.add_argument("--reuse", type=Path, default=None,
                   help="refit from a previous acquisition run's search results (no new search)")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    t_start = time.monotonic()
    bank = load_bank(args.bank)
    parent = bank.manifest()
    registry = json.loads(args.registry.read_text()) if args.registry.exists() else {"events": {}}
    log = dict(run_id=args.out.name, created_at=utc_now(), command=sys.argv, parent_bank=parent,
               provenance=provenance(), events=[])

    accepted = []
    for path in args.event:
        event = json.loads(path.read_text())
        entry = dict(event_file=str(path), event_id=event["event_id"], seed=event["environment_seed"])
        if event["event_hash"] in registry["events"] and args.reuse is None:
            entry["status"] = "duplicate_skipped"
        elif event["active_bank_hash"] != parent["bank_hash"]:
            entry["status"] = "stale_bank_skipped"
        else:
            prior = [e for e in registry["events"].values()
                     if e["task"] == event["task"] and e["reason_kind"] == event["reason"].split(" (")[0]]
            entry["status"] = "accepted"
            entry["history"] = "recurrence_of_known_kind" if prior else "new_kind"
            accepted.append(event)
        log["events"].append(entry)

    if not accepted:
        log["result"] = "no_event_no_learning"
        json_write(args.out / "acquisition_summary.json", log)
        print("No accepted event: nothing learned.")
        return log

    options = tuple((int(o.split(":")[0]), int(o.split(":")[1]) if ":" in o else 1) for o in args.options)
    from apc_maniskill.experience_loop import load_router
    parent_classes = list(load_router(bank.router).class_names)
    classes = (parent_classes + [args.candidate_name] if args.candidate_name not in parent_classes
               else parent_classes) if args.design == "extend" else list(MODULES)
    new_class = classes.index(args.candidate_name) if args.design == "extend" else TRANSIT
    xs_c, ys_c, xr_new, yr_new = [], [], [], []
    exclude = {}
    cost = dict(search_env_steps=0, replay_env_steps=0, search_wall_seconds=0.0)
    for event in accepted:
        ev_dir = args.out / event["event_id"]
        print(f"Acquiring for {event['event_id']} ({event['task']}:{event['environment_seed']}, "
              f"step {event['detected_at_step']})", flush=True)
        if args.reuse is not None:
            prev = json.loads((args.reuse / "acquisition_summary.json").read_text())
            pe = next(e for e in prev["events"] if e["event_id"] == event["event_id"])
            acq, replay = dict(pe["acquisition"]), dict(pe["acquired_replay"])
            shutil.copytree(args.reuse / event["event_id"], ev_dir)
            cost.setdefault("reused_from", str(args.reuse))
        else:
            acq = acquire(event, bank, ev_dir, options=options, horizon=args.horizon,
                          max_decisions=args.max_decisions, workers=args.workers,
                          start=args.start)
            replay = replay_acquired(event, bank, acq, ev_dir)
            cost["search_env_steps"] += acq["search_env_steps"]
            cost["search_wall_seconds"] += acq["search_wall_seconds"]
            cost["replay_env_steps"] += replay["steps"]
        rows = read_jsonl(ev_dir / "acquired_trajectory.jsonl")
        onset, end = acq["onset_step"], acq["window_end_step"]
        hist = [r for r in read_jsonl(Path(event["observation_history"]))
                if r["environment_seed"] == event["environment_seed"] and r["task"] == event["task"]]
        replay_ok = all(a["executed_id"] == b["executed_id"] and abs(
            a["reward_components"]["cube_goal_dist_xy_after_m"]
            - b["reward_components"]["cube_goal_dist_xy_after_m"]) < 1e-7
            for a, b in zip(rows[:onset], hist[:onset]))
        entry = next(e for e in log["events"] if e["event_id"] == event["event_id"])
        entry.update(acquisition={k: v for k, v in acq.items() if k != "prefix"},
                     acquired_replay=dict(success=replay["success"], steps=replay["steps"],
                                          final=replay["final"],
                                          module_counts=replay["module_counts"]),
                     prefix_replay_matches_history=bool(replay_ok))
        window = [r for r in rows if onset <= r["control_step"] < end]
        if not acq["chosen_options"]:
            entry["learning"] = "no_candidate: bank baseline best at onset"
            continue
        # Candidate data: decision states and the outcome-selected option there.
        by_step = {r["control_step"]: r for r in rows}
        for step, choice in zip(acq["decision_steps"], acq["chosen_options"]):
            xs_c.append(by_step[step]["candidate_x"])
            ys_c.append(options.index(tuple(choice)))
        after = [r for r in rows if r["control_step"] >= end and r["router_x"] is not None]
        if args.design == "replace":
            # Router outcome labels: window states -> transit slot (candidate) ...
            xr_new += [r["router_x"] for r in window]
            yr_new += [TRANSIT] * len(window)
            xr_new += [r["router_x"] for r in after]
        else:
            # Decision states -> the new candidate (holds are covered by option commitment).
            xr_new += [by_step[s]["router_x2"] for s in acq["decision_steps"]]
            yr_new += [new_class] * len(acq["decision_steps"])
            xr_new += [r["router_x2"] for r in after]
        # ... and after handoff, the parent router's decisions on the acquired path.
        yr_new += [r["router_class"] for r in after]
        exclude[(event["task"], event["environment_seed"])] = onset
        entry["learning"] = dict(candidate_rows=len(acq["decision_steps"]), router_outcome_rows=len(window),
                                 router_after_handoff_rows=len(after))

    if not xs_c:
        log["result"] = "no_candidate"
        json_write(args.out / "acquisition_summary.json", log)
        return log

    t_fit = time.monotonic()
    tree_c, mean, std = fit_cart(xs_c, ys_c, len(options), max_depth=args.candidate_depth, min_leaf=1)
    cand_path = args.out / f"{args.candidate_name}.pt"
    cand_meta = save_option_candidate(cand_path, tree_c, mean, std, options,
                                      source=dict(events=[e["event_id"] for e in accepted],
                                                  rows=len(ys_c)))
    schema = "router_v2" if args.design == "extend" else "router_v1"
    xr_old, yr_old, replay_steps = router_rows_from_replays(args.replay, exclude, schema)
    xr = np.asarray(xr_old + xr_new, dtype=np.float32)
    yr = np.asarray(yr_old + yr_new, dtype=np.int64)
    counts = np.bincount(yr, minlength=len(classes))
    if args.design == "replace":  # T32/R3-c weighting (class-balanced)
        weights = np.array([len(yr) / (4 * counts[c]) for c in yr])
        min_leaf = 2
    else:  # uniform parent rows, outcome decision rows up-weighted (few, direct evidence)
        weights = np.where(yr == new_class, args.outcome_weight, 1.0)
        min_leaf = 1
    tree_r, _, _ = fit_cart(xr, yr, len(classes), weights=weights, max_depth=args.router_depth,
                            min_leaf=min_leaf, normalize=False)
    router_path = args.out / "router_A.pt"
    save_router(router_path, tree_r, feature_schema=schema, class_names=classes, rows=int(len(yr)),
                class_counts=counts.tolist(), parent_router_sha256=sha256_file(bank.router))
    fit_seconds = time.monotonic() - t_fit
    train_acc_c = float((tree_c(torch.from_numpy((np.asarray(xs_c, np.float32) - mean) / std)).argmax(1).numpy()
                         == np.asarray(ys_c)).mean())
    train_acc_r = float((tree_r(torch.from_numpy(xr)).argmax(1).numpy() == yr).mean())
    notes = dict(events=[e["event_id"] for e in accepted], design=args.design,
                 replacement=("candidate_A replaces the transit slot" if args.design == "replace" else None))
    if args.design == "extend":
        roles = dict(base=bank.base, transit=bank.transit, place=bank.place)
        roles.update({k: v for k, v in bank.candidate_roles().items() if k != args.candidate_name})
        new = {args.candidate_name: cand_path}
        banks = {
            "bank_A_full": write_bank(args.out / "bank_A_full", dict(roles, router=router_path, **new),
                                      parent, notes),
            # class 4 falls back to Base when no candidate is present
            "bank_A_router_only": write_bank(args.out / "bank_A_router_only", dict(roles, router=router_path),
                                             parent, notes),
            # the old router has no class 4: identical behaviour to the parent bank
            "bank_A_candidate_only": write_bank(args.out / "bank_A_candidate_only", dict(
                roles, router=bank.router, **new), parent, notes),
        }
    else:
        banks = {
            "bank_A_full": write_bank(args.out / "bank_A_full", dict(base=bank.base, router=router_path,
                                      transit=cand_path, place=bank.place), parent, notes),
            "bank_A_router_only": write_bank(args.out / "bank_A_router_only", dict(
                base=bank.base, router=router_path, transit=bank.transit, place=bank.place), parent, notes),
            "bank_A_candidate_only": write_bank(args.out / "bank_A_candidate_only", dict(
                base=bank.base, router=bank.router, transit=cand_path, place=bank.place), parent, notes),
        }
    for e in accepted:
        registry["events"][e["event_hash"]] = dict(event_id=e["event_id"], task=e["task"],
                                                   seed=e["environment_seed"],
                                                   reason_kind=e["reason"].split(" (")[0],
                                                   learned_in=args.out.name,
                                                   new_bank_hash=banks["bank_A_full"]["bank_hash"])
    json_write(args.registry, registry)
    log.update(result="learned", candidate=dict(cand_meta, rows=len(ys_c), train_accuracy=train_acc_c,
                                                label_counts={str(list(options[k])): int(v) for k, v in zip(*np.unique(ys_c, return_counts=True))}),
               router=dict(path=str(router_path), sha256=sha256_file(router_path),
                           tree_nodes=int(tree_r.feature.shape[0]), rows=int(len(yr)),
                           parent_replay_rows=len(yr_old), outcome_rows=len(yr_new),
                           class_counts=counts.tolist(), train_accuracy=train_acc_r),
               banks={k: dict(bank_hash=v["bank_hash"], dir=str(args.out / k)) for k, v in banks.items()},
               cost=dict(cost, fit_seconds=fit_seconds, parent_replay_source_steps=replay_steps,
                         total_wall_seconds=time.monotonic() - t_start))
    json_write(args.out / "acquisition_summary.json", log)
    print(json.dumps({k: log[k] for k in ("result", "cost")}, indent=1))
    return log


if __name__ == "__main__":
    main()
