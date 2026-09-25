"""T32-R1: record executed transitions (incl. terminal step) on the shared execution path.

Runs a few (task, seed) episodes with one bank version and writes, into a new run
directory, ``transitions.jsonl`` (one record per env step), the online detector
events as JSON files, and a summary recomputed from the transition log.

Example:
  python scripts/run_t32r1_transitions.py --out runs/t32r1-transitions-20260926-a \
      --episode true_place:3011 --episode true_place:3014 --episode true_place:3114
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from apc_maniskill.deficit_detector import AutonomousDeficitDetector
from apc_maniskill.experience_loop import (Bank, EpisodeLog, build_policy, make_task_env, run_episode,
                                           summarize_transitions)
from apc_maniskill.runner import json_write, provenance, utc_now


def parse_episode(text):
    task, seed = text.split(":")
    return task, int(seed)


def write_event(out: Path, run_id: str, task: str, seed: int, bank: dict, event: dict,
                log_path: Path) -> Path:
    """Persist a detector event with the bank and observation-history reference."""
    body = dict(schema="apc_deficit_event_v2", run_id=run_id, task=task, environment_seed=seed,
                detected_at_step=event["step"], detector_status=event["status"],
                reason=event["reason"], streak_steps=event["streak_steps"],
                active_bank_hash=bank["bank_hash"], bank_files=bank["files"],
                observation_history=str(log_path), created_at=utc_now(),
                detector="AutonomousDeficitDetector(defaults)")
    digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    body["event_hash"] = digest
    body["event_id"] = f"EVT-{digest[:12]}"
    path = out / "events" / f"{body['event_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    json_write(path, body)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episode", action="append", type=parse_episode, required=True,
                        help="task:seed with task in pick/place/true_place")
    parser.add_argument("--bundle", type=Path, default=Path("dist_autonomous_bundle_v1"),
                        help="initial bundle dir, or a bank version dir with bank_manifest.json")
    parser.add_argument("--router", type=Path, default=None)
    parser.add_argument("--transit", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--no-states", action="store_true", help="omit s_t/s_tp1 dicts")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=False)
    if (args.bundle / "bank_manifest.json").exists():
        roles = json.loads((args.bundle / "bank_manifest.json").read_text())["roles"]
        bank = Bank.from_roles(args.bundle, roles)
    else:
        bank = Bank.initial(args.bundle)
    if args.router is not None:
        bank.router = args.router
    if args.transit is not None:
        bank.transit = args.transit
    bank_manifest = bank.manifest()
    run_id = args.out.name
    json_write(args.out / "manifest.json", dict(
        run_id=run_id, created_at=utc_now(), command=sys.argv, bank=bank_manifest,
        episodes=[dict(task=t, seed=s) for t, s in args.episode], max_steps=args.max_steps,
        success_definition="task success held for 20 consecutive control steps",
        provenance=provenance()))
    log_path = args.out / "transitions.jsonl"
    results = []
    envs = {}
    start = time.monotonic()
    for index, (task, seed) in enumerate(args.episode):
        if task not in envs:
            env = make_task_env(task, args.out / f"env_{task}")
            envs[task] = (env, build_policy(env, args.out / f"env_{task}", bank))
        env, policy = envs[task]
        log = EpisodeLog(log_path, run_id=run_id, episode_id=f"{index:03d}-{task}-{seed}", seed=seed,
                         task=task, bank_hash=bank_manifest["bank_hash"], store_states=not args.no_states)
        result = run_episode(env, policy, seed=seed, max_steps=args.max_steps, log=log,
                             detector=AutonomousDeficitDetector())
        log.close()
        row = dict(task=task, **result.as_dict())
        row["event_files"] = [str(write_event(args.out, run_id, task, seed, bank_manifest, e, log_path))
                              for e in result.events]
        results.append(row)
        print(f"{task}:{seed} success={result.success} steps={result.steps} "
              f"modules={result.module_counts} rejections={result.rejection_counts} "
              f"events={len(result.events)} ({result.wall_seconds:.1f}s)", flush=True)
    for env, _ in envs.values():
        env.close()
    recomputed = summarize_transitions(log_path)
    json_write(args.out / "summary.json", dict(
        run_id=run_id, finished_at=utc_now(), wall_seconds=time.monotonic() - start,
        episodes=results, recomputed_from_transitions=recomputed,
        consistency=dict(
            records_equal_env_steps=recomputed["records"] == sum(r["steps"] for r in results),
            all_terminal_recorded=all(e["terminal_recorded"] or r["stopped_early"]
                                      for e, r in zip(recomputed["episodes"].values(), results)))))


if __name__ == "__main__":
    main()
