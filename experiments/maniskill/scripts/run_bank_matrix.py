"""Evaluate bank versions x (task, seed) conditions on the shared execution path.

Used for T32-R4 (2x2 behaviour/router ablation and retention) and later matrices.
Each cell is one episode (max 1,200 steps); success is the 20-step consecutive
task success.  Results are recomputed from the per-cell transition logs.

Example:
  python scripts/run_bank_matrix.py --out runs/t32r4-matrix-20260926-a \
     --bank old=dist_autonomous_bundle_v1 --bank full=runs/.../bank_A_full \
     --episode true_place:3014 --episode pick:3201
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

from apc_maniskill.experience_loop import (Bank, EpisodeLog, build_policy, make_task_env, run_episode,
                                           summarize_transitions)
from apc_maniskill.runner import json_write, provenance, utc_now


def load_bank(path: Path) -> Bank:
    if (path / "bank_manifest.json").exists():
        m = json.loads((path / "bank_manifest.json").read_text())
        return Bank.from_roles(path, m["roles"])
    return Bank.initial(path)


def make_gate(spec):
    """'thr:0.015' (hand-designed diagnostic) or a learned place_gate.pt path."""
    if spec is None:
        return None
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_t33b_place_entry import PlaceGate, threshold_gate
    if spec.startswith("thr:"):
        return threshold_gate(float(spec[4:]))
    return PlaceGate(Path(spec))


def cell(job):
    out, name, bank_spec, task, seed, max_steps = job
    bank_dir, _, gate_spec = bank_spec.partition("@")
    bank = load_bank(Path(bank_dir))
    manifest = bank.manifest()
    cell_dir = Path(out) / "cells" / f"{name}__{task}_{seed}"
    env = make_task_env(task, cell_dir)
    veto = gate_spec == "veto_candidates"  # T48/T49 ablation: router refit without Candidate actions
    policy = build_policy(env, cell_dir, bank, place_gate=None if veto else make_gate(gate_spec or None),
                          candidate_gate=(lambda x2: False) if veto else None)
    log = EpisodeLog(cell_dir / "transitions.jsonl", run_id=Path(out).name,
                     episode_id=f"{name}-{task}-{seed}", seed=seed, task=task,
                     bank_hash=manifest["bank_hash"], store_states=False)
    try:
        result = run_episode(env, policy, seed=seed, max_steps=max_steps, log=log)
        row = dict(bank=name, bank_hash=manifest["bank_hash"], task=task, run_error=None,
                   **result.as_dict())
    except Exception as exc:  # recorded as run_error, never as task failure
        row = dict(bank=name, bank_hash=manifest["bank_hash"], task=task, seed=seed,
                   run_error=f"{type(exc).__name__}: {exc}", success=None, steps=None)
    finally:
        log.close()
        env.close()
    if row["run_error"] is None:
        row["recomputed"] = summarize_transitions(cell_dir / "transitions.jsonl")["episodes"]
    return row


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bank", action="append", required=True,
                   help="name=dir or name=dir@gate (gate: thr:<m>, place_gate.pt, or veto_candidates)")
    p.add_argument("--episode", action="append", required=True, help="task:seed")
    p.add_argument("--max-steps", type=int, default=1200)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    banks = dict(b.split("=", 1) for b in args.bank)
    episodes = [(e.split(":")[0], int(e.split(":")[1])) for e in args.episode]
    json_write(args.out / "manifest.json", dict(
        run_id=args.out.name, created_at=utc_now(), command=sys.argv,
        banks={k: dict(load_bank(Path(v.partition("@")[0])).manifest(), place_gate=v.partition("@")[2] or None)
               for k, v in banks.items()},
        episodes=episodes, max_steps=args.max_steps,
        success_definition="task success held for 20 consecutive control steps",
        provenance=provenance()))
    jobs = [(str(args.out), n, d, t, s, args.max_steps) for n, d in banks.items() for t, s in episodes]
    t0 = time.monotonic()
    rows = []
    with get_context("spawn").Pool(args.workers) as pool:
        for row in pool.imap_unordered(cell, jobs):
            rows.append(row)
            print(f"{row['bank']:24s} {row['task']}:{row['seed']} success={row['success']} "
                  f"steps={row['steps']} modules={row.get('module_counts')} "
                  f"rej={row.get('rejection_counts')} err={row['run_error']}", flush=True)
    order = {n: i for i, n in enumerate(banks)}
    rows.sort(key=lambda r: (order[r["bank"]], episodes.index((r["task"], r["seed"]))))
    table = {n: {f"{t}:{s}": next(r["success"] for r in rows if r["bank"] == n and r["task"] == t
                                  and r["seed"] == s) for t, s in episodes} for n in banks}
    json_write(args.out / "summary.json", dict(
        run_id=args.out.name, finished_at=utc_now(), wall_seconds=time.monotonic() - t0,
        env_steps=sum(r["steps"] or 0 for r in rows), table=table,
        successes={n: sum(bool(v) for v in t.values()) for n, t in table.items()}, cells=rows))


if __name__ == "__main__":
    main()
