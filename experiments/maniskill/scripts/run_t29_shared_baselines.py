"""T29: shared-model baselines (shared CART + replay, shared MLP + replay).

A single goal-conditioned model maps the shared schema-v5 state features to the 20
primitive IDs for every phase.  It receives the same information APC gets:

* replay rows: the executed decisions of the initial bank on the replayed
  development trajectories (a common distillation of the initial policy; the
  env steps that produced them are listed as cost), and
* optionally the acquisition window rows produced for APC (same new data).

The hand-designed grasp-recovery guard and the executor safety checks are kept for
both methods.  Output banks use a constant one-node router (always the model).

Example:
  python scripts/run_t29_shared_baselines.py --out runs/t29-shared-20260926-a \
      --replay runs/t32r1-transitions-20260926-a --window runs/t32r3-acquire-20260926-a
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from apc_maniskill.experience_loop import (Bank, fit_cart, read_jsonl, save_candidate, save_router,
                                           sha256_file)
from apc_maniskill.primitive_learning import SCHEMA_V5, network
from apc_maniskill.primitive_tree import CART
from apc_maniskill.primitives import NAMES20
from apc_maniskill.runner import json_write, provenance, utc_now


def constant_router(path: Path):
    tree = CART(1, 4)
    tree.scores.copy_(torch.tensor([[0.0, -18.0, -18.0, -18.0]]))
    save_router(path, tree, note="constant router: always the shared model (class 0)")


def replay_rows(runs, exclude):
    xs, ys, steps = [], [], 0
    for run in runs:
        for r in read_jsonl(Path(run) / "transitions.jsonl"):
            steps += 1
            key = (r["task"], r["environment_seed"])
            if key in exclude and r["control_step"] >= exclude[key]:
                continue
            xs.append(r["candidate_x"])
            ys.append(r["proposed_id"])
    return xs, ys, steps


def window_rows(acq_dirs):
    xs, ys, exclude = [], [], {}
    for d in acq_dirs:
        summary = json.loads((Path(d) / "acquisition_summary.json").read_text())
        for e in summary["events"]:
            acq = e.get("acquisition")
            if not acq or not acq["chosen_options"]:
                continue
            rows = read_jsonl(Path(d) / e["event_id"] / "acquired_trajectory.jsonl")
            onset, end = acq["onset_step"], acq["window_end_step"]
            win = [r for r in rows if onset <= r["control_step"] < end]
            xs += [r["candidate_x"] for r in win]
            ys += [r["proposed_id"] for r in win]
            exclude[(rows[0]["task"], rows[0]["environment_seed"])] = onset
    return xs, ys, exclude


def train_mlp(x, y, *, updates=4000, seed=0, lr=1e-3, batch=256):
    x = np.asarray(x, np.float32)
    y = np.asarray(y, np.int64)
    mean = x.mean(0)
    std = x.std(0)
    std[std < 1e-4] = 1.0
    xt = torch.from_numpy((x - mean) / std)
    yt = torch.from_numpy(y)
    torch.manual_seed(seed)
    model = network(x.shape[1], 20)
    counts = np.bincount(y, minlength=20).astype(np.float32)
    weight = torch.from_numpy(np.where(counts > 0, len(y) / (20 * np.maximum(counts, 1)), 0.0).astype(np.float32))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(updates):
        idx = torch.randint(0, len(yt), (batch,), generator=gen)
        loss = torch.nn.functional.cross_entropy(model(xt[idx]), yt[idx], weight=weight)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    acc = float((model(xt).argmax(1) == yt).float().mean())
    return model, mean, std, acc


def write_bank(dst: Path, model_path: Path, parent: Bank, notes: dict) -> dict:
    dst.mkdir(parents=True, exist_ok=False)
    constant_router(dst / "router.pt")
    (dst / "base.pt").write_bytes(model_path.read_bytes())
    bank = Bank(base=dst / "base.pt", router=dst / "router.pt")
    manifest = dict(schema="apc_bank_version_v1", created_at=utc_now(),
                    roles=dict(base="base.pt", router="router.pt"), **bank.manifest(), **notes)
    json_write(dst / "bank_manifest.json", manifest)
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--replay", type=Path, action="append", required=True)
    p.add_argument("--window", type=Path, action="append", default=[],
                   help="acquisition run dirs whose window rows are added")
    p.add_argument("--bank", type=Path, default=Path("dist_autonomous_bundle_v1"))
    p.add_argument("--tag", default="v0")
    p.add_argument("--cart-depth", type=int, default=14)
    p.add_argument("--mlp-updates", type=int, default=4000)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    parent = Bank.initial(args.bank)
    xw, yw, exclude = window_rows(args.window)
    xr, yr, replay_steps = replay_rows(args.replay, exclude)
    x, y = xr + xw, yr + yw
    task_meta = json.loads(json.dumps(torch.load(parent.base, weights_only=False)["task"]))
    result = dict(run_id=args.out.name, created_at=utc_now(), command=sys.argv, provenance=provenance(),
                  rows=dict(replay=len(yr), window=len(yw)), replay_source_env_steps=replay_steps)

    t0 = time.monotonic()
    tree, mean, std = fit_cart(x, y, 20, max_depth=args.cart_depth, min_leaf=2)
    cart_fit = time.monotonic() - t0
    cart_path = args.out / "shared_cart.pt"
    meta = save_candidate(cart_path, tree, mean, std, task_meta=task_meta,
                          source=dict(rows=len(y), sha256=""))
    cart_acc = float((tree(torch.from_numpy((np.asarray(x, np.float32) - mean) / std)).argmax(1).numpy()
                      == np.asarray(y)).mean())
    t0 = time.monotonic()
    model, m_mean, m_std, mlp_acc = train_mlp(x, y, updates=args.mlp_updates)
    mlp_fit = time.monotonic() - t0
    mlp_path = args.out / "shared_mlp.pt"
    torch.save({"schema": SCHEMA_V5, "primitive_names": list(NAMES20), "task": task_meta,
                "control_freq": 20.0, "state_dict": model.state_dict(), "mean": m_mean, "std": m_std,
                "source_steps_sha256": "", "model_kind": "mlp", "updates": args.mlp_updates,
                "allow_parameterized_goal_tasks": True}, mlp_path)
    notes = dict(method="shared_model_plus_replay", tag=args.tag, recovery_guard="kept (shared)")
    banks = {f"shared_cart_{args.tag}": write_bank(args.out / f"shared_cart_{args.tag}", cart_path, parent, notes),
             f"shared_mlp_{args.tag}": write_bank(args.out / f"shared_mlp_{args.tag}", mlp_path, parent, notes)}
    result.update(cart=dict(meta, fit_seconds=cart_fit, train_accuracy=cart_acc),
                  mlp=dict(path=str(mlp_path), sha256=sha256_file(mlp_path), bytes=mlp_path.stat().st_size,
                           parameters=sum(p.numel() for p in model.parameters()), fit_seconds=mlp_fit,
                           updates=args.mlp_updates, train_accuracy=mlp_acc),
                  banks={k: dict(dir=str(args.out / k), bank_hash=v["bank_hash"]) for k, v in banks.items()})
    json_write(args.out / "summary.json", result)
    print(json.dumps({k: result[k] for k in ("rows", "cart", "mlp")}, indent=1, default=str))


if __name__ == "__main__":
    main()
