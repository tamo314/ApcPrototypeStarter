"""T38 (minimal): run a bank version in a fresh process from its model files only.

The child process receives only the bank directory (models + bank_manifest.json):
no acquisition data, option outcomes, optimizer state or Temporary model.  It
verifies file hashes against the manifest, runs the given episodes, and reports
peak RSS.  A reference child that only imports the simulator stack and builds the
environment (no bank) gives the runtime baseline, so model-attributable memory is
separated from simulator/runtime memory.

Example:
  python scripts/run_t38_release_check.py --out runs/t38-release-20260926-a \
     --bank runs/t33c-acquire-20260926-b/bank_A_full --episode true_place:3014
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

CHILD = r"""
import json, resource, sys, time, hashlib
from pathlib import Path
mode, bank_dir, out, episodes = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), json.loads(sys.argv[4])
t0 = time.monotonic()
from apc_maniskill.experience_loop import Bank, build_policy, make_task_env, run_episode
result = {"mode": mode}
if mode == "bank":
    m = json.loads((bank_dir / "bank_manifest.json").read_text())
    bank = Bank.from_roles(bank_dir, m["roles"])
    result["hash_ok"] = all(hashlib.sha256((bank_dir / f).read_bytes()).hexdigest() == m["files"][r]["sha256"]
                            for r, f in m["roles"].items())
    result["files_present"] = sorted(p.name for p in bank_dir.iterdir())
rows = []
for task, seed in episodes:
    env = make_task_env(task, out / f"{mode}-{task}-{seed}")
    if mode == "bank":
        policy = build_policy(env, out / f"{mode}-{task}-{seed}", bank)
        t1 = time.monotonic()
        r = run_episode(env, policy, seed=seed, max_steps=1200)
        rows.append(dict(task=task, seed=seed, success=r.success, steps=r.steps,
                         seconds=time.monotonic() - t1, modules=r.module_counts))
    else:
        env.reset(seed=seed)
    env.close()
result.update(episodes=rows, peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              wall_seconds=time.monotonic() - t0)
print("RESULT" + json.dumps(result))
"""


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--episode", action="append", required=True)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    episodes = [[e.split(":")[0], int(e.split(":")[1])] for e in args.episode]
    manifest = json.loads((args.bank / "bank_manifest.json").read_text())
    sizes = {r: (args.bank / f).stat().st_size for r, f in manifest["roles"].items()}
    results = {}
    for mode in ("runtime_only", "bank"):
        proc = subprocess.run([sys.executable, "-c", CHILD, mode, str(args.bank), str(args.out),
                               json.dumps(episodes)], capture_output=True, text=True)
        line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT")), None)
        results[mode] = json.loads(line[6:]) if line else dict(run_error=proc.stderr[-2000:])
    summary = dict(bank=str(args.bank), bank_hash=manifest["bank_hash"], model_bytes=sizes,
                   model_bytes_total=sum(sizes.values()), results=results,
                   peak_rss_difference_mb=(results["bank"].get("peak_rss_mb", 0)
                                           - results["runtime_only"].get("peak_rss_mb", 0)),
                   note="Peak RSS difference includes episode execution buffers, not only model weights.")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
