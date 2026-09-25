"""Compositional Transfer & Ablation Matrix on Known & Held-out Task Combinations (W6: T18 & T19).

Evaluates zero-shot synthesis (0 additional updates) and rigorous ablation under the
IDENTICAL UnifiedRouterSelector configuration:
- Full Unified Router: Base + Recovery + Transit + Place
- Ablation 1 (No Recovery): Unified Router with --disable-recovery
- Ablation 2 (No Transit): Unified Router without transit checkpoint (falls back to Base)
- Ablation 3 (No Place): Unified Router without place checkpoint (falls back to Base)
- Ablation 4 (Baseline): Unified Router with all patches/recovery disabled (pure Base fallback)

Evaluated across:
1. Seed 3012 (Development / Known condition: used in prior analysis/training runs)
2. Seed 3014 (True Held-out condition: completely unseen in any prior training, split, or analysis)
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List

from apc_maniskill.runner import json_write, utc_now


def run_cmd(args: List[str], cwd: Path) -> subprocess.CompletedProcess:
    res = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[ERROR] Command failed with return code {res.returncode}")
        print("STDOUT:", res.stdout[-2000:])
        print("STDERR:", res.stderr[-2000:])
        raise RuntimeError(f"Command failed: {' '.join(args)}")
    return res


def analyze_run(run_dir: Path) -> Dict[str, Any]:
    ep_p = run_dir / "episodes.jsonl"
    steps_p = run_dir / "steps.jsonl"
    if not (ep_p.exists() and steps_p.exists()):
        return {"success": False, "steps": 0, "module_steps": {}, "transitions": 0}

    ep_info = json.loads(ep_p.read_text(encoding="utf-8").strip())
    module_counts = Counter()
    prev_module = None
    transitions = 0

    with open(steps_p, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            diag = d.get("info", {}).get("diagnostic", {})
            active_mod = diag.get("active_module")
            if active_mod in ("place", "transit", "grasp_recovery", "base"):
                mod = active_mod
            elif diag.get("patch_applied", False):
                mod = "place"
            elif diag.get("transit_applied", False) or diag.get("transit_guard_triggered", False):
                mod = "transit"
            elif diag.get("recovery_applied", False) or diag.get("recovery_active", False):
                mod = "grasp_recovery"
            else:
                mod = "base"

            module_counts[mod] += 1
            if prev_module is not None and mod != prev_module:
                transitions += 1
            prev_module = mod

    return {
        "success": ep_info.get("consecutive_success_achieved", False),
        "steps": ep_info.get("steps", 0),
        "module_steps": dict(sorted(module_counts.items())),
        "transitions": transitions,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path,
                        default=Path("runs/apc-t15-autonomous-loop-20260925-c/dist_autonomous_bundle_v1"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[3012, 3014])
    parser.add_argument("--max-steps", type=int, default=1200)
    args = parser.parse_args()

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = Path(__file__).resolve().parents[1]
    bundle_dir = (repo_dir / args.bundle_dir).resolve() if not args.bundle_dir.is_absolute() else args.bundle_dir.resolve()
    py = sys.executable

    base_ckpt = bundle_dir / "base_selector.pt"
    place_ckpt = bundle_dir / "place_candidate.pt"
    transit_ckpt = bundle_dir / "transit_candidate.pt"
    router_ckpt = bundle_dir / "unified_router.pt"

    for ckpt in [base_ckpt, place_ckpt, transit_ckpt, router_ckpt]:
        if not ckpt.exists():
            raise FileNotFoundError(f"Bundle checkpoint missing: {ckpt}")

    conditions = [
        ("full_router", "Full Unified Router (All Modules: Base + Recovery + Transit + Place)", {}),
        ("no_recovery", "Ablation: Disable Grasp Recovery Guard", {"disable_recovery": True}),
        ("no_transit", "Ablation: Disable Transit Module (omit transit checkpoint)", {"omit_transit": True}),
        ("no_place", "Ablation: Disable Place Module (omit place checkpoint)", {"omit_place": True}),
        ("baseline", "Baseline: Fallback Base Only (Recovery Disabled + No Transit + No Place)", {
            "disable_recovery": True,
            "omit_transit": True,
            "omit_place": True,
        }),
    ]

    seed_roles = {
        3012: "Known Development Condition (Evaluated in W1/W4 router training)",
        3014: "True Held-out Unseen Condition (Never seen in any training/tuning)",
    }

    report: Dict[str, Any] = {
        "experiment": "w6_t18_t19_compositional_transfer_matrix_rigorous",
        "started_at": utc_now(),
        "bundle_dir": str(bundle_dir),
        "additional_learning_updates": 0,
        "additional_training_samples": 0,
        "seed_descriptions": {str(s): seed_roles.get(s, "Evaluation Seed") for s in args.seeds},
        "results": {},
    }
    t_start = time.time()

    for seed in args.seeds:
        seed_label = f"seed_{seed}"
        role_label = seed_roles.get(seed, f"Seed {seed}")
        report["results"][seed_label] = {}
        print(f"\n{'='*80}\nEVALUATING CONDITION: SEED {seed} ({role_label})\n{'='*80}")

        for cond_key, cond_desc, cond_opts in conditions:
            cond_out = out_dir / f"eval_seed{seed}_{cond_key}"
            print(f"\n--- Running Condition: {cond_key} ({cond_desc}) on Seed {seed} ---")

            cmd = [
                py, str(repo_dir / "scripts" / "run_fetch_primitives.py"),
                "--out", str(cond_out),
                "--episodes", "1",
                "--seed", str(seed),
                "--max-steps", str(args.max_steps),
                "--consecutive-success-steps", "20",
                "--selector", "unified_router",
                "--checkpoint", str(base_ckpt),
                "--router-checkpoint", str(router_ckpt),
                "--true-place-goal",
                "--rotations", "--base", "--far-start",
                "--allow-task-mismatch",
            ]

            if not cond_opts.get("omit_place", False):
                cmd.extend(["--patch-checkpoint", str(place_ckpt)])
            if not cond_opts.get("omit_transit", False):
                cmd.extend(["--transit-checkpoint", str(transit_ckpt)])
            if cond_opts.get("disable_recovery", False):
                cmd.append("--disable-recovery")

            run_cmd(cmd, cwd=repo_dir)
            analysis = analyze_run(cond_out)
            print(f"Result for {cond_key} (seed {seed}): Success={analysis['success']}, Steps={analysis['steps']}, Transitions={analysis['transitions']}")
            print(f"Module steps: {analysis['module_steps']}")
            report["results"][seed_label][cond_key] = analysis

    report["completed_at"] = utc_now()
    report["wall_seconds"] = time.time() - t_start

    summary_file = out_dir / "compositional_transfer_report.json"
    json_write(summary_file, report)
    print("\n" + "="*80)
    print("COMPOSITIONAL TRANSFER & ABLATION EXPERIMENT COMPLETE")
    print(f"Summary saved to: {summary_file}")
    print("="*80)


if __name__ == "__main__":
    main()
