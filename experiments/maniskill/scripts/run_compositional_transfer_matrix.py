"""Compositional Transfer & Ablation Matrix on Held-out Task Combinations (W6: T18 & T19).

Evaluates zero-shot synthesis (0 additional updates) on unseen combined requirements:
Grasp Recovery (F1=1) -> Lifted Transit (F2=1) -> True Placement Release (F3=1).
Executes across seeds 3008 and 3009 with physical state continuity (no reset during execution)
and performs complete ablation controls (Full vs -Recovery vs -Transit vs -Place vs Baseline).
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
            if diag.get("patch_applied", False):
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
    parser.add_argument("--seeds", type=int, nargs="+", default=[3011, 3012])
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
        ("full_router", "Full Unified Router (All Modules: Base + Recovery + Transit + Place)"),
        ("no_transit", "Ablation: No Transit Correction (Base + Place Only)"),
        ("no_place", "Ablation: No Place Module (Base + Transit Only)"),
        ("baseline", "Baseline: Unadapted Base Only (No Modules)"),
    ]

    report: Dict[str, Any] = {
        "experiment": "w6_t18_t19_compositional_transfer_matrix",
        "started_at": utc_now(),
        "bundle_dir": str(bundle_dir),
        "additional_learning_updates": 0,
        "additional_training_samples": 0,
        "results": {},
    }
    t_start = time.time()

    for seed in args.seeds:
        report["results"][f"seed_{seed}"] = {}
        print(f"\n{'='*80}\nEVALUATING HELD-OUT COMBINATION A (SEED {seed})\n{'='*80}")

        for cond_key, cond_desc in conditions:
            cond_out = out_dir / f"eval_seed{seed}_{cond_key}"
            print(f"\n--- Running Condition: {cond_key} ({cond_desc}) on Seed {seed} ---")

            if cond_key == "full_router":
                cmd = [
                    py, str(repo_dir / "scripts" / "run_fetch_primitives.py"),
                    "--out", str(cond_out),
                    "--episodes", "1",
                    "--seed", str(seed),
                    "--max-steps", str(args.max_steps),
                    "--consecutive-success-steps", "20",
                    "--selector", "unified_router",
                    "--checkpoint", str(base_ckpt),
                    "--patch-checkpoint", str(place_ckpt),
                    "--transit-checkpoint", str(transit_ckpt),
                    "--router-checkpoint", str(router_ckpt),
                    "--true-place-goal",
                    "--rotations", "--base", "--far-start",
                    "--allow-task-mismatch",
                ]
            elif cond_key == "no_transit":
                # Composite patch with recovery guard but WITHOUT transit checkpoint
                cmd = [
                    py, str(repo_dir / "scripts" / "run_fetch_primitives.py"),
                    "--out", str(cond_out),
                    "--episodes", "1",
                    "--seed", str(seed),
                    "--max-steps", str(args.max_steps),
                    "--consecutive-success-steps", "20",
                    "--selector", "composite_patch",
                    "--checkpoint", str(base_ckpt),
                    "--patch-checkpoint", str(place_ckpt),
                    "--grasp-guard",
                    "--patch-mode", "place",
                    "--true-place-goal",
                    "--rotations", "--base", "--far-start",
                    "--allow-task-mismatch",
                ]
            elif cond_key == "no_place":
                # Transit patch with recovery guard but WITHOUT place checkpoint
                cmd = [
                    py, str(repo_dir / "scripts" / "run_fetch_primitives.py"),
                    "--out", str(cond_out),
                    "--episodes", "1",
                    "--seed", str(seed),
                    "--max-steps", str(args.max_steps),
                    "--consecutive-success-steps", "20",
                    "--selector", "composite_patch",
                    "--checkpoint", str(base_ckpt),
                    "--patch-checkpoint", str(transit_ckpt),
                    "--grasp-guard",
                    "--patch-mode", "pick",
                    "--true-place-goal",
                    "--rotations", "--base", "--far-start",
                    "--allow-task-mismatch",
                ]
            elif cond_key == "baseline":
                cmd = [
                    py, str(repo_dir / "scripts" / "run_fetch_primitives.py"),
                    "--out", str(cond_out),
                    "--episodes", "1",
                    "--seed", str(seed),
                    "--max-steps", str(args.max_steps),
                    "--consecutive-success-steps", "20",
                    "--selector", "composite_patch",
                    "--checkpoint", str(base_ckpt),
                    "--patch-checkpoint", str(base_ckpt),
                    "--patch-mode", "place",
                    "--true-place-goal",
                    "--rotations", "--base", "--far-start",
                    "--allow-task-mismatch",
                ]
            else:
                raise ValueError(f"Unknown condition: {cond_key}")

            run_cmd(cmd, cwd=repo_dir)
            analysis = analyze_run(cond_out)
            print(f"Result for {cond_key} (seed {seed}): Success={analysis['success']}, Steps={analysis['steps']}, Transitions={analysis['transitions']}")
            print(f"Module steps: {analysis['module_steps']}")
            report["results"][f"seed_{seed}"][cond_key] = analysis

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
