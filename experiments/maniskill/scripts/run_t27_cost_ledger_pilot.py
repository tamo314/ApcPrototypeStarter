"""T27 Pilot: Initial Assets & Cost Ledger Verification Run (T27).

Loads initial assets manifest and condition split manifest, executes a minimal pilot run
across known success and known failure cases, and automatically generates the cost ledger.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List

from apc_maniskill.runner import json_write, utc_now


def run_cmd(args: List[str], cwd: Path) -> subprocess.CompletedProcess:
    res = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[ERROR] Command failed with return code {res.returncode}")
        print("STDOUT:", res.stdout[-1500:])
        print("STDERR:", res.stderr[-1500:])
        raise RuntimeError(f"Command failed: {' '.join(args)}")
    return res


def analyze_run(run_dir: Path) -> Dict[str, Any]:
    ep_p = run_dir / "episodes.jsonl"
    steps_p = run_dir / "steps.jsonl"
    if not (ep_p.exists() and steps_p.exists()):
        return {
            "success": False,
            "steps": 0,
            "module_steps": {},
            "transitions": 0,
            "consecutive_success": 0,
            "first_success_step": None,
        }

    ep_info = json.loads(ep_p.read_text(encoding="utf-8").strip())
    module_counts = Counter()
    prev_module = None
    transitions = 0
    first_success = None

    with open(steps_p, encoding="utf-8") as f:
        for idx, line in enumerate(f):
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

            if d.get("info", {}).get("success", False) and first_success is None:
                first_success = idx + 1

    return {
        "success": ep_info.get("consecutive_success_achieved", False),
        "steps": ep_info.get("steps", 0),
        "module_steps": dict(sorted(module_counts.items())),
        "transitions": transitions,
        "consecutive_success": ep_info.get("consecutive_success_max", 0),
        "first_success_step": first_success,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/apc-t27-pilot-20260925-a"))
    parser.add_argument("--assets-manifest", type=Path,
                        default=Path("configs/t27_initial_assets_manifest.json"))
    parser.add_argument("--split-manifest", type=Path,
                        default=Path("configs/t27_condition_split_manifest.json"))
    parser.add_argument("--max-steps", type=int, default=1200)
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parents[1]
    out_dir = (repo_dir / args.out).resolve() if not args.out.is_absolute() else args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    # Load manifests
    assets_manifest_p = (repo_dir / args.assets_manifest).resolve()
    split_manifest_p = (repo_dir / args.split_manifest).resolve()
    if not assets_manifest_p.exists():
        raise FileNotFoundError(f"Missing assets manifest: {assets_manifest_p}")
    if not split_manifest_p.exists():
        raise FileNotFoundError(f"Missing split manifest: {split_manifest_p}")

    assets_data = json.loads(assets_manifest_p.read_text(encoding="utf-8"))
    split_data = json.loads(split_manifest_p.read_text(encoding="utf-8"))

    bundle_dir = repo_dir / assets_data["bundle_dir"]
    base_ckpt = bundle_dir / "base_selector.pt"
    place_ckpt = bundle_dir / "place_candidate.pt"
    transit_ckpt = bundle_dir / "transit_candidate.pt"
    router_ckpt = bundle_dir / "unified_router.pt"

    for ckpt in [base_ckpt, place_ckpt, transit_ckpt, router_ckpt]:
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing initial asset checkpoint: {ckpt}")

    # Pilot execution plan: 1 known success pick, 1 known success place, 1 known failure TruePlace (3014)
    pilot_cases = [
        {
            "case_id": "known_success_pick",
            "task": "APC-FetchPickCubeFar-v1",
            "seed": 3201,
            "type": "pick",
            "expected_outcome": "success",
            "description": "Known Pick baseline (seed 3201)"
        },
        {
            "case_id": "known_success_place",
            "task": "APC-FetchPlaceCubeFar-v1",
            "seed": 3001,
            "type": "place",
            "expected_outcome": "success",
            "description": "Known Place baseline (seed 3001)"
        },
        {
            "case_id": "known_failure_extrapolation",
            "task": "APC-FetchTruePlaceFar-v1",
            "seed": 3014,
            "type": "true_place",
            "expected_outcome": "failure",
            "description": "Known TruePlace extrapolation failure (seed 3014)"
        }
    ]

    ledger: Dict[str, Any] = {
        "task_name": "T27_initial_assets_and_cost_ledger_pilot",
        "started_at": utc_now(),
        "assets_manifest": str(assets_manifest_p.relative_to(repo_dir)),
        "split_manifest": str(split_manifest_p.relative_to(repo_dir)),
        "initial_models_footprint_bytes": assets_data["summary"]["total_model_file_size_bytes"],
        "initial_models_footprint_kb": assets_data["summary"]["total_model_file_size_kb"],
        "total_tree_nodes": assets_data["summary"]["total_tree_nodes"],
        "total_gradient_parameters": assets_data["summary"]["total_gradient_parameters"],
        "pilot_cases": [],
        "cost_summary": {}
    }

    t_global_start = time.time()
    total_env_steps = 0
    total_successful_cases = 0

    print("=" * 80)
    print("STARTING T27 PILOT: COST LEDGER & INITIAL ASSET VERIFICATION")
    print(f"Initial Asset Footprint: {assets_data['summary']['total_model_file_size_kb']} KB (4 CART models, 0 grad params)")
    print(f"Pilot Cases: {len(pilot_cases)}")
    print("=" * 80)

    for idx, tc in enumerate(pilot_cases, 1):
        c_dir = out_dir / f"case_{idx:02d}_{tc['case_id']}"
        print(f"\n[{idx}/{len(pilot_cases)}] Executing {tc['case_id']} (seed {tc['seed']}, expected: {tc['expected_outcome']})...")

        cmd = [
            py, str(repo_dir / "scripts" / "run_fetch_primitives.py"),
            "--out", str(c_dir),
            "--episodes", "1",
            "--seed", str(tc["seed"]),
            "--max-steps", str(args.max_steps),
            "--consecutive-success-steps", "20",
            "--selector", "unified_router",
            "--checkpoint", str(base_ckpt),
            "--router-checkpoint", str(router_ckpt),
            "--patch-checkpoint", str(place_ckpt),
            "--transit-checkpoint", str(transit_ckpt),
            "--rotations", "--base", "--far-start",
            "--allow-task-mismatch",
        ]
        if tc["type"] == "true_place":
            cmd.append("--true-place-goal")
        elif tc["type"] == "place":
            cmd.append("--place-goal")

        t_sub_start = time.time()
        run_cmd(cmd, cwd=repo_dir)
        analysis = analyze_run(c_dir)
        sub_wall = round(time.time() - t_sub_start, 2)
        total_env_steps += analysis["steps"]
        if analysis["success"]:
            total_successful_cases += 1

        entry = {
            "index": idx,
            "case_id": tc["case_id"],
            "task": tc["task"],
            "seed": tc["seed"],
            "expected_outcome": tc["expected_outcome"],
            "actual_outcome": "success" if analysis["success"] else "failure",
            "matches_expectation": (analysis["success"] == (tc["expected_outcome"] == "success")),
            "steps": analysis["steps"],
            "first_success_step": analysis["first_success_step"],
            "consecutive_success": analysis["consecutive_success"],
            "module_steps": analysis["module_steps"],
            "transitions": analysis["transitions"],
            "wall_seconds": sub_wall,
            "model_size_bytes": assets_data["summary"]["total_model_file_size_bytes"],
            "run_dir": str(c_dir.name)
        }
        ledger["pilot_cases"].append(entry)
        print(f"  -> Outcome: {entry['actual_outcome'].upper()} in {entry['steps']} steps ({sub_wall}s), matches expectation: {entry['matches_expectation']}")
        print(f"     Module allocation: {entry['module_steps']}")

    total_wall = round(time.time() - t_global_start, 2)
    ledger["completed_at"] = utc_now()
    ledger["cost_summary"] = {
        "total_pilot_episodes": len(pilot_cases),
        "total_environment_steps": total_env_steps,
        "standard_execution_steps": total_env_steps,
        "adaptation_exploration_steps": 0,
        "adaptation_training_wall_seconds": 0.0,
        "total_wall_seconds": total_wall,
        "average_steps_per_episode": round(total_env_steps / len(pilot_cases), 1),
        "successful_cases_count": total_successful_cases,
        "models_footprint_bytes": assets_data["summary"]["total_model_file_size_bytes"]
    }

    summary_file = out_dir / "t27_cost_ledger_summary.json"
    json_write(summary_file, ledger)

    print("\n" + "=" * 80)
    print("T27 PILOT COST LEDGER RUN COMPLETE")
    print(f"Ledger Output: {summary_file}")
    print(f"Total Steps: {total_env_steps}, Total Time: {total_wall}s, Matches Expectation: {all(c['matches_expectation'] for c in ledger['pilot_cases'])}")
    print("=" * 80)


if __name__ == "__main__":
    main()
