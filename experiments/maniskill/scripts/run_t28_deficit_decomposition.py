"""T28 Deficit Decomposition: Routing Deficiency vs Action Deficiency (T28).

Decomposes Candidate A (Seed 3014: transit progress recovery) and Candidate B (Seed 3013: place entry offset recovery)
by comparing baseline router against diagnostic forced module selection and threshold variants.
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
            "final_dist_xy": None,
            "final_cube_z": None,
        }

    ep_info = json.loads(ep_p.read_text(encoding="utf-8").strip())
    module_counts = Counter()
    prev_module = None
    transitions = 0
    first_success = None
    final_dist = None
    final_z = None

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

            pre = diag.get("pre_action_state", {})
            cp = pre.get("cube_position")
            gp = pre.get("goal_position")
            if cp and gp:
                final_dist = round(((cp[0]-gp[0])**2 + (cp[1]-gp[1])**2)**0.5, 4)
                final_z = round(cp[2], 4)

    return {
        "success": ep_info.get("consecutive_success_achieved", False),
        "steps": ep_info.get("steps", 0),
        "module_steps": dict(sorted(module_counts.items())),
        "transitions": transitions,
        "consecutive_success": ep_info.get("consecutive_success_max", 0),
        "first_success_step": first_success,
        "final_dist_xy": final_dist,
        "final_cube_z": final_z,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/apc-t28-decomposition-20260925-a"))
    parser.add_argument("--bundle-dir", type=Path, default=Path("dist_autonomous_bundle_v1"))
    parser.add_argument("--max-steps", type=int, default=1200)
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parents[1]
    out_dir = (repo_dir / args.out).resolve() if not args.out.is_absolute() else args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    bundle_dir = (repo_dir / args.bundle_dir).resolve()
    base_ckpt = bundle_dir / "base_selector.pt"
    place_ckpt = bundle_dir / "place_candidate.pt"
    transit_ckpt = bundle_dir / "transit_candidate.pt"
    router_ckpt = bundle_dir / "unified_router.pt"

    for ckpt in [base_ckpt, place_ckpt, transit_ckpt, router_ckpt]:
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

    # Decomposition diagnostic test matrix
    tests = [
        # Candidate A (Seed 3014): Extrapolated Transit Recovery
        {
            "id": "A1_baseline_seed3014",
            "candidate": "Deficit_A",
            "seed": 3014,
            "desc": "Seed 3014 Baseline Unified Router (Known failure: Transit unselected)",
            "flags": [
                "--selector", "unified_router",
                "--checkpoint", str(base_ckpt),
                "--router-checkpoint", str(router_ckpt),
                "--patch-checkpoint", str(place_ckpt),
                "--transit-checkpoint", str(transit_ckpt),
            ]
        },
        {
            "id": "A2_forced_transit_seed3014",
            "candidate": "Deficit_A",
            "seed": 3014,
            "desc": "Seed 3014 Forced Transit Guard (Diagnostic: Does existing transit solve 3014 if forced?)",
            "flags": [
                "--selector", "composite_patch",
                "--checkpoint", str(base_ckpt),
                "--patch-checkpoint", str(place_ckpt),
                "--transit-checkpoint", str(transit_ckpt),
                "--transit-guard", "--transit-mode", "guard",
            ]
        },

        # Candidate B (Seed 3013): Place Entry Offset Recovery
        {
            "id": "B1_baseline_seed3013",
            "candidate": "Deficit_B",
            "seed": 3013,
            "desc": "Seed 3013 Baseline Unified Router (Known failure: Early place latch at 2.4cm)",
            "flags": [
                "--selector", "unified_router",
                "--checkpoint", str(base_ckpt),
                "--router-checkpoint", str(router_ckpt),
                "--patch-checkpoint", str(place_ckpt),
                "--transit-checkpoint", str(transit_ckpt),
            ]
        },
        {
            "id": "B2_strict_threshold_seed3013",
            "candidate": "Deficit_B",
            "seed": 3013,
            "desc": "Seed 3013 Strict Place Threshold 0.015m (Diagnostic: Does delaying place entry prevent stall?)",
            "flags": [
                "--selector", "composite_patch",
                "--checkpoint", str(base_ckpt),
                "--patch-checkpoint", str(place_ckpt),
                "--transit-checkpoint", str(transit_ckpt),
                "--patch-threshold-xy", "0.015",
                "--transit-guard", "--transit-mode", "guard",
            ]
        },
    ]

    report: Dict[str, Any] = {
        "experiment": "T28_deficit_decomposition",
        "started_at": utc_now(),
        "bundle_dir": str(bundle_dir.relative_to(repo_dir)),
        "diagnostics": [],
        "conclusions": {}
    }

    t_start = time.time()
    print("=" * 80)
    print("STARTING T28 DEFICIT DECOMPOSITION EXPERIMENT")
    print(f"Testing Candidate A (3014) and Candidate B (3013) across {len(tests)} diagnostic runs")
    print("=" * 80)

    for idx, t in enumerate(tests, 1):
        r_dir = out_dir / f"run_{idx:02d}_{t['id']}"
        print(f"\n[{idx}/{len(tests)}] Running {t['id']} ({t['desc']})...")

        cmd = [
            py, str(repo_dir / "scripts" / "run_fetch_primitives.py"),
            "--out", str(r_dir),
            "--episodes", "1",
            "--seed", str(t["seed"]),
            "--max-steps", str(args.max_steps),
            "--consecutive-success-steps", "20",
            "--true-place-goal",
            "--rotations", "--base", "--far-start",
            "--allow-task-mismatch",
        ] + t["flags"]

        t_sub = time.time()
        run_cmd(cmd, cwd=repo_dir)
        analysis = analyze_run(r_dir)
        wall_s = round(time.time() - t_sub, 2)

        entry = {
            "index": idx,
            "test_id": t["id"],
            "candidate": t["candidate"],
            "seed": t["seed"],
            "description": t["desc"],
            "success": analysis["success"],
            "steps": analysis["steps"],
            "consecutive_success": analysis["consecutive_success"],
            "final_dist_xy": analysis["final_dist_xy"],
            "final_cube_z": analysis["final_cube_z"],
            "module_steps": analysis["module_steps"],
            "transitions": analysis["transitions"],
            "wall_seconds": wall_s,
            "run_dir": str(r_dir.name)
        }
        report["diagnostics"].append(entry)

        status_str = "SUCCESS" if entry["success"] else "FAILURE"
        print(f"  -> Result: {status_str} in {entry['steps']} steps ({wall_s}s), final_dist_xy={entry['final_dist_xy']}m, final_z={entry['final_cube_z']}m")
        print(f"     Module allocation: {entry['module_steps']}")

    # Formulate conclusion
    a1 = next(e for e in report["diagnostics"] if e["test_id"] == "A1_baseline_seed3014")
    a2 = next(e for e in report["diagnostics"] if e["test_id"] == "A2_forced_transit_seed3014")
    b1 = next(e for e in report["diagnostics"] if e["test_id"] == "B1_baseline_seed3013")
    b2 = next(e for e in report["diagnostics"] if e["test_id"] == "B2_strict_threshold_seed3013")

    # A evaluation: If forced transit still fails, it's action deficiency
    deficit_a_nature = "routing_deficiency" if a2["success"] else "action_deficiency"
    # B evaluation: If strict threshold / delayed place succeeds, it's threshold/routing deficiency; if it fails, it's action deficiency
    deficit_b_nature = "routing_and_threshold_deficiency" if b2["success"] else "action_deficiency"

    report["conclusions"] = {
        "candidate_A_seed3014": {
            "baseline_success": a1["success"],
            "forced_transit_success": a2["success"],
            "nature_of_deficit": deficit_a_nature,
            "interpretation": (
                "Existing transit module solves seed 3014 when forced; deficit is in router application condition."
                if a2["success"] else
                "Existing transit module fails even when forced; new behavioral candidate action is strictly required for far goal transit."
            )
        },
        "candidate_B_seed3013": {
            "baseline_success": b1["success"],
            "strict_threshold_success": b2["success"],
            "nature_of_deficit": deficit_b_nature,
            "interpretation": (
                "Delaying place entry threshold solves seed 3013; deficit is primarily in early latch threshold."
                if b2["success"] else
                "Strict threshold fails; new offset recovery candidate action is strictly required for place entry."
            )
        },
        "total_wall_seconds": round(time.time() - t_start, 2),
        "completed_at": utc_now()
    }

    out_file = out_dir / "t28_decomposition_summary.json"
    json_write(out_file, report)

    print("\n" + "=" * 80)
    print("T28 DEFICIT DECOMPOSITION EXPERIMENT COMPLETE")
    print(f"Summary JSON: {out_file}")
    print(f"Candidate A (Seed 3014): {deficit_a_nature}")
    print(f"Candidate B (Seed 3013): {deficit_b_nature}")
    print("=" * 80)


if __name__ == "__main__":
    main()
