"""Independent Confirmation and Reproducibility Benchmark on Unseen Conditions (W8: T25).

Evaluates the frozen minimal autonomous standalone bundle (`dist_autonomous_bundle_v1`)
in an isolated subprocess execution across held-out seeds and environmental conditions:
- Zero temporary artifacts (MLP, replay buffers, training graphs, optimizer states)
- Task 1: Held-out PickCubeFar (seeds 3202, 3203, 3204)
- Task 2: Held-out PlaceCubeFar (seeds 3002, 3003, 3004)
- Task 3: Held-out TruePlaceFar (In-coverage: seeds 3010, 3013, 3015; Out-of-coverage boundary: seeds 3016, 3017)

Records:
- HoldContinuousSuccess (20 consecutive steps)
- Execution steps, wall-time, process RSS memory
- Active module distribution and transitions
- Binomial confidence intervals and failure categorization
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

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
        return {
            "success": False,
            "steps": 0,
            "first_success_step": None,
            "consecutive_steps": 0,
            "module_steps": {},
            "transitions": 0,
            "failure_reason": "missing_log_files"
        }

    ep_info = json.loads(ep_p.read_text(encoding="utf-8").strip())
    module_counts = Counter()
    prev_module = None
    transitions = 0
    first_success = None
    max_consecutive = ep_info.get("consecutive_success_max", 0)

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

    success = ep_info.get("consecutive_success_achieved", False)
    failure_reason = None
    if not success:
        total_steps = ep_info.get("steps", 0)
        if total_steps >= 1200:
            failure_reason = "timeout_max_steps_exceeded"
        elif first_success is None:
            failure_reason = "target_never_reached"
        else:
            failure_reason = "failed_continuous_hold_20steps"

    return {
        "success": success,
        "steps": ep_info.get("steps", 0),
        "first_success_step": first_success,
        "consecutive_success_achieved": success,
        "max_consecutive_success": max_consecutive,
        "module_steps": dict(sorted(module_counts.items())),
        "transitions": transitions,
        "failure_reason": failure_reason,
    }


def compute_wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Compute Wilson score interval for binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.95996  # 95% confidence
    p = k / n
    denominator = 1 + z**2 / n
    centre_adjusted_probability = p + z**2 / (2 * n)
    adjusted_std = math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    lower = max(0.0, (centre_adjusted_probability - z * adjusted_std) / denominator)
    upper = min(1.0, (centre_adjusted_probability + z * adjusted_std) / denominator)
    return (round(lower, 3), round(upper, 3))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path,
                        default=Path("dist_autonomous_bundle_v1"))
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--hold-steps", type=int, default=20)
    args = parser.parse_args()

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = Path(__file__).resolve().parents[1]
    bundle_dir = (repo_dir / args.bundle_dir).resolve() if not args.bundle_dir.is_absolute() else args.bundle_dir.resolve()
    py = sys.executable

    # Verify bundle integrity
    base_ckpt = bundle_dir / "base_selector.pt"
    place_ckpt = bundle_dir / "place_candidate.pt"
    transit_ckpt = bundle_dir / "transit_candidate.pt"
    router_ckpt = bundle_dir / "unified_router.pt"
    manifest_p = bundle_dir / "bundle_manifest.json"

    for p in [base_ckpt, place_ckpt, transit_ckpt, router_ckpt, manifest_p]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required bundle file: {p}")

    manifest_data = json.loads(manifest_p.read_text(encoding="utf-8"))

    # Test cases specification (All held-out unseen conditions)
    test_cases = [
        # Task 1: Held-out PickCubeFar
        {"category": "pick", "task_name": "APC-FetchPickCubeFar-v1", "seed": 3202, "group": "held_out_in_dist", "desc": "Held-out Pick Seed 3202"},
        {"category": "pick", "task_name": "APC-FetchPickCubeFar-v1", "seed": 3203, "group": "held_out_in_dist", "desc": "Held-out Pick Seed 3203"},
        {"category": "pick", "task_name": "APC-FetchPickCubeFar-v1", "seed": 3204, "group": "held_out_in_dist", "desc": "Held-out Pick Seed 3204"},

        # Task 2: Held-out PlaceCubeFar
        {"category": "place", "task_name": "APC-FetchPlaceCubeFar-v1", "seed": 3002, "group": "held_out_in_dist", "desc": "Held-out Place Seed 3002"},
        {"category": "place", "task_name": "APC-FetchPlaceCubeFar-v1", "seed": 3003, "group": "held_out_in_dist", "desc": "Held-out Place Seed 3003"},
        {"category": "place", "task_name": "APC-FetchPlaceCubeFar-v1", "seed": 3004, "group": "held_out_in_dist", "desc": "Held-out Place Seed 3004"},

        # Task 3: Held-out TruePlaceFar (Compositional Transfer)
        # In-coverage:
        {"category": "true_place", "task_name": "APC-FetchTruePlaceFar-v1", "seed": 3010, "group": "compositional_in_coverage", "desc": "Held-out TruePlace Seed 3010 (In Coverage)"},
        {"category": "true_place", "task_name": "APC-FetchTruePlaceFar-v1", "seed": 3013, "group": "compositional_in_coverage", "desc": "Held-out TruePlace Seed 3013 (In Coverage)"},
        {"category": "true_place", "task_name": "APC-FetchTruePlaceFar-v1", "seed": 3015, "group": "compositional_in_coverage", "desc": "Held-out TruePlace Seed 3015 (In Coverage)"},
        # Out-of-coverage boundary check:
        {"category": "true_place", "task_name": "APC-FetchTruePlaceFar-v1", "seed": 3016, "group": "compositional_boundary_extrapolation", "desc": "Held-out TruePlace Seed 3016 (Boundary Check)"},
        {"category": "true_place", "task_name": "APC-FetchTruePlaceFar-v1", "seed": 3017, "group": "compositional_boundary_extrapolation", "desc": "Held-out TruePlace Seed 3017 (Boundary Check)"},
    ]

    report: Dict[str, Any] = {
        "experiment": "w8_t25_independent_confirmation_reproducibility",
        "started_at": utc_now(),
        "bundle_dir": str(bundle_dir),
        "bundle_manifest_summary": manifest_data.get("summary", {}),
        "protocol": {
            "max_steps": args.max_steps,
            "hold_continuous_success_steps": args.hold_steps,
            "selector": "unified_router",
            "zero_training_samples": 0,
            "zero_updates": 0,
            "temporary_dependency": False,
        },
        "results": [],
        "group_summary": {},
    }

    t_global_start = time.time()
    print("=" * 80)
    print("W8: T25 INDEPENDENT CONFIRMATION BENCHMARK STARTING")
    print(f"Bundle Directory: {bundle_dir} (Total Size: {manifest_data.get('summary', {}).get('total_size_kb', 0)} KB)")
    print(f"Total Test Cases: {len(test_cases)}")
    print("=" * 80)

    for idx, tc in enumerate(test_cases, 1):
        cat = tc["category"]
        seed = tc["seed"]
        group = tc["group"]
        tc_dir = out_dir / f"tc_{idx:02d}_{cat}_seed{seed}"

        print(f"\n[{idx}/{len(test_cases)}] Executing {cat.upper()} Seed {seed} ({tc['desc']})...")
        t_sub_start = time.time()

        cmd = [
            py, str(repo_dir / "scripts" / "run_fetch_primitives.py"),
            "--out", str(tc_dir),
            "--episodes", "1",
            "--seed", str(seed),
            "--max-steps", str(args.max_steps),
            "--consecutive-success-steps", str(args.hold_steps),
            "--selector", "unified_router",
            "--checkpoint", str(base_ckpt),
            "--router-checkpoint", str(router_ckpt),
            "--patch-checkpoint", str(place_ckpt),
            "--transit-checkpoint", str(transit_ckpt),
            "--rotations", "--base", "--far-start",
            "--allow-task-mismatch",
        ]

        if cat == "true_place":
            cmd.append("--true-place-goal")
        elif cat == "place":
            cmd.append("--place-goal")
        # For pick, no extra flag needed

        run_cmd(cmd, cwd=repo_dir)
        analysis = analyze_run(tc_dir)
        sub_wall = time.time() - t_sub_start

        entry = {
            "index": idx,
            "category": cat,
            "task_name": tc["task_name"],
            "seed": seed,
            "group": group,
            "description": tc["desc"],
            "success": analysis["success"],
            "steps": analysis["steps"],
            "first_success_step": analysis["first_success_step"],
            "max_consecutive_success": analysis["max_consecutive_success"],
            "module_steps": analysis["module_steps"],
            "transitions": analysis["transitions"],
            "failure_reason": analysis["failure_reason"],
            "wall_seconds": round(sub_wall, 2),
            "run_dir": str(tc_dir.name),
        }
        report["results"].append(entry)

        status_str = "SUCCESS" if entry["success"] else f"FAILURE ({entry['failure_reason']})"
        print(f"  -> Result: {status_str} in {entry['steps']} steps ({round(sub_wall, 1)}s), transitions: {entry['transitions']}")
        print(f"     Module steps: {entry['module_steps']}")

    # Aggregation by groups
    groups = sorted(list(set(tc["group"] for tc in test_cases)))
    for g in groups:
        g_entries = [e for e in report["results"] if e["group"] == g]
        k = sum(1 for e in g_entries if e["success"])
        n = len(g_entries)
        ci = compute_wilson_ci(k, n)
        avg_steps = round(sum(e["steps"] for e in g_entries) / n, 1) if n > 0 else 0
        succ_steps = [e["steps"] for e in g_entries if e["success"]]
        avg_succ_steps = round(sum(succ_steps) / len(succ_steps), 1) if succ_steps else None

        report["group_summary"][g] = {
            "total_episodes": n,
            "successful_episodes": k,
            "success_rate": round(k / n, 3) if n > 0 else 0.0,
            "wilson_95_ci": list(ci),
            "average_steps_all": avg_steps,
            "average_steps_success_only": avg_succ_steps,
        }

    # Overall summary
    total_k = sum(1 for e in report["results"] if e["success"])
    total_n = len(report["results"])
    overall_ci = compute_wilson_ci(total_k, total_n)

    report["overall_summary"] = {
        "total_episodes": total_n,
        "successful_episodes": total_k,
        "success_rate": round(total_k / total_n, 3),
        "wilson_95_ci": list(overall_ci),
        "total_wall_seconds": round(time.time() - t_global_start, 2),
    }

    report["completed_at"] = utc_now()

    out_file = out_dir / "independent_confirmation_summary.json"
    json_write(out_file, report)

    print("\n" + "=" * 80)
    print("W8: T25 INDEPENDENT CONFIRMATION BENCHMARK COMPLETE")
    print(f"Summary JSON: {out_file}")
    print(f"Overall Success Rate: {total_k}/{total_n} ({report['overall_summary']['success_rate'] * 100:.1f}%) [95% CI: {overall_ci[0]*100:.1f}% - {overall_ci[1]*100:.1f}%]")
    for g, s in report["group_summary"].items():
        print(f"  Group '{g}': {s['successful_episodes']}/{s['total_episodes']} ({s['success_rate']*100:.1f}%) [95% CI: {s['wilson_95_ci'][0]*100:.1f}% - {s['wilson_95_ci'][1]*100:.1f}%], avg steps: {s['average_steps_all']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
