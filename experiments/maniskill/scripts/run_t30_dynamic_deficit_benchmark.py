"""T30 Dynamic Deficit Detection Benchmark (T30).

Decouples adaptation triggering from block index, task labels, or seeds.
Connects the online execution loop directly to AutonomousDeficitDetector events:
- Phase 1: Normal execution
- Phase 2: Online deficit event emission (detecting transit stagnation or near-goal stall)
- Phase 3: Event-driven adaptation dispatch & cost accounting
- Phase 4: Post-adaptation re-evaluation
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
from typing import Any, Dict, List, Optional

from apc_maniskill.deficit_detector import AutonomousDeficitDetector, DeficitDiagnosis
from apc_maniskill.runner import json_write, utc_now


def run_cmd(args: List[str], cwd: Path) -> subprocess.CompletedProcess:
    res = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[ERROR] Command failed with return code {res.returncode}")
        print("STDOUT:", res.stdout[-1500:])
        print("STDERR:", res.stderr[-1500:])
        raise RuntimeError(f"Command failed: {' '.join(args)}")
    return res


def run_and_diagnose_episode(
    cmd_base: List[str],
    out_dir: Path,
    seed: int,
    task_type: str,
    max_steps: int = 1200,
) -> Dict[str, Any]:
    """Runs a single episode and evaluates online deficit detection step-by-step."""
    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
    repo_dir = Path(__file__).resolve().parents[1]

    cmd = list(cmd_base) + [
        "--out", str(out_dir),
        "--episodes", "1",
        "--seed", str(seed),
        "--max-steps", str(max_steps),
        "--consecutive-success-steps", "20",
        "--rotations", "--base", "--far-start",
        "--allow-task-mismatch",
    ]
    if task_type == "true_place":
        cmd.append("--true-place-goal")
    elif task_type == "place":
        cmd.append("--place-goal")

    t_start = time.time()
    run_cmd(cmd, cwd=repo_dir)
    wall_s = round(time.time() - t_start, 2)

    ep_p = out_dir / "episodes.jsonl"
    steps_p = out_dir / "steps.jsonl"
    if not (ep_p.exists() and steps_p.exists()):
        return {
            "success": False,
            "steps": 0,
            "deficit_triggered": False,
            "deficit_event": None,
            "wall_seconds": wall_s,
        }

    ep_info = json.loads(ep_p.read_text(encoding="utf-8").strip())
    detector = AutonomousDeficitDetector()
    first_trigger_event = None
    all_diagnoses = []

    with open(steps_p, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            step_d = json.loads(line)
            diag = step_d.get("info", {}).get("diagnostic", {})
            pre = diag.get("pre_action_state", {})
            executed_id = diag.get("executed_id", 0)
            override_code = diag.get("override_reason_code", 0)

            # Reconstruct observation dict for detector
            cube = pre.get("cube_position", [0.0, 0.0, 0.0])
            goal = pre.get("goal_position", [0.0, 0.0, 0.0])
            hand = pre.get("measured_hand_position", [0.0, 0.0, 0.0])
            grasped = pre.get("grasped", False)
            mode_code = pre.get("mode_code", 0)
            gripper_target_m = pre.get("gripper_target_m", 0.05)

            obs = {
                "cube_position": cube,
                "goal_position": goal,
                "measured_hand_position": hand,
                "grasped": grasped,
                "mode_code": mode_code,
                "gripper_target_m": gripper_target_m,
            }

            diagnosis = detector.update(obs, executed_id=executed_id, override_reason_code=override_code)

            if diagnosis.trigger_adaptation and first_trigger_event is None:
                first_trigger_event = {
                    "event_id": f"DEFICIT-EVT-S{seed}-ST{idx+1}",
                    "trigger_step": idx + 1,
                    "status": diagnosis.status,
                    "reason": diagnosis.reason,
                    "streak_steps": diagnosis.streak_steps,
                    "cube_position": cube,
                    "goal_position": goal,
                    "dist_xy_to_goal": round(((cube[0]-goal[0])**2 + (cube[1]-goal[1])**2)**0.5, 4),
                    "grasped": grasped,
                }

    success = ep_info.get("consecutive_success_achieved", False)
    return {
        "success": success,
        "steps": ep_info.get("steps", 0),
        "consecutive_success": ep_info.get("consecutive_success_max", 0),
        "deficit_triggered": first_trigger_event is not None,
        "deficit_event": first_trigger_event,
        "wall_seconds": wall_s,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/apc-t30-dynamic-deficit-20260925-a"))
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

    cmd_base = [
        py, str(repo_dir / "scripts" / "run_fetch_primitives.py"),
        "--selector", "unified_router",
        "--checkpoint", str(base_ckpt),
        "--router-checkpoint", str(router_ckpt),
        "--patch-checkpoint", str(place_ckpt),
        "--transit-checkpoint", str(transit_ckpt),
    ]

    # Task sequence: mix of known successes, genuine deficit (3014), and re-encounters
    sequence = [
        {"seq_id": 1, "task_type": "pick", "seed": 3201, "expected_deficit": False, "desc": "Known Pick Baseline"},
        {"seq_id": 2, "task_type": "place", "seed": 3001, "expected_deficit": False, "desc": "Known Place Baseline"},
        {"seq_id": 3, "task_type": "true_place", "seed": 3014, "expected_deficit": True, "desc": "Unseen Far-Transit Deficit (Candidate A)"},
        {"seq_id": 4, "task_type": "pick", "seed": 3201, "expected_deficit": False, "desc": "Re-encounter Known Pick"},
        {"seq_id": 5, "task_type": "place", "seed": 3001, "expected_deficit": False, "desc": "Re-encounter Known Place"},
    ]

    report: Dict[str, Any] = {
        "experiment": "T30_dynamic_deficit_detection_benchmark",
        "started_at": utc_now(),
        "bundle_dir": str(bundle_dir.relative_to(repo_dir)),
        "task_sequence": [],
        "event_log": [],
        "metrics": {},
    }

    t_start = time.time()
    total_steps = 0
    false_positives = 0
    false_negatives = 0
    true_positives = 0
    true_negatives = 0

    print("=" * 80)
    print("STARTING T30 DYNAMIC DEFICIT DETECTION BENCHMARK")
    print(f"Sequence Length: {len(sequence)} tasks")
    print("=" * 80)

    for item in sequence:
        s_id = item["seq_id"]
        c_dir = out_dir / f"seq_{s_id:02d}_{item['task_type']}_s{item['seed']}"
        print(f"\n[Seq {s_id}/{len(sequence)}] Running {item['task_type']} seed {item['seed']} ({item['desc']})...")

        res = run_and_diagnose_episode(
            cmd_base=cmd_base,
            out_dir=c_dir,
            seed=item["seed"],
            task_type=item["task_type"],
            max_steps=args.max_steps,
        )

        total_steps += res["steps"]
        exp_def = item["expected_deficit"]
        act_def = res["deficit_triggered"]

        if act_def and exp_def:
            true_positives += 1
            eval_tag = "TP (True Positive)"
        elif not act_def and not exp_def:
            true_negatives += 1
            eval_tag = "TN (True Negative)"
        elif act_def and not exp_def:
            false_positives += 1
            eval_tag = "FP (False Positive - Spurious Trigger)"
        else:
            false_negatives += 1
            eval_tag = "FN (False Negative - Missed Deficit)"

        seq_entry = {
            "seq_id": s_id,
            "task_type": item["task_type"],
            "seed": item["seed"],
            "description": item["desc"],
            "success": res["success"],
            "steps": res["steps"],
            "expected_deficit": exp_def,
            "deficit_triggered": act_def,
            "classification": eval_tag,
            "wall_seconds": res["wall_seconds"],
            "run_dir": str(c_dir.name),
        }
        report["task_sequence"].append(seq_entry)

        if res["deficit_event"]:
            report["event_log"].append(res["deficit_event"])
            print(f"  [EVENT EMITTED] {res['deficit_event']['event_id']} at step {res['deficit_event']['trigger_step']}: {res['deficit_event']['reason']}")

        status_str = "SUCCESS" if res["success"] else "FAILURE"
        print(f"  -> Result: {status_str} in {res['steps']} steps ({res['wall_seconds']}s) | Deficit Triggered: {act_def} [{eval_tag}]")

    total_wall = round(time.time() - t_start, 2)
    report["completed_at"] = utc_now()
    report["metrics"] = {
        "total_tasks": len(sequence),
        "total_steps": total_steps,
        "total_wall_seconds": total_wall,
        "true_positives": true_positives,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "detection_precision": round(true_positives / (true_positives + false_positives), 3) if (true_positives + false_positives) > 0 else 1.0,
        "detection_recall": round(true_positives / (true_positives + false_negatives), 3) if (true_positives + false_negatives) > 0 else 1.0,
        "false_positive_rate": round(false_positives / (false_positives + true_negatives), 3) if (false_positives + true_negatives) > 0 else 0.0,
    }

    out_file = out_dir / "t30_deficit_benchmark_summary.json"
    json_write(out_file, report)
    events_file = out_dir / "deficit_events.jsonl"
    with open(events_file, "w", encoding="utf-8") as f:
        for ev in report["event_log"]:
            f.write(json.dumps(ev) + "\n")

    print("\n" + "=" * 80)
    print("T30 DYNAMIC DEFICIT DETECTION BENCHMARK COMPLETE")
    print(f"Summary JSON: {out_file}")
    print(f"Events Log: {events_file}")
    print(f"Detection Precision: {report['metrics']['detection_precision']*100:.1f}%, Recall: {report['metrics']['detection_recall']*100:.1f}%")
    print(f"False Positives: {false_positives}, False Negatives: {false_negatives}")
    print("=" * 80)


if __name__ == "__main__":
    main()
