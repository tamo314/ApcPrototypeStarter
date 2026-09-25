import json
from pathlib import Path
import numpy as np
from apc_maniskill.deficit_detector import AutonomousDeficitDetector

TEST_CASES = [
    # 1. Genuine Deficits (Should trigger adaptation)
    ("runs/apc-ablation-scope-cart_continuous-seed3011-20260925-a", "genuine_deficit", "Transit Drift Failure (seed 3011 unpatched)"),
    ("runs/apc-ablation-scope-cart_continuous-seed3012-20260925-a", "genuine_deficit", "Transit Stagnation Failure (seed 3012 unpatched)"),

    # 2. Normal Successes (Should NEVER trigger adaptation - False Alarm = 0)
    ("runs/apc-t13-unified-router-eval-seed3011-20260925-a", "normal", "True Place Success (Unified Router seed 3011)"),
    ("runs/apc-t13-retention-place-seed3001-20260925-a", "normal", "Place Retention Success (seed 3001)"),
    ("runs/apc-t13-retention-pick-seed3201-20260925-a", "normal", "Pick Retention Success (seed 3201)"),

    # 3. Transient / Recoverable Failures (Should classify as transient or resolve without spurious expansion)
    ("runs/apc-t06-secure-seed3009-20260925-a", "transient_recoverable", "Pre-grasp Stalling (seed 3009)"),
    ("runs/apc-t06-secure-seed3008-20260925-b", "transient_recoverable", "Lost Grasp Slip (seed 3008)"),
]

def evaluate_detector():
    detector = AutonomousDeficitDetector()
    results = []

    print("================================================================================")
    print("Evaluating Autonomous Deficit Detector (W5: T14)")
    print("================================================================================")

    for run_dir, expected_category, label in TEST_CASES:
        path = Path(run_dir) / "steps.jsonl"
        if not path.exists():
            print(f"Skipping {label}: {path} not found")
            continue

        detector.reset()
        triggered_adaptation = False
        first_trigger_step = -1
        first_trigger_reason = ""
        transient_detected = False
        total_steps = 0

        with open(path) as f:
            for line in f:
                d = json.loads(line)
                total_steps += 1
                diag = d.get("info", {}).get("diagnostic", {})
                pre = diag.get("pre_action_state", {})
                if not pre:
                    continue

                executed_id = diag.get("executed_id", d.get("action", 0))
                override_code = diag.get("override_reason_code", 0)

                diag_res = detector.update(pre, executed_id, override_code)
                if diag_res.status == "transient_recoverable":
                    transient_detected = True
                if diag_res.trigger_adaptation and not triggered_adaptation:
                    triggered_adaptation = True
                    first_trigger_step = d["step"]
                    first_trigger_reason = diag_res.reason

        results.append({
            "label": label,
            "expected": expected_category,
            "total_steps": total_steps,
            "triggered_adaptation": triggered_adaptation,
            "first_trigger_step": first_trigger_step,
            "first_trigger_reason": first_trigger_reason,
            "transient_detected": transient_detected
        })

        status_str = "TRIGGERED" if triggered_adaptation else "NO_TRIGGER"
        print(f"\n[{expected_category.upper()}] {label}")
        print(f"  Steps: {total_steps} | Adaptation Triggered: {status_str} (Step {first_trigger_step})")
        if triggered_adaptation:
            print(f"  Reason: {first_trigger_reason}")
        if transient_detected:
            print(f"  Transient Recoverable Event Detected: Yes")

    # Summary metrics
    n_genuine = sum(1 for r in results if r["expected"] == "genuine_deficit")
    tp_genuine = sum(1 for r in results if r["expected"] == "genuine_deficit" and r["triggered_adaptation"])
    
    n_normal = sum(1 for r in results if r["expected"] == "normal")
    fp_normal = sum(1 for r in results if r["expected"] == "normal" and r["triggered_adaptation"])

    print("\n--------------------------------------------------------------------------------")
    print("Summary Metrics:")
    print(f"  Deficit Detection Rate (Sensitivity): {tp_genuine}/{n_genuine} ({tp_genuine/n_genuine*100:.1f}%)" if n_genuine > 0 else "N/A")
    print(f"  False Alarm Rate on Normal Tasks:     {fp_normal}/{n_normal} ({fp_normal/n_normal*100:.1f}%)" if n_normal > 0 else "N/A")
    print("--------------------------------------------------------------------------------")

if __name__ == "__main__":
    evaluate_detector()
