"""Re-aggregate stability metrics and consecutive success diagnostics (T00).

Distinguishes:
1. hold_complete (observation window completion after first success)
2. final_success (task success on the final step)
3. max_consecutive_success (maximum continuous steps of success)
4. final_consecutive_success (continuous success leading up to termination)
5. post-first-success success ratio and cause of interruption
"""
import json
from pathlib import Path
import argparse


def analyze_run(run_dir: Path):
    steps_file = run_dir / "steps.jsonl"
    episodes_file = run_dir / "episodes.jsonl"
    summary_file = run_dir / "summary.json"
    if not (steps_file.exists() and episodes_file.exists()):
        return None

    with open(steps_file, encoding="utf-8") as f:
        steps = [json.loads(line) for line in f if line.strip()]
    with open(episodes_file, encoding="utf-8") as f:
        ep_data = json.loads(f.read().strip())

    succ_seq = [bool(s.get("success", False)) for s in steps]
    first_succ = next((i for i, s in enumerate(succ_seq) if s), None)

    # Consecutive tracking
    cur_consec = 0
    max_consec = 0
    for s in succ_seq:
        if s:
            cur_consec += 1
            if cur_consec > max_consec:
                max_consec = cur_consec
        else:
            cur_consec = 0
    final_consec = cur_consec
    final_succ = succ_seq[-1] if succ_seq else False

    # Breakdown of post-first-success steps
    post_steps = []
    drop_reasons = []
    if first_succ is not None:
        post_steps = steps[first_succ:]
        for s in post_steps:
            if not s.get("success", False):
                info = s.get("info", {})
                placed = bool(info.get("is_obj_placed_surface", False) or info.get("is_placed", False))
                rel = bool(info.get("is_released", False))
                obj_stat = bool(info.get("is_obj_static", False))
                rob_stat = bool(info.get("is_robot_static", False))
                reasons = []
                if not placed: reasons.append("not_placed")
                if not rel: reasons.append("not_released")
                if not obj_stat: reasons.append("obj_moving")
                if not rob_stat: reasons.append("robot_moving")
                drop_reasons.append((s.get("step"), reasons))

    post_succ_cnt = sum(bool(s.get("success", False)) for s in post_steps)
    post_succ_ratio = (post_succ_cnt / len(post_steps)) if post_steps else 0.0

    return {
        "run": run_dir.name,
        "seed": ep_data.get("seed"),
        "total_steps": len(steps),
        "first_succ": first_succ,
        "final_succ": final_succ,
        "max_consec": max_consec,
        "final_consec": final_consec,
        "post_steps_count": len(post_steps),
        "post_succ_cnt": post_succ_cnt,
        "post_succ_ratio": post_succ_ratio,
        "hold_complete": ep_data.get("hold_complete"),
        "hold_20_recorded": ep_data.get("consecutive_success_final_20"),
        "drop_reasons": drop_reasons,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patterns", nargs="+", default=["apc-ablation-scope-*-20260925-a", "apc-regression-place-*-20260925-a"])
    parser.add_argument("--runs-dir", type=str, default="experiments/maniskill/runs")
    parser.add_argument("--out-json", type=str, default="experiments/maniskill/runs/reaggregated_stability_metrics.json")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    target_dirs = []
    for pat in args.patterns:
        target_dirs.extend(sorted(runs_dir.glob(pat)))

    results = []
    for d in target_dirs:
        res = analyze_run(d)
        if res:
            results.append(res)

    print(f"\nAnalyzed {len(results)} runs:")
    header = f"{'Run Name':<45} | {'Seed':<5} | {'Tot':<4} | {'1st':<4} | {'Fin':<5} | {'MaxC':<4} | {'FinC':<4} | {'Post':<4} | {'PSuc':<4} | {'HoldC':<5} | {'H20':<5}"
    print(header)
    print("-" * len(header))
    for r in results:
        fin_s = "TRUE" if r["final_succ"] else "FALSE"
        f1_s = str(r["first_succ"]) if r["first_succ"] is not None else "-"
        hc_s = "T" if r["hold_complete"] else "F"
        h20_s = "T" if r["hold_20_recorded"] else "F"
        print(f"{r['run']:<45} | {r['seed']:<5} | {r['total_steps']:<4} | {f1_s:<4} | {fin_s:<5} | {r['max_consec']:<4} | {r['final_consec']:<4} | {r['post_steps_count']:<4} | {r['post_succ_cnt']:<4} | {hc_s:<5} | {h20_s:<5}")

    print("\n\nBreakdown of Success Interruptions after First Success:")
    for r in results:
        if r["drop_reasons"]:
            print(f"- {r['run']} (Seed {r['seed']}): {len(r['drop_reasons'])} interruption steps out of {r['post_steps_count']}")
            for step, reas in r["drop_reasons"]:
                print(f"    Step {step}: {', '.join(reas)}")

    out_p = Path(args.out_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved re-aggregated metrics to {out_p}")


if __name__ == "__main__":
    main()
