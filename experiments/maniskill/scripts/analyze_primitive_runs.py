"""Summarize saved primitive runs, keeping solver failure flags separate."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil


def analyze(run):
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError(f"Run is not completed: {run}")
    episodes = [json.loads(line) for line in (run / "episodes.jsonl").read_text().splitlines()]
    groups = {e["episode"]: [] for e in episodes}
    steps_file = run / "steps.jsonl"
    for line in steps_file.read_text().splitlines():
        row = json.loads(line)
        groups[row["episode"]].append(row["info"]["diagnostic"])
    output = dict(run=str(run.resolve()), steps_sha256=hashlib.sha256(steps_file.read_bytes()).hexdigest(),
                  selector=manifest["policy_details"]["selector"], episodes=[])
    for episode in episodes:
        rows = groups[episode["episode"]]
        if len(rows) != episode["steps"]:
            raise ValueError(f"Saved step count mismatch: {run}")
        rejects = [d for d in rows if d["override_reason_code"] == 2]
        queried = [d for d in rows if "teacher_id" in d]
        contacts = [d["table_contact_force_norm_sum_n"] for d in rows
                    if "table_contact_force_norm_sum_n" in d]
        confusion = Counter((d["teacher_id"], d["proposed_id"]) for d in queried
                            if d["teacher_id"] != d["proposed_id"])
        # Failure classification according to APC_NEXT_RESEARCH_PLAN.md P1
        grasped_steps = sum(d["post_action_state"]["grasped"] for d in rows)
        last_grasped = rows[-1]["post_action_state"]["grasped"]
        cube_pos = rows[0]["post_action_state"].get("cube_position")
        goal_pos = rows[0]["post_action_state"].get("goal_position")
        table_path_fails = sum(not d["ik_table_clear"] for d in rejects)

        if episode["success_final"]:
            failure_mode = "success"
        elif table_path_fails > 100 and grasped_steps == 0:
            failure_mode = "path_table_reject"
        elif grasped_steps == 0:
            failure_mode = "grasp_failure"
        elif grasped_steps > 0 and not last_grasped:
            failure_mode = "grasp_loss"
        elif last_grasped:
            # Check if reached near goal
            final_p = rows[-1]["post_action_state"]
            fcube = final_p.get("cube_position")
            fgoal = final_p.get("goal_position")
            if fcube and fgoal:
                dist = sum((a - b) ** 2 for a, b in zip(fcube, fgoal)) ** 0.5
                if dist < 0.05:
                    failure_mode = "static_hold_failure"
                else:
                    failure_mode = "goal_unreached"
            else:
                failure_mode = "goal_unreached"
        else:
            failure_mode = "unknown"

        output["episodes"].append(dict(
            seed=episode["seed"], steps=len(rows), first=episode["success_ever"], final=episode["success_final"],
            failure_mode=failure_mode,
            cube_pos=cube_pos, goal_pos=goal_pos,
            wall_seconds=episode["wall_seconds"], grasped_steps=grasped_steps,
            arm_rejections=len(rejects),
            # These flags can overlap; they are not mutually exclusive causes.
            ik_failure_steps=sum(not d["ik_success"] for d in rejects),
            joint_limit_failure_steps=sum(not d["ik_within_limits"] for d in rejects),
            ik_solver_attempts=sum(d.get("ik_attempts", 0) +
                                   (d.get("translation_backoff_initial_solver") or {}).get("ik_attempts", 0)
                                   for d in rows),
            ik_reset_seed_attempts=sum(d.get("ik_reset_seed_attempts", 0) +
                                       (d.get("translation_backoff_initial_solver") or {}).get("ik_reset_seed_attempts", 0)
                                       for d in rows),
            ik_reset_seed_recovered_steps=sum(d.get("ik_reset_seed_used", False) for d in rows),
            translation_backoff_attempts=sum(d.get("translation_backoff_initial_solver") is not None for d in rows),
            translation_backoff_recovered_steps=sum(d.get("translation_backoff_used", False) for d in rows),
            pitch_schedule_descending_steps=sum(d.get("pitch_schedule_descending", False) for d in rows),
            table_path_failure_steps=table_path_fails,
            base_rejections=sum(d["override_reason_code"] == 3 for d in rows),
            retreat_choices=sum(d["proposed_id"] == 17 for d in rows),
            probe_steps=sum(d.get("probe_applied", False) for d in rows),
            contact_measurements=len(contacts),
            contact_steps=sum(force > 0 for force in contacts) if contacts else None,
            max_contact_n=max(contacts) if contacts else None,
            queried_steps=len(queried), mismatches=sum(confusion.values()) if queried else None,
            confusion=[dict(teacher=a, proposed=b, count=n) for (a, b), n in confusion.most_common()]))
    output["provenance"] = {
        "git_commit": manifest.get("git_commit"),
        "task": manifest.get("task"),
        "policy": manifest.get("policy"),
        "policy_details": manifest.get("policy_details"),
    }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    results = [analyze(run) for run in args.runs]
    args.out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.out / Path(__file__).name)
    (args.out / "report.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    
    # Generate Markdown summary table
    md_lines = ["# Primitive Evaluation Comparison\n"]
    for result in results:
        run_name = Path(result["run"]).name
        selector = result["selector"]
        rows = result["episodes"]
        n_succ = sum(r["final"] for r in rows)
        total_steps = sum(r["steps"] for r in rows)
        max_contact = max((r["max_contact_n"] for r in rows if r["max_contact_n"] is not None), default=0.0)
        
        md_lines.append(f"## Run: `{run_name}` ({selector})")
        md_lines.append(f"- Success: {n_succ}/{len(rows)} | Total Steps: {total_steps} | Max Contact: {max_contact:.2f} N\n")
        md_lines.append("| Seed | Final | Steps | Grasped | Mode | Goal Z | Rejects | Table Path Fails |")
        md_lines.append("|---|---|---|---|---|---|---|---|")
        for r in rows:
            gz = f"{r['goal_pos'][2]:.3f}" if r.get('goal_pos') else "-"
            md_lines.append(f"| {r['seed']} | {r['final']} | {r['steps']} | {r['grasped_steps']} | {r['failure_mode']} | {gz} | {r['arm_rejections']} | {r['table_path_failure_steps']} |")
        md_lines.append("\n")
        
        # Print to stdout
        print(f"[{run_name}] {selector}: {n_succ}/{len(rows)} success, {total_steps} steps, max contact {max_contact:.2f} N")
        for r in rows:
            print(f"  Seed {r['seed']}: final={r['final']}, steps={r['steps']}, mode={r['failure_mode']}, goal_z={r.get('goal_pos', [0,0,0])[2]:.3f}")
    
    (args.out / "comparison.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")



if __name__ == "__main__":
    main()
