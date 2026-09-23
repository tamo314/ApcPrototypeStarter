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
        output["episodes"].append(dict(
            seed=episode["seed"], steps=len(rows), first=episode["success_ever"], final=episode["success_final"],
            wall_seconds=episode["wall_seconds"], grasped_steps=sum(d["post_action_state"]["grasped"] for d in rows),
            arm_rejections=len(rejects),
            # These flags can overlap; they are not mutually exclusive causes.
            ik_failure_steps=sum(not d["ik_success"] for d in rejects),
            joint_limit_failure_steps=sum(not d["ik_within_limits"] for d in rejects),
            table_path_failure_steps=sum(not d["ik_table_clear"] for d in rejects),
            base_rejections=sum(d["override_reason_code"] == 3 for d in rows),
            retreat_choices=sum(d["proposed_id"] == 17 for d in rows),
            probe_steps=sum(d.get("probe_applied", False) for d in rows),
            contact_measurements=len(contacts),
            contact_steps=sum(force > 0 for force in contacts) if contacts else None,
            max_contact_n=max(contacts) if contacts else None,
            queried_steps=len(queried), mismatches=sum(confusion.values()) if queried else None,
            confusion=[dict(teacher=a, proposed=b, count=n) for (a, b), n in confusion.most_common()]))
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
    for result in results:
        rows = result["episodes"]
        print(Path(result["run"]).name, len(rows), sum(r["steps"] for r in rows),
              sum(r["first"] for r in rows), sum(r["final"] for r in rows),
              max((r["max_contact_n"] for r in rows if r["max_contact_n"] is not None), default=None))


if __name__ == "__main__":
    main()
