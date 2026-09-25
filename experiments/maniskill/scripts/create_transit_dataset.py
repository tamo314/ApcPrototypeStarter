"""Prepare transit teacher dataset for learning transit local capability.

Records provenance distinguishing hand-designed teacher corrections from autonomous discovery.
"""
import hashlib
import json
from pathlib import Path


def prepare_transit_dataset():
    source_run = Path("runs/apc-transit-teacher-seed3002-20260925-a")
    normal_run = Path("runs/test-transit-guard-seed3001-20260925-a")
    out_dir = Path("runs/dataset-transit-teacher-20260925-b")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_3002 = json.loads((source_run / "manifest.json").read_text(encoding="utf-8"))
    manifest_3001 = json.loads((normal_run / "manifest.json").read_text(encoding="utf-8"))
    
    steps_3002_bytes = (source_run / "steps.jsonl").read_bytes()
    steps_3001_bytes = (normal_run / "steps.jsonl").read_bytes()
    digest_3002 = hashlib.sha256(steps_3002_bytes).hexdigest()
    digest_3001 = hashlib.sha256(steps_3001_bytes).hexdigest()

    extracted_steps = []
    count_3002 = 0
    guard_count = 0
    count_3001 = 0

    # 1. Extract from seed 3002 (contains transit guard triggers)
    with open(source_run / "steps.jsonl", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            diag = data.get("info", {}).get("diagnostic", {})
            pre = diag.get("pre_action_state", {})
            grasped = pre.get("grasped", False)
            cz = pre.get("cube_position", [0, 0, 0])[2]
            cube_init_z = pre.get("cube_initial_z", 0.02)
            patch_applied = diag.get("patch_applied", False)
            
            # Transit phase: grasped, lifted above table, before place patch activation
            if grasped and cz > cube_init_z + 0.06 and not patch_applied:
                guard = diag.get("transit_guard_triggered", False)
                if guard:
                    # Explicit teacher correction label
                    data["info"]["diagnostic"]["proposed_id"] = diag.get("transit_id", diag.get("proposed_id"))
                    guard_count += 1
                data["episode"] = 0
                extracted_steps.append(data)
                count_3002 += 1

    # 2. Extract from seed 3001 (contains normal successful transit steps)
    with open(normal_run / "steps.jsonl", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            diag = data.get("info", {}).get("diagnostic", {})
            pre = diag.get("pre_action_state", {})
            grasped = pre.get("grasped", False)
            cz = pre.get("cube_position", [0, 0, 0])[2]
            cube_init_z = pre.get("cube_initial_z", 0.02)
            patch_applied = diag.get("patch_applied", False)
            
            if grasped and cz > cube_init_z + 0.06 and not patch_applied:
                data["episode"] = 0
                extracted_steps.append(data)
                count_3001 += 1

    print(f"Total extracted transit training steps: {len(extracted_steps)}")
    print(f"From seed 3002: {count_3002} (guard triggered: {guard_count})")
    print(f"From seed 3001: {count_3001}")

    # Write steps.jsonl
    with open(out_dir / "steps.jsonl", "w", encoding="utf-8") as f:
        for step in extracted_steps:
            f.write(json.dumps(step) + "\n")

    # Generate synthesized episodes.jsonl matching the environment steps
    ep_3002 = json.loads((source_run / "episodes.jsonl").read_text(encoding="utf-8").strip())
    total_env_steps = ep_3002["steps"] + json.loads((normal_run / "episodes.jsonl").read_text(encoding="utf-8").strip())["steps"]
    
    ep_data = dict(ep_3002)
    ep_data["episode"] = 0
    ep_data["steps"] = total_env_steps
    ep_data["extracted_training_samples"] = len(extracted_steps)
    ep_data["source_runs"] = [
        {"run": str(source_run), "seed": 3002, "env_steps": ep_3002["steps"], "extracted_samples": count_3002, "guard_triggers": guard_count, "sha256": digest_3002},
        {"run": str(normal_run), "seed": 3001, "env_steps": total_env_steps - ep_3002["steps"], "extracted_samples": count_3001, "guard_triggers": 0, "sha256": digest_3001}
    ]
    with open(out_dir / "episodes.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(ep_data) + "\n")

    # Construct manifest with complete provenance
    manifest = dict(manifest_3002)
    manifest["status"] = "completed"
    manifest["label_source"] = "hand_designed_transit_guard_correction"
    manifest["provenance"]["dataset_composition"] = {
        "description": "Local transit capability dataset extracted from hand-designed transit guard and baseline transit",
        "sources": ep_data["source_runs"],
        "total_environment_steps": total_env_steps,
        "extracted_training_samples": len(extracted_steps),
        "extraction_condition": "grasped and cube_z > cube_initial_z + 0.06 and not patch_applied"
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Dataset successfully created at {out_dir}")


if __name__ == "__main__":
    prepare_transit_dataset()
