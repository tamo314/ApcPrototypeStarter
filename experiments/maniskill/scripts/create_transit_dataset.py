"""Prepare transit teacher dataset for learning transit local capability."""
import json
import shutil
from pathlib import Path

def prepare_transit_dataset():
    source_run = Path("runs/apc-transit-teacher-seed3002-20260925-a")
    normal_run = Path("runs/test-transit-guard-seed3001-20260925-a")
    out_dir = Path("runs/dataset-transit-teacher-20260925-a")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read base manifest
    manifest = json.loads((source_run / "manifest.json").read_text(encoding="utf-8"))
    
    extracted_steps = []
    
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
                # If guard triggered, ensure proposed_id is the guarded action (transit_id)
                guard = diag.get("transit_guard_triggered", False)
                if guard:
                    data["info"]["diagnostic"]["proposed_id"] = diag.get("transit_id", diag.get("proposed_id"))
                extracted_steps.append(data)
                
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
                extracted_steps.append(data)

    print(f"Total extracted transit training steps: {len(extracted_steps)}")
    guard_count = sum(1 for d in extracted_steps if d["info"]["diagnostic"].get("transit_guard_triggered"))
    print(f"Steps where transit guard triggered: {guard_count}")

    # Write steps.jsonl
    with open(out_dir / "steps.jsonl", "w", encoding="utf-8") as f:
        for step in extracted_steps:
            f.write(json.dumps(step) + "\n")
            
    # Update and write manifest
    manifest["status"] = "completed"
    manifest["policy_details"]["learned"] = False  # treat as teacher dataset
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Dataset successfully created at {out_dir}")

if __name__ == "__main__":
    prepare_transit_dataset()
