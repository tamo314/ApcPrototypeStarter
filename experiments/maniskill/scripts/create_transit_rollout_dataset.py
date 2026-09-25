"""Extract rollout steps executed by Transit Temporary MLP for CART distillation."""
import json
from pathlib import Path

def extract_transit_rollout():
    source_run = Path("runs/apc-transit-temp-eval-seed3002-20260925-a")
    out_dir = Path("runs/dataset-transit-temp-rollout-20260925-a")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((source_run / "manifest.json").read_text(encoding="utf-8"))
    
    extracted = []
    with open(source_run / "steps.jsonl", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            diag = data.get("info", {}).get("diagnostic", {})
            pre = diag.get("pre_action_state", {})
            grasped = pre.get("grasped", False)
            cz = pre.get("cube_position", [0, 0, 0])[2]
            cube_init_z = pre.get("cube_initial_z", 0.02)
            patch_applied = diag.get("patch_applied", False)
            
            # Transit phase executed by Temporary MLP
            if grasped and cz > cube_init_z + 0.06 and not patch_applied:
                extracted.append(data)

    print(f"Extracted {len(extracted)} steps of Temporary MLP execution")
    with open(out_dir / "steps.jsonl", "w", encoding="utf-8") as f:
        for d in extracted:
            f.write(json.dumps(d) + "\n")

    manifest["status"] = "completed"
    manifest["policy_details"]["learned"] = False
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Copy episodes.jsonl
    with open(out_dir / "episodes.jsonl", "w", encoding="utf-8") as f:
        f.write((source_run / "episodes.jsonl").read_text(encoding="utf-8"))

    print(f"Distillation dataset created at {out_dir}")

if __name__ == "__main__":
    extract_transit_rollout()
