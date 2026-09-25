"""Extract rollout steps executed by Transit Temporary MLP for CART distillation.

Preserves policy details from the Temporary rollout while explicitly recording provenance.
"""
import hashlib
import json
from pathlib import Path


def extract_transit_rollout():
    source_run = Path("runs/apc-transit-temp-eval-seed3002-20260925-a")
    out_dir = Path("runs/dataset-transit-temp-rollout-20260925-b")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((source_run / "manifest.json").read_text(encoding="utf-8"))
    steps_bytes = (source_run / "steps.jsonl").read_bytes()
    steps_sha256 = hashlib.sha256(steps_bytes).hexdigest()
    
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

    episodes_raw = (source_run / "episodes.jsonl").read_text(encoding="utf-8").strip()
    ep_data = json.loads(episodes_raw)
    ep_data["extracted_distillation_samples"] = len(extracted)
    ep_data["source_run"] = str(source_run)
    ep_data["source_steps_sha256"] = steps_sha256
    with open(out_dir / "episodes.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(ep_data) + "\n")

    manifest["status"] = "completed"
    manifest["label_source"] = "transit_temporary_mlp_rollout"
    manifest["provenance"]["distillation_metadata"] = {
        "source_run": str(source_run),
        "source_steps_sha256": steps_sha256,
        "extracted_samples": len(extracted),
        "extraction_condition": "grasped and cube_z > cube_initial_z + 0.06 and not patch_applied",
        "description": "Rollout decisions produced by the learned transit temporary MLP before place patch handoff"
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Distillation dataset created at {out_dir}")


if __name__ == "__main__":
    extract_transit_rollout()
