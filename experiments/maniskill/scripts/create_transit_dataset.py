"""Prepare transit teacher dataset for learning transit local capability.

Records provenance distinguishing hand-designed teacher corrections from autonomous discovery.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def prepare_transit_dataset(source_run_paths, out_dir_path):
    out_dir = Path(out_dir_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    extracted_steps = []
    source_stats = []
    total_env_steps = 0
    first_manifest = None
    overall_action_counts = Counter()

    for rpath_str in source_run_paths:
        rpath = Path(rpath_str)
        if not rpath.exists():
            print(f"Warning: source path {rpath} does not exist, skipping.")
            continue

        manifest_p = rpath / "manifest.json"
        ep_p = rpath / "episodes.jsonl"
        steps_p = rpath / "steps.jsonl"

        if not (manifest_p.exists() and ep_p.exists() and steps_p.exists()):
            print(f"Warning: incomplete run at {rpath}, skipping.")
            continue

        manifest_data = json.loads(manifest_p.read_text(encoding="utf-8"))
        if first_manifest is None:
            first_manifest = manifest_data

        ep_data_raw = json.loads(ep_p.read_text(encoding="utf-8").strip())
        env_steps = ep_data_raw.get("steps", 0)
        total_env_steps += env_steps
        seed = ep_data_raw.get("seed", 0)

        steps_bytes = steps_p.read_bytes()
        digest = hashlib.sha256(steps_bytes).hexdigest()

        count_run = 0
        guard_count_run = 0
        run_action_counts = Counter()

        with open(steps_p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
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
                        guard_count_run += 1

                    chosen_act = data["info"]["diagnostic"].get("proposed_id")
                    if chosen_act is not None:
                        run_action_counts[chosen_act] += 1
                        overall_action_counts[chosen_act] += 1

                    data["episode"] = 0
                    extracted_steps.append(data)
                    count_run += 1

        print(f"Source {rpath.name} (seed {seed}): {count_run} extracted steps (guard triggered: {guard_count_run})")
        print(f"  Action distribution: {dict(sorted(run_action_counts.items()))}")
        source_stats.append({
            "run": str(rpath),
            "seed": seed,
            "env_steps": env_steps,
            "extracted_samples": count_run,
            "guard_triggers": guard_count_run,
            "sha256": digest,
        })

    print(f"\nTotal extracted transit training steps: {len(extracted_steps)}")
    print(f"Overall Action distribution: {dict(sorted(overall_action_counts.items()))}")

    # Write steps.jsonl
    with open(out_dir / "steps.jsonl", "w", encoding="utf-8") as f:
        for step in extracted_steps:
            f.write(json.dumps(step) + "\n")

    # Generate synthesized episodes.jsonl
    synthesized_ep = {
        "episode": 0,
        "seed": 0,
        "steps": total_env_steps,
        "extracted_training_samples": len(extracted_steps),
        "source_runs": source_stats,
    }
    with open(out_dir / "episodes.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(synthesized_ep) + "\n")

    # Construct manifest with complete provenance
    manifest = dict(first_manifest or {})
    manifest["status"] = "completed"
    manifest["label_source"] = "hand_designed_transit_guard_correction"
    if "provenance" not in manifest or not isinstance(manifest["provenance"], dict):
        manifest["provenance"] = {}
    manifest["provenance"]["dataset_composition"] = {
        "description": "Multi-seed transit capability dataset with X/Y transit corrections and baseline transit",
        "sources": source_stats,
        "total_environment_steps": total_env_steps,
        "extracted_training_samples": len(extracted_steps),
        "extraction_condition": "grasped and cube_z > cube_initial_z + 0.06 and not patch_applied",
        "action_distribution": dict(sorted(overall_action_counts.items())),
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Dataset successfully created at {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", default=[
        "runs/apc-transit-teacher-seed3002-20260925-a",
        "runs/test-transit-guard-seed3001-20260925-a",
        "runs/apc-matrix-config1_guard-seed3007-20260925-a",
    ])
    parser.add_argument("--out", type=str, default="runs/dataset-transit-teacher-20260925-c")
    args = parser.parse_args()

    prepare_transit_dataset(args.sources, args.out)


if __name__ == "__main__":
    main()
