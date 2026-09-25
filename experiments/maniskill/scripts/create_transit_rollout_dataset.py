"""Extract rollout steps executed by Transit Temporary MLP for CART distillation.

Preserves policy details from the Temporary rollout while explicitly recording provenance.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def extract_transit_rollouts(source_run_paths, out_dir_path):
    out_dir = Path(out_dir_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    extracted = []
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

                # Transit phase executed by Temporary MLP
                if grasped and cz > cube_init_z + 0.06 and not patch_applied:
                    chosen_act = diag.get("transit_id", diag.get("proposed_id"))
                    if chosen_act is not None:
                        # Ensure the distillation label reflects the Temporary MLP decision
                        data["info"]["diagnostic"]["proposed_id"] = chosen_act
                        run_action_counts[chosen_act] += 1
                        overall_action_counts[chosen_act] += 1
                    data["episode"] = 0
                    extracted.append(data)
                    count_run += 1

        print(f"Source {rpath.name} (seed {seed}): {count_run} extracted steps")
        print(f"  Action distribution: {dict(sorted(run_action_counts.items()))}")
        source_stats.append({
            "run": str(rpath),
            "seed": seed,
            "env_steps": env_steps,
            "extracted_samples": count_run,
            "sha256": digest,
        })

    print(f"\nTotal extracted Temporary MLP rollout steps: {len(extracted)}")
    print(f"Overall Action distribution: {dict(sorted(overall_action_counts.items()))}")

    with open(out_dir / "steps.jsonl", "w", encoding="utf-8") as f:
        for d in extracted:
            f.write(json.dumps(d) + "\n")

    synthesized_ep = {
        "episode": 0,
        "seed": 0,
        "steps": total_env_steps,
        "extracted_distillation_samples": len(extracted),
        "source_runs": source_stats,
    }
    with open(out_dir / "episodes.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(synthesized_ep) + "\n")

    manifest = dict(first_manifest or {})
    manifest["status"] = "completed"
    manifest["label_source"] = "transit_temporary_mlp_rollout"
    if "provenance" not in manifest or not isinstance(manifest["provenance"], dict):
        manifest["provenance"] = {}
    manifest["provenance"]["distillation_metadata"] = {
        "source_runs": source_stats,
        "total_environment_steps": total_env_steps,
        "extracted_samples": len(extracted),
        "extraction_condition": "grasped and cube_z > cube_initial_z + 0.06 and not patch_applied",
        "description": "Multi-seed rollout decisions produced by the learned transit temporary MLP before place patch handoff",
        "action_distribution": dict(sorted(overall_action_counts.items())),
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Distillation dataset created at {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", default=[
        "runs/apc-transit-temp-eval-seed3002-20260925-a",
        "runs/apc-transit-temp-eval-seed3007-20260925-b",
    ])
    parser.add_argument("--out", type=str, default="runs/dataset-transit-temp-rollout-20260925-c")
    args = parser.parse_args()

    extract_transit_rollouts(args.sources, args.out)


if __name__ == "__main__":
    main()
