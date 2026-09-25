"""Run 3x3 comparison matrix on unseen seeds (3005, 3006, 3007) across 3 configurations."""
import json
import subprocess
import sys
from pathlib import Path

PYTHON_BIN = "/home/tamot/.venvs/apc-maniskill-wsl-py312/bin/python"

BASE_CKPT = "runs/single-goal-conditioned-cart-20260925-a/selector.pt"
PLACE_CKPT = "runs/apc-cycle-true-place-20260925-c/primitive_bank/primitives/fetch_true_place_patch_v1_consolidated_selector.pt"
TRANSIT_TEMP_CKPT = "runs/apc-transit-temp-train-20260925-a/selector.pt"
TRANSIT_CAND_CKPT = "runs/apc-transit-cand-cart-20260925-a/selector.pt"

SEEDS = [3005, 3006, 3007]
CONFIGS = [
    ("config1_guard", ["--transit-guard"]),
    ("config2_temp_mlp", ["--transit-checkpoint", TRANSIT_TEMP_CKPT]),
    ("config3_cand_cart", ["--transit-checkpoint", TRANSIT_CAND_CKPT]),
]

def run_experiment(config_name, extra_args, seed):
    out_dir = Path(f"runs/apc-matrix-{config_name}-seed{seed}-20260925-a")
    if out_dir.exists() and (out_dir / "summary.json").exists():
        print(f"Skipping existing run: {out_dir}")
        return out_dir

    cmd = [
        PYTHON_BIN, "scripts/run_fetch_primitives.py",
        "--episodes", "1",
        "--max-steps", "1200",
        "--seed", str(seed),
        "--rotations",
        "--base",
        "--far-start",
        "--true-place-goal",
        "--post-success-steps", "20",
        "--selector", "composite_patch",
        "--patch-mode", "place",
        "--patch-threshold-xy", "0.016",
        "--checkpoint", BASE_CKPT,
        "--patch-checkpoint", PLACE_CKPT,
        "--allow-task-mismatch",
        "--out", str(out_dir),
    ] + extra_args

    print(f"\n=== Running {config_name} on seed {seed} -> {out_dir} ===")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error running {out_dir}:\n{res.stderr}")
    else:
        print(f"Completed {out_dir}")
    return out_dir

def main():
    results = []
    for seed in SEEDS:
        for config_name, extra_args in CONFIGS:
            out_dir = run_experiment(config_name, extra_args, seed)
            
            # Read summary and final info
            summary_path = out_dir / "summary.json"
            episodes_path = out_dir / "episodes.jsonl"
            
            if summary_path.exists() and episodes_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                ep_data = json.loads(episodes_path.read_text(encoding="utf-8").strip())
                final_info = ep_data.get("final_info", {})
                diag = final_info.get("diagnostic", {})
                
                # Check active modules in steps
                active_modules = set()
                with open(out_dir / "steps.jsonl", encoding="utf-8") as f:
                    for line in f:
                        step_data = json.loads(line)
                        step_diag = step_data.get("info", {}).get("diagnostic", {})
                        mod = step_diag.get("active_module")
                        if mod:
                            active_modules.add(mod)
                
                results.append({
                    "config": config_name,
                    "seed": seed,
                    "success": summary.get("success_rate_over_observed", 0) == 1.0,
                    "hold_complete": summary.get("hold_complete_episodes", 0) == 1,
                    "steps": summary.get("steps", 0),
                    "dist_xy_m": final_info.get("cube_goal_dist_xy_m", None),
                    "active_modules": sorted(active_modules),
                    "place_reached": "place" in active_modules or final_info.get("diagnostic", {}).get("patch_applied", False),
                })
            else:
                results.append({
                    "config": config_name,
                    "seed": seed,
                    "success": False,
                    "error": "Run failed or missing summary",
                })

    print("\n\n=================== COMPARISON MATRIX RESULTS ===================")
    print(f"{'Config':<20} | {'Seed':<6} | {'Success':<8} | {'Hold20':<8} | {'Steps':<6} | {'PlaceReached':<12} | {'Modules':<25}")
    print("-" * 95)
    for r in results:
        succ = "TRUE" if r.get("success") else "FALSE"
        hold = "TRUE" if r.get("hold_complete") else "FALSE"
        steps = str(r.get("steps", "-"))
        reached = "YES" if r.get("place_reached") else "NO"
        mods = ",".join(r.get("active_modules", []))
        print(f"{r['config']:<20} | {r['seed']:<6} | {succ:<8} | {hold:<8} | {steps:<6} | {reached:<12} | {mods:<25}")

    out_json = Path("runs/apc-matrix-unseen-comparison-20260925.json")
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved structured results to {out_json}")

if __name__ == "__main__":
    main()
