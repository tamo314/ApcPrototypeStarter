"""Run comparison matrix across configurations with rigorous metrics logging and run validation."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

PYTHON_BIN = "/home/tamot/.venvs/apc-maniskill-wsl-py312/bin/python"

BASE_CKPT = "runs/single-goal-conditioned-cart-20260925-a/selector.pt"
PLACE_CKPT = "runs/apc-cycle-true-place-20260925-c/primitive_bank/primitives/fetch_true_place_patch_v1_consolidated_selector.pt"
DEFAULT_TRANSIT_TEMP_CKPT = "runs/apc-transit-temp-train-20260925-b/selector.pt"
DEFAULT_TRANSIT_CAND_CKPT = "runs/apc-transit-cand-cart-20260925-b/selector.pt"


def get_file_sha256(path_str):
    p = Path(path_str)
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check_run_compatibility(out_dir, expected_args):
    """Verify that an existing run directory matches the requested configuration and completed successfully."""
    summary_path = out_dir / "summary.json"
    episodes_path = out_dir / "episodes.jsonl"
    steps_path = out_dir / "steps.jsonl"
    run_args_path = out_dir / "run_args.json"

    if not (summary_path.exists() and episodes_path.exists() and steps_path.exists()):
        return False

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "completed" or summary.get("steps", 0) <= 0:
            return False

        if run_args_path.exists():
            recorded = json.loads(run_args_path.read_text(encoding="utf-8"))
            if recorded.get("extra_args") != expected_args:
                return False
            # Check sha256 of transit checkpoint if specified
            transit_ckpt = recorded.get("transit_checkpoint")
            if transit_ckpt:
                current_sha = get_file_sha256(transit_ckpt)
                if current_sha and recorded.get("transit_checkpoint_sha256") != current_sha:
                    return False
        return True
    except Exception:
        return False


def run_experiment(config_name, extra_args, seed, out_prefix, force=False):
    out_dir = Path(f"runs/{out_prefix}-{config_name}-seed{seed}-20260925-a")

    if not force and check_run_compatibility(out_dir, extra_args):
        print(f"Reusing verified completed run: {out_dir}")
        return out_dir

    if force and out_dir.exists():
        print(f"Cleaning existing directory for force rerun: {out_dir}")
        shutil.rmtree(out_dir, ignore_errors=True)

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
        return None

    # Record run args and checkpoint sha256 for exact provenance checking
    transit_ckpt_val = None
    if "--transit-checkpoint" in extra_args:
        idx = extra_args.index("--transit-checkpoint")
        if idx + 1 < len(extra_args):
            transit_ckpt_val = extra_args[idx + 1]

    run_meta = {
        "config_name": config_name,
        "seed": seed,
        "extra_args": extra_args,
        "transit_checkpoint": transit_ckpt_val,
        "transit_checkpoint_sha256": get_file_sha256(transit_ckpt_val) if transit_ckpt_val else None,
    }
    (out_dir / "run_args.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    print(f"Completed {out_dir}")
    return out_dir


def analyze_run(out_dir, config_name, seed):
    if out_dir is None:
        return {
            "config": config_name,
            "seed": seed,
            "success_final": False,
            "hold_20": False,
            "error": "Subprocess execution failed (non-zero exit code)",
        }

    summary_path = out_dir / "summary.json"
    episodes_path = out_dir / "episodes.jsonl"
    steps_path = out_dir / "steps.jsonl"

    if not (summary_path.exists() and episodes_path.exists() and steps_path.exists()):
        return {
            "config": config_name,
            "seed": seed,
            "success_final": False,
            "hold_20": False,
            "error": "Run incomplete or missing output files",
        }

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    ep_data = json.loads(episodes_path.read_text(encoding="utf-8").strip())
    final_info = ep_data.get("final_info", {})

    success_final = bool(ep_data.get("success_final", False))
    hold_complete = bool(ep_data.get("hold_complete", False))
    consecutive_success_final_20 = bool(ep_data.get("consecutive_success_final_20", False))
    max_consecutive_success = int(ep_data.get("max_consecutive_success", 0))
    total_steps = int(summary.get("steps", 0))

    final_dist_raw = final_info.get("cube_goal_dist_xy_m")
    if isinstance(final_dist_raw, list) and len(final_dist_raw) > 0:
        final_dist_xy_m = float(final_dist_raw[0])
    elif isinstance(final_dist_raw, (int, float)):
        final_dist_xy_m = float(final_dist_raw)
    else:
        final_dist_xy_m = None

    module_counts = {"base": 0, "transit": 0, "place": 0, "other": 0}
    min_dist_xy_m = float("inf")

    with open(steps_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            sdata = json.loads(line)
            info = sdata.get("info", {})
            diag = info.get("diagnostic", {})

            mod = diag.get("active_module")
            if not mod:
                if diag.get("patch_applied", False):
                    mod = "place"
                elif diag.get("transit_applied", False) or diag.get("transit_guard_triggered", False):
                    mod = "transit"
                else:
                    mod = "base"

            if mod in module_counts:
                module_counts[mod] += 1
            else:
                module_counts["other"] += 1

            d_raw = info.get("cube_goal_dist_xy_m")
            if isinstance(d_raw, list) and len(d_raw) > 0:
                d = float(d_raw[0])
                if d < min_dist_xy_m:
                    min_dist_xy_m = d
            elif isinstance(d_raw, (int, float)):
                d = float(d_raw)
                if d < min_dist_xy_m:
                    min_dist_xy_m = d

    if min_dist_xy_m == float("inf"):
        min_dist_xy_m = None

    place_reached = module_counts["place"] > 0 or bool(final_info.get("diagnostic", {}).get("patch_applied", False))

    return {
        "config": config_name,
        "seed": seed,
        "success_final": success_final,
        "hold_complete": hold_complete,
        "consecutive_success_final_20": consecutive_success_final_20,
        "hold_20": consecutive_success_final_20,
        "max_consecutive_success": max_consecutive_success,
        "steps": total_steps,
        "min_dist_xy_m": min_dist_xy_m,
        "final_dist_xy_m": final_dist_xy_m,
        "module_steps": module_counts,
        "place_reached": place_reached,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[3005, 3006, 3007])
    parser.add_argument("--out-prefix", type=str, default="apc-matrix")
    parser.add_argument("--transit-temp-ckpt", type=str, default=DEFAULT_TRANSIT_TEMP_CKPT)
    parser.add_argument("--transit-cand-ckpt", type=str, default=DEFAULT_TRANSIT_CAND_CKPT)
    parser.add_argument("--out-json", type=str, default="runs/apc-matrix-comparison-summary.json")
    parser.add_argument("--comparison-mode", choices=["3way", "scope_ablation"], default="3way",
                        help="3way (guard vs temp vs cand) or scope_ablation (continuous vs guarded intervention)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.comparison_mode == "scope_ablation":
        configs = [
            ("hand_guard", ["--transit-guard"]),
            ("mlp_continuous", ["--transit-checkpoint", args.transit_temp_ckpt, "--transit-mode", "patch"]),
            ("mlp_guarded", ["--transit-checkpoint", args.transit_temp_ckpt, "--transit-mode", "guard"]),
            ("cart_continuous", ["--transit-checkpoint", args.transit_cand_ckpt, "--transit-mode", "patch"]),
            ("cart_guarded", ["--transit-checkpoint", args.transit_cand_ckpt, "--transit-mode", "guard"]),
        ]
    else:
        configs = [
            ("config1_guard", ["--transit-guard"]),
            ("config2_temp_mlp", ["--transit-checkpoint", args.transit_temp_ckpt]),
            ("config3_cand_cart", ["--transit-checkpoint", args.transit_cand_ckpt]),
        ]

    results = []
    for seed in args.seeds:
        for config_name, extra_args in configs:
            out_dir = run_experiment(config_name, extra_args, seed, args.out_prefix, force=args.force)
            res = analyze_run(out_dir, config_name, seed)
            results.append(res)

    print("\n\n=================== COMPARISON MATRIX RESULTS ===================")
    header = f"{'Config':<18} | {'Seed':<5} | {'SuccFin':<8} | {'Hold20':<7} | {'Steps':<6} | {'MinDist(m)':<11} | {'FinDist(m)':<11} | {'Base/Trans/Place':<18}"
    print(header)
    print("-" * len(header))
    for r in results:
        if "error" in r and r.get("error"):
            print(f"{r['config']:<18} | {r['seed']:<5} | ERROR: {r['error']}")
            continue
        succ = "TRUE" if r["success_final"] else "FALSE"
        hold = "TRUE" if r["consecutive_success_final_20"] else "FALSE"
        steps = str(r["steps"])
        min_d = f"{r['min_dist_xy_m']:.4f}" if r["min_dist_xy_m"] is not None else "-"
        fin_d = f"{r['final_dist_xy_m']:.4f}" if r["final_dist_xy_m"] is not None else "-"
        mods = f"{r['module_steps']['base']}/{r['module_steps']['transit']}/{r['module_steps']['place']}"
        print(f"{r['config']:<18} | {r['seed']:<5} | {succ:<8} | {hold:<7} | {steps:<6} | {min_d:<11} | {fin_d:<11} | {mods:<18}")

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved structured results to {out_path}")


if __name__ == "__main__":
    main()
