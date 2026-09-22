"""CLI: diagnose the host, collect real trajectories, and inspect results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4

from .runner import WORKSPACE, RunConfig, collect, command, provenance, summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Print host/package diagnostics; does not validate simulation")
    rollout = commands.add_parser("rollout", help="Run one real environment with a diagnostic policy")
    rollout.add_argument("--config", type=Path, required=True)
    rollout.add_argument("--out", type=Path, help="New directory only; defaults to workspace/runs/<uuid>")
    rollout.add_argument("--episodes", type=int)
    rollout.add_argument("--max-steps", type=int)
    rollout.add_argument("--seed", type=int)
    rollout.add_argument("--policy", choices=["random", "zero", "fetch_goal", "fetch_random", "fetch_zero",
                                              "fetch_bc", "fetch_pick_bc"])
    rollout.add_argument("--checkpoint", type=str)
    rollout.add_argument("--post-success-steps", type=int,
                         help="Fetch-only diagnostic: continue N steps after first success")
    rollout.add_argument("--next-goal-offset", type=float, nargs=2, metavar=("X", "Y"),
                         help="Manually chain a second Fetch goal at this world offset (metres)")
    rollout.add_argument("--sim-backend", choices=["physx_cpu", "physx_cuda"])
    rollout.add_argument("--video", action="store_true", default=None)
    summary = commands.add_parser("summarize", help="Print a summary, including failed/partial status")
    summary.add_argument("run_dir", type=Path)
    bc = commands.add_parser("train-bc", help="Clone scripted Fetch base actions with a small CPU MLP")
    bc.add_argument("--demo-run", type=Path, required=True)
    bc.add_argument("--out", type=Path, required=True)
    bc.add_argument("--updates", type=int, default=1000)
    bc.add_argument("--seed", type=int, default=0)
    bc.add_argument("--stop-weight", type=float, default=1.0,
                    help="Sampling weight for zero base-command demonstrations (>=1)")
    operation_bc = commands.add_parser(
        "train-operation-bc", help="Clone scripted Fetch pick-place actions with a small CPU MLP")
    operation_bc.add_argument("--demo-run", type=Path, required=True)
    operation_bc.add_argument("--extra-demo-run", type=Path, action="append", default=[])
    operation_bc.add_argument("--out", type=Path, required=True)
    operation_bc.add_argument("--updates", type=int, default=1000)
    operation_bc.add_argument("--seed", type=int, default=0)
    operation_bc.add_argument("--successful-only", action="store_true",
                              help="Train only on teacher episodes that reached success")
    dagger = commands.add_parser(
        "collect-operation-dagger", help="Run an operation policy and save separate scripted labels")
    dagger.add_argument("--checkpoint", type=Path, required=True)
    dagger.add_argument("--out", type=Path, required=True)
    dagger.add_argument("--episodes", type=int, default=3)
    dagger.add_argument("--max-steps", type=int, default=350)
    dagger.add_argument("--seed", type=int, default=0)
    distill = commands.add_parser("distill", help="Fit a separate compact policy to a frozen neural teacher")
    distill.add_argument("--teacher-checkpoint", type=Path, required=True)
    distill.add_argument("--state-run", type=Path, action="append", required=True)
    distill.add_argument("--out", type=Path, required=True)
    distill.add_argument("--hidden-width", type=int, default=8)
    distill.add_argument("--updates", type=int, default=1000)
    distill.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.command == "doctor":
        report = provenance()
        report["vulkan"] = command(["vulkaninfo", "--summary"])
        try:
            import torch
            report["torch_cuda_available"] = torch.cuda.is_available()
            report["torch_cuda_version"] = torch.version.cuda
            if torch.cuda.is_available():
                x = torch.ones(4, device="cuda")
                report["torch_cuda_arithmetic"] = float((x * x).sum().item())
        except Exception as exc:
            report["torch_error"] = repr(exc)
        report["note"] = "Diagnostics only. CUDA availability does not prove PhysX/Vulkan compatibility."
        print(json.dumps(report, indent=2))
    elif args.command == "rollout":
        values = json.loads(args.config.read_text(encoding="utf-8"))
        for key in ("episodes", "max_steps", "seed", "policy", "checkpoint", "sim_backend", "video",
                    "post_success_steps", "next_goal_offset"):
            value = getattr(args, key)
            if value is not None:
                values[key] = value
        config = RunConfig(**values)
        output = args.out or WORKSPACE / "runs" / uuid4().hex
        print(f"Run directory: {output.resolve()}", flush=True)
        collect(config, output)
        print(json.dumps(summarize(output), indent=2))
    elif args.command == "distill":
        from .bc import train
        output = train(args.state_run[0], args.out, extra_runs=args.state_run[1:],
                       teacher_checkpoint=args.teacher_checkpoint, hidden_width=args.hidden_width,
                       updates=args.updates, seed=args.seed)
        print((output / "training.json").read_text(encoding="utf-8"))
    elif args.command == "train-bc":
        from .bc import train
        output = train(args.demo_run, args.out, updates=args.updates, seed=args.seed,
                       stop_weight=args.stop_weight)
        print((output / "training.json").read_text(encoding="utf-8"))
    elif args.command == "train-operation-bc":
        from .operation_bc import train
        output = train(args.demo_run, args.out, updates=args.updates, seed=args.seed,
                       successful_only=args.successful_only, extra_runs=args.extra_demo_run)
        print((output / "training.json").read_text(encoding="utf-8"))
    elif args.command == "collect-operation-dagger":
        from .operation_dagger import OperationDaggerPolicy
        from .runner import json_write
        config = RunConfig(env_id="APC-FetchPickCube-v1", robot_uids="fetch", policy="external",
                           episodes=args.episodes, max_steps=args.max_steps, seed=args.seed,
                           env_max_steps=args.max_steps, task_label="operation_dagger_relabel")
        collect(config, args.out, policy_factory=lambda env, output: OperationDaggerPolicy(
            env, output, args.checkpoint))
        result = summarize(args.out)
        json_write(args.out / "summary.json", result)
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(summarize(args.run_dir), indent=2))


if __name__ == "__main__":
    main()
