"""Run a short manual ten-primitive Fetch rollout with normal runner evidence."""
import argparse
from pathlib import Path
import shutil

from apc_maniskill.primitive_policy import BaseDemoSelector, PickPlaceSelector, PrimitivePolicy
from apc_maniskill.runner import RunConfig, collect, json_write, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1300)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--selector", choices=["manual", "pick_place", "mlp", "base_demo"], default="manual")
    parser.add_argument("--rotations", action="store_true")
    parser.add_argument("--base", action="store_true")
    parser.add_argument("--post-success-steps", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--query-teacher", action="store_true")
    args = parser.parse_args()
    if (args.selector == "mlp") != (args.checkpoint is not None):
        parser.error("mlp selector requires --checkpoint, other selectors do not")
    if args.selector == "mlp" and not args.rotations:
        parser.error("the first MLP checkpoint requires --rotations")
    if args.base and not args.rotations:
        parser.error("base candidates require --rotations to preserve IDs 0..15")
    if args.base and args.selector != "base_demo":
        parser.error("the first 20-ID experiment uses --selector base_demo")
    if args.selector == "base_demo" and not args.base:
        parser.error("base_demo requires --base")
    if args.query_teacher and args.selector != "mlp":
        parser.error("teacher queries are only recorded alongside mlp execution")
    config = RunConfig(env_id="APC-FetchPickCube-v1", robot_uids="fetch", policy="external",
                       episodes=args.episodes, seed=args.seed, max_steps=args.max_steps,
                       env_max_steps=args.max_steps,
                       post_success_steps=args.post_success_steps,
                       task_label=args.selector + ("_fetch20_primitives" if args.base else
                                                   "_fetch16_primitives" if args.rotations else
                                                   "_fetch10_primitives"))

    def factory(env, output):
        shutil.copy2(__file__, output / "run_fetch_primitives.py")
        if args.selector == "mlp":
            from apc_maniskill.primitive_learning import MLPSelector
            copied = output / "selector.pt"
            shutil.copy2(args.checkpoint, copied)
            selector = MLPSelector(copied, env.unwrapped.experiment_metadata(),
                                   float(env.unwrapped.sim_config.control_freq))
        else:
            selector = (PickPlaceSelector(use_rotation=args.rotations) if args.selector == "pick_place"
                        else BaseDemoSelector() if args.selector == "base_demo" else None)
        return PrimitivePolicy(env, output, allow_rotation=args.rotations, allow_base=args.base,
                               selector=selector, query_teacher=args.query_teacher)

    collect(config, args.out, policy_factory=factory)
    json_write(args.out / "summary.json", summarize(args.out))


if __name__ == "__main__":
    main()
