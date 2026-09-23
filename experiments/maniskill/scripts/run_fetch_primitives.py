"""Run Fetch primitive selectors with normal runner evidence."""
import argparse
from pathlib import Path
import shutil

from apc_maniskill.primitive_policy import (BaseDemoSelector, BaseRecoverPickSelector,
                                           BaseThenPickSelector,
                                           PickPlaceSelector, PitchGuardSelector, PrimitivePolicy,
                                           WaitThenPickSelector)
from apc_maniskill.runner import RunConfig, collect, json_write, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1300)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--selector", choices=["manual", "pick_place", "mlp", "mlp_pitch_guard",
                                               "base_demo", "base_then_pick", "base_recover_pick",
                                               "wait_then_pick"], default="manual")
    parser.add_argument("--rotations", action="store_true")
    parser.add_argument("--base", action="store_true")
    parser.add_argument("--post-success-steps", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--query-teacher", action="store_true")
    args = parser.parse_args()
    if (args.selector in ("mlp", "mlp_pitch_guard")) != (args.checkpoint is not None):
        parser.error("mlp selectors require --checkpoint, other selectors do not")
    if args.selector in ("mlp", "mlp_pitch_guard") and not args.rotations:
        parser.error("the 16-ID MLP checkpoint requires --rotations")
    if args.selector == "wait_then_pick" and not args.rotations:
        parser.error("wait_then_pick requires --rotations")
    if args.base and not args.rotations:
        parser.error("base candidates require --rotations to preserve IDs 0..15")
    if args.base and args.selector not in ("base_demo", "base_then_pick", "base_recover_pick"):
        parser.error("the 20-ID manual selectors require --selector base_demo, base_then_pick, or base_recover_pick")
    if args.selector in ("base_demo", "base_then_pick", "base_recover_pick") and not args.base:
        parser.error("20-ID manual selectors require --base")
    if args.query_teacher and args.selector not in ("mlp", "mlp_pitch_guard"):
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
        if args.selector in ("mlp", "mlp_pitch_guard"):
            from apc_maniskill.primitive_learning import MLPSelector
            copied = output / "selector.pt"
            shutil.copy2(args.checkpoint, copied)
            selector = MLPSelector(copied, env.unwrapped.experiment_metadata(),
                                   float(env.unwrapped.sim_config.control_freq))
            if args.selector == "mlp_pitch_guard":
                selector = PitchGuardSelector(selector)
        else:
            selector = (PickPlaceSelector(use_rotation=args.rotations) if args.selector == "pick_place"
                        else BaseDemoSelector() if args.selector == "base_demo"
                        else BaseThenPickSelector() if args.selector == "base_then_pick"
                        else BaseRecoverPickSelector() if args.selector == "base_recover_pick"
                        else WaitThenPickSelector() if args.selector == "wait_then_pick" else None)
        return PrimitivePolicy(env, output, allow_rotation=args.rotations, allow_base=args.base,
                               selector=selector, query_teacher=args.query_teacher)

    collect(config, args.out, policy_factory=factory)
    json_write(args.out / "summary.json", summarize(args.out))


if __name__ == "__main__":
    main()
