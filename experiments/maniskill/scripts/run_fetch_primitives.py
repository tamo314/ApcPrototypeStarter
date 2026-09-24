"""Run Fetch primitive selectors with normal runner evidence."""
import argparse
from pathlib import Path
import shutil

from apc_maniskill.primitive_policy import (BaseApproachPickSelector, BaseReadyPickSelector,
                                           BaseReadySettledPickSelector, BaseReadyRecoverPickSelector, BaseReadyAxisRetryPickSelector,
                                           BaseDemoSelector, BaseRecoverPickSelector,
                                           BaseThenPickSelector,
                                           PickPlaceSelector, PitchGuardSelector, PrimitivePolicy, RotationProbeSelector,
                                           WaitThenPickSelector)
from apc_maniskill.runner import RunConfig, collect, json_write, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1300)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--selector", choices=["manual", "pick_place", "mlp", "tree", "tree_probe", "mlp_pitch_guard",
                                               "base_demo", "base_then_pick", "base_recover_pick", "base_approach_pick",
                                               "base_ready_pick", "base_ready_recover_pick", "base_ready_axis_retry_pick", "base_ready_settled_pick",
                                               "wait_then_pick"], default="manual")
    parser.add_argument("--rotations", action="store_true")
    parser.add_argument("--base", action="store_true")
    parser.add_argument("--far-start", action="store_true")
    parser.add_argument("--post-success-steps", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--query-teacher", action="store_true")
    parser.add_argument("--pitch-deg", type=float, default=15)
    parser.add_argument("--base-switch-x-m", type=float, default=0.195)
    parser.add_argument("--ik-reset-seed", action="store_true")
    parser.add_argument("--translation-backoff", action="store_true")
    parser.add_argument("--descend-pitch-deg", type=float)
    parser.add_argument("--grasp-height-m", type=float, default=.012)
    args = parser.parse_args()
    if args.pitch_deg != 15 and args.selector != "base_ready_pick":
        parser.error("pitch diagnostic is supported only by base_ready_pick")
    if args.base_switch_x_m != 0.195 and args.selector != "base_ready_pick":
        parser.error("base switch diagnostic is supported only by base_ready_pick")
    if args.descend_pitch_deg is not None and args.selector != "base_ready_pick":
        parser.error("pitch schedule requires base_ready_pick")
    if args.grasp_height_m != .012 and args.selector != "base_ready_pick":
        parser.error("grasp height diagnostic requires base_ready_pick")
    if (args.selector in ("mlp", "tree", "tree_probe", "mlp_pitch_guard")) != (args.checkpoint is not None):
        parser.error("learned selectors require --checkpoint, other selectors do not")
    if args.selector in ("mlp", "tree", "tree_probe", "mlp_pitch_guard") and not args.rotations:
        parser.error("learned checkpoints require --rotations")
    if args.selector == "wait_then_pick" and not args.rotations:
        parser.error("wait_then_pick requires --rotations")
    if args.base and not args.rotations:
        parser.error("base candidates require --rotations to preserve IDs 0..15")
    if args.base and args.selector not in ("base_demo", "base_then_pick", "base_recover_pick", "base_approach_pick", "base_ready_pick", "base_ready_recover_pick", "base_ready_axis_retry_pick", "base_ready_settled_pick", "mlp", "tree", "tree_probe", "mlp_pitch_guard"):
        parser.error("the 20-ID manual selectors require a base selector")
    if args.selector in ("base_demo", "base_then_pick", "base_recover_pick", "base_approach_pick", "base_ready_pick", "base_ready_recover_pick", "base_ready_axis_retry_pick", "base_ready_settled_pick") and not args.base:
        parser.error("20-ID manual selectors require --base")
    if args.base and args.selector == "mlp_pitch_guard" and not args.far_start:
        parser.error("the 20-ID pitch diagnostic requires --far-start")
    if args.selector in ("base_approach_pick", "base_ready_pick", "base_ready_recover_pick", "base_ready_axis_retry_pick", "base_ready_settled_pick") and not args.far_start:
        parser.error("base approach selectors require --far-start")
    if args.query_teacher and args.selector not in ("mlp", "tree", "tree_probe", "mlp_pitch_guard"):
        parser.error("teacher queries are only recorded alongside learned execution")
    if args.base and args.query_teacher and not args.far_start:
        parser.error("the 20-ID teacher requires --far-start")
    config = RunConfig(env_id="APC-FetchPickCubeFar-v1" if args.far_start else "APC-FetchPickCube-v1",
                       robot_uids="fetch", policy="external",
                       episodes=args.episodes, seed=args.seed, max_steps=args.max_steps,
                       env_max_steps=args.max_steps,
                       post_success_steps=args.post_success_steps,
                       task_label=args.selector + ("_fetch20_primitives" if args.base else
                                                   "_fetch16_primitives" if args.rotations else
                                                   "_fetch10_primitives"))

    def factory(env, output):
        shutil.copy2(__file__, output / "run_fetch_primitives.py")
        if args.selector in ("mlp", "tree", "tree_probe", "mlp_pitch_guard"):
            from apc_maniskill.primitive_learning import LearnedSelector
            copied = output / "selector.pt"
            shutil.copy2(args.checkpoint, copied)
            selector = LearnedSelector(copied, env.unwrapped.experiment_metadata(),
                                   float(env.unwrapped.sim_config.control_freq))
            if (args.selector in ("tree", "tree_probe")) != (selector.model_kind == "cart"):
                raise ValueError("Runner selector kind does not match checkpoint model")
            if selector.primitive_count != (20 if args.base else 16):
                raise ValueError("Checkpoint primitive count does not match runner bank")
            if args.selector == "tree_probe":
                selector = RotationProbeSelector(selector)
            if args.selector == "mlp_pitch_guard":
                selector = PitchGuardSelector(selector)
        else:
            selector = (PickPlaceSelector(use_rotation=args.rotations) if args.selector == "pick_place"
                        else BaseDemoSelector() if args.selector == "base_demo"
                        else BaseThenPickSelector() if args.selector == "base_then_pick"
                        else BaseRecoverPickSelector() if args.selector == "base_recover_pick"
                        else BaseApproachPickSelector() if args.selector == "base_approach_pick"
                        else BaseReadyPickSelector(desired_pitch_deg=args.pitch_deg,
                                                   descend_pitch_deg=args.descend_pitch_deg,
                                                   grasp_height_m=args.grasp_height_m,
                                                   base_switch_x_m=args.base_switch_x_m) if args.selector == "base_ready_pick"
                        else BaseReadyAxisRetryPickSelector() if args.selector == "base_ready_axis_retry_pick"
                        else BaseReadyRecoverPickSelector() if args.selector == "base_ready_recover_pick"
                        else BaseReadySettledPickSelector() if args.selector == "base_ready_settled_pick"
                        else WaitThenPickSelector() if args.selector == "wait_then_pick" else None)
        return PrimitivePolicy(env, output, allow_rotation=args.rotations, allow_base=args.base,
                               selector=selector, query_teacher=args.query_teacher,
                               ik_reset_seed=args.ik_reset_seed, translation_backoff=args.translation_backoff)

    collect(config, args.out, policy_factory=factory)
    json_write(args.out / "summary.json", summarize(args.out))


if __name__ == "__main__":
    main()
