"""Run a short hand-designed Fetch arm diagnostic, saving normal runner evidence."""
import argparse
from pathlib import Path
import shutil

from apc_maniskill.arm_ik import ArmIKPolicy
from apc_maniskill.runner import RunConfig, collect, json_write, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--offset", type=float, nargs=3, default=[0, 0, 0.02])
    parser.add_argument("--protocol", choices=["track", "pick"], default="track")
    parser.add_argument("--pitch-deg", type=float, default=90)
    parser.add_argument("--torso-ik", action="store_true")
    args = parser.parse_args()
    config = RunConfig(env_id="PickCube-v1", robot_uids="fetch", episodes=args.episodes,
                       max_steps=args.max_steps, seed=args.seed, policy="external",
                       env_max_steps=args.max_steps,
                       task_label="scripted_fetch_arm_" + args.protocol)

    def factory(env, output):
        shutil.copy2(__file__, output / "run_fetch_arm.py")
        return ArmIKPolicy(env, output, offset=args.offset, protocol=args.protocol,
                           pitch_deg=args.pitch_deg, torso_ik=args.torso_ik)

    collect(config, args.out, policy_factory=factory)
    json_write(args.out / "summary.json", summarize(args.out))


if __name__ == "__main__":
    main()
