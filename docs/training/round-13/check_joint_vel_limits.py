#!/usr/bin/env python3
"""Print the joint velocity limits the sim actually holds for the leg joints.

``SPOT_WITH_ARM_CFG`` sets ``effort_limit`` but never ``velocity_limit``, so the
value comes from whatever the URDF importer wrote into the USD. Round 12 measured
a front-left knee peak of 52 rad/s, so this checks whether anything is supposed
to be clamping it.

Example::

    python docs/training/round-13/check_joint_vel_limits.py
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Dump joint velocity limits.")
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

import spot_arm_locomotion  # noqa: F401  — registers Gym tasks

TASK = "Isaac-Velocity-Rough-Spot-Arm-v0"


def main() -> None:
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(TASK, num_envs=args_cli.num_envs)
    env = gym.make(TASK, cfg=env_cfg).unwrapped
    env.reset()

    robot = env.scene["robot"]
    vel_sim = robot.data.joint_vel_limits.torch[0]
    eff_sim = robot.data.joint_effort_limits.torch[0]
    print(f"\n{'joint':>22} {'vel limit':>10} {'effort limit':>13}")
    print("-" * 48)
    for i, name in enumerate(robot.joint_names):
        if "hip" not in name and "knee" not in name:
            continue
        print(f"{name:>22} {float(vel_sim[i]):10.2f} {float(eff_sim[i]):13.2f}")
    print("\nRound-12 measured knee peaks: front_left 52.6, others 6.3-11.7 rad/s.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
