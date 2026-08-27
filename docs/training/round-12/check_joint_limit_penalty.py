#!/usr/bin/env python3
"""Split the ``dof_pos_limits`` penalty into leg (actionable) and arm (constant).

``dof_pos_limits`` has no ``asset_cfg``, so it covers all 19 joints. The arm is
PD-held at the stow pose, which the policy cannot move. If the stow pose sits
outside the arm's soft limits, that part of the penalty is a constant the policy
can never reduce — raising the weight would only lower the baseline return.

Example::

    python docs/training/round-12/check_joint_limit_penalty.py
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Split dof_pos_limits into leg and arm parts.")
parser.add_argument("--num_envs", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import spot_arm_locomotion  # noqa: F401  — registers Gym tasks

TASK = "Isaac-Velocity-Rough-Spot-Arm-v0"


def main() -> None:
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(TASK, num_envs=args_cli.num_envs)
    env = gym.make(TASK, cfg=env_cfg).unwrapped
    env.reset()

    robot = env.scene["robot"]
    names = robot.joint_names
    default = robot.data.default_joint_pos.torch[0]
    soft = robot.data.soft_joint_pos_limits.torch[0]
    hard = robot.data.joint_pos_limits.torch[0]

    below = (soft[:, 0] - default).clamp(min=0.0)
    above = (default - soft[:, 1]).clamp(min=0.0)
    violation = below + above

    print(f"\n{'joint':>22} {'default':>8} {'hard lo':>9} {'hard hi':>9} {'soft lo':>9} {'soft hi':>9} {'violation':>10}")
    print("-" * 90)
    for i, name in enumerate(names):
        mark = "  <-- outside" if float(violation[i]) > 1e-6 else ""
        print(
            f"{name:>22} {float(default[i]):8.3f} {float(hard[i, 0]):9.3f} {float(hard[i, 1]):9.3f}"
            f" {float(soft[i, 0]):9.3f} {float(soft[i, 1]):9.3f} {float(violation[i]):10.3f}{mark}"
        )

    leg = torch.tensor([("hip" in n or "knee" in n) for n in names])
    arm = ~leg
    print(f"\nviolation at the default pose (rad, summed):")
    print(f"  legs (policy can move these) : {float(violation[leg].sum()):.3f}")
    print(f"  arm  (PD-held, constant)     : {float(violation[arm].sum()):.3f}")
    print(f"  total                        : {float(violation.sum()):.3f}")
    print("\nround-10/11 logged Episode_Reward/dof_pos_limits was -0.398 at weight -1.0,")
    print("i.e. a mean per-step violation of 0.398 rad summed over all 19 joints.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
