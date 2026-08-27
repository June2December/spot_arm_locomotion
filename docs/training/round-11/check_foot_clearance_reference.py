#!/usr/bin/env python3
"""Measure the round-10 ``foot_clearance`` reference bug on the rough generator.

Spawns the training env, holds the default pose (zero action) so no foot is
actually lifted, and prints the term both ways:

* ``world``   — round 10: foot z above a constant ``_FOOT_RADIUS``
* ``terrain`` — round 11: foot z above the terrain under that foot

A standing robot should score ~0 either way. Anything above 0 in the ``world``
column is reward paid for the sub-terrain sitting above z=0.

Example::

    /home/june/IsaacLab/isaaclab.sh -p docs/training/round-11/check_foot_clearance_reference.py
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Compare world-z vs terrain-relative foot clearance.")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=30)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.managers import SceneEntityCfg

import spot_arm_locomotion  # noqa: F401  — registers Gym tasks
from spot_arm_locomotion.tasks.locomotion.mdp.rewards import (
    _FOOT_RADIUS,
    _foot_pos_vel,
    _ground_height_under_feet,
)

TASK = "Isaac-Velocity-Rough-Spot-Arm-v0"


def main() -> None:
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(TASK, num_envs=args_cli.num_envs)
    env = gym.make(TASK, cfg=env_cfg).unwrapped

    feet_cfg = SceneEntityCfg("robot", body_names=".*_lower_leg")
    feet_cfg.resolve(env.scene)
    scan_cfg = SceneEntityCfg("height_scanner")
    scan_cfg.resolve(env.scene)

    env.reset()
    action = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    for _ in range(args_cli.steps):
        env.step(action)

    foot_pos, _ = _foot_pos_vel(env, feet_cfg)
    span = max(0.08 - _FOOT_RADIUS, 1e-3)
    world = torch.clamp((foot_pos[:, :, 2] - _FOOT_RADIUS) / span, 0.0, 1.0).mean(dim=1)
    terrain_z = _ground_height_under_feet(env, foot_pos, scan_cfg)
    relative = torch.clamp((foot_pos[:, :, 2] - terrain_z - _FOOT_RADIUS) / span, 0.0, 1.0).mean(dim=1)

    origin_z = env.scene.env_origins[:, 2]
    print(f"\n{args_cli.num_envs} envs, zero action, {args_cli.steps} steps — standing, no foot lifted")
    print(f"{'sub-terrain origin z':>22} | {'envs':>5} | {'world lift':>10} | {'terrain lift':>12}")
    print("-" * 60)
    for label, mask in [
        ("below -0.05 m", origin_z < -0.05),
        ("within +-0.05 m", origin_z.abs() <= 0.05),
        ("above +0.05 m", origin_z > 0.05),
        ("all", torch.ones_like(origin_z, dtype=torch.bool)),
    ]:
        if not bool(mask.any()):
            continue
        print(
            f"{label:>22} | {int(mask.sum()):5d} | {float(world[mask].mean()):10.3f} |"
            f" {float(relative[mask].mean()):12.3f}"
        )
    print(f"\nenv origin z range: {float(origin_z.min()):+.2f} .. {float(origin_z.max()):+.2f} m")
    print(f"scanner-vs-origin z residual (mean abs): {float((terrain_z.mean(dim=1) - origin_z).abs().mean()):.3f} m")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
