#!/usr/bin/env python3
"""Measure the four gait complaints from the round-11 GUI rollout.

1. Robots that stop and hold one rear foot up  -> per-foot contact fraction,
   split by whether the command term marked the env as standing.
2. Knee barely moves, hip does the work        -> per-joint range of motion.
3. Drifts diagonally on a straight command     -> lateral / forward speed ratio
   and yaw rate under a forced (vx, 0, 0) command.
4. Feet planted under the belly, no stance     -> foot xy in the body frame vs
   the nominal hip locations from the URDF.

Example::

    python docs/training/round-11/diagnose_gait.py \
        --checkpoint logs/rsl_rl/rough_spot_with_arm/2026-08-25_15-27-52/model_5649.pt
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Measure Spot-with-Arm gait pathologies.")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Spot-Arm-Play-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--warmup", type=int, default=100)
parser.add_argument(
    "--force_straight",
    type=float,
    default=None,
    help="Overwrite the command every step with (vx, 0, 0) at this vx.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
# Hydra parses sys.argv itself and rejects the flags above.
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import spot_arm_locomotion  # noqa: F401  — registers Gym tasks
from spot_arm_locomotion.tasks.locomotion.mdp.rewards import _foot_pos_vel

# URDF: hip_x at y=+-0.055, hip_y adds +-0.110945, hips at x=+-0.29785.
NOMINAL_HIP_Y = 0.055 + 0.110945
NOMINAL_HIP_X = 0.29785
FOOT_ORDER = ["front_left", "front_right", "rear_left", "rear_right"]


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg) -> None:
    try:
        import importlib.metadata as metadata

        installed_version = metadata.version("rsl-rl-lib")
    except Exception:
        installed_version = "0.0.0"
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg)
    inner = env.unwrapped
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=inner.device)

    feet_cfg = SceneEntityCfg("robot", body_names=".*_lower_leg")
    feet_cfg.resolve(inner.scene)
    foot_names = [inner.scene["robot"].body_names[i].replace("_lower_leg", "") for i in feet_cfg.body_ids]
    contact = inner.scene["contact_forces"]
    contact_cfg = SceneEntityCfg("contact_forces", body_names=".*_lower_leg")
    contact_cfg.resolve(inner.scene)
    cmd_term = inner.command_manager.get_term("base_velocity")
    robot = inner.scene["robot"]
    joint_names = robot.joint_names

    rec: dict[str, list[torch.Tensor]] = {k: [] for k in
                                          ("qpos", "foot_b", "in_contact", "vel_b", "ang_b", "cmd", "standing")}

    obs = wrapped.get_observations()
    for step in range(args_cli.warmup + args_cli.steps):
        if args_cli.force_straight is not None:
            cmd_term.vel_command_b[:, 0] = args_cli.force_straight
            cmd_term.vel_command_b[:, 1] = 0.0
            cmd_term.vel_command_b[:, 2] = 0.0
            cmd_term.is_standing_env[:] = False
        with torch.inference_mode():
            obs, _, dones, _ = wrapped.step(policy(obs))
            policy.reset(dones)
        if step < args_cli.warmup:
            continue
        foot_pos_w, _ = _foot_pos_vel(inner, feet_cfg)
        root_pos = robot.data.root_pos_w.torch.unsqueeze(1)
        root_quat = yaw_quat(robot.data.root_quat_w.torch).unsqueeze(1).expand(-1, foot_pos_w.shape[1], -1)
        rec["foot_b"].append(quat_apply_inverse(root_quat.reshape(-1, 4), (foot_pos_w - root_pos).reshape(-1, 3))
                             .view(foot_pos_w.shape).cpu())
        rec["qpos"].append(robot.data.joint_pos.torch.clone().cpu())
        forces = contact.data.net_forces_w_history.torch[:, :, contact_cfg.body_ids]
        rec["in_contact"].append((forces.norm(dim=-1).max(dim=1).values > 1.0).cpu())
        rec["vel_b"].append(robot.data.root_lin_vel_b.torch.clone().cpu())
        rec["ang_b"].append(robot.data.root_ang_vel_b.torch.clone().cpu())
        rec["cmd"].append(cmd_term.vel_command_b.clone().cpu())
        rec["standing"].append(cmd_term.is_standing_env.clone().cpu())

    d = {k: torch.stack(v) for k, v in rec.items()}  # (T, N, ...)
    moving = d["cmd"][..., :2].norm(dim=-1) > 0.1
    standing = d["standing"]

    print("\n" + "=" * 78)
    label = f"forced straight vx={args_cli.force_straight}" if args_cli.force_straight else "sampled commands"
    print(f"{args_cli.task}  {args_cli.num_envs} envs x {args_cli.steps} steps  ({label})")
    print("=" * 78)

    print("\n[2] Joint range of motion — moving envs only (rad)")
    print(f"{'joint':>22} {'default':>8} {'mean':>8} {'std':>8} {'p5':>8} {'p95':>8} {'range':>8}")
    default = robot.data.default_joint_pos.torch[0].cpu()
    for j, name in enumerate(joint_names):
        if "hip" not in name and "knee" not in name:
            continue
        q = d["qpos"][..., j][moving]
        p5, p95 = torch.quantile(q, torch.tensor([0.05, 0.95]))
        print(f"{name:>22} {float(default[j]):8.3f} {float(q.mean()):8.3f} {float(q.std()):8.3f}"
              f" {float(p5):8.3f} {float(p95):8.3f} {float(p95 - p5):8.3f}")

    print("\n[4] Foot position in the body yaw frame — moving envs (m)")
    print(f"{'foot':>14} {'nominal x':>10} {'x':>8} {'nominal y':>10} {'y':>8} {'z':>8}")
    fb = d["foot_b"]
    mean_y = {}
    for i, name in enumerate(foot_names):
        sel = fb[:, :, i, :][moving]
        nom_x = NOMINAL_HIP_X * (1 if name.startswith("front") else -1)
        nom_y = NOMINAL_HIP_Y * (1 if name.endswith("left") else -1)
        mean_y[name] = float(sel[:, 1].mean())
        print(f"{name:>14} {nom_x:10.3f} {float(sel[:, 0].mean()):8.3f} {nom_y:10.3f}"
              f" {mean_y[name]:8.3f} {float(sel[:, 2].mean()):8.3f}")
    front_w = mean_y["front_left"] - mean_y["front_right"]
    rear_w = mean_y["rear_left"] - mean_y["rear_right"]
    print(f"\n  stance width  front {front_w:.3f} m   rear {rear_w:.3f} m"
          f"   nominal {2 * NOMINAL_HIP_Y:.3f} m")
    print(f"  front is {100 * front_w / (2 * NOMINAL_HIP_Y):.0f}% of nominal,"
          f" rear {100 * rear_w / (2 * NOMINAL_HIP_Y):.0f}%")

    print("\n[1] Foot contact fraction")
    print(f"{'foot':>14} {'moving':>10} {'standing':>10}")
    ic = d["in_contact"]
    for i, name in enumerate(foot_names):
        c = ic[:, :, i]
        mv = float(c[moving].float().mean()) if bool(moving.any()) else float("nan")
        st = float(c[standing].float().mean()) if bool(standing.any()) else float("nan")
        print(f"{name:>14} {mv:10.3f} {st:10.3f}")
    n_down = ic.sum(dim=-1)
    if bool(standing.any()):
        print(f"\n  standing envs: mean feet down {float(n_down[standing].float().mean()):.2f} / 4;"
              f" frames with a foot up {100 * float((n_down[standing] < 4).float().mean()):.1f}%")
    print(f"  moving envs:   mean feet down {float(n_down[moving].float().mean()):.2f} / 4")

    print("\n[3] Straight-line tracking — moving envs")
    v = d["vel_b"][moving]
    w = d["ang_b"][moving]
    c = d["cmd"][moving]
    print(f"  commanded  vx {float(c[:, 0].mean()):+.3f}  vy {float(c[:, 1].mean()):+.3f}"
          f"  wz {float(c[:, 2].mean()):+.3f}")
    print(f"  actual     vx {float(v[:, 0].mean()):+.3f}  vy {float(v[:, 1].mean()):+.3f}"
          f"  wz {float(w[:, 2].mean()):+.3f}")
    print(f"  |vy| mean {float(v[:, 1].abs().mean()):.3f} m/s"
          f"  = {100 * float(v[:, 1].abs().mean() / v[:, 0].abs().mean().clamp(min=1e-6)):.0f}% of |vx|")
    drift = torch.rad2deg(torch.atan2(v[:, 1], v[:, 0].clamp(min=1e-6)))
    print(f"  crab angle atan2(vy, vx): mean {float(drift.mean()):+.1f} deg,"
          f" mean |angle| {float(drift.abs().mean()):.1f} deg")
    print(f"  |wz| mean {float(w[:, 2].abs().mean()):.3f} rad/s (command wz"
          f" {float(c[:, 2].abs().mean()):.3f})")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
