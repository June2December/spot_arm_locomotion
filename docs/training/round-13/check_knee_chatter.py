#!/usr/bin/env python3
"""Separate a fast swing from a high-frequency oscillation, per leg joint.

Round 12 measured ``front_left_knee`` at 50 rad/s p99 while the other three knees
sat at 5 rad/s. Two very different things produce that:

* a genuinely fast swing  -> velocity sign flips about twice per stride (~2.7 Hz),
  spectral energy at the stride frequency, large net travel per flip.
* a PD/limit oscillation  -> velocity sign flips every 1-2 policy steps (~10-25 Hz),
  spectral energy far above the stride frequency, tiny net travel per flip.

The second one is what reads as "teleporting" in a 50 Hz viewport.

Example::

    python docs/training/round-13/check_knee_chatter.py \
        --checkpoint logs/rsl_rl/rough_spot_with_arm/2026-08-25_16-31-31/model_7148.pt \
        --force_straight 1.0
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Chatter vs swing for the leg joints.")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Spot-Arm-Play-v0")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=512)
parser.add_argument("--warmup", type=int, default=150)
parser.add_argument("--force_straight", type=float, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import spot_arm_locomotion  # noqa: F401  — registers Gym tasks

DT = 0.02
NYQUIST = 0.5 / DT  # 25 Hz


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

    robot = inner.scene["robot"]
    cmd_obj = inner.command_manager.get_term("base_velocity")
    joint_names = robot.joint_names
    leg_ids = [i for i, n in enumerate(joint_names) if "hip" in n or "knee" in n]
    act_ids = list(range(inner.action_manager.total_action_dim))

    qpos, qvel, acts = [], [], []
    obs = wrapped.get_observations()
    for step in range(args_cli.warmup + args_cli.steps):
        if args_cli.force_straight is not None:
            cmd_obj.vel_command_b[:, 0] = args_cli.force_straight
            cmd_obj.vel_command_b[:, 1:] = 0.0
            cmd_obj.is_standing_env[:] = False
        with torch.inference_mode():
            action = policy(obs)
            obs, _, dones, _ = wrapped.step(action)
            policy.reset(dones)
        if step < args_cli.warmup:
            continue
        qpos.append(robot.data.joint_pos.torch[:, leg_ids].cpu())
        qvel.append(robot.data.joint_vel.torch[:, leg_ids].cpu())
        acts.append(action.cpu())

    q = torch.stack(qpos).numpy()   # (T, N, 12)
    v = torch.stack(qvel).numpy()
    a = torch.stack(acts).numpy()   # (T, N, 12) target offsets, scale 0.4
    T = q.shape[0]

    print("\n" + "=" * 100)
    label = f"forced straight vx={args_cli.force_straight}" if args_cli.force_straight else "sampled commands"
    print(f"chatter check  {args_cli.num_envs} envs x {T} steps ({T * DT:.0f} s)  ({label})")
    print(f"policy runs at {1 / DT:.0f} Hz, so the highest representable motion is {NYQUIST:.0f} Hz")
    print("=" * 100)

    # Spectrum of the joint angle, mean over envs, ignoring DC.
    freqs = np.fft.rfftfreq(T, d=DT)
    print(f"\n{'joint':>22} {'flips/s':>8} {'peak Hz':>8} {'>8Hz energy':>12}"
          f" {'|dq|/step p99':>14} {'travel/flip':>12} {'|v| p99':>8}")
    print("-" * 100)
    for k, jid in enumerate(leg_ids):
        vk = v[:, :, k]
        sign_flip = (np.diff(np.sign(vk), axis=0) != 0).mean(axis=0) / DT  # flips per second
        spec = np.abs(np.fft.rfft(q[:, :, k] - q[:, :, k].mean(axis=0), axis=0)).mean(axis=1)
        peak_hz = freqs[1 + int(np.argmax(spec[1:]))]
        hi = float(spec[freqs > 8.0].sum() / spec[1:].sum())
        dq = np.abs(np.diff(q[:, :, k], axis=0))
        flips_per_s = float(sign_flip.mean())
        travel = float(np.abs(vk).mean() / max(flips_per_s, 1e-6))  # rad covered between reversals
        print(f"{joint_names[jid]:>22} {flips_per_s:8.1f} {peak_hz:8.2f} {hi:12.3f}"
              f" {float(np.percentile(dq, 99)):14.4f} {travel:12.4f} {float(np.percentile(np.abs(vk), 99)):8.2f}")

    print("\nAction (policy output, before the 0.4 scale) — saturation means the PD target is railed")
    print(f"{'joint':>22} {'mean':>8} {'std':>8} {'p1':>8} {'p99':>8} {'|a|>1':>8} {'flips/s':>8}")
    print("-" * 100)
    for k in act_ids:
        ak = a[:, :, k]
        flips = float((np.diff(np.sign(ak - ak.mean(axis=0)), axis=0) != 0).mean(axis=0).mean() / DT)
        print(f"{joint_names[leg_ids[k]]:>22} {ak.mean():8.3f} {ak.std():8.3f}"
              f" {np.percentile(ak, 1):8.3f} {np.percentile(ak, 99):8.3f}"
              f" {float((np.abs(ak) > 1.0).mean()):8.3f} {flips:8.1f}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
