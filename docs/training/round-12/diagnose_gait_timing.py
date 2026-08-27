#!/usr/bin/env python3
"""Measure gait timing, per-leg symmetry, and body bounce.

Answers two things the joint/stance measurement in ``diagnose_gait.py`` cannot:

* "one leg snaps / teleports"  -> per-foot swing duration, stride frequency and
  peak foot speed, so an outlier leg shows up as a number.
* "the others look floaty"     -> base height oscillation, vertical velocity and
  duty factor, plus how close the contact pattern is to a trot.

Example::

    python docs/training/round-12/diagnose_gait_timing.py \
        --checkpoint logs/rsl_rl/rough_spot_with_arm/2026-08-25_16-31-31/model_7148.pt \
        --force_straight 1.0
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Measure Spot-with-Arm gait timing and symmetry.")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Spot-Arm-Play-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=800)
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

from isaaclab.managers import SceneEntityCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import spot_arm_locomotion  # noqa: F401  — registers Gym tasks
from spot_arm_locomotion.tasks.locomotion.mdp.rewards import _foot_pos_vel, _ground_height_under_feet

DT = 0.02  # policy step
# Boston Dynamics Spot on hardware: trot around 1.5-2.5 Hz, swing 0.20-0.35 s.
REAL_STRIDE_HZ = (1.5, 2.5)
REAL_SWING_S = (0.20, 0.35)


def runs_of_true(mask: np.ndarray) -> list[int]:
    """Lengths of consecutive True runs in a 1-D boolean array, ends dropped."""
    idx = np.flatnonzero(np.diff(mask.astype(np.int8)))
    if len(idx) < 2:
        return []
    starts = idx[:-1] + 1
    lengths = np.diff(idx)
    return [int(n) for s, n in zip(starts, lengths) if mask[s]]


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
    contact_cfg = SceneEntityCfg("contact_forces", body_names=".*_lower_leg")
    contact_cfg.resolve(inner.scene)
    scan_cfg = SceneEntityCfg("height_scanner")
    scan_cfg.resolve(inner.scene)
    contact = inner.scene["contact_forces"]
    robot = inner.scene["robot"]
    cmd_term = inner.command_manager.get_command
    cmd_obj = inner.command_manager.get_term("base_velocity")
    joint_names = robot.joint_names
    leg_ids = [i for i, n in enumerate(joint_names) if "hip" in n or "knee" in n]

    rec = {k: [] for k in ("contact", "foot_z_rel", "foot_speed", "base_h", "base_vz", "qvel", "moving")}
    obs = wrapped.get_observations()
    for step in range(args_cli.warmup + args_cli.steps):
        if args_cli.force_straight is not None:
            cmd_obj.vel_command_b[:, 0] = args_cli.force_straight
            cmd_obj.vel_command_b[:, 1:] = 0.0
            cmd_obj.is_standing_env[:] = False
        with torch.inference_mode():
            obs, _, dones, _ = wrapped.step(policy(obs))
            policy.reset(dones)
        if step < args_cli.warmup:
            continue
        foot_pos, foot_vel = _foot_pos_vel(inner, feet_cfg)
        ground = _ground_height_under_feet(inner, foot_pos, scan_cfg)
        root = robot.data.root_pos_w.torch
        base_ground = _ground_height_under_feet(inner, root.unsqueeze(1), scan_cfg)[:, 0]
        forces = contact.data.net_forces_w_history.torch[:, :, contact_cfg.body_ids]
        rec["contact"].append((forces.norm(dim=-1).max(dim=1).values > 1.0).cpu())
        rec["foot_z_rel"].append((foot_pos[:, :, 2] - ground).cpu())
        rec["foot_speed"].append(foot_vel.norm(dim=-1).cpu())
        rec["base_h"].append((root[:, 2] - base_ground).cpu())
        rec["base_vz"].append(robot.data.root_lin_vel_w.torch[:, 2].cpu())
        rec["qvel"].append(robot.data.joint_vel.torch[:, leg_ids].abs().cpu())
        rec["moving"].append((cmd_term("base_velocity")[:, :2].norm(dim=-1) > 0.1).cpu())

    d = {k: torch.stack(v).numpy() for k, v in rec.items()}
    contact_np = d["contact"]  # (T, N, 4)
    moving_env = d["moving"].mean(axis=0) > 0.9  # envs commanded to walk the whole window

    label = f"forced straight vx={args_cli.force_straight}" if args_cli.force_straight else "sampled commands"
    print("\n" + "=" * 92)
    print(f"{args_cli.task}  {int(moving_env.sum())}/{args_cli.num_envs} walking envs"
          f" x {args_cli.steps} steps ({args_cli.steps * DT:.0f} s)  ({label})")
    print("=" * 92)

    print("\n[A] Per-foot gait timing   (real Spot: stride 1.5-2.5 Hz, swing 0.20-0.35 s)")
    print(f"{'foot':>14} {'duty':>7} {'stride Hz':>10} {'swing s':>9} {'swing max':>10}"
          f" {'stance s':>9} {'peak spd':>9} {'peak z':>8}")
    print("-" * 92)
    summary = {}
    for i, name in enumerate(foot_names):
        swings, stances = [], []
        for e in np.flatnonzero(moving_env):
            c = contact_np[:, e, i]
            swings += runs_of_true(~c)
            stances += runs_of_true(c)
        swing_s = np.array(swings) * DT if swings else np.array([np.nan])
        stance_s = np.array(stances) * DT if stances else np.array([np.nan])
        duty = float(contact_np[:, moving_env, i].mean())
        n_td = sum(len(runs_of_true(~contact_np[:, e, i])) for e in np.flatnonzero(moving_env))
        stride_hz = n_td / (int(moving_env.sum()) * args_cli.steps * DT)
        air = ~contact_np[:, moving_env, i]
        peak_spd = float(np.percentile(d["foot_speed"][:, moving_env, i][air], 99)) if air.any() else float("nan")
        peak_z = float(np.percentile(d["foot_z_rel"][:, moving_env, i][air], 99)) if air.any() else float("nan")
        summary[name] = (stride_hz, float(np.nanmean(swing_s)), peak_spd)
        print(f"{name:>14} {duty:7.3f} {stride_hz:10.2f} {float(np.nanmean(swing_s)):9.3f}"
              f" {float(np.nanmax(swing_s)):10.3f} {float(np.nanmean(stance_s)):9.3f}"
              f" {peak_spd:9.2f} {peak_z:8.3f}")

    hz = np.array([v[0] for v in summary.values()])
    sw = np.array([v[1] for v in summary.values()])
    sp = np.array([v[2] for v in summary.values()])
    print(f"\n  spread across legs: stride Hz {hz.min():.2f}-{hz.max():.2f}"
          f" ({100 * (hz.max() - hz.min()) / hz.mean():.0f}% of mean),"
          f" swing {sw.min():.3f}-{sw.max():.3f} s, peak speed {sp.min():.2f}-{sp.max():.2f} m/s")

    print("\n[B] Contact pattern — a clean trot pairs the diagonals")
    n_down = contact_np[:, moving_env, :].sum(axis=-1)
    for k in range(5):
        print(f"  {k} feet down: {100 * float((n_down == k).mean()):5.1f}%")
    diag_a = contact_np[:, moving_env, 0] & contact_np[:, moving_env, 3]   # FL + RR
    diag_b = contact_np[:, moving_env, 1] & contact_np[:, moving_env, 2]   # FR + RL
    print(f"  FL+RR both down {100 * float(diag_a.mean()):.1f}%   FR+RL both down {100 * float(diag_b.mean()):.1f}%")
    print(f"  exactly one diagonal down (trot signature): {100 * float((diag_a ^ diag_b).mean()):.1f}%")

    print("\n[C] Body bounce")
    bh = d["base_h"][:, moving_env]
    vz = d["base_vz"][:, moving_env]
    print(f"  height above terrain: mean {bh.mean():.3f} m, std {bh.std():.3f} m,"
          f" p1-p99 {np.percentile(bh, 1):.3f}-{np.percentile(bh, 99):.3f} m")
    print(f"  vertical velocity   : std {vz.std():.3f} m/s, p99 |vz| {np.percentile(np.abs(vz), 99):.3f} m/s")
    print(f"  nominal standing height is 0.54 m (body z with the foot spheres on the ground)")

    print("\n[D] Leg joint velocity (rad/s, moving envs)")
    qv = d["qvel"][:, moving_env]
    print(f"{'joint':>22} {'mean':>8} {'p99':>8} {'max':>8} {'limit':>8}")
    limits = {"hip_x": 17.647, "hip_y": 17.647, "knee": 12.0}
    for k, jid in enumerate(leg_ids):
        name = joint_names[jid]
        lim = next(v for key, v in limits.items() if name.endswith(key))
        print(f"{name:>22} {qv[:, :, k].mean():8.2f} {np.percentile(qv[:, :, k], 99):8.2f}"
              f" {qv[:, :, k].max():8.2f} {lim:8.1f}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
