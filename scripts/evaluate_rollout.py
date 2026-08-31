#!/usr/bin/env python3
"""Headless rollout metrics for a trained locomotion policy.

GUI로 보는 것(넘어짐, 걸음 모양) 말고, 숫자로 비교할 지표를 한 번에 뽑는다.
``diagnose_gait.py`` + ``diagnose_gait_timing.py`` 를 합치고, 속도 추종·액션 포화를 추가했다.

Example::

    source scripts/isaaclab_env.sh
    python scripts/evaluate_rollout.py \\
        --checkpoint logs/rsl_rl/rough_spot_with_arm/2026-08-27_10-20-22/model_10146.pt \\
        --json docs/training/round-15/data/round15_rollout.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a Spot-with-Arm policy rollout (headless).")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Velocity-Rough-Spot-Arm-RoughTest-v0",
    help="PLAY task (default: rough test, vx=1.0 on all envs).",
)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=800, help="Measured steps after warmup.")
parser.add_argument("--warmup", type=int, default=150)
parser.add_argument(
    "--force_straight",
    type=float,
    default=None,
    help="Overwrite command to (vx, 0, 0) every step. Omit to use the task's PLAY command.",
)
parser.add_argument("--json", type=str, default=None, help="Write metrics JSON to this path.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import spot_arm_locomotion  # noqa: F401
from spot_arm_locomotion.tasks.locomotion.mdp.rewards import _foot_pos_vel, _ground_height_under_feet

DT = 0.02
NOMINAL_HIP_Y = 0.055 + 0.110945
NOMINAL_HIP_X = 0.29785
JOINT_VEL_LIMITS = {"hip_x": 17.647, "hip_y": 17.647, "knee": 12.0}
KNEE_NAMES = ("front_left_knee", "front_right_knee", "rear_left_knee", "rear_right_knee")


def runs_of_true(mask: np.ndarray) -> list[int]:
    idx = np.flatnonzero(np.diff(mask.astype(np.int8)))
    if len(idx) < 2:
        return []
    starts = idx[:-1] + 1
    lengths = np.diff(idx)
    return [int(n) for s, n in zip(starts, lengths) if mask[s]]


def pct(x: float) -> float:
    return round(100.0 * float(x), 2)


def joint_stats(q: torch.Tensor, default: float) -> dict:
    p5, p95 = torch.quantile(q, torch.tensor([0.05, 0.95]))
    return {
        "default": round(float(default), 4),
        "mean": round(float(q.mean()), 4),
        "std": round(float(q.std()), 4),
        "p5": round(float(p5), 4),
        "p95": round(float(p95), 4),
        "range_p95_p5": round(float(p95 - p5), 4),
    }


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg) -> None:
    try:
        import importlib.metadata as metadata

        installed_version = metadata.version("rsl-rl-lib")
    except Exception:
        installed_version = "0.0.0"
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    clip_actions = float(agent_cfg.clip_actions) if agent_cfg.clip_actions is not None else None

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
    cmd_term = inner.command_manager.get_term("base_velocity")
    joint_names = robot.joint_names
    leg_ids = [i for i, n in enumerate(joint_names) if "hip" in n or "knee" in n]
    action_joint_names = [n for n in joint_names if n.endswith("_hip_x")]
    action_joint_names += [n for n in joint_names if n.endswith("_hip_y")]
    action_joint_names += [n for n in joint_names if n.endswith("_knee")]
    default_pos = robot.data.default_joint_pos.torch[0].cpu()

    rec: dict[str, list] = {
        "contact": [],
        "foot_b": [],
        "foot_z_rel": [],
        "foot_speed": [],
        "base_h": [],
        "base_vz": [],
        "qpos": [],
        "qvel": [],
        "vel_b": [],
        "ang_b": [],
        "cmd": [],
        "moving": [],
        "actions": [],
        "dones": [],
    }

    obs = wrapped.get_observations()
    for step in range(args_cli.warmup + args_cli.steps):
        if args_cli.force_straight is not None:
            cmd_term.vel_command_b[:, 0] = args_cli.force_straight
            cmd_term.vel_command_b[:, 1:] = 0.0
            cmd_term.is_standing_env[:] = False
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = wrapped.step(actions)
            policy.reset(dones)
        if step < args_cli.warmup:
            continue

        foot_pos, foot_vel = _foot_pos_vel(inner, feet_cfg)
        ground = _ground_height_under_feet(inner, foot_pos, scan_cfg)
        root = robot.data.root_pos_w.torch
        root_pos = root.unsqueeze(1)
        root_quat = yaw_quat(robot.data.root_quat_w.torch).unsqueeze(1).expand(-1, foot_pos.shape[1], -1)
        base_ground = _ground_height_under_feet(inner, root.unsqueeze(1), scan_cfg)[:, 0]
        forces = contact.data.net_forces_w_history.torch[:, :, contact_cfg.body_ids]

        rec["contact"].append((forces.norm(dim=-1).max(dim=1).values > 1.0).cpu())
        rec["foot_b"].append(
            quat_apply_inverse(root_quat.reshape(-1, 4), (foot_pos - root_pos).reshape(-1, 3))
            .view(foot_pos.shape)
            .cpu()
        )
        rec["foot_z_rel"].append((foot_pos[:, :, 2] - ground).cpu())
        rec["foot_speed"].append(foot_vel.norm(dim=-1).cpu())
        rec["base_h"].append((root[:, 2] - base_ground).cpu())
        rec["base_vz"].append(robot.data.root_lin_vel_w.torch[:, 2].cpu())
        rec["qpos"].append(robot.data.joint_pos.torch.clone().cpu())
        rec["qvel"].append(robot.data.joint_vel.torch[:, leg_ids].abs().cpu())
        rec["vel_b"].append(robot.data.root_lin_vel_b.torch.clone().cpu())
        rec["ang_b"].append(robot.data.root_ang_vel_b.torch.clone().cpu())
        rec["cmd"].append(cmd_term.vel_command_b.clone().cpu())
        rec["moving"].append((cmd_term.vel_command_b[:, :2].norm(dim=-1) > 0.1).cpu())
        rec["actions"].append(actions.cpu())
        rec["dones"].append(dones.cpu())

    d = {k: torch.stack(v) for k, v in rec.items()}
    contact_np = d["contact"].numpy()
    moving = d["cmd"][..., :2].norm(dim=-1) > 0.1
    moving_env = d["moving"].float().mean(dim=0) > 0.9
    n_walk = int(moving_env.sum())

    label = f"forced straight vx={args_cli.force_straight}" if args_cli.force_straight is not None else "task PLAY command"
    print("\n" + "=" * 92)
    print(f"{args_cli.task}  {n_walk}/{args_cli.num_envs} walking envs"
          f" x {args_cli.steps} steps ({args_cli.steps * DT:.0f} s)  ({label})")
    print(f"checkpoint: {args_cli.checkpoint}")
    if clip_actions is not None:
        print(f"clip_actions: {clip_actions}")
    print("=" * 92)

    # --- velocity tracking ---
    v = d["vel_b"][moving]
    c = d["cmd"][moving]
    w = d["ang_b"][moving]
    err_vx = (v[:, 0] - c[:, 0]).abs()
    err_vy = v[:, 1].abs()
    err_wz = w[:, 2].abs()
    vx_ratio = float(v[:, 0].mean() / c[:, 0].mean().clamp(min=1e-6))
    velocity = {
        "cmd_vx_mean": round(float(c[:, 0].mean()), 4),
        "cmd_vy_mean": round(float(c[:, 1].mean()), 4),
        "actual_vx_mean": round(float(v[:, 0].mean()), 4),
        "actual_vy_mean": round(float(v[:, 1].mean()), 4),
        "vx_tracking_ratio": round(vx_ratio, 4),
        "error_vx_mean": round(float(err_vx.mean()), 4),
        "error_vy_mean": round(float(err_vy.mean()), 4),
        "error_wz_mean": round(float(err_wz.mean()), 4),
        "crab_angle_deg_mean": round(float(torch.rad2deg(torch.atan2(v[:, 1], v[:, 0].clamp(min=1e-6))).mean()), 2),
        "crab_angle_deg_abs_mean": round(
            float(torch.rad2deg(torch.atan2(v[:, 1], v[:, 0].clamp(min=1e-6))).abs().mean()), 2
        ),
    }
    print("\n[A] Velocity tracking (moving frames)")
    print(f"  commanded  vx {velocity['cmd_vx_mean']:+.3f}  vy {velocity['cmd_vy_mean']:+.3f}")
    print(f"  actual     vx {velocity['actual_vx_mean']:+.3f}  vy {velocity['actual_vy_mean']:+.3f}"
          f"  (ratio {velocity['vx_tracking_ratio']:.3f})")
    print(f"  |vx-cmd| mean {velocity['error_vx_mean']:.3f} m/s   |vy| mean {velocity['error_vy_mean']:.3f} m/s"
          f"   |wz| mean {velocity['error_wz_mean']:.3f} rad/s")
    print(f"  crab angle mean {velocity['crab_angle_deg_mean']:+.1f} deg"
          f"  (|angle| {velocity['crab_angle_deg_abs_mean']:.1f} deg)")

    # --- contact pattern ---
    n_down = contact_np[:, moving_env.numpy(), :].sum(axis=-1)
    diag_a = contact_np[:, moving_env.numpy(), 0] & contact_np[:, moving_env.numpy(), 3]
    diag_b = contact_np[:, moving_env.numpy(), 1] & contact_np[:, moving_env.numpy(), 2]
    diagonal_trot = float((diag_a ^ diag_b).mean())
    contact_pattern = {
        "feet_down_0_pct": pct((n_down == 0).mean()),
        "feet_down_1_pct": pct((n_down == 1).mean()),
        "feet_down_2_pct": pct((n_down == 2).mean()),
        "feet_down_3_pct": pct((n_down == 3).mean()),
        "feet_down_4_pct": pct((n_down == 4).mean()),
        "diagonal_trot_contact_pct": pct(diagonal_trot),
        "fl_rr_both_down_pct": pct(diag_a.mean()),
        "fr_rl_both_down_pct": pct(diag_b.mean()),
    }
    print("\n[B] Contact pattern")
    for k in range(5):
        print(f"  {k} feet down: {contact_pattern[f'feet_down_{k}_pct']:5.1f}%")
    print(f"  diagonal trot contact %: {contact_pattern['diagonal_trot_contact_pct']:.1f}%"
          f"  (FL+RR {contact_pattern['fl_rr_both_down_pct']:.1f}%"
          f"  FR+RL {contact_pattern['fr_rl_both_down_pct']:.1f}%)")

    # --- joint pose ---
    joints: dict[str, dict] = {}
    for j, name in enumerate(joint_names):
        if "hip" not in name and "knee" not in name:
            continue
        joints[name] = joint_stats(d["qpos"][..., j][moving], float(default_pos[j]))
    knees = {n: joints[n] for n in KNEE_NAMES if n in joints}
    print("\n[C] Leg joint pose — moving envs (rad)")
    print(f"{'joint':>22} {'default':>8} {'mean':>8} {'range':>8}")
    for name in KNEE_NAMES:
        if name not in joints:
            continue
        s = joints[name]
        print(f"{name:>22} {s['default']:8.3f} {s['mean']:8.3f} {s['range_p95_p5']:8.3f}")

    # --- stance ---
    fb = d["foot_b"]
    mean_y: dict[str, float] = {}
    stance_rows = []
    for i, name in enumerate(foot_names):
        sel = fb[:, :, i, :][moving]
        nom_x = NOMINAL_HIP_X * (1 if name.startswith("front") else -1)
        nom_y = NOMINAL_HIP_Y * (1 if name.endswith("left") else -1)
        mean_y[name] = float(sel[:, 1].mean())
        stance_rows.append(
            {
                "foot": name,
                "x_mean": round(float(sel[:, 0].mean()), 4),
                "y_mean": round(mean_y[name], 4),
                "nominal_x": round(nom_x, 4),
                "nominal_y": round(nom_y, 4),
            }
        )
    front_w = mean_y["front_left"] - mean_y["front_right"]
    rear_w = mean_y["rear_left"] - mean_y["rear_right"]
    nominal_w = 2 * NOMINAL_HIP_Y
    stance = {
        "front_stance_width_m": round(front_w, 4),
        "rear_stance_width_m": round(rear_w, 4),
        "front_stance_pct_of_nominal": round(100 * front_w / nominal_w, 1),
        "rear_stance_pct_of_nominal": round(100 * rear_w / nominal_w, 1),
        "feet": stance_rows,
    }
    print("\n[D] Stance width (body yaw frame)")
    print(f"  front {front_w:.3f} m ({stance['front_stance_pct_of_nominal']:.0f}% of nominal)"
          f"   rear {rear_w:.3f} m ({stance['rear_stance_pct_of_nominal']:.0f}%)")

    # --- gait timing ---
    gait_timing: dict[str, dict] = {}
    print("\n[E] Per-foot gait timing")
    print(f"{'foot':>14} {'duty':>7} {'stride Hz':>10} {'swing s':>9} {'peak spd':>9}")
    for i, name in enumerate(foot_names):
        swings, stances = [], []
        for e in np.flatnonzero(moving_env.numpy()):
            cfoot = contact_np[:, e, i]
            swings += runs_of_true(~cfoot)
            stances += runs_of_true(cfoot)
        swing_s = np.array(swings) * DT if swings else np.array([np.nan])
        duty = float(contact_np[:, moving_env.numpy(), i].mean())
        n_td = sum(len(runs_of_true(~contact_np[:, e, i])) for e in np.flatnonzero(moving_env.numpy()))
        stride_hz = n_td / (n_walk * args_cli.steps * DT) if n_walk else float("nan")
        air = ~contact_np[:, moving_env.numpy(), i]
        peak_spd = float(np.percentile(d["foot_speed"][:, moving_env, i][air], 99)) if air.any() else float("nan")
        gait_timing[name] = {
            "duty": round(duty, 4),
            "stride_hz": round(stride_hz, 4),
            "swing_s_mean": round(float(np.nanmean(swing_s)), 4),
            "peak_foot_speed_m_s": round(peak_spd, 4),
        }
        print(f"{name:>14} {duty:7.3f} {stride_hz:10.2f} {gait_timing[name]['swing_s_mean']:9.3f}"
              f" {peak_spd:9.2f}")

    # --- body ---
    bh = d["base_h"][:, moving_env].numpy()
    vz = d["base_vz"][:, moving_env].numpy()
    body = {
        "height_above_terrain_mean_m": round(float(bh.mean()), 4),
        "height_above_terrain_std_m": round(float(bh.std()), 4),
        "height_p1_m": round(float(np.percentile(bh, 1)), 4),
        "height_p99_m": round(float(np.percentile(bh, 99)), 4),
        "vertical_velocity_std_m_s": round(float(vz.std()), 4),
        "nominal_standing_height_m": 0.54,
    }
    print("\n[F] Body")
    print(f"  height above terrain: mean {body['height_above_terrain_mean_m']:.3f} m"
          f"  std {body['height_above_terrain_std_m']:.3f} m"
          f"  (nominal stand 0.54 m)")

    # --- joint velocity ---
    qv = d["qvel"][:, moving_env]
    joint_velocity: dict[str, dict] = {}
    print("\n[G] Leg joint |velocity| (rad/s)")
    print(f"{'joint':>22} {'p99':>8} {'max':>8} {'limit':>8}")
    for k, jid in enumerate(leg_ids):
        name = joint_names[jid]
        lim = next(v for key, v in JOINT_VEL_LIMITS.items() if name.endswith(key))
        joint_velocity[name] = {
            "p99": round(float(torch.quantile(qv[:, :, k], 0.99)), 4),
            "max": round(float(qv[:, :, k].max()), 4),
            "limit": lim,
        }
        print(f"{name:>22} {joint_velocity[name]['p99']:8.2f} {joint_velocity[name]['max']:8.2f} {lim:8.1f}")

    # --- actions ---
    act = d["actions"][:, moving_env]
    action_stats = {
        "mean_abs": round(float(act.abs().mean()), 4),
        "p99_abs": round(float(torch.quantile(act.abs(), 0.99)), 4),
        "max_abs": round(float(act.abs().max()), 4),
    }
    if clip_actions is not None:
        sat = act.abs() > (0.95 * clip_actions)
        action_stats["clip_actions"] = clip_actions
        action_stats["saturated_pct"] = pct(sat.float().mean())
        action_stats["saturated_per_joint"] = {
            action_joint_names[i]: pct(sat[:, :, i].float().mean()) for i in range(act.shape[-1])
        }
    print("\n[H] Actions (policy output, pre-env-clip)")
    print(f"  |a| mean {action_stats['mean_abs']:.3f}  p99 {action_stats['p99_abs']:.3f}"
          f"  max {action_stats['max_abs']:.3f}")
    if clip_actions is not None:
        print(f"  saturated (|a| > 0.95×{clip_actions}): {action_stats['saturated_pct']:.1f}%")

    terminations = {
        "episode_done_frames_pct": pct(d["dones"].float().mean()),
    }
    print(f"\n[I] Terminations during window: {terminations['episode_done_frames_pct']:.1f}% of env-steps")

    report = {
        "checkpoint": str(args_cli.checkpoint),
        "task": args_cli.task,
        "num_envs": args_cli.num_envs,
        "steps": args_cli.steps,
        "warmup": args_cli.warmup,
        "dt": DT,
        "walking_envs": n_walk,
        "command_label": label,
        "velocity_tracking": velocity,
        "contact_pattern": contact_pattern,
        "knees": knees,
        "leg_joints": joints,
        "stance": stance,
        "gait_timing": gait_timing,
        "body": body,
        "joint_velocity": joint_velocity,
        "actions": action_stats,
        "terminations": terminations,
    }

    if args_cli.json:
        out = Path(args_cli.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\n[INFO] Wrote {out}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
