"""Reward helpers for Spot with Arm."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply
from isaaclab_tasks.manager_based.locomotion.velocity.mdp.rewards import feet_air_time as _feet_air_time
from isaaclab_tasks.manager_based.locomotion.velocity.mdp.rewards import feet_slide as _feet_slide

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv

# Sphere center in each ``*_lower_leg`` frame (URDF). Standing z ≈ 0.036 m.
_FOOT_IN_LOWER_LEG = (0.0, 0.0, -0.3365)
_FOOT_RADIUS = 0.036


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Same as Isaac Lab ``feet_air_time``, but short steps are 0 instead of a penalty.

    Lab's term is ``(air_time - threshold)`` on first contact. Air time below the
    threshold is negative and punished standing / contact chatter.
    """
    return torch.clamp(_feet_air_time(env, command_name, sensor_cfg, threshold), min=0.0)


def track_lin_vel_xy_dot(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward commanded-direction speed: ``cmd_xy · v_xy``.

    Exponential tracking is almost flat at v=0 when the command is ~1 m/s.
    This term is 0 when standing and rises as soon as the body moves along
    the command, so the first centimetre of motion is a real credit.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)[:, :2]
    vel = asset.data.root_lin_vel_b.torch[:, :2]
    return torch.clamp(torch.sum(cmd * vel, dim=1), min=-2.0, max=2.0)


def _foot_pos_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> tuple[torch.Tensor, torch.Tensor]:
    """World position / velocity of the four foot-sphere centers."""
    asset: RigidObject = env.scene[asset_cfg.name]
    pos = asset.data.body_pos_w.torch[:, asset_cfg.body_ids]
    quat = asset.data.body_quat_w.torch[:, asset_cfg.body_ids]
    vel = asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids]
    ang = asset.data.body_ang_vel_w.torch[:, asset_cfg.body_ids]
    n_envs, n_feet, _ = pos.shape
    offset = pos.new_tensor(_FOOT_IN_LOWER_LEG).view(1, 1, 3).expand(n_envs, n_feet, 3)
    offset_w = quat_apply(quat.reshape(-1, 4), offset.reshape(-1, 3)).view(n_envs, n_feet, 3)
    foot_pos = pos + offset_w
    foot_vel = vel + torch.cross(ang, offset_w, dim=-1)
    return foot_pos, foot_vel


def feet_slide(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_speed_threshold: float = 0.4,
) -> torch.Tensor:
    """Penalize foot xy speed in contact, only after the body is already moving."""
    slide = _feet_slide(env, sensor_cfg, asset_cfg)
    robot: RigidObject = env.scene["robot"]
    speed = torch.linalg.norm(robot.data.root_lin_vel_b.torch[:, :2], dim=1)
    return slide * (speed > body_speed_threshold).float()


def foot_clearance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    target_height: float = 0.08,
) -> torch.Tensor:
    """Reward commanded robots for raising a foot off the ground.

    The swing-velocity gate used in 6a zeroed this term while standing, so
    there was no gradient to pick a foot up. Command-gated height is 0 when
    the command is ~0 or the feet stay on the ground, and rises as soon as
    a foot leaves the sphere radius.
    """
    foot_pos, _foot_vel = _foot_pos_vel(env, asset_cfg)
    ground = _FOOT_RADIUS
    lift = torch.clamp(
        (foot_pos[:, :, 2] - ground) / max(target_height - ground, 1e-3), min=0.0, max=1.0
    )
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    return torch.mean(lift, dim=1) * (cmd > 0.1).float()
