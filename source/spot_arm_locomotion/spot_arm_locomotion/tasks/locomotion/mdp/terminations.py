"""Termination helpers for Spot with Arm."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def root_height_above_maximum(
    env: ManagerBasedRLEnv,
    maximum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the root flies too far above the terrain."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w.torch[:, 2] > maximum_height


def root_lin_vel_too_large(
    env: ManagerBasedRLEnv,
    max_speed: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the root linear speed explodes (pre-NaN)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    speed = torch.linalg.norm(asset.data.root_lin_vel_w.torch, dim=-1)
    return torch.nan_to_num(speed, nan=max_speed + 1.0, posinf=max_speed + 1.0) > max_speed
