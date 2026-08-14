"""Observation helpers for Spot with Arm."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def replace_nonfinite(data: torch.Tensor, fill: float = 0.0) -> torch.Tensor:
    """Replace NaN/Inf so PPO never sees non-finite observations."""
    return torch.nan_to_num(data, nan=fill, posinf=fill, neginf=fill)


def height_scan(
    env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5
) -> torch.Tensor:
    """Height scan with non-finite hits replaced.

    Isaac Lab's scan is ``sensor_z - hit_z - offset``. Misses are Inf; a
    flying/fallen robot can make both terms Inf, so ``Inf - Inf`` becomes NaN
    and ``clip`` cannot recover it.
    """
    scan = mdp.height_scan(env, sensor_cfg, offset=offset)
    return torch.nan_to_num(scan, nan=0.0, posinf=1.0, neginf=-1.0)
