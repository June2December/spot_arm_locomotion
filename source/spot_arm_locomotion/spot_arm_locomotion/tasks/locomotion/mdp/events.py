"""Event terms for Spot with Arm."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv


def reset_arm_to_default(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset arm joints to the folded default pose and hold that PD target.

    Leg joints are randomized separately. Arm joints are not in the action space,
    so writing the default position as the implicit-PD target keeps the arm folded
    for the rest of the episode.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    default_pos = asset.data.default_joint_pos.torch[env_ids][:, asset_cfg.joint_ids]
    default_vel = asset.data.default_joint_vel.torch[env_ids][:, asset_cfg.joint_ids]
    asset.write_joint_position_to_sim_index(
        position=default_pos, joint_ids=asset_cfg.joint_ids, env_ids=env_ids
    )
    asset.write_joint_velocity_to_sim_index(
        velocity=default_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids
    )
    asset.set_joint_position_target_index(
        target=default_pos, joint_ids=asset_cfg.joint_ids, env_ids=env_ids
    )
