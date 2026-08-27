"""Rough-terrain PLAY. Not used for training."""

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils.configclass import configclass

from .spot_arm_rough_cfg import SpotArmRoughEnvCfg


@configclass
class SpotArmRoughTerrainPlayCfg(SpotArmRoughEnvCfg):
    """Lab 10×20 rough grid, world camera, same forward command on every robot."""

    def __post_init__(self):
        super().__post_init__()

        # One robot per column-band so stairs / boxes / noise are all occupied
        # at once. 10×20 patches are 8 m; the mesh is ~120×200 m with border.
        self.scene.num_envs = 80
        self.scene.env_spacing = 2.5

        # Static world camera. Chase (asset_root) locked the viewport to env 0
        # every frame, so you could not pan to another patch.
        self.viewer.origin_type = "world"
        self.viewer.asset_name = None
        self.viewer.eye = (0.0, -55.0, 48.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG.replace(
            num_rows=10, num_cols=20, curriculum=False
        )
        self.scene.terrain.max_init_terrain_level = None
        self.curriculum.terrain_levels = None

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.reset_base.params["pose_range"] = {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (0.0, 0.0)}
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)

        # Training samples vx 0.5–1.5, vy ±0.4, yaw ±0.8, and 10% stand.
        # PLAY uses one command so speed/gait differences are terrain + policy,
        # not a different joystick per robot.
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
