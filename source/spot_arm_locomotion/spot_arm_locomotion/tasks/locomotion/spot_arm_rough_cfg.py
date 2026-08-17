"""Velocity-tracking env for Spot with Arm.

Round 10: round-9 walk + rough terrain curriculum from level 0.
"""

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils.configclass import configclass

from .spot_arm_env_cfg import SpotArmEnvCfg


@configclass
class SpotArmRoughEnvCfg(SpotArmEnvCfg):
    """Velocity tracking. Round 10: rough generator, curriculum from 0."""

    def __post_init__(self):
        super().__post_init__()

        self.rewards.dof_torques_l2.weight = -0.0001
        self.rewards.dof_pos_limits.weight = -1.0
        self.rewards.track_lin_vel_xy_dot.weight = 1.5
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.feet_air_time.weight = 1.5
        self.rewards.foot_clearance.weight = 2.0
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.arm_joint_deviation.weight = -1.0
        self.rewards.flat_orientation_l2.weight = -1.0

        # Round 9 walks on a plane. Round 1 died on stairs before it could
        # stand — start the Lab 10×20 grid at level 0 and let curriculum climb.
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG.replace(
            num_rows=10, num_cols=20, curriculum=True
        )
        self.scene.terrain.max_init_terrain_level = 0

        # Round 1/2a: random spawn vel + joint scale knocked the robot over in
        # <0.3 s even after PD held a zero-action stand. Keep the default pose.
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params["pose_range"] = {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (0.0, 0.0)}
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.push_robot = None
        self.events.add_base_mass = None
        if getattr(self.events, "base_com", None) is not None:
            self.events.base_com = None
        self.commands.base_velocity.rel_standing_envs = 0.1
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)
        self.commands.base_velocity.vel_xy_success_threshold = 0.35


@configclass
class SpotArmRoughEnvCfg_PLAY(SpotArmRoughEnvCfg):
    """Plane PLAY — fewer envs, no domain randomization."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 16
        self.scene.env_spacing = 8.0
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
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
