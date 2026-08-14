"""Rough-terrain reward scales and terrain curriculum for Spot with Arm."""

from isaaclab.utils.configclass import configclass

from .spot_arm_env_cfg import SpotArmEnvCfg


@configclass
class SpotArmRoughEnvCfg(SpotArmEnvCfg):
    """Rough-terrain velocity tracking with terrain-level curriculum.

    Reward weights follow ``legged_gym`` ``SpotWithArmRoughCfg`` (torques, joint
    limits) plus the inherited locomotion terms. Terrain curriculum is the
    standard Isaac Lab velocity generator (pyramid stairs, boxes, random rough,
    slopes) with increasing difficulty as the robot walks farther.
    """

    def __post_init__(self):
        super().__post_init__()

        # --- reward scales (SpotWithArmRoughCfg.rewards.scales) ---
        self.rewards.dof_torques_l2.weight = -0.0002
        # -10.0 matches the old Isaac Gym scale but overflows PPO std on this stack.
        self.rewards.dof_pos_limits.weight = -1.0
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 0.5
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.arm_joint_deviation.weight = -1.0
        self.rewards.flat_orientation_l2.weight = 0.0

        # --- terrain curriculum ---
        # Parent LocomotionVelocityRoughEnvCfg already enables terrain_levels when
        # the generator is present. Tighten the initial spawn level and keep the
        # 10x20 grid from ROUGH_TERRAINS_CFG (matches legged_gym num_rows/cols).
        self.scene.terrain.max_init_terrain_level = 5
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = True
            self.scene.terrain.terrain_generator.num_rows = 10
            self.scene.terrain.terrain_generator.num_cols = 20


@configclass
class SpotArmRoughEnvCfg_PLAY(SpotArmRoughEnvCfg):
    """Fewer environments, no domain randomization — used by ``play.py``."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 16
        self.scene.env_spacing = 8.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            # One robot per terrain cell. Origins are reused when num_envs > rows*cols
            # (that is why play with 32 envs on a 5x5 grid stacked robots in the same pit).
            n = max(self.scene.num_envs, 32)
            cols = 8
            rows = max(4, (n + cols - 1) // cols)
            self.scene.terrain.terrain_generator.num_rows = rows
            self.scene.terrain.terrain_generator.num_cols = cols
            self.scene.terrain.terrain_generator.curriculum = False

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.curriculum.terrain_levels = None
        # Standing spawn: no random yaw/velocity that knocks the robot over.
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
