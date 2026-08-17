"""Rough-terrain PLAY test for a policy trained on flat ground.

Not used for training. Spawns the round-9 (plane) policy on Isaac Lab's
rough generator so we can see whether velocity tracking survives stairs,
boxes, and noise. Expect falls — it never trained here.
"""

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils.configclass import configclass

from .spot_arm_rough_cfg import SpotArmRoughEnvCfg


@configclass
class SpotArmRoughTerrainPlayCfg(SpotArmRoughEnvCfg):
    """Same robot/commands as plane PLAY, terrain is the Lab rough generator."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5

        # Parent training cfg now uses a generator. Widen to the Lab/legged_gym
        # 10×20 grid (8 m patches → ~80×160 m plus 20 m border).
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
