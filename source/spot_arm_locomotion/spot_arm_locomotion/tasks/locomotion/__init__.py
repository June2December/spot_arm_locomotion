"""Register Gymnasium locomotion tasks for Spot with Arm."""
import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Velocity-Rough-Spot-Arm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.spot_arm_rough_cfg:SpotArmRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpotArmRoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Spot-Arm-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.spot_arm_rough_cfg:SpotArmRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpotArmRoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Spot-Arm-RoughTest-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.spot_arm_rough_test_cfg:SpotArmRoughTerrainPlayCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpotArmRoughPPORunnerCfg",
    },
)
