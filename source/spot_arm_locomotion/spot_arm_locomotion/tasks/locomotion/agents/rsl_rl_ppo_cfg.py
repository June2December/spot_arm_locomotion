"""RSL-RL PPO runner configuration for Spot with Arm locomotion."""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class SpotArmRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """On-policy PPO settings aligned with ``SpotWithArmRoughCfgPPO``."""

    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "rough_spot_with_arm"
    store_code_state = False
    # Round 13 clip=1.0 locked the knee at default ±0.4 rad (ㄱ pose) and,
    # because entropy is computed before the clip, log-std ran to 203.
    # 2.5 × scale 0.4 = ±1.0 rad, enough for a stride (12차 ROM 0.5–1.0)
    # while still cutting the |a|~6.5 teleport. Resume 7148, not 8647.
    clip_actions = 2.5
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        # log-space std cannot go negative (scalar std crashed training at iter ~351).
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.4, std_type="log"),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # Round 13: clip=1.0 made sampled actions all hit ±1, so entropy was
        # paid on unused Gaussian mass and log-std ran to 203. Clip is now 2.5
        # (outside 3σ of std 0.7) and this is halved as a backstop.
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
