# Spot with Arm — Isaac Lab locomotion RL

Isaac Lab manager-based velocity-tracking environment for **Boston Dynamics Spot with Arm**.
The policy commands the **12 leg joints**. The **7 arm joints** are held at the folded default pose
and are included in observations.

Gym ids:

- `Isaac-Velocity-Rough-Spot-Arm-v0` — training (rough terrain + curriculum)
- `Isaac-Velocity-Rough-Spot-Arm-Play-v0` — playback (50 envs, no randomization)

## Layout

```
spot_arm_locomotion/
  assets/spot_with_arm/          # symlink to legged_gym URDF + meshes
  scripts/train.py               # RSL-RL OnPolicyRunner training
  scripts/play.py                # load checkpoint and roll out
  source/spot_arm_locomotion/
    spot_arm_locomotion/
      robots/spot_with_arm.py    # ArticulationCfg (URDF spawn, 12+7 actuators)
      tasks/locomotion/
        spot_arm_env_cfg.py      # ManagerBasedRLEnv: scene, obs, action, events
        spot_arm_rough_cfg.py    # reward scales + terrain curriculum + PLAY
        agents/rsl_rl_ppo_cfg.py
```

## Install

Use the same Python interpreter as Isaac Lab:

```bash
# from this project root
/home/june/IsaacLab/isaaclab.sh -p -m pip install -e source/spot_arm_locomotion
```

The robot URDF is resolved from `assets/spot_with_arm` (symlink to
`/home/june/legged_gym/resources/robots/spot_with_arm`). Isaac Lab converts it to USD on first spawn
and caches the result under `assets/spot_with_arm/usd/`.

## Train

```bash
cd /home/june/isaac_projects/spot_arm_locomotion
/home/june/IsaacLab/isaaclab.sh -p scripts/train.py \
    --task Isaac-Velocity-Rough-Spot-Arm-v0 \
    --num_envs 4096 \
    --headless
```

Checkpoints land in `logs/rsl_rl/rough_spot_with_arm/<timestamp>/`.

## Play

```bash
/home/june/IsaacLab/isaaclab.sh -p scripts/play.py \
    --task Isaac-Velocity-Rough-Spot-Arm-Play-v0 \
    --num_envs 32
```

Pass `--checkpoint path/to/model_XXXX.pt` to load a specific run.

## Control / observation summary

| Channel | Dim | Source |
|---|---|---|
| Action | 12 | leg `hip_x`, `hip_y`, `knee` position targets, scale 0.25 |
| Policy obs | 3+3+3+3+19+19+12+height | base vel, gravity, command, all joints, last action, height scan |
| Arm | 7 | implicit PD at default pose (not in the action space) |

PD gains (legged_gym `SpotWithArmRoughCfg`): legs Kp=20, Kd=0.5. Arm is stiffer (Kp=80, Kd=2.0) so it stays folded.
