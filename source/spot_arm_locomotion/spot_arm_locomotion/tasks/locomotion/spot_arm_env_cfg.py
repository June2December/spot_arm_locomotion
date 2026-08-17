"""Isaac Lab manager-based environment for Boston Dynamics Spot with Arm.

Policy actions command the 12 leg joints. Arm joints are held at the folded
default pose and included in observations so the policy can account for the
extra mass and kinematics.
"""

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab.utils.modifiers import ModifierCfg
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import EventsCfg, LocomotionVelocityRoughEnvCfg

from spot_arm_locomotion.robots.spot_with_arm import ARM_JOINT_NAMES, LEG_JOINT_NAMES, SPOT_WITH_ARM_CFG
from spot_arm_locomotion.tasks.locomotion import mdp as spot_arm_mdp

_SANITIZE = [ModifierCfg(func=spot_arm_mdp.replace_nonfinite)]

# Body-name regex used by contact rewards / terminations (URDF link names).
# Foot collision geometry lives on ``*_lower_leg`` (sphere at the foot).
FEET_BODY_NAMES = ".*_lower_leg"
PENALIZED_BODY_NAMES = [".*_upper_leg", ".*_hip", "arm_link_.*"]
TERMINATE_BODY_NAMES = ["body", "arm_link_.*"]


@configclass
class SpotArmActionsCfg:
    """12-DoF joint-position actions for the legs only."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINT_NAMES,
        scale=0.4,
        use_default_offset=True,
    )


@configclass
class SpotArmObservationsCfg:
    """Observations: base state, commands, all 19 joints (legs + arm), last 12 actions, height scan."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor observations (order is preserved)."""

        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1), modifiers=_SANITIZE, clip=(-100.0, 100.0)
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), modifiers=_SANITIZE, clip=(-100.0, 100.0)
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05), modifiers=_SANITIZE, clip=(-1.0, 1.0)
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        # All joints so the policy sees the (held) arm configuration.
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            modifiers=_SANITIZE,
            clip=(-10.0, 10.0),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            modifiers=_SANITIZE,
            clip=(-100.0, 100.0),
        )
        actions = ObsTerm(func=mdp.last_action, modifiers=_SANITIZE, clip=(-10.0, 10.0))
        height_scan = ObsTerm(
            func=spot_arm_mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            modifiers=_SANITIZE,
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged critic observations (same terms, no noise)."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, modifiers=_SANITIZE, clip=(-100.0, 100.0))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, modifiers=_SANITIZE, clip=(-100.0, 100.0))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, modifiers=_SANITIZE, clip=(-1.0, 1.0))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
            modifiers=_SANITIZE,
            clip=(-10.0, 10.0),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
            modifiers=_SANITIZE,
            clip=(-100.0, 100.0),
        )
        actions = ObsTerm(func=mdp.last_action, modifiers=_SANITIZE, clip=(-10.0, 10.0))
        height_scan = ObsTerm(
            func=spot_arm_mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            modifiers=_SANITIZE,
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class SpotArmRewardsCfg:
    """Reward terms mapped from ``legged_gym`` SpotWithArm / LeggedRobotCfg."""

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": 1.0}
    )
    # Standing is 0; any speed along the command is positive. Gives a gradient
    # at v=0 that the exponential kernel does not.
    track_lin_vel_xy_dot = RewTerm(
        func=spot_arm_mdp.track_lin_vel_xy_dot,
        weight=1.0,
        params={"command_name": "base_velocity"},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": 1.0}
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-0.0002,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    feet_air_time = RewTerm(
        func=spot_arm_mdp.feet_air_time,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_NAMES),
            "command_name": "base_velocity",
            "threshold": 0.1,
        },
    )
    # Round 5: body moves, feet_air_time stays 0. Only new term vs that run:
    # pay for foot height while a walk command is on. No slide, no friction change.
    foot_clearance = RewTerm(
        func=spot_arm_mdp.foot_clearance,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_NAMES),
            "command_name": "base_velocity",
            "target_height": 0.08,
        },
    )
    feet_slide = RewTerm(
        func=spot_arm_mdp.feet_slide,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_NAMES),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_NAMES),
            "body_speed_threshold": 0.4,
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=PENALIZED_BODY_NAMES),
            "threshold": 1.0,
        },
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)
    # Keep the unused arm near the folded default pose.
    arm_joint_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    # Round 1/2a: mean return rose because episodes got shorter. Dying must be worse
    # than staying up with small per-step penalties (H1/G1/Cassie use -200).
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)


@configclass
class SpotArmEventsCfg(EventsCfg):
    """Reset / randomization events. Arm joints are pinned to the folded pose."""

    reset_arm_joints = EventTerm(
        func=spot_arm_mdp.reset_arm_to_default,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)},
    )


@configclass
class SpotArmTerminationsCfg:
    """Episode termination: timeout, body/arm illegal contact."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=TERMINATE_BODY_NAMES),
            "threshold": 1.0,
        },
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})
    root_too_high = DoneTerm(func=spot_arm_mdp.root_height_above_maximum, params={"maximum_height": 5.0})
    root_too_fast = DoneTerm(func=spot_arm_mdp.root_lin_vel_too_large, params={"max_speed": 50.0})


@configclass
class SpotArmEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Manager-based velocity-tracking environment for Spot with Arm.

    Inherits terrain, commands, events, and simulation presets from
    :class:`LocomotionVelocityRoughEnvCfg`, then swaps in the Spot-with-Arm
    asset and 12-DoF leg actions.
    """

    observations: SpotArmObservationsCfg = SpotArmObservationsCfg()
    actions: SpotArmActionsCfg = SpotArmActionsCfg()
    rewards: SpotArmRewardsCfg = SpotArmRewardsCfg()
    terminations: SpotArmTerminationsCfg = SpotArmTerminationsCfg()
    events: SpotArmEventsCfg = SpotArmEventsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Match legged_gym: dt=0.005, decimation=4 → 50 Hz policy.
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation

        # Robot
        self.scene.robot = SPOT_WITH_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Current URDF importer nests the torso at Geometry/body.
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/Geometry/body"

        # Domain randomization / reset: Spot root link is "body".
        self.events.add_base_mass.params["asset_cfg"].body_names = "body"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "body"
        if self.events.base_com.default is not None:
            self.events.base_com.default.params["asset_cfg"].body_names = "body"

        # Randomize only the 12 leg joints on reset; arm is handled by reset_arm_joints.
        self.events.reset_robot_joints.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)
        self.events.reset_robot_joints.params["position_range"] = (0.9, 1.1)

        # Do not spawn with roll/pitch/z velocity — that dumps the robot on its side.
        self.events.reset_base.params["velocity_range"] = {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (-0.5, 0.5),
        }

        # Match legged_gym domain_rand.friction_range.
        self.events.physics_material.params["static_friction_range"] = (0.5, 1.25)
        self.events.physics_material.params["dynamic_friction_range"] = (0.5, 1.25)
