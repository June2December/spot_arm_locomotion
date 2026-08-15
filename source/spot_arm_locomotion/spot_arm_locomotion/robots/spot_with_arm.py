"""Boston Dynamics Spot with Arm articulation configuration.

The policy commands the 12 leg joints. Arm joints (7 DoF) are held at the folded
default pose by a stiff implicit PD actuator and are included in observations.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

def _find_project_root() -> str:
    """Walk up from this file until ``assets/spot_with_arm`` is found."""
    here = os.path.abspath(os.path.dirname(__file__))
    for _ in range(8):
        if os.path.isfile(os.path.join(here, "assets", "spot_with_arm", "urdf", "spot_with_arm.urdf")):
            return here
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    raise FileNotFoundError(
        "Could not locate assets/spot_with_arm. Expected a project-root folder containing "
        "assets/spot_with_arm/urdf/spot_with_arm.urdf (symlink to legged_gym is fine)."
    )


_PROJECT_ROOT = _find_project_root()
SPOT_WITH_ARM_URDF_PATH = os.path.join(_PROJECT_ROOT, "assets", "spot_with_arm", "urdf", "spot_with_arm.urdf")
SPOT_WITH_ARM_USD_DIR = os.path.join(_PROJECT_ROOT, "assets", "spot_with_arm", "usd_lab")

# 12 locomotion joints (hip_x, hip_y, knee on each of 4 legs)
LEG_JOINT_NAMES = [
    ".*_hip_x",
    ".*_hip_y",
    ".*_knee",
]
# 7 arm / gripper joints held at the folded (stow) pose
ARM_JOINT_NAMES = ["arm_.*"]

# Stay a hair inside ±π: exactly ±π can wrap in USD back to the URDF zero (arm straight).
# BD stow: first link (hr0) flat on the deck pointing aft; elbow fully folded so the
# forearm lies on top of it pointing forward; gripper closed at the front.
#   sh1 → lower limit  -π   (horizontal back, parallel to the body top)
#   el0 → upper limit  +π   (180 deg fold, second link stacked on the first)
#   f1x → 0 closed (finger along the jaw); -1.57 = open
ARM_STOW_JOINT_POS = {
    "arm_sh0": 0.0,
    "arm_sh1": -3.12,  # ~-179 deg, first link flat along the back (limit -π)
    "arm_el0": 3.12,  # ~179 deg, second link folded onto the first (limit +π)
    "arm_el1": 0.0,
    "arm_wr0": 0.0,
    "arm_wr1": 0.0,
    "arm_f1x": 0.0,  # closed (0 = shut; -1.57 = open)
}


def _activate_nested_contact_sensors(root_prim) -> None:
    """Add PhysX contact reporters on nested rigid bodies.

    The current URDF importer nests child links under the torso. Isaac Lab's
    default ``activate_contact_sensors`` stops at the first RigidBodyAPI, so
    legs/arm would otherwise have no contact reports.
    """
    from pxr import UsdPhysics

    from isaaclab.sim.utils import safe_set_attribute_on_usd_prim

    stack = [root_prim]
    while stack:
        prim = stack.pop()
        stack.extend(prim.GetChildren())
        path = str(prim.GetPath())
        parent = prim.GetParent()
        # Skip materials and visual-mesh duplicates (Geometry/body/body) that are
        # not articulation links — tagging them floods PhysX "no rigid body" warnings.
        if "/Materials/" in path or (parent and prim.GetName() == parent.GetName()):
            continue
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        applied = prim.GetAppliedSchemas()
        if "PhysxRigidBodyAPI" not in applied:
            prim.AddAppliedSchema("PhysxRigidBodyAPI")
        if "PhysxContactReportAPI" not in applied:
            prim.AddAppliedSchema("PhysxContactReportAPI")
        safe_set_attribute_on_usd_prim(prim, "physxRigidBody:sleepThreshold", 0.0, camel_case=False)
        safe_set_attribute_on_usd_prim(prim, "physxContactReport:threshold", 0.0, camel_case=False)


def spawn_spot_with_arm(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn the URDF asset, then tag nested links for contact sensing.

    If a cached USD already exists, skip the URDF importer. A stale
    ``.asset_hash`` otherwise retriggers conversion, which needs
    ``newton_usd_schemas`` and currently fails in this Isaac Sim install.
    """
    usd_path = os.path.join(SPOT_WITH_ARM_USD_DIR, "spot_with_arm", "spot_with_arm.usda")
    skip_convert = os.path.isfile(usd_path) and not getattr(cfg, "force_usd_conversion", False)
    if skip_convert:
        from isaaclab.sim.converters.urdf_converter import UrdfConverter

        _orig_convert = UrdfConverter._convert_asset
        UrdfConverter._convert_asset = lambda self, _cfg: None
        try:
            prim = sim_utils.spawn_from_urdf(
                prim_path, cfg, translation=translation, orientation=orientation, **kwargs
            )
        finally:
            UrdfConverter._convert_asset = _orig_convert
    else:
        prim = sim_utils.spawn_from_urdf(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    _activate_nested_contact_sensors(prim)
    return prim


SPOT_WITH_ARM_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        func=spawn_spot_with_arm,
        asset_path=SPOT_WITH_ARM_URDF_PATH,
        usd_dir=SPOT_WITH_ARM_USD_DIR,
        usd_file_name="spot_with_arm.usd",
        fix_base=False,
        merge_fixed_joints=True,
        make_instanceable=True,
        force_usd_conversion=False,
        activate_contact_sensors=True,
        self_collision=False,
        robot_type="Quadruped",
        run_asset_transformer=False,
        run_multi_physics_conversion=False,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None),
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.54),
        joint_pos={
            # legs — standing pose from legged_gym SpotWithArmRoughCfg
            "front_left_hip_x": 0.0,
            "front_left_hip_y": 0.8,
            "front_left_knee": -1.5,
            "front_right_hip_x": 0.0,
            "front_right_hip_y": 0.8,
            "front_right_knee": -1.5,
            "rear_left_hip_x": 0.0,
            "rear_left_hip_y": 0.8,
            "rear_left_knee": -1.5,
            "rear_right_hip_x": 0.0,
            "rear_right_hip_y": 0.8,
            "rear_right_knee": -1.5,
            # arm — BD stow (folded onto the back). URDF zeros point the arm forward.
            **ARM_STOW_JOINT_POS,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=LEG_JOINT_NAMES,
            effort_limit_sim={
                ".*_hip_x": 45.0,
                ".*_hip_y": 45.0,
                ".*_knee": 115.0,
            },
            stiffness=20.0,
            damping=0.5,
        ),
        "arm": ImplicitActuatorCfg(
            joint_names_expr=ARM_JOINT_NAMES,
            effort_limit_sim={
                "arm_sh0": 90.9,
                "arm_sh1": 181.8,
                "arm_el0": 90.9,
                "arm_el1": 30.3,
                "arm_wr0": 30.3,
                "arm_wr1": 30.3,
                "arm_f1x": 15.32,
            },
            stiffness=120.0,
            damping=4.0,
        ),
    },
)
"""Spot with Arm: 12-DoF leg PD control, arm joints held at the folded default pose."""
