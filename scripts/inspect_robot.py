#!/usr/bin/env python3
"""Spawn one Spot-with-Arm on a flat plane: standing, feet on the ground, arm stowed.

Specs used here (Boston Dynamics public Spot / Spot Arm + this URDF)
-------------------------------------------------------------------
Spot (quadruped, BD published + this URDF)
  Mass:            ~32.7 kg base (URDF body link 32.86 kg; arm adds ~6 kg)
  Standing height: ~0.84 m overall; sitting ~0.51 m
  Length / width:  1.10 m / 0.50 m
  Legs:            12 DoF — hip_x, hip_y, knee × 4
  Foot:            sphere r=0.036 m at (0, 0, -0.3365) in each *_lower_leg
  Stand pose:      hip_y=0.8 rad, knee=-1.5 rad (legged_gym SpotWithArmRoughCfg)
  Body z so the foot sphere sits on z=0: **0.54 m**
                   (FK: foot center is 0.499 m below body origin; +0.036 m radius)

Spot Arm (6 DoF + gripper), mount on body at xyz=(0.292, 0, 0.188)
  Joint ranges (BD IFU / URDF, rad):
    sh0 yaw    [-2.62,  3.14]   0 = sagittal, arm in the body mid-plane
    sh1 pitch  [-3.14,  0.52]   0 = stretched FORWARD along +X
                                negative = pitch up and BACK over the body
    el0 elbow  [ 0.00,  3.14]   0 = straight; larger = folded
    el1 twist  [-2.79,  2.79]
    wr0 pitch  [-1.83,  1.83]
    wr1 twist  [-2.88,  2.88]
    f1x gripper[-1.57,  0.00]   0 = closed, -1.57 = open

  BD named stow (firmware, not published as numbers):
    Shoulder + elbow fully folded, arm centered on the back, elbow toward the
    rear, gripper overhanging the front. Ready-to-walk pose.
  Numeric stow used here (stay inside limits; avoid ±π which USD can wrap to 0):
      sh0=0.00  sh1=-3.12  el0=3.12  el1=0  wr0=0  wr1=0  f1x=0.00 (closed)

This script overwrites root + joint state every physics step (so soft leg PD
does not sag the feet into the plane) and snaps the lowest foot sphere onto z=0.

Example::

    source scripts/isaaclab_env.sh
    python scripts/inspect_robot.py --viz kit
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Inspect Spot-with-Arm on a flat plane (1 robot).")
parser.add_argument(
    "--pd-hold",
    action="store_true",
    help="Do not overwrite joints every step. Hold the default pose with PD only (stand test).",
)
parser.add_argument("--duration", type=float, default=0.0, help="Exit after this many seconds (0 = until window close).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply

from spot_arm_locomotion.robots.spot_with_arm import ARM_STOW_JOINT_POS, SPOT_WITH_ARM_CFG

FOOT_IN_LOWER_LEG = torch.tensor([0.0, 0.0, -0.3365])
FOOT_RADIUS = 0.036


@configclass
class InspectSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    robot: ArticulationCfg = SPOT_WITH_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def hold_pose(robot, root_pose: torch.Tensor) -> None:
    q = robot.data.default_joint_pos.torch.clone()
    dq = robot.data.default_joint_vel.torch.clone()
    robot.write_root_pose_to_sim_index(root_pose=root_pose)
    robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros_like(robot.data.root_vel_w.torch))
    robot.write_joint_position_to_sim_index(position=q)
    robot.write_joint_velocity_to_sim_index(velocity=dq)
    robot.set_joint_position_target_index(target=q)
    robot.set_joint_velocity_target_index(target=dq)
    robot.write_data_to_sim()


def snap_root_so_feet_on_ground(robot, root_pose: torch.Tensor) -> torch.Tensor:
    names = list(robot.body_names)
    ids = [i for i, name in enumerate(names) if "lower_leg" in name]
    if not ids:
        print("[WARN] No *_lower_leg bodies; skip foot snap.")
        return root_pose
    pos = robot.data.body_pos_w.torch[0, ids]
    quat = robot.data.body_quat_w.torch[0, ids]
    local = FOOT_IN_LOWER_LEG.to(device=pos.device, dtype=pos.dtype).expand(len(ids), 3)
    foot_w = pos + quat_apply(quat, local)
    min_z = float(foot_w[:, 2].min())
    dz = FOOT_RADIUS - min_z
    root_pose = root_pose.clone()
    root_pose[0, 2] += dz
    print(
        f"[INFO] Foot-sphere min z before snap={min_z:.3f} m; "
        f"lifted root by {dz:.3f} m (root z now {float(root_pose[0, 2]):.3f})."
    )
    return root_pose


def print_state(robot) -> None:
    names = list(robot.joint_names)
    q = robot.data.joint_pos.torch[0]
    print("[INFO] Joint positions (rad):")
    for name, value in zip(names, q.tolist()):
        marker = "  <- stow" if name in ARM_STOW_JOINT_POS else ""
        print(f"    {name:24s} {value: .4f}{marker}")
    if hasattr(robot.data, "joint_stiffness"):
        kp = robot.data.joint_stiffness.torch[0]
        print("[INFO] Joint stiffness:", {n: float(k) for n, k in zip(names, kp.tolist())})
    bodies = list(robot.body_names)
    pos = robot.data.body_pos_w.torch[0]
    for i, name in enumerate(bodies):
        if "lower_leg" in name or name.endswith("body") or "arm_link" in name:
            x, y, z = pos[i].tolist()
            print(f"[INFO] body {name:28s}  z={z: .3f}  xy=({x: .3f}, {y: .3f})")


def apply_pd_targets(robot, scene: InteractiveScene) -> None:
    q = robot.data.default_joint_pos.torch.clone()
    dq = robot.data.default_joint_vel.torch.clone()
    robot.set_joint_position_target_index(target=q)
    robot.set_joint_velocity_target_index(target=dq)
    scene.write_data_to_sim()


def projected_gravity_xy(robot) -> tuple[float, float, float]:
    """Return (gx, gy, |g_xy|) of projected gravity in the body frame, env 0."""
    g = robot.data.projected_gravity_b.torch[0]
    gx, gy, gz = (float(g[0]), float(g[1]), float(g[2]))
    return gx, gy, (gx * gx + gy * gy) ** 0.5


def run_simulator(sim: SimulationContext, scene: InteractiveScene, root_pose: torch.Tensor) -> int:
    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()
    printed = False
    elapsed = 0.0
    next_log = 1.0
    max_tilt = 0.0
    min_z = float("inf")
    while simulation_app.is_running():
        if args_cli.pd_hold:
            apply_pd_targets(robot, scene)
        else:
            hold_pose(robot, root_pose)
        if not printed:
            print_state(robot)
            printed = True
        sim.step()
        scene.update(sim_dt)
        elapsed += sim_dt
        if args_cli.pd_hold:
            _, _, gxy = projected_gravity_xy(robot)
            max_tilt = max(max_tilt, gxy)
            z = float(robot.data.root_pos_w.torch[0, 2])
            min_z = min(min_z, z)
            if elapsed >= next_log:
                print(
                    f"[PD-HOLD] t={elapsed:.1f}s  body_z={z:.3f} m  "
                    f"|g_xy|={gxy:.3f} (0=upright, 1=sideways)"
                )
                next_log += 1.0
        if args_cli.duration > 0 and elapsed >= args_cli.duration:
            break

    if args_cli.pd_hold and args_cli.duration > 0:
        print(f"[PD-HOLD] done. min body_z={min_z:.3f} m  max |g_xy|={max_tilt:.3f}")
        # Fail if it tipped past ~35 deg (|g_xy|~0.57) or body dropped below 0.35 m
        # (standing spawn is ~0.52 m; 0.35 m is still on the feet, not the belly).
        if max_tilt > 0.57 or min_z < 0.35:
            print("[PD-HOLD] FAIL: pose not held by PD.")
            return 1
        print("[PD-HOLD] PASS: default pose held by PD.")
        return 0
    return 0


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Side-on, low camera so feet-on-ground and folded arm are both visible.
    sim.set_camera_view([1.6, 1.7, 0.45], [0.05, 0.0, 0.28])

    scene = InteractiveScene(InspectSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()

    robot = scene["robot"]
    root_pose = robot.data.default_root_pose.torch.clone()
    root_pose[:, :3] += scene.env_origins
    hold_pose(robot, root_pose)
    sim.step()
    scene.update(sim.get_physics_dt())
    root_pose = snap_root_so_feet_on_ground(robot, root_pose)
    hold_pose(robot, root_pose)

    print("[INFO] Inspect: standing Spot-with-Arm, arm stow sh1=-3.12 el0=3.12 gripper closed, feet on plane.")
    if args_cli.pd_hold:
        print("[INFO] PD-hold: joints are NOT overwritten each step.")
    print("[INFO] Close the Isaac Sim window to exit.")
    rc = run_simulator(sim, scene, root_pose)
    simulation_app.close()
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
