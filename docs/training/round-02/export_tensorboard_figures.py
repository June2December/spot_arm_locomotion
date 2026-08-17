#!/usr/bin/env python3
"""Export TensorBoard scalars from round-1 training into PNG figures + JSON."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parent
RUN = Path(
    "/home/june/isaac_projects/spot_arm_locomotion/logs/rsl_rl/rough_spot_with_arm/2026-08-15_13-54-47"
)
FIG = ROOT / "figures"
DATA = ROOT / "data"
DT_POLICY = 0.02  # sim.dt 0.005 * decimation 4


def load_scalars(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    ea = EventAccumulator(str(path), size_guidance={"scalars": 0})
    ea.Reload()
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for tag in ea.Tags().get("scalars", []):
        evs = ea.Scalars(tag)
        out[tag] = (
            np.array([e.step for e in evs], dtype=np.int32),
            np.array([e.value for e in evs], dtype=np.float64),
        )
    return out


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 140,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def downsample(y: np.ndarray, n: int = 60) -> list[float]:
    if len(y) <= n:
        return [float(v) for v in y]
    idx = np.linspace(0, len(y) - 1, n).round().astype(int)
    return [float(y[i]) for i in idx]


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    style()
    s = load_scalars(RUN)

    def y(tag: str) -> np.ndarray:
        return s[tag][1]

    def x(tag: str) -> np.ndarray:
        return s[tag][0]

    # --- dashboard ---
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2))
    fig.suptitle("Round 1 PPO — 2026-08-15_13-54-47  (1500 iter, 2048 envs)", fontsize=13)

    ax = axes[0, 0]
    ax.plot(x("Train/mean_episode_length"), y("Train/mean_episode_length") * DT_POLICY, color="#1d4ed8")
    ax.axhline(20.0, color="#9ca3af", ls="--", lw=1, label="timeout 20 s")
    ax.set_title("Mean episode length")
    ax.set_xlabel("iteration")
    ax.set_ylabel("seconds (policy dt=20 ms)")
    ax.legend(loc="upper right")

    ax = axes[0, 1]
    ax.plot(x("Episode_Termination/bad_orientation"), y("Episode_Termination/bad_orientation"), label="bad_orientation", color="#b91c1c")
    ax.plot(x("Episode_Termination/base_contact"), y("Episode_Termination/base_contact"), label="base_contact", color="#c2410c")
    ax.plot(x("Episode_Termination/time_out"), y("Episode_Termination/time_out"), label="time_out", color="#15803d")
    ax.set_title("Termination mix")
    ax.set_xlabel("iteration")
    ax.set_ylabel("fraction of dones")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="center right")

    ax = axes[0, 2]
    ax.plot(x("Train/mean_reward"), y("Train/mean_reward"), color="#111827")
    ax.axhline(0.0, color="#9ca3af", ls=":", lw=1)
    ax.set_title("Mean episode reward")
    ax.set_xlabel("iteration")
    ax.set_ylabel("return")

    ax = axes[1, 0]
    ax.plot(x("Episode_Reward/track_lin_vel_xy_exp"), y("Episode_Reward/track_lin_vel_xy_exp"), label="lin_xy", color="#1d4ed8")
    ax.plot(x("Episode_Reward/track_ang_vel_z_exp"), y("Episode_Reward/track_ang_vel_z_exp"), label="ang_z", color="#7c3aed")
    ax.set_title("Velocity-tracking rewards")
    ax.set_xlabel("iteration")
    ax.set_ylabel("episode mean")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(x("Curriculum/terrain_levels"), y("Curriculum/terrain_levels"), color="#0f766e")
    ax.set_title("Terrain curriculum level")
    ax.set_xlabel("iteration")
    ax.set_ylabel("mean level")

    ax = axes[1, 2]
    ax.plot(x("Policy/mean_std"), y("Policy/mean_std"), color="#374151")
    ax.set_title("Policy action std")
    ax.set_xlabel("iteration")
    ax.set_ylabel("mean std")

    fig.tight_layout()
    fig.savefig(FIG / "00-dashboard.png")
    plt.close(fig)

    # --- individual figures ---
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    steps = y("Train/mean_episode_length")
    ax.plot(x("Train/mean_episode_length"), steps, label="env steps", color="#1d4ed8")
    ax.plot(x("Train/mean_episode_length"), steps * DT_POLICY, label="seconds", color="#b91c1c")
    ax.set_title("Mean episode length never reaches timeout")
    ax.set_xlabel("iteration")
    ax.set_ylabel("length")
    ax.legend()
    fig.savefig(FIG / "01-episode-length.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.stackplot(
        x("Episode_Termination/bad_orientation"),
        y("Episode_Termination/bad_orientation"),
        y("Episode_Termination/base_contact"),
        y("Episode_Termination/time_out"),
        labels=["bad_orientation", "base_contact", "time_out"],
        colors=["#fecaca", "#fed7aa", "#bbf7d0"],
    )
    ax.plot(x("Episode_Termination/bad_orientation"), y("Episode_Termination/bad_orientation"), color="#b91c1c", lw=1.2)
    ax.set_title("Almost every episode ends by tipping (bad_orientation)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("fraction of terminations")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="center right")
    fig.savefig(FIG / "02-terminations.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.plot(x("Train/mean_reward"), y("Train/mean_reward"), color="#111827")
    ax.axhline(0.0, color="#9ca3af", ls=":")
    ax.set_title("Mean reward stays negative — no walking return")
    ax.set_xlabel("iteration")
    ax.set_ylabel("mean episode reward")
    fig.savefig(FIG / "03-mean-reward.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    reward_tags = [
        "Episode_Reward/track_lin_vel_xy_exp",
        "Episode_Reward/track_ang_vel_z_exp",
        "Episode_Reward/feet_air_time",
        "Episode_Reward/lin_vel_z_l2",
        "Episode_Reward/ang_vel_xy_l2",
        "Episode_Reward/dof_torques_l2",
        "Episode_Reward/action_rate_l2",
        "Episode_Reward/undesired_contacts",
        "Episode_Reward/flat_orientation_l2",
    ]
    for tag in reward_tags:
        ax.plot(x(tag), y(tag), label=tag.split("/")[-1], lw=1.1)
    ax.axhline(0.0, color="#9ca3af", ls=":", lw=1)
    ax.set_title("Per-term episode rewards (flat_orientation is identically 0)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("episode mean")
    ax.legend(ncol=2, fontsize=8)
    fig.savefig(FIG / "04-reward-terms.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.plot(x("Metrics/base_velocity/error_vel_xy"), y("Metrics/base_velocity/error_vel_xy"), label="xy vel error", color="#1d4ed8")
    ax.plot(x("Metrics/base_velocity/error_vel_yaw"), y("Metrics/base_velocity/error_vel_yaw"), label="yaw vel error", color="#7c3aed")
    ax.set_title("Command tracking error (xy got worse)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("error")
    ax.legend()
    fig.savefig(FIG / "05-velocity-error.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.plot(x("Curriculum/terrain_levels"), y("Curriculum/terrain_levels"), color="#0f766e")
    ax.set_title("Curriculum collapsed to level 0 (robots never walked far enough)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("mean terrain level")
    fig.savefig(FIG / "06-curriculum.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.plot(x("Policy/mean_std"), y("Policy/mean_std"), color="#374151")
    ax.set_title("Action std shrank while still failing (collapsed to a bad pose)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("mean std")
    fig.savefig(FIG / "07-policy-std.png")
    plt.close(fig)

    last = {k: float(v[1][-1]) for k, v in s.items()}
    first = {k: float(v[1][0]) for k, v in s.items()}
    metrics = {
        "run": str(RUN),
        "iterations": int(x("Train/mean_reward")[-1]) + 1,
        "policy_dt_s": DT_POLICY,
        "episode_timeout_s": 20.0,
        "first": first,
        "last": last,
        "highlights": {
            "episode_length_steps_last": last["Train/mean_episode_length"],
            "episode_length_s_last": last["Train/mean_episode_length"] * DT_POLICY,
            "bad_orientation_last": last["Episode_Termination/bad_orientation"],
            "time_out_last": last["Episode_Termination/time_out"],
            "mean_reward_last": last["Train/mean_reward"],
            "track_lin_vel_last": last["Episode_Reward/track_lin_vel_xy_exp"],
            "terrain_level_last": last["Curriculum/terrain_levels"],
            "success_rate_last": last["Metrics/success_rate"],
        },
        "canvas_series": {
            "iteration": downsample(x("Train/mean_reward"), 50),
            "episode_length_s": downsample(y("Train/mean_episode_length") * DT_POLICY, 50),
            "mean_reward": downsample(y("Train/mean_reward"), 50),
            "bad_orientation": downsample(y("Episode_Termination/bad_orientation"), 50),
            "base_contact": downsample(y("Episode_Termination/base_contact"), 50),
            "time_out": downsample(y("Episode_Termination/time_out"), 50),
            "track_lin_xy": downsample(y("Episode_Reward/track_lin_vel_xy_exp"), 50),
            "terrain_level": downsample(y("Curriculum/terrain_levels"), 50),
            "mean_std": downsample(y("Policy/mean_std"), 50),
        },
    }
    (DATA / "round1_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("wrote figures to", FIG)
    print("episode_length_s_last", metrics["highlights"]["episode_length_s_last"])
    print("bad_orientation_last", metrics["highlights"]["bad_orientation_last"])


if __name__ == "__main__":
    main()
