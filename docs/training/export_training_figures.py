#!/usr/bin/env python3
"""Export standard TensorBoard training figures for any round.

Produces the graphs that were missing from several round folders (reward
decomposition, PPO health, terminations) in a consistent layout.

Usage::

    python docs/training/export_training_figures.py \\
        logs/rsl_rl/rough_spot_with_arm/2026-08-27_10-20-22 \\
        --out docs/training/round-15 \\
        --baseline logs/rsl_rl/rough_spot_with_arm/2026-08-26_12-52-45 \\
        --title "Round 15"
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOGS = Path("/home/june/isaac_projects/spot_arm_locomotion/logs/rsl_rl/rough_spot_with_arm")
DT_POLICY = 0.02
C_BASE = "#9ca3af"
C_RUN = "#0f766e"


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
            "figure.dpi": 140,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def plot_series(ax, scalars: dict, tag: str, mult: float, color: str, label: str, offset: bool) -> bool:
    if tag not in scalars:
        return False
    step, val = scalars[tag]
    x = step - step[0] if offset else step
    ax.plot(x, val * mult, color=color, lw=1.2, label=label)
    return True


def reward_tags(scalars: dict) -> list[str]:
    return sorted(k for k in scalars if k.startswith("Episode_Reward/"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export standard training TensorBoard figures.")
    parser.add_argument("run_dir", type=str, nargs="?", default=None, help="TensorBoard run directory.")
    parser.add_argument("--out", type=str, required=True, help="Output folder (e.g. docs/training/round-15).")
    parser.add_argument("--baseline", type=str, default=None, help="Optional baseline run for overlay.")
    parser.add_argument("--title", type=str, default=None, help="Figure title prefix.")
    args = parser.parse_args()

    run = Path(args.run_dir) if args.run_dir else sorted(p for p in LOGS.iterdir() if p.is_dir())[-1]
    out = Path(args.out)
    fig_dir = out / "figures"
    data_dir = out / "data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    style()

    s_run = load_scalars(run)
    s_base = load_scalars(Path(args.baseline)) if args.baseline else None
    title = args.title or run.name
    pairs = [(s_base, C_BASE, "baseline")] if s_base else []
    pairs.append((s_run, C_RUN, "run"))

    # --- 00 dashboard ---
    panels = [
        ("Train/mean_episode_length", DT_POLICY, "Episode length (s)"),
        ("Episode_Termination/time_out", 1.0, "Termination: time_out"),
        ("Episode_Termination/base_contact", 1.0, "Termination: base_contact"),
        ("Curriculum/terrain_levels", 1.0, "Terrain level"),
        ("Policy/mean_std", 1.0, "Policy mean_std"),
        ("Metrics/base_velocity/error_vel_xy", 1.0, "error_vel_xy"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2))
    fig.suptitle(f"{title} — training dashboard ({run.name})", fontsize=12)
    for ax, (tag, mult, panel_title) in zip(axes.ravel(), panels):
        for s, color, name in pairs:
            plot_series(ax, s, tag, mult, color, name, offset=s_base is not None)
        ax.set_title(panel_title)
        ax.set_xlabel("iterations since resume" if s_base else "iteration")
        if tag == "Policy/mean_std":
            ax.set_ylim(0, max(3.0, float(s_run[tag][1].max()) * 1.1) if tag in s_run else 3.0)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "00-dashboard.png")
    plt.close(fig)

    # --- 01 all reward terms ---
    tags = reward_tags(s_run)
    if tags:
        n = len(tags)
        cols = 3
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(12.5, 2.6 * rows))
        fig.suptitle(f"{title} — episode reward terms (do not sum across rounds with different weights)", fontsize=11)
        for ax, tag in zip(axes.ravel(), tags):
            for s, color, name in pairs:
                plot_series(ax, s, tag, 1.0, color, name, offset=s_base is not None)
            ax.set_title(tag.split("/")[-1], fontsize=9)
            ax.axhline(0.0, color="#9ca3af", ls=":", lw=0.8)
            ax.set_xlabel("iter", fontsize=8)
        for ax in axes.ravel()[len(tags) :]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(fig_dir / "01-reward-terms.png")
        plt.close(fig)

    # --- 02 terminations stack ---
    term_tags = sorted(k for k in s_run if k.startswith("Episode_Termination/"))
    if term_tags:
        fig, ax = plt.subplots(figsize=(9.0, 4.5))
        x0 = s_run[term_tags[0]][0]
        if s_base:
            x0 = x0 - x0[0]
        stacks = [s_run[t][1] for t in term_tags]
        ax.stackplot(
            x0,
            *stacks,
            labels=[t.split("/")[-1] for t in term_tags],
            alpha=0.85,
        )
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{title} — termination mix")
        ax.set_xlabel("iterations since resume" if s_base else "iteration")
        ax.legend(loc="center right", fontsize=8)
        fig.savefig(fig_dir / "02-terminations.png")
        plt.close(fig)

    # --- 03 PPO health ---
    ppo_panels = [
        ("Policy/mean_std", 1.0, "mean_std (stop if >>2)"),
        ("Loss/entropy", 1.0, "entropy"),
        ("Loss/value", 1.0, "value loss"),
        ("Loss/surrogate", 1.0, "surrogate"),
        ("Loss/learning_rate", 1.0, "learning rate"),
        ("Train/mean_reward", 1.0, "mean_reward (cross-round invalid)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.0))
    fig.suptitle(f"{title} — PPO / optimization health", fontsize=12)
    for ax, (tag, mult, panel_title) in zip(axes.ravel(), ppo_panels):
        for s, color, name in pairs:
            plot_series(ax, s, tag, mult, color, name, offset=s_base is not None)
        ax.set_title(panel_title)
        ax.set_xlabel("iterations since resume" if s_base else "iteration")
    fig.tight_layout()
    fig.savefig(fig_dir / "03-ppo-health.png")
    plt.close(fig)

    # --- 04 tracking + gait ---
    extra = [
        ("Metrics/base_velocity/error_vel_xy", 1.0, "error_vel_xy"),
        ("Metrics/base_velocity/error_vel_yaw", 1.0, "error_vel_yaw"),
        ("Episode_Reward/gait", 1.0, "gait reward"),
        ("Episode_Reward/foot_clearance", 1.0, "foot_clearance"),
        ("Episode_Reward/leg_joint_deviation", 1.0, "leg_joint_deviation"),
        ("Episode_Reward/track_lin_vel_xy_dot", 1.0, "track_lin_vel_xy_dot"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.0))
    fig.suptitle(f"{title} — tracking & locomotion rewards", fontsize=12)
    for ax, (tag, mult, panel_title) in zip(axes.ravel(), extra):
        plotted = False
        for s, color, name in pairs:
            plotted |= plot_series(ax, s, tag, mult, color, name, offset=s_base is not None)
        if not plotted:
            ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(panel_title)
        ax.set_xlabel("iterations since resume" if s_base else "iteration")
    fig.tight_layout()
    fig.savefig(fig_dir / "04-tracking-gait.png")
    plt.close(fig)

    stem = re.sub(r"[^a-zA-Z0-9]+", "_", out.name).strip("_").lower()
    metrics = {
        "run": str(run),
        "baseline": str(args.baseline) if args.baseline else None,
        "iterations": int(s_run["Train/mean_reward"][0][-1]) + 1 if "Train/mean_reward" in s_run else 0,
        "first": {k: float(v[1][0]) for k, v in s_run.items()},
        "last": {k: float(v[1][-1]) for k, v in s_run.items()},
        "reward_tags": tags,
    }
    (data_dir / f"{stem}_training_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"wrote {fig_dir}  ({len(tags)} reward terms)")


if __name__ == "__main__":
    main()
