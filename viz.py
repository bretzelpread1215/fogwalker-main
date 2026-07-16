"""render episode panels and gifs."""

import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter

import sensor
from agent import EpisodeResult
from belief import BeliefGrid
from world import World

C_PATH = "#d62728"
C_PLAN = "#1f77b4"
C_AGENT = "#ff7f0e"
C_GOAL = "#2ca02c"


def _setup_axes(fig):
    axes = fig.subplots(1, 3)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].set_title("true world (hidden)")
    axes[1].set_title("agent belief  P(obstacle)")
    axes[2].set_title("path + current plan")
    return axes


def _draw_frame(axes, world: World, belief: BeliefGrid, pos, path_so_far, plan):
    for ax in axes:
        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].set_title("true world (hidden)")
    axes[1].set_title("agent belief  P(obstacle)")
    axes[2].set_title("path + current plan")

    axes[0].imshow(world.grid, cmap="gray_r", vmin=0, vmax=1)

    axes[1].imshow(belief.p_obstacle(), cmap="gray_r", vmin=0, vmax=1)

    known_view = np.full(world.shape, 0.5)
    known = belief.known_mask()
    known_view[known] = belief.p_obstacle()[known].round()
    axes[2].imshow(known_view, cmap="gray_r", vmin=0, vmax=1)
    if len(path_so_far) > 1:
        rr, cc = zip(*path_so_far)
        axes[2].plot(cc, rr, color=C_PATH, lw=2, label="walked")
    if plan and len(plan) > 1:
        rr, cc = zip(*plan)
        axes[2].plot(cc, rr, color=C_PLAN, lw=1.5, ls="--", label="plan")
    axes[2].legend(loc="upper right", fontsize=7, framealpha=0.9)

    for ax in axes:
        ax.plot(world.goal[1], world.goal[0], "*", color=C_GOAL, ms=12)
        ax.plot(world.start[1], world.start[0], "s", color=C_PATH, ms=6, mfc="none")
        ax.plot(pos[1], pos[0], "o", color=C_AGENT, ms=8)


def save_episode_gif(
    world: World,
    result: EpisodeResult,
    out_path: str,
    sensor_radius: int = 2,
    fps: int = 8,
) -> None:
    """replay an episode into a gif.

    rebuilds belief from the deterministic sensor instead of snapshots.
    """
    belief = BeliefGrid(world.shape)
    fig = plt.figure(figsize=(12, 4.2), constrained_layout=True)
    axes = _setup_axes(fig)

    writer = PillowWriter(fps=fps)
    with writer.saving(fig, out_path, dpi=100):
        for i, rec in enumerate(result.records):
            belief.update(sensor.sense(world, rec.position, sensor_radius))
            _draw_frame(
                axes,
                world,
                belief,
                rec.position,
                result.path[: i + 1],
                rec.planned_path,
            )
            writer.grab_frame()
        # hold the ending
        for _ in range(fps):
            writer.grab_frame()
    plt.close(fig)
