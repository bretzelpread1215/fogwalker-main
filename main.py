"""cli entry point."""

import argparse
import math
import os

import agent
import stats
import viz
import world


def run_one(
    map_name: str, size: int, seed: int, radius: int, lam: float, gif: str | None
) -> stats.EpisodeStats:
    w = world.MAP_GENERATORS[map_name](size, seed)
    result = agent.run_episode(w, sensor_radius=radius, lam=lam)
    if gif:
        os.makedirs(os.path.dirname(gif) or ".", exist_ok=True)
        viz.save_episode_gif(w, result, gif, sensor_radius=radius)
        print(f"wrote {gif}")
    return stats.summarize(map_name, seed, w, result)


def main() -> None:
    p = argparse.ArgumentParser(description="Belief-state gridworld navigation")
    p.add_argument("--map", choices=list(world.MAP_GENERATORS), default="simple")
    p.add_argument(
        "--size", type=int, default=25, help="grid side length; ignored by simple"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--radius", type=int, default=2, help="sensor radius")
    p.add_argument(
        "--lam",
        type=float,
        default=0.0,
        help="risk aversion: unknown-cell cost = 1 + lam * P(obstacle)",
    )
    p.add_argument("--gif", default=None, help="write episode GIF to this path")
    p.add_argument(
        "--all",
        action="store_true",
        help="run benchmark maps across several seeds; print stats table",
    )
    args = p.parse_args()

    if args.radius < 1:
        p.error("--radius must be at least 1")
    if not math.isfinite(args.lam) or args.lam < 0:
        p.error("--lam must be finite and nonnegative")
    if args.all or args.map == "open":
        if args.size < 8:
            p.error("--size must be at least 8 for open maps")
    elif args.map != "simple" and args.size < 3:
        p.error("--size must be at least 3")

    if args.all:
        rows = [
            run_one(name, args.size, seed, args.radius, args.lam, gif=None)
            for name in ("open", "maze", "clutter")
            for seed in range(3)
        ]
        print(stats.format_table(rows))
        return

    s = run_one(args.map, args.size, args.seed, args.radius, args.lam, args.gif)
    print(stats.format_table([s]))


if __name__ == "__main__":
    main()
