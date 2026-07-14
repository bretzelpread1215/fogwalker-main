"""episode metrics and the full-map baseline."""

from collections import deque
from dataclasses import dataclass

from agent import EpisodeResult
from world import FREE, World


def optimal_path_length(world: World) -> int | None:
    """return the shortest full-map path, or ``None`` if unsolvable."""
    seen = {world.start: 0}
    queue = deque([world.start])
    while queue:
        cell = queue.popleft()
        if cell == world.goal:
            return seen[cell]
        r, c = cell
        for nbr in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if world.in_bounds(nbr) and world.grid[nbr] == FREE and nbr not in seen:
                seen[nbr] = seen[cell] + 1
                queue.append(nbr)
    return None


@dataclass
class EpisodeStats:
    map_name: str
    seed: int
    outcome: str
    steps: int
    optimal: int | None
    replans: int
    cells_revealed: int
    fraction_known: float

    @property
    def ratio(self) -> float | None:
        """return steps divided by the full-map optimum."""
        if self.optimal in (None, 0) or self.outcome != "success":
            return None
        return self.steps / self.optimal


def summarize(
    map_name: str, seed: int, world: World, result: EpisodeResult
) -> EpisodeStats:
    return EpisodeStats(
        map_name=map_name,
        seed=seed,
        outcome=result.outcome,
        steps=result.steps,
        optimal=optimal_path_length(world),
        replans=result.replans,
        cells_revealed=result.cells_revealed,
        fraction_known=result.fraction_known,
    )


def format_table(rows: list[EpisodeStats]) -> str:
    """format a fixed-width stats table."""
    header = (
        f"{'map':<9}{'seed':>5}{'outcome':>9}{'steps':>7}{'optimal':>9}"
        f"{'ratio':>7}{'replans':>9}{'revealed':>10}{'known':>7}"
    )
    lines = [header, "-" * len(header)]
    for s in rows:
        ratio = f"{s.ratio:.2f}" if s.ratio is not None else "-"
        optimal = str(s.optimal) if s.optimal is not None else "-"
        lines.append(
            f"{s.map_name:<9}{s.seed:>5}{s.outcome:>9}{s.steps:>7}{optimal:>9}"
            f"{ratio:>7}{s.replans:>9}{s.cells_revealed:>10}{s.fraction_known:>7.0%}"
        )
    return "\n".join(lines)
