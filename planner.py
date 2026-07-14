"""a* planning over belief.

unknown cells stay traversable. ``lam`` adds probability-weighted risk.
"""

import heapq
import itertools
import math

from belief import BeliefGrid

# 4-way motion keeps manhattan exact
MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    """return the 4-way grid distance."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def plan(
    belief: BeliefGrid,
    start: tuple[int, int],
    goal: tuple[int, int],
    lam: float = 0.0,
) -> list[tuple[int, int]] | None:
    """find a path over the current belief.

    returns ``[start, ..., goal]``, or ``None`` if unreachable even with
    unknown cells open.
    """
    if not math.isfinite(lam) or lam < 0:
        raise ValueError("lam must be finite and nonnegative")

    rows, cols = belief.log_odds.shape
    p = belief.p_obstacle()
    blocked = belief.known_obstacle()

    if blocked[goal]:
        return None

    # counter keeps heap ties stable and comparable
    counter = itertools.count()
    frontier: list[tuple[float, int, tuple[int, int]]] = []
    heapq.heappush(frontier, (manhattan(start, goal), next(counter), start))

    g_cost: dict[tuple[int, int], float] = {start: 0.0}
    came_from: dict[tuple[int, int], tuple[int, int]] = {}

    while frontier:
        _, _, current = heapq.heappop(frontier)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for dr, dc in MOVES:
            nbr = (current[0] + dr, current[1] + dc)
            if not (0 <= nbr[0] < rows and 0 <= nbr[1] < cols):
                continue
            if blocked[nbr]:
                # fog stays traversable
                continue

            step = 1.0 + lam * p[nbr]

            tentative_g = g_cost[current] + step
            if tentative_g < g_cost.get(nbr, float("inf")):
                g_cost[nbr] = tentative_g
                came_from[nbr] = current
                f = tentative_g + manhattan(nbr, goal)
                heapq.heappush(frontier, (f, next(counter), nbr))

    return None
