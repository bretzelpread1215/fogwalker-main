"""hidden grid world and map generators."""

from collections import deque
from dataclasses import dataclass

import numpy as np

FREE = 0
OBSTACLE = 1


@dataclass(frozen=True)
class World:
    """immutable obstacle grid with start and goal."""

    grid: np.ndarray  # 1 blocked, 0 free
    start: tuple[int, int]  # (row, col)
    goal: tuple[int, int]

    @property
    def shape(self) -> tuple[int, int]:
        return self.grid.shape

    def in_bounds(self, cell: tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < self.grid.shape[0] and 0 <= c < self.grid.shape[1]

    def is_obstacle(self, cell: tuple[int, int]) -> bool:
        return bool(self.grid[cell] == OBSTACLE)


def simple_map() -> World:
    """return a small map that forces a replan."""
    grid = np.zeros((10, 12), dtype=int)
    # vertical wall with a low gap
    grid[0:7, 6] = OBSTACLE
    # block the direct route
    grid[6, 3:7] = OBSTACLE
    return World(grid=grid, start=(2, 1), goal=(2, 10))


def _connected(grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> bool:
    """return whether the true grid connects start and goal."""
    if grid[start] == OBSTACLE or grid[goal] == OBSTACLE:
        return False
    rows, cols = grid.shape
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            return True
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (
                0 <= nr < rows
                and 0 <= nc < cols
                and grid[nr, nc] == FREE
                and (nr, nc) not in seen
            ):
                seen.add((nr, nc))
                queue.append((nr, nc))
    return False


def open_map(size: int = 25, seed: int = 0) -> World:
    """return a sparse, connected map."""
    if size < 8:
        raise ValueError("size must be at least 8")

    rng = np.random.default_rng(seed)
    start, goal = (1, 1), (size - 2, size - 2)
    for _ in range(200):  # retry until connected
        grid = np.zeros((size, size), dtype=int)
        for _ in range(size // 4):
            r, c = rng.integers(1, size - 1, size=2)
            length = int(rng.integers(3, size // 2))
            if rng.random() < 0.5:
                grid[r, c : min(c + length, size)] = OBSTACLE
            else:
                grid[r : min(r + length, size), c] = OBSTACLE
        grid[start] = grid[goal] = FREE
        if _connected(grid, start, goal):
            return World(grid, start, goal)
    raise RuntimeError("could not generate a connected open map")


def maze_map(size: int = 25, seed: int = 0) -> World:
    """return a connected backtracker maze.

    even sizes round up.
    """
    if size < 3:
        raise ValueError("size must be at least 3")

    size += (size + 1) % 2  # odd size keeps walls aligned
    rng = np.random.default_rng(seed)
    grid = np.ones((size, size), dtype=int)

    # carve odd-cell passages with dfs
    start_cell = (1, 1)
    grid[start_cell] = FREE
    stack = [start_cell]
    while stack:
        r, c = stack[-1]
        nbrs = [
            (nr, nc)
            for nr, nc in ((r - 2, c), (r + 2, c), (r, c - 2), (r, c + 2))
            if 0 < nr < size and 0 < nc < size and grid[nr, nc] == OBSTACLE
        ]
        if not nbrs:
            stack.pop()
            continue
        nr, nc = nbrs[rng.integers(len(nbrs))]
        grid[(r + nr) // 2, (c + nc) // 2] = FREE  # open the wall between
        grid[nr, nc] = FREE
        stack.append((nr, nc))

    # add alternate routes
    for _ in range(size // 3):
        r = int(rng.integers(1, size - 1))
        c = int(rng.integers(1, size - 1))
        grid[r, c] = FREE

    start, goal = (1, 1), (size - 2, size - 2)
    grid[goal] = FREE
    if not _connected(grid, start, goal):  # defensive check
        raise RuntimeError("maze generation failed")
    return World(grid, start, goal)


def cluttered_map(size: int = 25, seed: int = 0, density: float = 0.25) -> World:
    """return a connected random map at the given density."""
    if size < 3:
        raise ValueError("size must be at least 3")
    if not 0.0 <= density < 1.0:
        raise ValueError("density must be in [0, 1)")

    rng = np.random.default_rng(seed)
    start, goal = (1, 1), (size - 2, size - 2)
    for _ in range(500):
        grid = (rng.random((size, size)) < density).astype(int)
        grid[start] = grid[goal] = FREE
        if _connected(grid, start, goal):
            return World(grid, start, goal)
    raise RuntimeError("could not generate a connected cluttered map")


MAP_GENERATORS = {
    "simple": lambda size, seed: simple_map(),
    "open": open_map,
    "maze": maze_map,
    "clutter": cluttered_map,
}
