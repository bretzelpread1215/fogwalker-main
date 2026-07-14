"""probabilistic occupancy grid in log-odds.

unknown cells start at 0. observations add signed evidence and saturate.
"""

import numpy as np

# finite saturation avoids infinities
# one perfect hit marks a cell known
L_SAT = 10.0
L_HIT = 10.0


class BeliefGrid:
    """fixed-grid obstacle belief in log-odds."""

    def __init__(self, shape: tuple[int, int]):
        # 0 log-odds means 50/50
        self.log_odds = np.zeros(shape, dtype=float)

    def update(self, observations: list[tuple[tuple[int, int], bool]]) -> int:
        """add ``(cell, is_obstacle)`` observations.

        returns the number of newly known cells.
        """
        newly_revealed = 0
        for cell, is_obstacle in observations:
            was_known = self.is_known(cell)
            self.log_odds[cell] += L_HIT if is_obstacle else -L_HIT
            # keep repeated hits finite
            self.log_odds[cell] = np.clip(self.log_odds[cell], -L_SAT, L_SAT)
            if not was_known and self.is_known(cell):
                newly_revealed += 1
        return newly_revealed

    def p_obstacle(self) -> np.ndarray:
        """return obstacle probabilities."""
        return 1.0 / (1.0 + np.exp(-self.log_odds))

    def is_known(self, cell: tuple[int, int]) -> bool:
        """return whether a cell reached saturation."""
        return bool(abs(self.log_odds[cell]) >= L_SAT)

    def known_obstacle(self) -> np.ndarray:
        """return the known-obstacle mask."""
        return self.log_odds >= L_SAT

    def known_mask(self) -> np.ndarray:
        """return the known-cell mask."""
        return np.abs(self.log_odds) >= L_SAT

    def fraction_known(self) -> float:
        return float(self.known_mask().mean())
