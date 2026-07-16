import numpy as np
import pytest

from stats import optimal_path_length
from world import FREE, cluttered_map, maze_map, open_map


@pytest.mark.parametrize("factory", [open_map, maze_map, cluttered_map])
def test_generated_maps_are_reproducible_and_connected(factory):
    first = factory(size=15, seed=7)
    second = factory(size=15, seed=7)

    assert np.array_equal(first.grid, second.grid)
    assert first.grid[first.start] == FREE
    assert first.grid[first.goal] == FREE
    assert optimal_path_length(first) is not None


def test_generators_reject_invalid_parameters():
    with pytest.raises(ValueError, match="size"):
        open_map(size=7)
    with pytest.raises(ValueError, match="size"):
        maze_map(size=2)
    with pytest.raises(ValueError, match="density"):
        cluttered_map(density=1.0)
