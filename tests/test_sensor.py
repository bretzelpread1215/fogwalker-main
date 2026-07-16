import numpy as np
import pytest

from sensor import sense
from world import World


def test_sensor_rejects_invalid_inputs():
    world = World(np.zeros((2, 2), dtype=int), (0, 0), (1, 1))

    with pytest.raises(ValueError, match="radius"):
        sense(world, (0, 0), -1)
    with pytest.raises(ValueError, match="bounds"):
        sense(world, (2, 0), 1)
