import numpy as np

from belief import BeliefGrid


def test_updates_reveal_each_cell_once():
    belief = BeliefGrid((2, 2))

    assert np.allclose(belief.p_obstacle(), 0.5)
    assert belief.update([((0, 0), False), ((1, 1), True)]) == 2
    assert belief.update([((0, 0), False), ((1, 1), True)]) == 0
    assert belief.is_known((0, 0))
    assert belief.is_known((1, 1))
    assert not belief.known_obstacle()[0, 0]
    assert belief.known_obstacle()[1, 1]
