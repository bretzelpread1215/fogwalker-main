import pytest

from belief import BeliefGrid
from planner import plan


def test_plan_routes_around_known_walls():
    belief = BeliefGrid((3, 3))
    belief.update([((1, 1), True)])

    path = plan(belief, (1, 0), (1, 2))

    assert path is not None
    assert path[0] == (1, 0)
    assert path[-1] == (1, 2)
    assert (1, 1) not in path


def test_plan_uses_unknown_cells_but_rejects_a_known_barrier():
    belief = BeliefGrid((3, 3))

    assert plan(belief, (0, 0), (2, 0)) is not None

    belief.update([((1, col), True) for col in range(3)])

    assert plan(belief, (0, 0), (2, 0)) is None


@pytest.mark.parametrize("lam", [-0.1, float("inf"), float("nan")])
def test_plan_rejects_invalid_risk(lam):
    with pytest.raises(ValueError):
        plan(BeliefGrid((2, 2)), (0, 0), (1, 1), lam=lam)
