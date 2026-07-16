import numpy as np
import pytest

from agent import run_episode
from world import FREE, OBSTACLE, World, simple_map


def test_simple_episode_succeeds_and_stays_in_free_space():
    world = simple_map()

    result = run_episode(world)

    assert result.outcome == "success"
    assert result.path[0] == world.start
    assert result.path[-1] == world.goal
    assert all(world.grid[cell] == FREE for cell in result.path)
    assert (
        sum(record.newly_revealed for record in result.records) == result.cells_revealed
    )


def test_episode_records_one_legal_planned_move_per_step():
    world = simple_map()

    result = run_episode(world)

    assert [record.position for record in result.records] == result.path
    for index, (current, next_cell) in enumerate(zip(result.path, result.path[1:])):
        assert abs(current[0] - next_cell[0]) + abs(current[1] - next_cell[1]) == 1
        assert world.in_bounds(next_cell)
        assert world.grid[next_cell] == FREE
        assert result.records[index].planned_path is not None
        assert result.records[index].planned_path[:2] == [current, next_cell]


def test_repeated_episodes_start_with_fresh_state():
    world = simple_map()

    first = run_episode(world)
    second = run_episode(world)

    assert first == second


def test_goal_on_last_allowed_move_is_success():
    world = World(np.zeros((1, 2), dtype=int), (0, 0), (0, 1))

    result = run_episode(world, sensor_radius=1, max_steps=1)

    assert result.outcome == "success"
    assert result.steps == 1


def test_starting_on_goal_succeeds_without_moving():
    world = World(np.zeros((1, 1), dtype=int), (0, 0), (0, 0))

    result = run_episode(world, sensor_radius=1, max_steps=0)

    assert result.outcome == "success"
    assert result.path == [(0, 0)]
    assert result.steps == 0
    assert len(result.records) == 1


def test_timeout_preserves_the_last_planned_position():
    world = World(np.zeros((1, 4), dtype=int), (0, 0), (0, 3))

    result = run_episode(world, sensor_radius=1, max_steps=2)

    assert result.outcome == "timeout"
    assert result.path == [(0, 0), (0, 1), (0, 2)]
    assert [record.position for record in result.records] == result.path


def test_known_barrier_returns_stuck_without_moving():
    grid = np.zeros((3, 3), dtype=int)
    grid[1, :] = OBSTACLE
    world = World(grid, (0, 0), (2, 0))

    result = run_episode(world, sensor_radius=3)

    assert result.outcome == "stuck"
    assert result.path == [world.start]
    assert result.steps == 0
    assert result.records[0].planned_path is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sensor_radius": 0}, "sensor_radius"),
        ({"max_steps": -1}, "max_steps"),
    ],
)
def test_episode_rejects_invalid_limits(kwargs, message):
    with pytest.raises(ValueError, match=message):
        run_episode(simple_map(), **kwargs)
