"""run the agent loop and record each step."""

from dataclasses import dataclass, field

import planner
import sensor
from belief import BeliefGrid
from world import World


@dataclass
class StepRecord:
    """one loop snapshot."""

    position: tuple[int, int]
    planned_path: list[tuple[int, int]] | None
    newly_revealed: int
    replanned: bool


@dataclass
class EpisodeResult:
    outcome: str  # "success", "stuck", or "timeout"
    path: list[tuple[int, int]] = field(default_factory=list)
    records: list[StepRecord] = field(default_factory=list)
    replans: int = 0
    cells_revealed: int = 0
    fraction_known: float = 0.0

    @property
    def steps(self) -> int:
        return len(self.path) - 1  # moves, not visited cells


def run_episode(
    world: World,
    sensor_radius: int = 2,
    lam: float = 0.0,
    max_steps: int = 5000,
) -> EpisodeResult:
    """run an episode and return its trace.

    a replan counts only when the new path differs from the unused old path.
    ``max_steps`` limits moves.
    """
    if sensor_radius < 1:
        raise ValueError("sensor_radius must be at least 1")
    if max_steps < 0:
        raise ValueError("max_steps must be nonnegative")

    belief = BeliefGrid(world.shape)
    pos = world.start
    path_taken = [pos]
    records: list[StepRecord] = []
    previous_plan: list[tuple[int, int]] | None = None
    replans = 0
    cells_revealed = 0

    for step_index in range(max_steps + 1):
        # only the sensor sees ground truth
        obs = sensor.sense(world, pos, sensor_radius)
        newly_revealed = belief.update(obs)
        cells_revealed += newly_revealed

        # plan optimistically over the belief
        current_plan = planner.plan(belief, pos, world.goal, lam=lam)

        # count only real route changes
        replanned = False
        if previous_plan is not None and current_plan is not None:
            expected_rest = previous_plan[1:]
            if current_plan != expected_rest:
                replanned = True
                replans += 1

        records.append(StepRecord(pos, current_plan, newly_revealed, replanned))

        if current_plan is None:
            # no path even if all fog is free
            return EpisodeResult(
                "stuck",
                path_taken,
                records,
                replans,
                cells_revealed,
                belief.fraction_known(),
            )

        if pos == world.goal:
            return EpisodeResult(
                "success",
                path_taken,
                records,
                replans,
                cells_revealed,
                belief.fraction_known(),
            )

        if step_index == max_steps:
            break

        # take one move, then check again
        pos = current_plan[1]
        path_taken.append(pos)
        previous_plan = current_plan

    return EpisodeResult(
        "timeout",
        path_taken,
        records,
        replans,
        cells_revealed,
        belief.fraction_known(),
    )
