"""limited-radius ground-truth sensor."""

from world import World


def sense(
    world: World, pos: tuple[int, int], radius: int
) -> list[tuple[tuple[int, int], bool]]:
    """return ``(cell, is_obstacle)`` pairs within radius.

    sensing is perfect and unoccluded.
    """
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    if not world.in_bounds(pos):
        raise ValueError("pos must be in bounds")

    r0, c0 = pos
    observations = []
    for r in range(r0 - radius, r0 + radius + 1):
        for c in range(c0 - radius, c0 + radius + 1):
            if not world.in_bounds((r, c)):
                continue
            if (r - r0) ** 2 + (c - c0) ** 2 <= radius**2:
                observations.append(((r, c), world.is_obstacle((r, c))))
    return observations
