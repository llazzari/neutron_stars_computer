from dataclasses import dataclass

from ..star.structure import Star
from ..star.tov_solver import Array


def phase_splitting_interface(
    transitional_pressure: float, r: float, y: Array
) -> float:
    """Event to compute the location of the interface inside the star."""
    return y[2] - transitional_pressure


@dataclass(slots=True)
class HybridStar(Star):
    core_radius: float = 0.0
    core_mass: float = 0.0
