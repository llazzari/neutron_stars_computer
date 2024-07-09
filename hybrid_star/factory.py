from typing import Any
from functools import partial
from dataclasses import dataclass, field

from ..star.structure import InternalProfiles, Star
from ..star.stability import RadOscInput, solve_rad_osc
from ..star.tov_solver import Array, TOVInput
from ..star.factory import StarFactory, StarStabilityFactory
from .stability import set_Lagragian_vars_at_interface
from .structure import HybridStar, phase_splitting_interface


@dataclass(slots=True)
class HybridStarFactory:
    """Creates a hybrid star using Maxwell's construction."""

    tov_input: TOVInput
    fac: StarFactory = field(init=False)

    def __post_init__(self) -> None:
        self.tov_input.events = (
            partial(
                phase_splitting_interface,
                self.tov_input.eos.transitional_pressure,  # type: ignore
            ),
        )
        self.fac = StarFactory(self.tov_input)

    def create_star(self, central_pressure: float) -> HybridStar:
        star: Star = self.fac.create_star(central_pressure)

        return HybridStar(
            central_pressure,
            star.radius,
            star.mass,
            star.mode,
            star.omega_squared,
            self.set_core_radius(self.fac.tov_solution),
            self.set_core_mass(self.fac.tov_solution),
        )

    def set_core_radius(self, tov_solution: Any) -> float:
        return float(tov_solution.t_events[1][0])

    def set_core_mass(self, tov_solution: Any) -> float:
        return float(tov_solution.y_events[1][0][1])

    def set_internal_profiles(self) -> InternalProfiles:
        return self.fac.set_internal_profiles()


@dataclass
class HybridStarStabilityFactory(StarStabilityFactory):
    conversion_speed: str
    """Creates a hybrid star using a Maxwell construction
        and computes its eigenfrequency in the radial oscillations formalism
        of Gondek, Zdunik and Haensel (1997)."""

    def __post_init__(self) -> None:
        self.star_fac = HybridStarFactory(self.central_ro_in.tov_input)

    def create_star(self, central_pressure: float) -> Any:
        return super().create_star(central_pressure)

    def set_eval_radius(self) -> Array:
        eval_radius: Array = super().set_eval_radius()
        return eval_radius[eval_radius <= self.structure.core_radius]

    def set_eval_radius_upper(self) -> Array:
        eval_radius: Array = super().set_eval_radius()
        return eval_radius[eval_radius >= self.structure.core_radius]

    def set_rad_osc_input(
        self, eval_radius: Array, initial_integration_vector: tuple[float, float]
    ) -> RadOscInput:
        """Set 2nd rad_osc_input in order to integrate
        from the interface to the star' surface."""
        rad_osc_input1: RadOscInput = super().set_rad_osc_input(
            eval_radius, initial_integration_vector
        )

        self._rad_osc_sol1 = solve_rad_osc(self.omega_squared_guess, rad_osc_input1)

        return super().set_rad_osc_input(
            eval_radius=self.set_eval_radius_upper(),
            initial_integration_vector=set_Lagragian_vars_at_interface(
                rad_osc_input1, self._rad_osc_sol1, self.conversion_speed
            ),
        )
