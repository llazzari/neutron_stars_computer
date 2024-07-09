import functools
import numpy as np
from typing import Any
from dataclasses import dataclass, field

from .tov_solver import Array, TOVInput, solve_tov
from .stability import CentralRadOscInput, RadOscInput, Stability
from .structure import InternalProfiles, Star, interpol_tov, tov_coeffs


@dataclass(slots=True)
class StarFactory:
    """Creates a star from a continuous EOS."""

    tov_input: TOVInput
    tov_solution: Any = field(init=False)

    def create_star(self, central_pressure: float) -> Star:
        self.tov_solution: Any = solve_tov(self.tov_input, central_pressure)

        radius = float(self.tov_solution.t_events[0][0])
        mass = float(self.tov_solution.y_events[0][0][1])

        return Star(central_pressure, radius, mass)

    def set_internal_profiles(self) -> InternalProfiles:
        return InternalProfiles(self.tov_solution.t, *self.tov_solution.y)


@dataclass
class StarStabilityFactory:
    """Creates a star from a continuous EOS and
    computes its eigenfrequency in the radial oscillations formalism of
    Gondek, Zdunik and Haensel (1997)."""

    central_ro_in: CentralRadOscInput
    omega_squared_guess: float

    def __post_init__(self) -> None:
        self.star_fac = StarFactory(self.central_ro_in.tov_input)

    def create_star(self, central_pressure: float) -> Any:
        self.structure: Any = self.star_fac.create_star(central_pressure)
        print(self.structure)
        self.ip: InternalProfiles = self.star_fac.set_internal_profiles()

        initial_integration_vector: tuple[float, float] = (
            self.central_ro_in.central_xi,
            self.central_ro_in.central_delta_p(self.structure.central_pressure),
        )
        eval_radius = self.set_eval_radius()

        rad_osc_input: RadOscInput = self.set_rad_osc_input(
            eval_radius, initial_integration_vector
        )
        self.stability = Stability()
        self.stability.find_frequency(
            rad_osc_input,
            self.omega_squared_guess,
        )

        return Star(
            self.structure.central_pressure,
            self.structure.radius,
            self.structure.mass,
            self.stability.mode,
            self.stability.omega_squared,
        )

    def set_eval_radius(self) -> Array:
        return self.ip.radial_coord

    def set_rad_osc_input(
        self, eval_radius: Array, initial_integration_vector: tuple[float, float]
    ) -> RadOscInput:

        interpolated_tov = interpol_tov(self.ip)

        return RadOscInput(
            tov_input=self.central_ro_in.tov_input,
            tov_spline=functools.partial(tov_coeffs, interpolated_tov=interpolated_tov),
            eval_radius=eval_radius,
            radial_interval=(eval_radius[0], eval_radius[-1]),
            initial_integration_vector=initial_integration_vector,
            INT_ABS_TOL=self.central_ro_in.INT_ABS_TOL,
            INT_REL_TOL=self.central_ro_in.INT_REL_TOL,
            ROOT_ABS_TOL=self.central_ro_in.ROOT_ABS_TOL,
            ROOT_REL_TOL=self.central_ro_in.ROOT_REL_TOL,
        )

    def set_internal_profiles(self) -> InternalProfiles:
        sol: Any = self.stability.rad_osc_sol
        radial_coord = sol.t
        xi, Delta_p = sol.y

        # correct the size of the internal profiles to match 'rad_osc_sol'
        pressures = np.delete(self.ip.pressures, -1)
        masses = np.delete(self.ip.masses, -1)
        time_metric_fn = np.delete(self.ip.time_metric_fn, -1)

        return InternalProfiles(
            radial_coord, time_metric_fn, masses, pressures, xi=xi, Delta_p=Delta_p
        )
