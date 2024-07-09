import functools
from typing import Any, Optional
from dataclasses import dataclass
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

from .tov_solver import radial_metric_fn, Array


@dataclass(slots=True)
class InternalProfiles:
    """Class containing time_metric_fn, mass [km] and pressure [MeV/fm³]
    as functions of the radial coordinate.
    If the stability is computed using radial oscillations,
    it will also contain the Lagragian variables \\xi and \\Delta p[MeV/fm³]"""

    radial_coord: Array
    time_metric_fn: Array
    masses: Array
    pressures: Array
    xi: Optional[Array] = None
    Delta_p: Optional[Array] = None

    def __post_init__(self) -> None:
        self.time_metric_fn: Array = self.correct_time_metric_fn()

    def correct_time_metric_fn(self) -> Array:
        lmbda: float = radial_metric_fn(self.radial_coord[-1], self.masses[-1])
        return self.time_metric_fn - (float(self.time_metric_fn[-1]) + lmbda)


@dataclass(slots=True)
class Star:
    central_pressure: float
    radius: float
    mass: float
    mode: Optional[int] = None
    omega_squared: Optional[float] = None


def interpol_tov(profile: InternalProfiles) -> tuple[Any, ...]:
    """Interpolates the TOV solution using cubic splines."""

    time_metric_fn_spl = Spline(profile.radial_coord, profile.time_metric_fn)
    mass_spl = Spline(profile.radial_coord, profile.masses)
    pressure_spl = Spline(profile.radial_coord, profile.pressures)

    return (time_metric_fn_spl, mass_spl, pressure_spl)


@functools.lru_cache
def tov_coeffs(r: float, interpolated_tov: tuple[Any, ...]) -> list[float]:
    return [float(tov_spl_at(r)) for tov_spl_at in interpolated_tov]
