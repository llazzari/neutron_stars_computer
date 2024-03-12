import functools
import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from . import conversionfactors as cf
from equationsofstate.eos import EquationOfState

Array = np.ndarray
Event = Callable[[float, Array], float]


@dataclass(slots=True)
class TOVInput:
    """Dataclass that contains all necessary inputs to solve TOV eqs."""
    eos: EquationOfState
    MIN_RADIUS: float = 1e-6
    MAX_RADIUS: float = 1e5
    RELATIVE_TOLERANCE: float = 1e-6
    ABSOLUTE_TOLERANCE: list[float] = field(
        default_factory=lambda: [1e-4, 1e-4, 1e-15])
    events: Iterable[Event] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.MIN_RADIUS <= 0.:
            raise ValueError(
                'Minimum radius has to be larger than zero '
                + 'to integrate TOV equations.'
            )
        if self.MAX_RADIUS <= self. MIN_RADIUS:
            raise ValueError(
                'Maximum radius has to be larger than minimum radius.')


def solve_tov(tov_input: TOVInput, central_pressure: float) -> Any:
    """Solves the TOV equations for a given central pressure and EOS.
    Returns a bunch object, see solve_ivp documentation for more details."""

    if central_pressure <= 0:
        raise ValueError('Central pressure has to be a larger than zero.')

    def compute_taylor_expansion_at_center() -> tuple[float, float, float]:
        """Taylor expansion near the stellar center (r ~ 0)."""
        r0: float = tov_input.MIN_RADIUS
        p0: float = central_pressure
        # v0: float = 0
        e0: float = tov_input.eos.energy_density_from(pressure=p0)
        gamma0: float = tov_input.eos.adiabatic_index_from(pressure=p0)

        v2: float = 8*np.pi/3*(e0 + 3*p0)*cf.MEV_FM3_TO_KM_2
        p2: float = -4*np.pi/3*(e0 + p0)*(e0 + 3*p0)*cf.MEV_FM3_TO_KM_2
        e2: float = p2*(e0 + p0)/(gamma0*p0)

        v4: float = 4*np.pi/5*(e2 + 5*p2) + 64*np.pi**2/9*e0 * \
            (e0 + 3*p0)*cf.MEV_FM3_TO_KM_2
        v4 *= cf.MEV_FM3_TO_KM_2
        p4: float = -2*np.pi/5*(e0 + p0)*(e2 + 5*p2)
        p4 -= 2*np.pi/3*(e2 + p2)*(p0 + 3*p0)
        p4 -= 32*np.pi**2/9*e0*(e0 + p0)*(e0 + 3*p0)*cf.MEV_FM3_TO_KM_2
        p4 *= cf.MEV_FM3_TO_KM_2

        vc: float = 0.5*v2*r0**2 + 0.25*v4*r0**4  # + v0
        mc: float = 4*np.pi*r0**3*e0/3
        pc: float = p0 + 0.5*p2*r0**2 + 0.25*p4*r0**4
        # ec: float = e0 + 0.5*e2*r0**2

        return (vc, mc, pc)

    def boundary_event(r: float, y: Array) -> float:
        """Defines the boundary of the star by reaching p(r=R) = 0."""
        return y[2]

    initial_integration_vector = compute_taylor_expansion_at_center()

    boundary_event.terminal = True
    boundary_event.direction = -1

    def tov_equations(r: float, y: Array) -> tuple[float, float, float]:
        """TOV equations. System of differential equations for pressure, mass
        and time metric function.
        ------------------------------
        Units:
        -------
        r: radius in km.
        p, e: pressure and energy density in MeV/fm^3.
        m: mass in km.
        v: nu (time metric function) is dimensionless.
        """

        v, m, p = y
        e: float = tov_input.eos.energy_density_from(pressure=p)

        dvdr: float = time_metric_fn_derivative(r, p, m)
        dmdr: float = mass_derivative(r, e)
        dpdr: float = pressure_derivative(e, p, dvdr)

        return (dvdr, dmdr, dpdr)

    tov_integration: Any = solve_ivp(
        tov_equations,
        (tov_input.MIN_RADIUS, tov_input.MAX_RADIUS),
        initial_integration_vector,
        method='DOP853',
        dense_output=True,
        rtol=tov_input.RELATIVE_TOLERANCE,
        atol=tov_input.ABSOLUTE_TOLERANCE,
        events=(boundary_event, *tov_input.events)
    )

    def check_integration_validity(success: bool, status: int) -> None:
        if not success:
            raise TOVIntegrationError(
                'The integration of the TOV eqs. was not successfull...')

        if status != 1:
            raise BoundaryNotFoundError(
                "The integration was successfull, "
                + "but the star's boundary was not reached. "
                + "Perhaps you TOVInput.MAXRADIUS is too small?"
            )

    check_integration_validity(tov_integration.success, tov_integration.status)

    return tov_integration


@functools.lru_cache
def time_metric_fn_derivative(r: float, p: float, m: float) -> float:
    exp_lambda: float = np.exp(-2*radial_metric_fn(r, m))
    return (4*np.pi*r*p*cf.MEV_FM3_TO_KM_2 + m/r**2) / exp_lambda


@functools.lru_cache
def mass_derivative(r: float, e: float) -> float:
    return 4.*np.pi*r**2.*e*cf.MEV_FM3_TO_KM_2


@functools.lru_cache
def pressure_derivative(e: float, p: float, dvdr: float) -> float:
    return -(e+p) * dvdr


@functools.lru_cache
def radial_metric_fn(radius: float, mass: float) -> float:
    return -np.log(1-2*mass/radius) / 2


class TOVIntegrationError(Exception):
    pass


class BoundaryNotFoundError(Exception):
    pass
