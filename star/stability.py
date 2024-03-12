import numpy as np
from typing import Any, Callable
from dataclasses import dataclass
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar, RootResults

from star import conversionfactors as cf
from star.tov_solver import (
    radial_metric_fn,
    time_metric_fn_derivative,
    TOVInput,
    Array
)


class RadialOscillationsIntegrationError(Exception):
    pass


@dataclass(slots=True)
class RadOscInput:
    """Necessary inputs to solve the TOV eqs. and
      the radial oscillation eqs.
      Partially defined by the user."""
    tov_input: TOVInput
    tov_spline: Callable[[float], list[float]]
    eval_radius: Array
    radial_interval: tuple[float, float]
    initial_integration_vector: tuple[float, float]
    INT_ABS_TOL: float
    INT_REL_TOL: float
    ROOT_ABS_TOL: float
    ROOT_REL_TOL: float


def solve_rad_osc(w2: float, rad_osc_input: RadOscInput) -> Any:
    """Solves the radial oscillation eqs. in the
      Gondek, Haensel and Zdunik (1997) formalism."""
    tov_input: TOVInput = rad_osc_input.tov_input

    def rad_osc_eqs(r: float, y: Array, w2: float) -> tuple[float, float]:
        """Radial oscillation eqs. System of differential equations for the
        Lagragian variables \\xi and \\Delta p.
        ------------------------------
        Units:
        -------
        xi: dimensionless.
        Dp: Delta p in MeV/fm³.
        w2: omega_squared in km⁻².
        """
        xi, Dp = y  # floats

        v, m, p = rad_osc_input.tov_spline(r)  # floats

        e: float = tov_input.eos.energy_density_from(pressure=p)
        gamma: float = tov_input.eos.adiabatic_index_from(pressure=p)

        lamb: float = radial_metric_fn(radius=r, mass=m)
        dvdr: float = time_metric_fn_derivative(r, p, m)

        dxidr: float = -(3*xi + Dp/(p*gamma))/r + dvdr*xi
        dDpdr: float = r*(e + p)*(np.exp(2*lamb)*(
            w2*np.exp(-2*v) - 8*np.pi*p*cf.MEV_FM3_TO_KM_2)
            + dvdr*(dvdr + 4/r))*xi
        dDpdr -= (dvdr + 4*np.pi*r*(e + p)
                  * cf.MEV_FM3_TO_KM_2*np.exp(2*lamb))*Dp

        return (dxidr, dDpdr)

    def nodes(r: float, y: Array, w2: float) -> float:
        """Computes how many times xi(r) is zero."""
        return y[0]

    return solve_ivp(
        rad_osc_eqs,
        y0=rad_osc_input.initial_integration_vector,
        t_span=rad_osc_input.radial_interval,
        t_eval=rad_osc_input.eval_radius,
        method='DOP853',
        args=(w2,),
        dense_output=True,
        atol=rad_osc_input.INT_ABS_TOL,
        rtol=rad_osc_input.INT_REL_TOL,
        events=nodes,
    )


@dataclass
class Stability:
    def find_frequency(
        self,
        rad_osc_input: RadOscInput,
        omega_squared_guess: float,
    ) -> RootResults:
        """Finds the frequency that satisfies \\Delta p (r=R) ~ 0."""
        w2: float = omega_squared_guess
        self._find_eigenfrequency = root_scalar(
            self.delta_p_at_surface,
            method='secant',
            x0=w2,
            x1=w2+np.abs(w2)*1e-3,
            args=(rad_osc_input,),
            xtol=rad_osc_input.ROOT_ABS_TOL,
            rtol=rad_osc_input.ROOT_REL_TOL
        )

        if not self._find_eigenfrequency.converged:
            raise ValueError('Delta p did not converge at the surface.')

        return self._find_eigenfrequency

    def delta_p_at_surface(
            self, w2: float, rad_osc_input: RadOscInput) -> float:
        self.rad_osc_sol: Any = solve_rad_osc(w2, rad_osc_input)
        return float(self.rad_osc_sol.y[1][-1])

    @property
    def omega_squared(self) -> float:
        return self._find_eigenfrequency.root

    @property
    def mode(self) -> int:
        return np.size(self.rad_osc_sol.t_events)


@dataclass(slots=True)
class CentralRadOscInput:
    """User defined input to solve the radial oscillations and TOV eqs."""
    tov_input: TOVInput
    central_xi: float = 1.
    INT_ABS_TOL: float = 1e-6
    INT_REL_TOL: float = 1e-3
    ROOT_ABS_TOL: float = 1e-15
    ROOT_REL_TOL: float = 2e-6

    def central_delta_p(self, central_pressure: float) -> float:
        central_gamma = self.tov_input.eos.adiabatic_index_from(
            central_pressure
        )
        return -3*central_gamma*central_pressure*self.central_xi

    def initial_integration_vector(
            self, central_pressure: float) -> tuple[float, float]:
        return (self.central_xi, self.central_delta_p(central_pressure))
