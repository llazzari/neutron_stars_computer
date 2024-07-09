from typing import Any

from .hybrid_eos import HybridEOS
from ..star.stability import RadOscInput
from ..star.tov_solver import pressure_derivative


def set_Lagragian_vars_at_interface(
    rad_osc_input: RadOscInput,
    rad_osc_sol: Any,
    conversion_speed: str,
) -> tuple[float, float]:

    # set variables before (minus) the interface
    xi_minus, Delta_p_minus = rad_osc_sol.y  # np.ndarrays

    # set variables after (plus) the interface
    xi_plus, Delta_p_plus = (float(xi_minus[-1]), float(Delta_p_minus[-1]))
    if conversion_speed == "rapid":
        xi_plus += rapid_conversion(Delta_p_plus, rad_osc_input)

    return (xi_plus, Delta_p_plus)


def rapid_conversion(Delta_p_plus: float, rad_osc_input: RadOscInput) -> float:
    # get core structure
    core_radius: float = rad_osc_input.eval_radius[-1]
    core_mass: float = rad_osc_input.tov_spline(core_radius)[1]

    # abbreviate Equation of State (eos)
    eos: HybridEOS = rad_osc_input.tov_input.eos  # type: ignore

    # Compute energy density before (minus) and after (plus) the interface
    e_plus: float = eos.energy_density_from(eos.transitional_pressure)
    e_minus: float = eos.energy_density_from(eos.transitional_pressure * (1 + 1e-5))

    # Compute pressure dervative before (minus) and after (plus) the interface
    dpdr_plus: float = pressure_derivative(
        core_radius, e_plus, eos.transitional_pressure, core_mass
    )
    dpdr_minus: float = pressure_derivative(
        core_radius, e_minus, eos.transitional_pressure, core_mass
    )

    return Delta_p_plus / core_radius * (1 / dpdr_plus - 1 / dpdr_minus)
