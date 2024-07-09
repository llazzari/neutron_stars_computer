from pathlib import Path
from functools import lru_cache
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from ..star.conversionfactors import GCM3_TO_MEVFM3

from .eos import EquationOfState

Array = np.ndarray
Coeff = tuple[float, ...]
Coeffs = tuple[Coeff, ...]
TupleListFloat = tuple[list[float], ...]
CrustCoreData = tuple[float, float, tuple[float, ...]]
CrustCoreCoeffs = tuple[float, float, tuple[float, ...], float, float]


@lru_cache
def pressure_in_gcm3(pressure: float) -> float:
    return abs(pressure) / GCM3_TO_MEVFM3


@lru_cache
def select_coeff(coeff: Coeffs, p_in_gcm3: float) -> Coeff:

    PRESS_dd: Coeff = coeff[0]
    ind: np.intp = np.max(np.argwhere(np.array(PRESS_dd) < p_in_gcm3))
    K, gamma, Lambda, a = [c[ind] for c in coeff[1:]]

    return (K, gamma, Lambda, a)


@dataclass(slots=True)
class GPP(EquationOfState):
    """Generalized Piecewise Polytropes from Boyle et al., PRD 102 083027 (2020).
    Input eos as a string, using the keys from table III of the aforementioned paper."""

    eos: str
    _coeffs: Coeffs = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self._coeffs: Coeffs = coefficients_at_dividing_densities(self.eos)

    def baryon_density_from(self, pressure: float) -> float:
        """Computes baryon number density in fm^-3 from the pressure."""

        p_in_gcm3: float = pressure_in_gcm3(pressure)
        K, gamma, Lambda = select_coeff(self._coeffs, p_in_gcm3)[:3]
        rho = ((p_in_gcm3 - Lambda) / K) ** (1 / gamma)

        return rho * GCM3_TO_MEVFM3 / 938

    def energy_density_from(self, pressure: float) -> float:
        """Computes energy density in MeV/fm^3 from the pressure."""

        p_in_gcm3: float = pressure_in_gcm3(pressure)
        K, gamma, Lambda, a = select_coeff(self._coeffs, p_in_gcm3)
        e = (p_in_gcm3 - Lambda) / (gamma - 1)
        e += (1 + a) * ((p_in_gcm3 - Lambda) / K) ** (1 / gamma) - Lambda

        return e * GCM3_TO_MEVFM3

    def adiabatic_index_from(self, pressure: float) -> float:
        """Computes the adiabatic index from the pressure."""

        p_in_gcm3: float = pressure_in_gcm3(pressure)
        gamma, Lambda = select_coeff(self._coeffs, p_in_gcm3)[1:3]

        return gamma * (p_in_gcm3 - Lambda) / p_in_gcm3

    def sound_speed_squared_from(self, pressure: float) -> float:
        """Computes the sound speed squared from the pressure."""

        p_in_gcm3: float = pressure_in_gcm3(pressure)
        K, gamma, Lambda, a = select_coeff(self._coeffs, p_in_gcm3)

        return 1 / (
            1 / (gamma - 1)
            + (1 + a) / (K * gamma) * ((p_in_gcm3 - Lambda) / K) ** (1 / gamma - 1)
        )


@lru_cache
def coefficients_at_dividing_densities(eos: str) -> Coeffs:
    # obtain crust and core coefficients
    crust: Coeffs = Sly4_crust()
    core: Coeffs = core_coefficients(crust[1:], eos)

    # concatenate crust and core coefficients
    crust_core: list[Coeff] = [c1 + c2 for c1, c2 in zip(crust, core)]

    k_dd, gamma_dd, lambda_dd, a_dd = tuple(crust_core[1:])

    def pressure_at_dividing_densities() -> Array:
        rho_arr, k_arr, gamma_arr, lambda_arr = [np.array(cc) for cc in crust_core[:4]]
        return k_arr * rho_arr**gamma_arr + lambda_arr

    pressure_dd: tuple[float, ...] = tuple(pressure_at_dividing_densities())

    return (pressure_dd, k_dd, gamma_dd, lambda_dd, a_dd)


def Sly4_crust() -> Coeffs:
    rho_crust: Coeff = (0.0, 6.285e5, 1.826e8, 3.350e11, 5.317e11)
    K_crust: Coeff = (5.214e-9, 5.726e-8, 1.662e-6, -7.957e29, 1.746e-8)
    gamma_crust: Coeff = (1.611, 1.440, 1.269, -1.841, 1.382)
    lambda_crust: Coeff = (0.0, -1.354, -6.025e3, 1.193e9, 7.077e8)
    a_crust: Coeff = (0.0, -1.861e-5, -5.278e-4, 1.035e-2, 8.208e-3)

    return (rho_crust, K_crust, gamma_crust, lambda_crust, a_crust)


def connect_crust_core(crust: Coeffs, eos: str) -> CrustCoreCoeffs:
    def read_crust_core_coeff() -> CrustCoreData:
        path = Path.cwd() / "src" / "neutron_stars_computer" / "equationsofstate"
        path = path / "tabulated_eos/GPP/TableIII_Boyle.dat"
        df: pd.DataFrame = pd.read_csv(path, sep=" ")

        eos_df: pd.DataFrame = df[df["EOS"] == eos]

        rho0: float = float(10 ** (eos_df["log_rho0"].iloc[0]))
        K1: float = float(10 ** (eos_df["log_K1"].iloc[0]))
        gamma_i: tuple[float, ...] = tuple(
            [float(eos_df[f"gamma{i}"].iloc[0]) for i in np.arange(1, 4)]
        )

        return (rho0, K1, gamma_i)

    def set_crust_coeff(crust: Coeffs) -> list[float]:
        return [c[-1] for c in crust]

    rho0, k1, gamma_i = read_crust_core_coeff()
    k_crust, gamma_crust, lambda_crust, a_crust = set_crust_coeff(crust)

    def set_crust_pressure() -> float:
        return k_crust * rho0**gamma_crust + lambda_crust

    def set_crust_energy_density() -> float:
        e_crust: float = k_crust * rho0**gamma_crust / (gamma_crust - 1)
        e_crust += (1 + a_crust) * rho0 - lambda_crust
        return e_crust

    p_crust: float = set_crust_pressure()
    e_crust: float = set_crust_energy_density()

    lambda1: float = p_crust - k1 * rho0 ** gamma_i[0]
    a1: float = e_crust / rho0 - k1 / (gamma_i[0] - 1) * rho0 ** (gamma_i[0] - 1)
    a1 += lambda1 / rho0 - 1

    return (rho0, k1, gamma_i, lambda1, a1)


def core_coefficients(crust: Coeffs, eos: str) -> Coeffs:
    rho0, k1, gamma_i, lambda1, a1 = connect_crust_core(crust, eos)

    def set_rho_core() -> tuple[float, ...]:
        return (
            (rho0, 10**14.45, 10**14.58)
            if "HEB" in eos
            else (rho0, 10**14.87, 10**14.99)
        )

    rho_core: tuple[float, ...] = set_rho_core()

    def set_core_lists() -> TupleListFloat:
        k_core: list[float] = [k1] * len(gamma_i)
        lambda_core: list[float] = [lambda1] * len(gamma_i)
        a_core: list[float] = [a1] * len(gamma_i)

        def set_k_core(i: int) -> float:
            return (
                k_core[i]
                * (gamma_i[i] / gamma_i[i + 1])
                * rho_core[i + 1] ** (gamma_i[i] - gamma_i[i + 1])
            )

        def set_lambda_core(i: int) -> float:
            return lambda_core[i] + (1 - gamma_i[i] / gamma_i[i + 1]) * k_core[i] * (
                rho_core[i + 1] ** gamma_i[i]
            )

        def set_a_core(i: int) -> float:
            return a_core[i] + gamma_i[i] * (
                (gamma_i[i + 1] - gamma_i[i])
                / ((gamma_i[i + 1] - 1) * (gamma_i[i] - 1))
            ) * (k_core[i] * rho_core[i + 1] ** (gamma_i[i] - 1))

        for i in np.arange(len(gamma_i) - 1):
            k_core[i + 1] = set_k_core(i)
            lambda_core[i + 1] = set_lambda_core(i)
            a_core[i + 1] = set_a_core(i)

        return (k_core, lambda_core, a_core)

    k_c, lambda_c, a_c = [tuple(c) for c in set_core_lists()]

    return (rho_core, k_c, gamma_i, lambda_c, a_c)
