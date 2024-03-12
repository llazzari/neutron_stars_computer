import numpy as np
import pandas as pd
import pytest
from scipy.integrate import OdeSolution
from equationsofstate.eos import EquationOfState
from equationsofstate.massless_mit_bm import MasslessMITBM
from star.factory import StarFactory
from star.structure import Star
from star.tov_solver import TOVInput, radial_metric_fn, solve_tov


@pytest.fixture()
def tov_input() -> TOVInput:
    eos: EquationOfState = MasslessMITBM()
    return TOVInput(eos)


@pytest.fixture()
def central_pressure() -> float:
    return 300.


# @pytest.fixture()
# def star(tov_input: TOVInput, central_pressure: float) -> Star:
#     fac = StarFactory(tov_input, central_pressure)
#     star = fac.create()
#     return star


# @pytest.fixture()
# def tov_solution(tov_input: TOVInput, central_pressure: float) -> OdeSolution:
#     return solve_tov(tov_input, central_pressure)


# def test_star_compute_structure(star: Star, tov_solution: OdeSolution) -> None:
#     assert star.structure.tov_solution.y[0].all() == tov_solution.y[0].all()


# def test_star_structure_mass(star: Star) -> None:
#     assert star.structure.mass == 2.9745109607039497


# def test_star_structure_radius(star: Star) -> None:
#     assert star.structure.radius == 10.95535523689543


# def test_star_structure_internal_profile_time_metric_fn(
#         star_structure: Structure) -> None:
#     df: pd.DataFrame = star_structure.internal_profile
#     assert df['time_metric_fn'].iloc[-1] == -radial_metric_fn(
#         star_structure.radius, star_structure.mass)


# def test_tov_coeffs(star: Star) -> None:
#     tov_spline = TOVSpline(star.structure.internal_profile)
#     tov_coeff: list[float, float, float] = tov_spline.tov_coeffs(
#         r=star.tov_input.MIN_RADIUS)
#     assert (np.isclose(tov_coeff[2], star.central_pressure)) & (
#         np.isclose(tov_coeff[1], 0))
