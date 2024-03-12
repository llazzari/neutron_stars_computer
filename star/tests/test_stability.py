import pytest
# from scipy.integrate import OdeSolution
from equationsofstate.eos import EquationOfState
from equationsofstate.massless_mit_bm import MasslessMITBM

from star.stability import CentralRadOscInput, RadOscInput
from star.structure import Star
from star.tov_solver import TOVInput


@pytest.fixture()
def rad_osc_input() -> RadOscInput:
    return CentralRadOscInput()


@pytest.fixture()
def eos() -> EquationOfState:
    return MasslessMITBM()


@pytest.fixture()
def tov_input(eos: EquationOfState) -> TOVInput:
    return TOVInput(eos)


@pytest.fixture()
def central_pressure() -> float:
    return 300.


@pytest.fixture()
def star(tov_input: TOVInput, central_pressure: float) -> Star:
    star = Star(tov_input, central_pressure)
    star.compute_structure()
    return star


# def test_solve_tov_initial_vector(
#         rad_osc_solution: OdeSolution, central_pressure) -> None:
#     time_metric_fn, mass, pressure = rad_osc_solution.y
#     assert (pressure[0] <= central_pressure) & (mass[0] > 0) & (
#         time_metric_fn[0] > 0)

# def test_solve_tov_success(rad_osc_solution: OdeSolution) -> None:
#     assert rad_osc_solution.success


# def test_solve_tov_status(rad_osc_solution: OdeSolution) -> None:
#     assert rad_osc_solution.status == 1
