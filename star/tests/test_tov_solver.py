import numpy as np
import pytest
from typing import Any
from equationsofstate.massless_mit_bm import MasslessMITBM
from equationsofstate.eos import EquationOfState
from star.tov_solver import (
    TOVInput,
    solve_tov,
    time_metric_fn_derivative,
    mass_derivative,
    pressure_derivative,
    radial_metric_fn
)


def test_time_metric_fn_derivative_null_args_raises_error() -> None:
    with pytest.raises(ZeroDivisionError):
        assert time_metric_fn_derivative(0, 0, 0)


def test_time_metric_fn_derivative_sign() -> None:
    assert time_metric_fn_derivative(10, 1, 2) > 0


def test_radial_metric_fn_mass_null_radius_raises_error() -> None:
    with pytest.raises(ZeroDivisionError):
        assert radial_metric_fn(0, 1)


def test_radial_metric_fn_mass_null_mass() -> None:
    assert np.isclose(radial_metric_fn(1, 0), 0)


def test_radial_metric_fn() -> None:
    assert radial_metric_fn(20, 1) == -np.log(0.9)/2


def test_pressure_derivative_null() -> None:
    assert pressure_derivative(0, 0, 0) == 0


def test_pressure_derivative_sign() -> None:
    assert pressure_derivative(300, 100, 0.1) < 0


def test_mass_derivative_null() -> None:
    assert mass_derivative(0, 0) == 0


def test_mass_derivative_sign() -> None:
    assert mass_derivative(1, 300) > 0


@pytest.fixture()
def eos() -> EquationOfState:
    return MasslessMITBM()


def test_invalid_min_radius_tov_input(eos: EquationOfState) -> None:
    with pytest.raises(ValueError):
        assert TOVInput(eos, MIN_RADIUS=0)


def test_invalid_max_radius_tov_input(eos: EquationOfState) -> None:
    with pytest.raises(ValueError):
        assert TOVInput(eos, MIN_RADIUS=1e-5, MAX_RADIUS=0)


@pytest.fixture()
def tov_input(eos: EquationOfState) -> TOVInput:
    return TOVInput(eos)


def test_solve_tov_invalid_central_pressure(tov_input: TOVInput) -> None:
    with pytest.raises(ValueError):
        assert solve_tov(tov_input, 0)


@pytest.fixture()
def central_pressure() -> float:
    return 300.


@pytest.fixture()
def tov_solution(tov_input: TOVInput, central_pressure: float) -> Any:
    return solve_tov(tov_input, central_pressure)


def test_solve_tov_initial_vector(tov_solution: Any, central_pressure) -> None:
    time_metric_fn, mass, pressure = tov_solution.y
    assert (pressure[0] <= central_pressure) & (mass[0] > 0) & (
        time_metric_fn[0] > 0)


def test_solve_tov_surface_pressure(tov_solution: Any) -> None:
    pressure = tov_solution.y[2]
    assert np.isclose(pressure[-1], 0)
