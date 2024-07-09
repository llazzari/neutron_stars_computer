import pytest
from equationsofstate.css import CSS


@pytest.fixture
def eos() -> CSS:
    return CSS(
        transitional_density=450.0,
        delta_epsilon=100.0,
        sound_speed_squared=0.5,
        transitional_pressure=20.0,
    )


def test_sound_speed_squared(eos: CSS) -> None:
    assert (
        0 <= eos.sound_speed_squared <= 1
    ), "Sound speed squared should be between 0 and 1"


def test_energy_density_from_pressure(eos: CSS) -> None:
    pressure = 3.0
    energy_density = eos.energy_density_from(pressure)
    assert energy_density > 0, "Energy density should be > 0"
    assert energy_density > pressure, "Energy density should be > pressure"


def test_adiabatic_index_from_pressure(eos: CSS) -> None:
    pressure = 3.0
    adiabatic_index = eos.adiabatic_index_from(pressure)
    assert isinstance(adiabatic_index, float), "Adiabatic index should be a float"


def test_transitional_density_positive(eos: CSS) -> None:
    assert eos.transitional_density > 0, "Transitional density should be > 0"


def test_delta_epsilon_positive(eos: CSS) -> None:
    assert eos.delta_epsilon > 0, "Delta epsilon should be > 0"


def test_transitional_pressure_positive(eos: CSS) -> None:
    assert eos.transitional_pressure > 0, "Transitional pressure should be > 0"
