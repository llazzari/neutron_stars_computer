import pytest
from hypothesis import given, strategies as st
from ..eos import EquationOfState


class TestEquationOfState:
    @pytest.fixture
    def eos(self) -> EquationOfState:
        raise NotImplementedError(
            "This fixture should be implemented by subclasses to provide an EOS instance"
        )

    @given(st.floats(min_value=0, max_value=1e10))
    def test_energy_density_from(self, eos: EquationOfState, pressure: float) -> None:
        result: float = eos.energy_density_from(pressure)
        assert result >= 0, "Energy density should be non-negative"

    @given(st.floats(min_value=0, max_value=1e10))
    def test_adiabatic_index_from(self, eos: EquationOfState, pressure: float) -> None:
        result: float = eos.adiabatic_index_from(pressure)
        assert result >= 0, "Adiabatic index should be non-negative"

    @given(st.floats(min_value=0, max_value=1e10))
    def test_sound_speed_squared_from(
        self, eos: EquationOfState, pressure: float
    ) -> None:
        result: float = eos.sound_speed_squared_from(pressure)
        assert 0 <= result <= 1, "Sound speed squared should be between 0 and 1"
