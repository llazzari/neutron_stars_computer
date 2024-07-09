import pytest
from hypothesis import given, strategies as st
from ..gpp import (
    GPP,
    select_coeff,
    coefficients_at_dividing_densities,
)
from .test_base_eos import TestEquationOfState


# Strategies
coeff_strategy = st.tuples(
    st.floats(min_value=-1e10, max_value=1e10),
    st.floats(min_value=-1e10, max_value=1e10),
    st.floats(min_value=-1e10, max_value=1e10),
    st.floats(min_value=-1e10, max_value=1e10),
)
coeffs_strategy = st.tuples(
    coeff_strategy, coeff_strategy, coeff_strategy, coeff_strategy, coeff_strategy
)


class TestGPP(TestEquationOfState):
    @pytest.fixture
    def eos(self) -> GPP:
        return GPP(eos="SLy4", _coeffs=coefficients_at_dividing_densities("SLy4"))


@given(coeffs_strategy, st.floats(min_value=0, max_value=1e10))
def test_select_coeff(coeffs: tuple[tuple[float, ...], ...], p_in_gcm3: float) -> None:
    result = select_coeff(coeffs, p_in_gcm3)
    assert (
        isinstance(result, tuple) and len(result) == 4
    ), "select_coeff should return a tuple of length 4"


def test_coefficients_at_dividing_densities() -> None:
    eos_name = "HEB"  # Example EOS name
    coeffs: tuple[tuple[float, ...], ...] = coefficients_at_dividing_densities(eos_name)
    assert (
        isinstance(coeffs, tuple) and len(coeffs) == 5
    ), "coefficients_at_dividing_densities should return a tuple of length 5"
