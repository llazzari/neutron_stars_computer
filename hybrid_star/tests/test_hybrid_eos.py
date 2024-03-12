import pytest
from equationsofstate.bps_fit import BPS_fit
from equationsofstate.massless_mit_bm import MasslessMITBM
from hybrid_star.hybrid_eos import HybridEOS


@pytest.fixture()
def hybrid_eos() -> HybridEOS:
    return HybridEOS(
        eos1=BPS_fit(),
        eos2=MasslessMITBM(),
        transitional_pressure=4.47e-4
    )


def test_energy_density_from_eos(hybrid_eos: HybridEOS) -> None:
    assert hybrid_eos.eos1.energy_density_from(
        hybrid_eos.transitional_pressure) == 0.20112872868514242


def test_energy_density_from_qm_eos(hybrid_eos: HybridEOS) -> None:
    assert hybrid_eos.eos2.energy_density_from(
        hybrid_eos.transitional_pressure) == 228.001341


def test_adiabatic_index_from_eos(hybrid_eos: HybridEOS) -> None:
    assert hybrid_eos.eos1.adiabatic_index_from(
        hybrid_eos.transitional_pressure) == 1.2675662930990597


def test_adiabatic_index_from_qm_eos(hybrid_eos: HybridEOS) -> None:
    assert hybrid_eos.eos2.adiabatic_index_from(
        hybrid_eos.transitional_pressure) == 170023.70469798654
