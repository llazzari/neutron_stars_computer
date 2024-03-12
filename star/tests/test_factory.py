import pytest
from star.structure import Star
from star.factory import StarFactory
from star.tov_solver import TOVInput
from equationsofstate.massless_mit_bm import MasslessMITBM


@pytest.fixture()
def star() -> Star:
    tov_input = TOVInput(eos=MasslessMITBM())
    fac = StarFactory(tov_input)
    return fac.create_star(central_pressure=300)


def test_star_factory(star: Star) -> None:
    assert hasattr(star, 'radius') & hasattr(star, 'mass')
