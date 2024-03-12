from icecream import ic
from equationsofstate.massless_mit_bm import MasslessMITBM

from hybrid_star.factory import HybridStarFactory, HybridStarStabilityFactory
from hybrid_star.hybrid_eos import HybridEOS
from star.constellation import create_stellar_family
from star.stability import CentralRadOscInput
from star.tov_solver import TOVInput
from equationsofstate.bps_fit import BPS_fit
from equationsofstate.eos import EquationOfState

import numpy as np
import pandas as pd


def main() -> None:
    eos: EquationOfState = BPS_fit()
    qm_eos: EquationOfState = MasslessMITBM()
    hybrid_eos: EquationOfState = HybridEOS(
        eos1=eos,
        eos2=qm_eos,
        transitional_pressure=4.7e-4
    )

    tov_input = TOVInput(hybrid_eos, MAX_RADIUS=1e5)
    central_ro_in = CentralRadOscInput(tov_input)
    central_pressures = np.geomspace(5e-4, 7e-4, 2)

    fac = HybridStarStabilityFactory(
        central_ro_in,
        omega_squared_guess=-4.9e-8,
        conversion_speed='slow'
    )
    # fac = HybridStarFactory(tov_input)
    hybrid_stars: pd.DataFrame = create_stellar_family(
        fac,
        central_pressures
    )
    ic(hybrid_stars)


if __name__ == '__main__':
    main()
