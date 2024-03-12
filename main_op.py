from star.stability import CentralRadOscInput
from star.tov_solver import TOVInput
from star.factory import StarStabilityFactory, StarFactory
from star.constellation import create_stellar_family
from equationsofstate.eos import EquationOfState
from equationsofstate.massless_mit_bm import MasslessMITBM

import time
import numpy as np
import pandas as pd


def main() -> None:
    print('Starting...')
    start = time.time()

    eos: EquationOfState = MasslessMITBM()
    central_pressures = np.linspace(5, 300, 100)
    tov_input = TOVInput(
        eos,
        ABSOLUTE_TOLERANCE=[1e-4, 1e-4, 1e-6],
        RELATIVE_TOLERANCE=1e-3
    )

    # c_ro_input = CentralRadOscInput(tov_input)
    # omega_squared_guess = -1e-1

    fac = StarFactory(tov_input)
    # fac = StarStabilityFactory(c_ro_input, omega_squared_guess)
    stars: pd.DataFrame = create_stellar_family(
        fac,
        central_pressures
    )
    print(stars)

    end = time.time()
    print(f"Time to complete: {end - start:.2f} s")


if __name__ == '__main__':
    main()
