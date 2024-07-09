from dataclasses import dataclass
import numpy as np

from .eos import EquationOfState


@dataclass(slots=True, frozen=True)
class MasslessMITBM(EquationOfState):
    BAG_PRESS: float = 57

    def energy_density_from(self, pressure: float) -> float:
        return 3 * pressure + 4 * self.BAG_PRESS

    def adiabatic_index_from(self, pressure: float) -> float:
        if pressure == 0.0:
            return np.inf
        return 4 / 3 * (1 + self.BAG_PRESS / pressure)

    def sound_speed_squared_from(self, pressure: float) -> float:
        return 1 / 3
