from dataclasses import dataclass
import numpy as np

from equationsofstate.eos import EquationOfState
import star.conversionfactors as cf


@dataclass(slots=True, frozen=True)
class BPS_fit(EquationOfState):
    a: float = -15.8306
    b: float = 11.2974
    c: float = 0.00664824
    d: float = 16.9824

    def energy_density_from_pressure_in_MeV4(self, p: float) -> float:
        p = np.abs(p)*cf.MEV_FM3_TO_MEV4
        return 10**(
            self.a+self.b*np.sqrt(
                1.+self.c*(self.d + np.log10(p))**2
            )
        )

    def energy_density_from(self, pressure: float) -> float:
        return self.energy_density_from_pressure_in_MeV4(
            pressure)/cf.MEV_FM3_TO_MEV4

    def sound_speed_squared_from(self, pressure: float) -> float:
        p = np.abs(pressure)*cf.MEV_FM3_TO_MEV4
        return p*(np.sqrt(1.+self.c*(self.d + np.log10(p))**2)) / (
            self.energy_density_from_pressure_in_MeV4(p/cf.MEV_FM3_TO_MEV4) *
            self.b*self.c*(self.d + np.log10(p)))
