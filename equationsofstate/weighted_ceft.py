from dataclasses import dataclass
from pathlib import Path

from .interpolate import RIPEOS
from .eos import EquationOfState


@dataclass()
class WeightedCEFT(EquationOfState):
    weight: float

    def __post_init__(self) -> None:
        path = (
            Path.cwd()
            / "src"
            / "neutron_stars_computer"
            / "equationsofstate"
            / "tabulated_eos"
            / "Ha_EOS"
        )
        self._eos1 = RIPEOS(str(path / "Hebelerstiff.csv"))
        self._eos2 = RIPEOS(str(path / "Hebelersoft.csv"))

    def energy_density_from(self, pressure: float) -> float:
        return self._eos1.energy_density_from(
            pressure
        ) * self.weight + self._eos2.energy_density_from(pressure) * (1 - self.weight)

    def adiabatic_index_from(self, pressure: float) -> float:
        return self._eos1.adiabatic_index_from(
            pressure
        ) * self.weight + self._eos2.adiabatic_index_from(pressure) * (1 - self.weight)

    def sound_speed_squared_from(self, pressure: float) -> float:
        return self._eos1.sound_speed_squared_from(
            pressure
        ) * self.weight + self._eos2.sound_speed_squared_from(pressure) * (
            1 - self.weight
        )
