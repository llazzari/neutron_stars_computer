from dataclasses import dataclass
from ..equationsofstate.eos import EquationOfState


@dataclass(frozen=True, slots=True)
class HybridEOS(EquationOfState):
    """Defines a hybrid equation of state using the Maxwell construction."""

    eos1: EquationOfState
    eos2: EquationOfState
    transitional_pressure: float

    def energy_density_from(self, pressure: float) -> float:
        return (
            self.eos1.energy_density_from(pressure)
            if (pressure <= self.transitional_pressure)
            else self.eos2.energy_density_from(pressure)
        )

    def adiabatic_index_from(self, pressure: float) -> float:
        return (
            self.eos1.adiabatic_index_from(pressure)
            if (pressure <= self.transitional_pressure)
            else self.eos2.adiabatic_index_from(pressure)
        )
