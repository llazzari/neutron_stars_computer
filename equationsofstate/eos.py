from abc import ABC, abstractmethod


class EquationOfState(ABC):
    @abstractmethod
    def energy_density_from(self, pressure: float) -> float:
        """Computes energy density from pressure."""

    def adiabatic_index_from(self, pressure: float) -> float:
        """Computes adiabatic index from pressure."""
        en_dens: float = self.energy_density_from(pressure)
        sound_speed_squared: float = self.sound_speed_squared_from(pressure)

        return (1 + en_dens / pressure) * sound_speed_squared

    def sound_speed_squared_from(self, pressure: float) -> float:
        """Computes sound speed squared from pressure
        using central differences."""
        h: float = pressure * 1e-4

        en_dens_plus_h: float = self.energy_density_from(pressure + h)
        en_dens_minus_h: float = self.energy_density_from(pressure - h)
        dedp: float = (en_dens_plus_h - en_dens_minus_h) / (2 * h)

        return 1 / dedp
