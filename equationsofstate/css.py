from .eos import EquationOfState


class CSS(EquationOfState):
    def __init__(
        self,
        transitional_density: float,
        delta_epsilon: float,
        sound_speed_squared: float,
        transitional_pressure: float,
    ) -> None:
        self.transitional_density: float = transitional_density
        self.delta_epsilon: float = delta_epsilon
        self.sound_speed_squared: float = sound_speed_squared
        self.transitional_pressure: float = transitional_pressure

    def energy_density_from(self, pressure: float) -> float:
        """Computes energy density from pressure."""
        return (
            self.transitional_density
            + self.delta_epsilon
            + (pressure - self.transitional_pressure) / self.sound_speed_squared
        )

    def sound_speed_squared_from(self, pressure: float) -> float:
        return self.sound_speed_squared
