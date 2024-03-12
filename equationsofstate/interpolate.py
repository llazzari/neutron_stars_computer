import pandas as pd
from dataclasses import dataclass
from typing import Any
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

from equationsofstate.eos import EquationOfState


def read_table(file_path: str) -> pd.DataFrame:
    """
    Read table contained in a .csv file whose path is specified in file_path.
    Table header must be like: e p n cs2 gamma (a single white space).
    Data from the table is expected to be separated by a white space.
    """
    if not file_path.endswith('.csv'):
        raise ValueError('File format is not valid.')
    return pd.read_csv(file_path, delim_whitespace=True)


def interpolate_table(df: pd.DataFrame) -> dict[str, Spline]:
    """
    Interpolate Equation Of State from data table.
    Pressure (p) and energy density (e) are expected to be in MeV fm^-3!
    Baryon number density (n, optional) is expected to be in fm^-3!
    Sound speed squared (cs2, optional) is expected to be dimensionless!
    Adiabatic index (gamma, optional) is expected to be dimensionless!
    """

    cols: list[str] = df.columns.to_list()
    cols.remove('p')

    return {col: Spline(df['p'], df[col], k=1) for col in cols}


@dataclass
class RIPEOS(EquationOfState):
    """Read and InterPolate Equation Of State from .csv file."""

    file_path: str

    def __post_init__(self) -> None:
        df: pd.DataFrame = read_table(self.file_path)
        self.interpolations: dict[str, Any] = interpolate_table(df)

        if 'gamma' not in self.interpolations.keys():
            self.interpolations['gamma'] = super().adiabatic_index_from

        if 'cs2' not in self.interpolations.keys():
            self.interpolations['cs2'] = super().sound_speed_squared_from

    def energy_density_from(self, pressure: float) -> float:
        return float(self.interpolations['e'](pressure))

    def baryon_density_from(self, pressure: float) -> float:
        return float(self.interpolations['n'](pressure))

    def adiabatic_index_from(self, pressure: float) -> float:
        return float(self.interpolations['gamma'](pressure))

    def sound_speed_squared_from(self, pressure: float) -> float:
        return float(self.interpolations['cs2'](pressure))
