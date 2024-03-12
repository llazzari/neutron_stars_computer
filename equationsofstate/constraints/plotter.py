import numpy as np
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from pathlib import Path
from matplotlib.axes import Axes


CONSTRAINTS_PATH = Path.cwd() / 'equationsofstate' / 'constraints' / 'data'
CEFT_SOFT_PATH = CONSTRAINTS_PATH / 'cEFT_soft.csv'
CEFT_STIFF_PATH = CONSTRAINTS_PATH / 'cEFT_stiff.csv'
PQCD_X1_PATH = CONSTRAINTS_PATH / 'FKV_X1.csv'
PQCD_X4_PATH = CONSTRAINTS_PATH / 'FKV_X4.csv'

KK_PATH = CONSTRAINTS_PATH / 'ep_Kurkela_Komoltsev.csv'

CEFT_EN_DENS = np.linspace(100, 165, 30)
PQCD_EN_DENS = np.linspace(1.2e4, 2.1e4, 30)


def process_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=' ', usecols=['e', 'p'])

    def compute_trace_anomaly(df: pd.DataFrame) -> pd.DataFrame:
        df['Delta'] = 1/3 - df['p']/df['e']
        return df
    return compute_trace_anomaly(df)


def interpolate_trace(df: pd.DataFrame) -> Spline:
    return Spline(df['e'], df['Delta'])


def set_splines(paths: tuple[str, str]) -> list[Spline]:
    return [interpolate_trace(process_data(path)) for path in paths]


def fill(ax: Axes, paths: tuple[str, str], en_dens: np.ndarray) -> None:
    ys_splines = set_splines(paths)
    ys = [y_spl(en_dens) for y_spl in ys_splines]
    ax.fill_between(en_dens, y1=ys[0], y2=ys[1], color='grey')  # type: ignore


def plot(ax: Axes, desired_constraints: list[str]) -> None:
    if 'cEFT' in desired_constraints:
        paths = (str(CEFT_SOFT_PATH), str(CEFT_STIFF_PATH))
        fill(ax, paths, CEFT_EN_DENS)

    if 'pQCD' in desired_constraints:
        paths = (str(PQCD_X1_PATH), str(PQCD_X4_PATH))
        fill(ax, paths, PQCD_EN_DENS)

    if 'KK' in desired_constraints:
        df = process_data(str(KK_PATH))
        ax.fill(df['e'], df['Delta'], color='orange', hatch='x', alpha=0.2)
