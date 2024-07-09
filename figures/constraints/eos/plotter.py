import numpy as np
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from pathlib import Path
from matplotlib.axes import Axes


CONSTRAINTS_PATH: Path = Path.cwd() / "equationsofstate" / "constraints" / "data"
CONSTRAINTS_FILES: dict[str, list[str]] = {
    "cEFT": ["cEFT_soft", "cEFT_stiff"],
    "pQCD": ["FKV_X1", "FKV_X4"],
    "KK": ["ep_Kurkela_Komoltsev"],
}
CONSTRAINTS_PATHS: dict[str, list[Path]] = {
    constraint: [CONSTRAINTS_PATH / f"{file}.csv" for file in files]
    for constraint, files in CONSTRAINTS_FILES.items()
}

CEFT_EN_DENS = np.linspace(100, 165, 30)
PQCD_EN_DENS = np.linspace(1.2e4, 2.1e4, 30)


def process_data(path: Path) -> pd.DataFrame:
    df: pd.DataFrame = pd.read_csv(
        path, sep=" ", usecols=["e", "p"], dtype={"e": float, "p": float}
    )

    def compute_trace_anomaly(df: pd.DataFrame) -> pd.DataFrame:
        df["Delta"] = 1 / 3 - df["p"] / df["e"]
        return df

    return compute_trace_anomaly(df)


def fill(ax: Axes, paths: list[Path], en_dens: np.ndarray) -> None:
    def set_splines() -> list[Spline]:
        def interpolate_trace(df: pd.DataFrame) -> Spline:
            return Spline(df["e"], df["Delta"])

        return [interpolate_trace(process_data(path)) for path in paths]

    ys_splines: list[Spline] = set_splines()
    ys = [y_spl(en_dens) for y_spl in ys_splines]
    ax.fill_between(en_dens, y1=ys[0], y2=ys[1], color="grey")  # type: ignore


def plot_eos_constraints(ax: Axes, desired_constraints: list[str]) -> None:
    for constraint in desired_constraints:
        if constraint == "KK":
            df: pd.DataFrame = process_data(CONSTRAINTS_PATHS[constraint][0])
            ax.fill(df["e"], df["Delta"], color="orange")
            continue
        fill(
            ax,
            CONSTRAINTS_PATHS[constraint],
            CEFT_EN_DENS if constraint == "cEFT" else PQCD_EN_DENS,
        )
