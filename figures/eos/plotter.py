import pandas as pd
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from ..utils.line_plot import plot_hadronic, plot_hybrid

MARKERS: list[str] = ["o", "v", "^", "<", ">", "s", "*", "P", "D", "X"]
MARKER_OPTIONS: dict = {
    "markersize": 7,
    "markeredgecolor": "black",
    "markerfacecolor": "black",
}


def set_markers(
    ax: Axes, df_had: pd.DataFrame, df_hyb: pd.DataFrame, x: str, y: str
) -> None:
    transitional_pressures = df_hyb["Transitional pressure"].unique()

    marker: dict[int, str] = dict(zip(transitional_pressures, MARKERS))

    for eos in df_hyb["EOS"].unique():
        df: pd.DataFrame = df_hyb[df_hyb["EOS"] == eos]
        transitional_pressure = int(df["Transitional pressure"].iloc[0])

        had: dict[str, float] = hadronic_structure_at(transitional_pressure, df_had)

        ax.plot(had[x], had[y], **MARKER_OPTIONS)
        ax.plot(
            df[x].iloc[0],
            df[y].iloc[0],
            marker=marker[transitional_pressure],
            **MARKER_OPTIONS
        )


def hadronic_structure_at(
    transitional_pressure: float, df_had: pd.DataFrame
) -> dict[str, float]:

    return {
        key: float(
            Spline(df_had["central_pressure"], df_had[key])(transitional_pressure)
        )
        for key in df_had.columns
        if key != "central_pressure"
    }


def plot_eos(
    ax: Axes, df_had: pd.DataFrame, df_hyb: pd.DataFrame, x: str, y: str
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    plot_hadronic(ax, x, y, df_had)
    # plot_hybrid(ax, x, y, df_hyb, "rapid")
