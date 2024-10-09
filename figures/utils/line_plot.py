import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes


def plot_hybrid(ax: Axes, x: str, y: str, df: pd.DataFrame, conv_speed: str) -> None:

    style: str | None = "Stability" if conv_speed == "rapid" else None
    style_order: list[str] | None = (
        ["Stable", "Unstable"] if conv_speed == "rapid" else None
    )

    sns.lineplot(
        df,
        x=x,
        y=y,
        hue="EOS",
        style=style,
        style_order=style_order,
        sort=False,
        palette="Set1",
        ax=ax,
        linewidth=2,
    )


def plot_hadronic(ax: Axes, x: str, y: str, df: pd.DataFrame) -> None:
    ax.plot(df[x], df[y], color="black", lw=2, label="Pure hadronic")
