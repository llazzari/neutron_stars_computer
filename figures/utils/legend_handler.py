from matplotlib.axes import Axes


def handle_legend(ax: Axes, c_leg) -> None:
    han, lab = ax.get_legend_handles_labels()
    stability_index = lab.index("Stability")

    ax.legend(
        han[stability_index : stability_index + 3],
        lab[stability_index : stability_index + 3],
        fontsize=15,
        loc="upper right",
        ncol=1,
    )

    ax.add_artist(c_leg)
