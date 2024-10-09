from matplotlib.axes import Axes


MEV_FM_3 = "[\\mathrm{MeV fm^{-3}}]"
KM = "[\\mathrm{km}]"
MSUN = "[\\mathrm{M}_\\odot]"

LABELS: dict[str, str] = {
    "central_pressure": f"$p_c~{MEV_FM_3}$",
    "energy_density": f"$\\epsilon~{MEV_FM_3}$",
    "trace_anomaly": "$\\Delta$",
    "mass": f"$M~{MSUN}$",
    "radius": f"$R~{KM}$",
}


def set_ax_labels(ax: Axes, x: str, y: str) -> None:
    ax.set_ylabel(LABELS[y], fontsize=16)
    ax.set_xlabel(LABELS[x], fontsize=16)


def customize_ax(ax: Axes) -> None:
    ax.legend(fontsize=15)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid()
