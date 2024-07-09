from matplotlib.axes import Axes
import pandas as pd
from pathlib import Path

CONSTRAINTS_ARGS: dict[str, list[str | None]] = {
    "GW170817_UR.M1R1": ["GW170817 (UR)", "//", "b"],
    "GW170817_UR.M2R2": [None, "//", "b"],
    "GW170817_EOS.M1R1": ["GW170817 (spec EOS)", "//", "b"],
    "GW170817_EOS.M2R2": [None, "//", "b"],
    "Miller_PSR_J0030.MR": ["PSR J0030+0451 (Miller)", "|", "g"],
    "Riley_PSR_J0030.MR": ["PSR J0030+0451 (Riley)", "|", "g"],
    "Miller_PSR_J0740.MR": ["PSR J0740+6620 (Miller)", "-", "r"],
    "Riley_PSR_J0740.MR": ["PSR J0740+6620 (Riley)", "-", "r"],
    "HESS_J1731-347.csv": ["HESS J1731-347", "x", "orange"],
}
CONSTRAINTS_PATH: Path = Path.cwd() / "constraints" / "data"


def plot_constraint(axmr: Axes, desired_constraints: list[str] = []) -> None:

    if not desired_constraints:
        desired_constraints = list(CONSTRAINTS_ARGS.keys())
    else:
        for gw_t1, gw_t2 in zip(["UR", "EOS"], ["EOS", "UR"]):
            if f"GW170817_{gw_t1}.M1R1" in desired_constraints:
                desired_constraints.append(f"GW170817_{gw_t1}.M2R2")
                if f"GW170817_{gw_t2}.M1R1" not in desired_constraints:
                    CONSTRAINTS_ARGS[f"GW170817_{gw_t1}.M1R1"][0] = "GW170817"

        for n1, n2 in zip(["Miller", "Riley"], ["Riley", "Miller"]):
            for psr_f, psr_l in zip(
                ["PSR_J0030", "PSR_J0740"], ["PSR J0030+0451", "PSR J0740+6620"]
            ):
                if (f"{n1}_{psr_f}.MR" in desired_constraints) and (
                    f"{n2}_{psr_f}.MR" not in desired_constraints
                ):
                    CONSTRAINTS_ARGS[f"{n1}_{psr_f}.MR"][0] = psr_l

    for f in desired_constraints:
        constraints_df = pd.read_csv(CONSTRAINTS_PATH / f, sep=" ")
        axmr.plot(
            constraints_df["r1"],
            constraints_df["m1"],
            c=CONSTRAINTS_ARGS[f][2],
            lw=0.25,
        )
        axmr.fill(
            constraints_df["r1"],
            constraints_df["m1"],
            label=CONSTRAINTS_ARGS[f][0],
            hatch=CONSTRAINTS_ARGS[f][1],
            c=CONSTRAINTS_ARGS[f][2],
            alpha=0.15,
        )
