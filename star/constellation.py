from dataclasses import asdict
from typing import Any, Iterable, Protocol
from concurrent.futures import ProcessPoolExecutor

import pandas as pd


class Factory(Protocol):
    def create_star(self, central_pressure: float) -> Any:
        """Creates a star."""


def create_stellar_family(
    fac: Factory, central_pressures: Iterable[float]
) -> pd.DataFrame:
    """Creates a list of stars/hybrid stars for a given EOS and
    array of central pressures."""
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(fac.create_star, cp) for cp in central_pressures]

    stars = pd.concat([pd.DataFrame(asdict(f.result()), index=[0]) for f in futures])
    stars.reset_index(drop=True, inplace=True)
    stars.dropna(axis=1, inplace=True)

    return stars
