import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(
        path,
        delim_whitespace=True,
        usecols=['p', 'e'],
        dtype={'p': float, 'e': float}
    )


def compute_trace_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    df['Delta'] = 1/3 - df['p']/df['e']
    return df


def process_data(path: str) -> pd.DataFrame:
    df: pd.DataFrame = load_data(path)
    return compute_trace_anomaly(df)
