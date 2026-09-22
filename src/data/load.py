from pathlib import Path
import pandas as pd


def load_historical_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Churn" not in df.columns:
        raise ValueError("El dataset histórico debe contener la columna Churn.")
    return df
