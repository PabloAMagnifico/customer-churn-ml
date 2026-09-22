from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import DATA_PATH, FIGURES_DIR, METRICS_DIR  # noqa: E402
from src.data.load import load_historical_data  # noqa: E402


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_historical_data(DATA_PATH)

    quality = {
        "rows": int(df.shape[0]), "columns": int(df.shape[1]),
        "duplicates": int(df.duplicated().sum()),
        "target_counts": df["Churn"].value_counts().to_dict(),
        "missing_by_column": df.isna().sum().sort_values(ascending=False).to_dict(),
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        "numeric_summary": df.select_dtypes(include="number").describe().round(3).to_dict(),
    }
    (METRICS_DIR / "eda_summary.json").write_text(json.dumps(quality, indent=2, default=str), encoding="utf-8")

    # Target distribution
    df["Churn"].value_counts().reindex(["No", "Yes"]).plot(kind="bar", rot=0)
    plt.title("Distribución del target Churn")
    plt.xlabel("Churn")
    plt.ylabel("Cantidad")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "target_distribution.png", dpi=150)
    plt.close()

    # Missing values
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if not missing.empty:
        missing.plot(kind="bar")
        plt.title("Valores faltantes por variable")
        plt.ylabel("Cantidad")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "missing_values.png", dpi=150)
        plt.close()

    # Churn by contract: useful business variable
    churn_contract = pd.crosstab(df["Contract"], df["Churn"], normalize="index")
    churn_contract["Yes"].sort_values(ascending=False).plot(kind="bar")
    plt.title("Proporción de churn por tipo de contrato")
    plt.xlabel("Contrato")
    plt.ylabel("Proporción Churn=Yes")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "churn_by_contract.png", dpi=150)
    plt.close()
    print(json.dumps(quality, indent=2, default=str))


if __name__ == "__main__":
    main()
