from pathlib import Path
import pandas as pd
from src.features.pipeline import build_preprocessor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_exists_and_target_present():
    path = ROOT / "data" / "raw" / "customer_churn_historical.csv"
    df = pd.read_csv(path)
    assert df.shape == (7043, 21)
    assert "Churn" in df.columns
    assert df["Churn"].isin(["Yes", "No"]).all()


def test_preprocessing_and_model_pipeline_fit():
    df = pd.read_csv(ROOT / "data" / "raw" / "customer_churn_historical.csv")
    X = df.drop(columns=["Churn", "customerID"])
    y = (df["Churn"] == "Yes").astype(int)
    pipe = Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", LogisticRegression(max_iter=2000, solver="liblinear", random_state=42)),
    ])
    pipe.fit(X.head(500), y.head(500))
    p = pipe.predict_proba(X.head(5))[:, 1]
    assert len(p) == 5
    assert ((p >= 0) & (p <= 1)).all()
