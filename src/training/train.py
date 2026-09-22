from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import DATA_PATH, FIGURES_DIR, METRICS_DIR, MODELS_DIR, RANDOM_STATE, TARGET, TEST_SIZE  # noqa: E402
from src.data.load import load_historical_data  # noqa: E402
from src.evaluation.metrics import evaluate_binary  # noqa: E402
from src.features.pipeline import build_preprocessor  # noqa: E402
from src.training.models import get_model_specs  # noqa: E402


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_historical_data(DATA_PATH)
    X = df.drop(columns=[TARGET, "customerID"])
    y = (df[TARGET] == "Yes").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    split_info = {
        "total_rows": len(df), "train_rows": len(X_train), "test_rows": len(X_test),
        "test_size": TEST_SIZE, "random_state": RANDOM_STATE,
        "train_positive_rate": float(y_train.mean()), "test_positive_rate": float(y_test.mean()),
    }
    (METRICS_DIR / "split.json").write_text(json.dumps(split_info, indent=2), encoding="utf-8")

    results = []
    for run_name, spec in get_model_specs().items():
        pipe = Pipeline([
            ("preprocess", build_preprocessor()),
            ("model", spec["estimator"]),
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        metrics = evaluate_binary(y_test, y_pred, y_prob)
        row = {"run_name": run_name, **metrics, **{f"param_{k}": v for k, v in spec["params"].items()}}
        results.append(row)
        if run_name == "logistic_c1":
            joblib.dump(pipe, MODELS_DIR / "candidate_logistic_c1.joblib")

    result_df = pd.DataFrame(results).sort_values(["recall", "f1", "roc_auc"], ascending=False)
    result_df.to_csv(METRICS_DIR / "model_comparison.csv", index=False)
    selected = result_df[result_df["run_name"] == "logistic_c1"].iloc[0].to_dict()
    (METRICS_DIR / "selected_model.json").write_text(json.dumps(selected, indent=2, default=str), encoding="utf-8")

    print(result_df[["run_name", "precision", "recall", "f1", "roc_auc", "fn", "fp"]].to_string(index=False))
    print("\nModelo candidato guardado en models/candidate_logistic_c1.joblib")


if __name__ == "__main__":
    main()
