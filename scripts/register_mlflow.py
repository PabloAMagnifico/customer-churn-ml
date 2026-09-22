"""Ejecuta y registra los experimentos en MLflow.

Requiere MLflow instalado. Está separado del entrenamiento verificable local para que
el proyecto pueda ejecutarse aun cuando la herramienta MLOps no esté disponible.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import DATA_PATH, RANDOM_STATE, TARGET, TEST_SIZE, MLFLOW_DB  # noqa: E402
from src.data.load import load_historical_data  # noqa: E402
from src.evaluation.metrics import evaluate_binary  # noqa: E402
from src.features.pipeline import build_preprocessor  # noqa: E402
from src.training.models import get_model_specs  # noqa: E402


def main():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_DB}"))
    mlflow.set_experiment("customer-churn-entrega-1")
    df = load_historical_data(DATA_PATH)
    X = df.drop(columns=[TARGET, "customerID"])
    y = (df[TARGET] == "Yes").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    for run_name, spec in get_model_specs().items():
        with mlflow.start_run(run_name=run_name) as run:
            pipe = Pipeline([("preprocess", build_preprocessor()), ("model", spec["estimator"])])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            y_prob = pipe.predict_proba(X_test)[:, 1]
            metrics = evaluate_binary(y_test, y_pred, y_prob)
            mlflow.log_params({"random_state": RANDOM_STATE, "test_size": TEST_SIZE, **spec["params"]})
            if run_name == "logistic_c1":
                mlflow.set_tag("candidate", "true")
            mlflow.log_metrics({k: v for k, v in metrics.items() if k in {"accuracy", "precision", "recall", "f1", "roc_auc"}})
            out = ROOT / "reports" / "metrics" / f"{run_name}.json"
            out.write_text(json.dumps({"run_id": run.info.run_id, "run_name": run_name, **metrics}, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(out), artifact_path="metrics")
            mlflow.sklearn.log_model(pipe, name="model", skops_trusted_types=["numpy.dtype"])

    print("Runs registrados. Abrí MLflow y usa el mejor Run para registrar el candidato en Model Registry.")


if __name__ == "__main__":
    main()
