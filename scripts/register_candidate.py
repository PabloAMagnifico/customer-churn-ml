"""Registra como candidato el modelo originado por un Run de MLflow.
Ejecutar después de scripts/register_mlflow.py.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "mlflow.db"


def main():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{DB}"))
    client = MlflowClient()
    exp = client.get_experiment_by_name("customer-churn-entrega-1")
    if exp is None:
        raise RuntimeError("No existe el experimento. Ejecuta primero scripts/register_mlflow.py")
    runs = client.search_runs([exp.experiment_id], order_by=["metrics.recall DESC", "metrics.f1 DESC", "metrics.roc_auc DESC"])
    candidate = next((r for r in runs if r.data.tags.get("candidate") == "true"), None)
    if candidate is None:
        # Se toma el Run logistic_c1 como candidato documentado del proyecto.
        candidate = next((r for r in runs if r.data.tags.get("mlflow.runName") == "logistic_c1"), None)
    if candidate is None:
        raise RuntimeError("No se encontró el Run candidato.")
    model_uri = f"runs:/{candidate.info.run_id}/model"
    name = os.getenv("MLFLOW_REGISTERED_MODEL", "customer-churn")
    mv = mlflow.register_model(model_uri, name)
    client.set_model_version_tag(name, mv.version, "source_run_id", candidate.info.run_id)
    client.set_model_version_tag(name, mv.version, "stage", "candidate")
    print(f"Modelo registrado: {name} versión {mv.version}; source_run_id={candidate.info.run_id}")


if __name__ == "__main__":
    main()
