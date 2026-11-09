from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient

from app.main import MODEL_PATH, app

SAMPLE = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_model_file_exists_and_loads():
    assert Path(MODEL_PATH).is_file(), (
        "Falta el modelo. Ejecutar: python -m src.training.train"
    )
    model = joblib.load(MODEL_PATH)
    assert hasattr(model, "predict_proba")


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert isinstance(body["model_version"], str)


def test_valid_prediction(client):
    response = client.post("/predict", json=SAMPLE)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in {"CHURN", "NO_CHURN"}
    assert 0 <= body["probability"] <= 1
    assert body["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert body["model_version"]
    assert body["request_id"]


def test_response_types(client):
    body = client.post("/predict", json=SAMPLE).json()
    assert isinstance(body["prediction"], str)
    assert isinstance(body["probability"], (int, float))
    assert isinstance(body["risk_level"], str)
    assert isinstance(body["model_version"], str)
    assert isinstance(body["request_id"], str)


def test_invalid_category_returns_422(client):
    invalid = {**SAMPLE, "PaymentMethod": "Cryptocurrency"}
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422


def test_missing_required_field_returns_422(client):
    invalid = dict(SAMPLE)
    invalid.pop("Contract")
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422


def test_extra_field_returns_422(client):
    invalid = {**SAMPLE, "customerID": "7590-VHVEG"}
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422


def test_optional_total_charges_can_be_missing(client):
    request_data = {**SAMPLE, "TotalCharges": None}
    response = client.post("/predict", json=request_data)
    assert response.status_code == 200
    assert 0 <= response.json()["probability"] <= 1
