from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Literal, Optional
from uuid import uuid4

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(ROOT / "models" / "candidate_logistic_c1.joblib")))
MODEL_VERSION = os.getenv("MODEL_VERSION", "logistic_c1-candidate")
PREDICTION_THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.50"))

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("customer-churn-api")


class CustomerFeatures(BaseModel):
    """Entrada con las mismas 19 columnas que usa el pipeline de entrenamiento."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
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
        },
    )

    gender: Literal["Female", "Male"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=120)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: Optional[float] = Field(default=None, ge=0)


class PredictionResponse(BaseModel):
    prediction: Literal["CHURN", "NO_CHURN"]
    probability: float = Field(ge=0, le=1)
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    model_version: str
    request_id: str


def load_model():
    """Carga una sola vez el pipeline serializado completo (preprocessing + estimator)."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"No se encontró el modelo en {MODEL_PATH}. "
            "Ejecutá 'python -m src.training.train' para generarlo."
        )
    model = joblib.load(MODEL_PATH)
    if not hasattr(model, "predict_proba"):
        raise TypeError("El artefacto cargado no expone predict_proba().")
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()
    logger.info("service_started model_version=%s model_path=%s", MODEL_VERSION, MODEL_PATH)
    yield
    app.state.model = None
    logger.info("service_stopped")


app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Servicio REST para estimar la probabilidad de abandono de clientes. "
        "Utiliza el pipeline scikit-learn candidato logistic_c1."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["operación"])
def health(request: Request):
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="El modelo no está cargado.")
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict", response_model=PredictionResponse, tags=["inferencia"])
def predict(payload: CustomerFeatures, request: Request):
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="El modelo no está cargado.")

    request_id = str(uuid4())
    started = perf_counter()
    try:
        row = payload.model_dump()
        # La columna TotalCharges admite ausencia; convertir None a NaN numérico
        # permite que el SimpleImputer del pipeline aplique la mediana aprendida.
        if row["TotalCharges"] is None:
            row["TotalCharges"] = float("nan")
        features = pd.DataFrame([row])
        probability = float(model.predict_proba(features)[0][1])
    except Exception as exc:
        logger.exception("prediction_failed request_id=%s", request_id)
        raise HTTPException(status_code=500, detail="No se pudo ejecutar la predicción.") from exc

    prediction = "CHURN" if probability >= PREDICTION_THRESHOLD else "NO_CHURN"
    if probability < 0.30:
        risk_level = "LOW"
    elif probability < 0.60:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    elapsed_ms = round((perf_counter() - started) * 1000, 2)
    logger.info(
        "prediction request_id=%s model_version=%s probability=%.6f prediction=%s latency_ms=%.2f",
        request_id,
        MODEL_VERSION,
        probability,
        prediction,
        elapsed_ms,
    )
    return PredictionResponse(
        prediction=prediction,
        probability=probability,
        risk_level=risk_level,
        model_version=MODEL_VERSION,
        request_id=request_id,
    )
