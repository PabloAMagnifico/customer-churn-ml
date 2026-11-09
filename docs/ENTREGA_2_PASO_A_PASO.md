# Entrega 2 - Servicio de inferencia

Este módulo agrega FastAPI + Pydantic, pytest, Docker y Docker Compose sobre el modelo candidato `logistic_c1` que ya genera `src.training.train`.

## Modelo requerido

El servicio carga una sola vez `models/candidate_logistic_c1.joblib`, que debe contener el pipeline completo (preprocesamiento + regresión logística). Si no existe, desde la raíz del repositorio ejecutar:

```bash
python -m src.training.train
```

No se incluye ningún token ni secreto en el código o la imagen.

## Instalación y tests en Windows / Git Bash

```bash
python -m pip install -r requirements-api.txt
python scripts/pin_api_versions.py
python -m pytest -q tests/test_api.py
```

`pin_api_versions.py` reemplaza las dependencias del modelo en `requirements-api.txt` por las versiones que se usaron para serializar el `.joblib`, reduciendo incompatibilidades al abrirlo en Docker.

## Ejecución local

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Abrir `http://127.0.0.1:8000/docs`. Ejecutar `GET /health` y `POST /predict`. El ejemplo de entrada está en `examples/predict_example.json`.

## Ejecución con Docker

Primero detener Uvicorn local con `Ctrl+C`. Verificar que Docker Desktop esté iniciado y luego ejecutar desde la raíz:

```bash
docker build -t customer-churn-api:entrega-2 .
docker compose up --build -d
docker compose ps
```

Abrir `http://localhost:8000/docs` y repetir `GET /health` y `POST /predict`.

Para observar logs y detener:

```bash
docker compose logs -f
docker compose down
```

## Contrato de la API

- `GET /health`: `{ "status": "ok", "model_version": "logistic_c1-candidate" }`.
- `POST /predict`: recibe las 19 variables sin `customerID` ni `Churn`; responde `prediction`, `probability`, `risk_level`, `model_version` y `request_id`.
- Pydantic devuelve HTTP 422 para campos faltantes, categorías inválidas o campos extra.

## Decisiones

- `prediction`: CHURN cuando la probabilidad estimada es >= 0.50; en caso contrario NO_CHURN.
- `risk_level`: LOW si p < 0.30; MEDIUM si 0.30 <= p < 0.60; HIGH si p >= 0.60. Son bandas operativas iniciales para la demostración académica, no umbrales calibrados con costos reales. Deberían validarse con el negocio y con la matriz de confusión.
- El modelo es el pipeline serializado completo de `logistic_c1`, de modo que el mismo preprocesamiento del entrenamiento se aplica en inferencia.

## Preparación para Git

El `.gitignore` original excluye `models/*.joblib`. Como el Dockerfile copia el archivo dentro de la imagen, para que otro integrante pueda clonar y reconstruir la imagen hay que versionar explícitamente el artefacto:

```bash
git add app tests/test_api.py Dockerfile .dockerignore compose.yaml requirements-api.txt examples/predict_example.json scripts/pin_api_versions.py docs/ENTREGA_2_PASO_A_PASO.md
git add -f models/candidate_logistic_c1.joblib
git add README.md
git commit -m "Entrega 2 - FastAPI, tests y Docker Compose"
git push origin main
git tag -f entrega-2
git push origin -f entrega-2
```

No agregar CSV al Git. Los datos siguen separados mediante DVC.
