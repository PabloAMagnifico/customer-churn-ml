## Entrega 2 - Servicio de inferencia

La API FastAPI carga el pipeline candidato `logistic_c1` desde `models/candidate_logistic_c1.joblib`. Expone `GET /health` y `POST /predict`, con contratos Pydantic, validación de entradas y salida con predicción, probabilidad, nivel de riesgo, versión del modelo e identificador de solicitud.

Instalación y pruebas:

```bash
python -m pip install -r requirements-api.txt
python scripts/pin_api_versions.py
python -m pytest -q tests/test_api.py
```

Ejecución local: `uvicorn app.main:app --host 127.0.0.1 --port 8000` y abrir `http://127.0.0.1:8000/docs`.

Ejecución contenerizada: `docker build -t customer-churn-api:entrega-2 .` y `docker compose up --build -d`. La API queda publicada en el puerto 8000. Más detalle en `docs/ENTREGA_2_PASO_A_PASO.md`.
