# Customer Churn ML — Entrega 1

Proyecto Integrador de Laboratorio de Minería de Datos — Pablo Magnifico.

## Objetivo

Construir un flujo reproducible para estimar la probabilidad de abandono (`Churn`) de clientes de telecomunicaciones.

## Estructura

```text
customer-churn-ml/
├── data/
│   ├── raw/customer_churn_historical.csv        
│   ├── raw/customer_churn_historical.csv.dvc
│   ├── production/                              
│   └── scoring/
├── metadata/
├── src/
│   ├── data/
│   ├── features/
│   ├── training/
│   └── evaluation/
├── scripts/
├── tests/
├── models/
├── reports/
│   ├── figures/
│   └── metrics/
├── dvc.yaml
├── params.yaml
├── requirements.txt
└── README.md
```

## Datos

El histórico contiene 7.043 observaciones, 21 columnas y la variable objetivo `Churn`. Hay 26 faltantes en `TotalCharges`. `customerID` no se utiliza como predictor. El lote `data/production/customer_churn_current.csv` no se usa para entrenar y queda reservado para monitoreo/drift.

## Instalación

Se recomienda Python 3.11–3.12 para minimizar diferencias de entorno con las herramientas MLOps.

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

## EDA

```bash
python scripts/run_eda.py
```

Genera `reports/eda.md`, `reports/metrics/eda_summary.json` y gráficos en `reports/figures/`.

## Entrenamiento reproducible

```bash
python -m src.training.train
```

La ejecución hace un split 80/20 estratificado con seed 42 y entrena seis alternativas:

1. DummyClassifier — majority
2. DummyClassifier — stratified
3. LogisticRegression — C=0.1
4. LogisticRegression — C=1.0
5. RandomForest — 250 árboles, profundidad 6
6. RandomForest — 250 árboles, profundidad 12

La tabla final queda en `reports/metrics/model_comparison.csv` y la candidata en `models/candidate_logistic_c1.joblib`.



## MLflow + Model Registry

Una vez instalado MLflow:

```bash
python scripts/register_mlflow.py
python scripts/register_candidate.py
```

El primer script registra parámetros, métricas, artefactos y modelos para seis runs bajo el experimento `customer-churn-entrega-1`. El segundo registra el modelo candidato en el Model Registry y agrega `source_run_id` para mantener el lineage.

Para inspeccionar localmente:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Luego abrir el host/puerto indicado por MLflow.



