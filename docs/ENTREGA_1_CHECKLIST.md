# Checklist — Entrega 1 / Primer Parcial

| Requisito | Estado en este proyecto | Evidencia |
|---|---|---|
| Repositorio estructurado | Listo | README.md + src/ + tests/ + scripts/ |
| README con reproducción | Listo | README.md |
| EDA razonable | Listo | reports/eda.md + reports/figures/ |
| Split reproducible train/test | Listo | src/training/train.py + reports/metrics/split.json |
| Pipeline sklearn | Listo | src/features/pipeline.py |
| Baseline + lineal + árbol | Listo | src/training/models.py |
| ≥ 6 runs razonados | Preparado y ejecutado localmente sin MLflow | reports/metrics/model_comparison.csv |
| DVC | Estructura lista | data/raw/*.dvc + dvc.yaml |
| Remote DagsHub | Requiere credenciales/repositorio del equipo | README.md |
| MLflow | Código de registro listo | scripts/register_mlflow.py |
| Model Registry | Flujo de registro listo; ejecutar con MLflow | scripts/register_candidate.py |
| Entrenamiento sin notebook | Listo | python -m src.training.train |
| Tag entrega-1 | Crear en GitHub sobre commit final | README.md |
