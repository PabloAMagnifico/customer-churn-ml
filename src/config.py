from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "customer_churn_historical.csv"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
MODELS_DIR = ROOT / "models"
MLRUNS_DIR = ROOT / "mlruns"
MLFLOW_DB = ROOT / "mlflow.db"
RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "Churn"
ID_COLUMN = "customerID"
