from pathlib import Path

# Base directory for the project
BASE_DIR = Path(__file__).resolve().parents[1]

# Data locations
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
TRAINING_DATA_PATH = RAW_DATA_DIR / "telemetry_training.csv"
SCORING_DATA_PATH = RAW_DATA_DIR / "telemetry_scoring.csv"

PREDICTIONS_DIR = BASE_DIR / "data" / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

# MLflow configuration
MLFLOW_TRACKING_URI = f"file:{(BASE_DIR / 'mlruns').as_posix()}"
EXPERIMENT_NAME = "telemetry_failure_risk"
