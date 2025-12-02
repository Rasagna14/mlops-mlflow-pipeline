# mlops-mlflow-pipeline

# Telemetry Failure Risk – MLflow MLOps Demo

This repo contains a small but realistic MLOps-style workflow based on vehicle telemetry data.  
The idea is similar to what I’ve worked on in simulation pipelines: we take aggregated driving features and train a model to estimate the short-term risk of a failure event.

## Project Goals

- Show how I structure a lightweight MLOps project around MLflow.
- Demonstrate feature engineering on telemetry-style data.
- Track experiments and metrics in MLflow.
- Support batch inference with a "latest good model" pattern.

## Architecture

- **Data**: synthetic telemetry dataset (`avg_speed`, `max_speed`, braking, rain, night flag).
- **Features**: normalized rates and interaction terms (brake rate, hard-brake ratio, heavy rain flag).
- **Model**: RandomForest classifier predicting `failure_within_24h`.
- **Tracking**: MLflow experiment with parameters, metrics, artifacts, and serialized model.
- **Serving pattern**: batch scoring script that always loads the latest model for the experiment.

## Project Layout

```text
data/
  raw/
    telemetry_training.csv
    telemetry_scoring.csv
  predictions/
    telemetry_predictions.csv        # created by batch_inference.py

src/
  config.py          # paths + MLflow configuration
  utils.py           # shared feature engineering
  train_model.py     # training + tracking
  batch_inference.py # batch scoring using latest model

requirements.txt
README.md

Running Locally

Create a virtual environment and install dependencies:

pip install -r requirements.txt


Train a model and log it to MLflow:

python src/train_model.py

This will create an mlruns/ directory locally and register the run under the
telemetry_failure_risk experiment.

Run batch inference with the latest model:

python src/batch_inference.py

This reads data/raw/telemetry_scoring.csv and writes predictions to:

data/predictions/telemetry_predictions.csv
