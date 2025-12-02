import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import config
from utils import build_features


def main():
    # Configure MLflow
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)

    # Load data
    df = pd.read_csv(config.TRAINING_DATA_PATH)

    X = build_features(df)
    y = df["failure_within_24h"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    params = {
        "n_estimators": 200,
        "max_depth": 4,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": 42,
    }

    with mlflow.start_run(run_name="rf_telemetry_risk"):
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # Validation metrics
        val_probs = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_probs)

        # Log parameters + metrics
        mlflow.log_params(params)
        mlflow.log_metric("val_auc", float(auc))

        # Feature importances as an artifact
        importance_df = (
            pd.DataFrame(
                {
                    "feature": X.columns,
                    "importance": model.feature_importances_,
                }
            )
            .sort_values("importance", ascending=False)
        )
        importance_df.to_csv("feature_importances.csv", index=False)
        mlflow.log_artifact("feature_importances.csv")

        # Log model
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"Training complete. Validation AUC: {auc:.3f}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
