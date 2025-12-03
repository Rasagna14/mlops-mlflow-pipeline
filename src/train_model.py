import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import os


def load_data(path: str):
    return pd.read_csv(path)


def train_model(df: pd.DataFrame):

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run():

        mlflow.autolog()  # ← automatically logs params, metrics, model

        model = RandomForestRegressor(
            n_estimators=120,
            max_depth=10,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Evaluate
        preds = model.predict(X_test)
        rmse = mean_squared_error(y_test, preds, squared=False)

        # Manual logging for visibility
        mlflow.log_metric("rmse", rmse)

        # Register model in MLflow Registry
        model_uri = mlflow.get_artifact_uri("model")
        mlflow.register_model(model_uri, "telemetry_model")  # ← creates versioned model

        print(f"Model trained and registered with RMSE: {rmse:.4f}")


if __name__ == "__main__":
    data_path = "data/processed/telemetry_features.csv"
    df = load_data(data_path)
    train_model(df)
