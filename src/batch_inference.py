import mlflow
import pandas as pd
from utils import load_csv, preprocess
from config import Config

if __name__ == "__main__":
    df = load_csv(Config.SCORING_PATH)
    df = preprocess(df)

    model = mlflow.sklearn.load_model("models:/telemetry-model/latest")
    predictions = model.predict(df)

    output = pd.DataFrame({"prediction": predictions})
    output.to_csv(f"{Config.PREDICTIONS_OUTPUT}/batch_predictions.csv", index=False)

    print("Inference complete. Predictions saved.")
