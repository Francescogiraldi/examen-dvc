import argparse
import json
import os
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def main(proc_dir: str, model_path: str, pred_path: str, metrics_path: str):
    X_test = pd.read_csv(os.path.join(proc_dir, "X_test_scaled.csv"))
    y_test = pd.read_csv(os.path.join(proc_dir, "y_test.csv")).squeeze("columns")

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    pd.DataFrame({"y_true": y_test.values, "y_pred": y_pred}).to_csv(pred_path, index=False)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--proc_dir", type=str, default="data/processed")
    p.add_argument("--model_path", type=str, default="models/model.pkl")
    p.add_argument("--pred_path", type=str, default="data/predictions.csv")
    p.add_argument("--metrics_path", type=str, default="metrics/scores.json")
    args = p.parse_args()
    main(args.proc_dir, args.model_path, args.pred_path, args.metrics_path)
