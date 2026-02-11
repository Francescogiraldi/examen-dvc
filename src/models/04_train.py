import argparse
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor


def main(proc_dir: str, params_path: str, model_path: str, random_state: int):
    X_train = pd.read_csv(os.path.join(proc_dir, "X_train_scaled.csv"))
    y_train = pd.read_csv(os.path.join(proc_dir, "y_train.csv")).squeeze("columns")

    best_params = joblib.load(params_path)

    model = RandomForestRegressor(random_state=random_state, **best_params)
    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--proc_dir", type=str, default="data/processed")
    p.add_argument("--params_path", type=str, default="models/best_params.pkl")
    p.add_argument("--model_path", type=str, default="models/model.pkl")
    p.add_argument("--random_state", type=int, default=42)
    args = p.parse_args()
    main(args.proc_dir, args.params_path, args.model_path, args.random_state)
