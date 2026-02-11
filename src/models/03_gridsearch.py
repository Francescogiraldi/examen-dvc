import argparse
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


def main(proc_dir: str, out_path: str, cv: int, n_jobs: int, random_state: int):
    X_train = pd.read_csv(os.path.join(proc_dir, "X_train_scaled.csv"))
    y_train = pd.read_csv(os.path.join(proc_dir, "y_train.csv")).squeeze("columns")

    model = RandomForestRegressor(random_state=random_state)

    param_grid = {
        "n_estimators": [200, 500],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=cv,
        n_jobs=n_jobs,
        verbose=1,
    )
    gs.fit(X_train, y_train)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(gs.best_params_, out_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--proc_dir", type=str, default="data/processed")
    p.add_argument("--out_path", type=str, default="models/best_params.pkl")
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--n_jobs", type=int, default=-1)
    p.add_argument("--random_state", type=int, default=42)
    args = p.parse_args()
    main(args.proc_dir, args.out_path, args.cv, args.n_jobs, args.random_state)
