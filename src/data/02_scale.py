import argparse
import os
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

def main(proc_dir: str, scaler_path: str):
    X_train = pd.read_csv(os.path.join(proc_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(proc_dir, "X_test.csv"))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    X_train_scaled.to_csv(os.path.join(proc_dir, "X_train_scaled.csv"), index=False)
    X_test_scaled.to_csv(os.path.join(proc_dir, "X_test_scaled.csv"), index=False)

    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--proc_dir", type=str, default="data/processed")
    parser.add_argument("--scaler_path", type=str, default="models/scaler.pkl")
    args = parser.parse_args()
    main(args.proc_dir, args.scaler_path)
