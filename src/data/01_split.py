import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def main(input_csv: str, out_dir: str, test_size: float, random_state: int):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(input_csv)

    target = "silica_concentrate"
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    X_train.to_csv(os.path.join(out_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(out_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(out_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(out_dir, "y_test.csv"), index=False)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="data/processed")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--random_state", type=int, default=42)
    args = p.parse_args()
    main(args.input_csv, args.out_dir, args.test_size, args.random_state)

