#!/usr/bin/env python3
"""
Quick sanity-check of the XGBoost tuning pipeline on a small sample.
"""
import argparse
import json
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
import xgboost as xgb

def main():
    parser = argparse.ArgumentParser(
        description="Simple XGBoost hyperparam test with logging"
    )
    parser.add_argument(
        "--parquet", type=str, required=True,
        help="Path to the Parquet file"
    )
    parser.add_argument(
        "--sample-size", type=int, default=1000,
        help="Number of rows to sample for quick test"
    )
    parser.add_argument(
        "--output", type=str, default="best_params_simple.json",
        help="Where to save best params + score as JSON"
    )
    args = parser.parse_args()

    print(f"[INFO] Loading data from {args.parquet}", flush=True)
    df = pd.read_parquet(args.parquet)
    print(f"[INFO] Original DataFrame shape: {df.shape}", flush=True)

    n = min(args.sample_size, len(df))
    df = df.sample(n=n, random_state=42)
    print(f"[INFO] Sampled {n} rows", flush=True)

    X = df.select_dtypes(include=[np.number]).drop(columns=["TARGET"], errors="ignore")
    y = df["TARGET"]
    print(f"[INFO] Features shape: {X.shape}", flush=True)

    print("[INFO] Defining CV and parameter distribution", flush=True)
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    param_dist = {
        "n_estimators":    [50, 100],
        "max_depth":       [3, 5],
        "learning_rate":   [0.1, 0.2],
    }
    print(f"[INFO] Parameter distribution: {param_dist}", flush=True)

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        #use_label_encoder=False,
        tree_method="hist",   # fast CPU
        random_state=42
    )
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=4,
        scoring="neg_log_loss",
        cv=cv,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    print("[INFO] Starting RandomizedSearchCV", flush=True)
    search.fit(X, y)
    print("[INFO] Search done", flush=True)
    best_params = search.best_params_
    best_score  = -search.best_score_
    print(f"[RESULT] Best params: {best_params}", flush=True)
    print(f"[RESULT] Best log-loss: {best_score:.5f}", flush=True)

    # Ensure output directory exists
    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Save to JSON
    with open(args.output, "w") as f:
        json.dump({
            "best_params": best_params,
            "best_log_loss": best_score
        }, f, indent=2)
    print(f"[INFO] Saved results to {args.output}", flush=True)

if __name__ == "__main__":
    main()