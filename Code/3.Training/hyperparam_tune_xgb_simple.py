#!/usr/bin/env python3
"""
Quick sanity-check of the XGBoost tuning pipeline on a small sample.
"""
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
import xgboost as xgb

def main():
    parser = argparse.ArgumentParser(
        description="Simple XGBoost hyperparam test"
    )
    parser.add_argument("--parquet",      type=str, required=True,
                        help="Path to Parquet file")
    parser.add_argument("--sample-size",  type=int, default=1000,
                        help="Number of rows to sample for quick test")
    args = parser.parse_args()

    # 1) Load a small random sample in pandas
    df = pd.read_parquet(args.parquet).sample(n=args.sample_size, random_state=42)

    # 2) Build feature matrix & target
    #    keep only numeric columns and drop the target itself
    X = df.select_dtypes(include=[np.number]).drop(columns=["TARGET"], errors="ignore")
    y = df["TARGET"]

    # 3) Simple CV splitter
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    # 4) Small grid for testing
    param_dist = {
        "n_estimators": [50, 100],
        "max_depth":    [3, 5],
        "learning_rate":[0.1, 0.2]
    }

    # 5) XGBoost on CPU for speed
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        tree_method="hist",          # fast CPU method
        random_state=42
    )

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=4,
        scoring="neg_log_loss",
        cv=cv,
        n_jobs=-1,                   # use all local cores
        random_state=42,
        verbose=1
    )

    # 6) Fit & show best params
    search.fit(X, y)
    print("Best parameters found:", search.best_params_)

if __name__ == "__main__":
    main()
