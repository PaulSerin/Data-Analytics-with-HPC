#!/usr/bin/env python3
"""
Quick sanity-check of XGBoost hyperparameter tuning using a small local Dask cluster.
"""
import argparse
import json
import pandas as pd
import numpy as np
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
from dask_ml.model_selection import RandomizedSearchCV
import xgboost as xgb

def main():
    # 0) parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Local Dask cluster XGBoost hyperparam tuning"
    )
    parser.add_argument("--parquet", type=str, required=True,
                        help="Path to your Parquet file")
    parser.add_argument("--sample-size", type=int, default=1000,
                        help="Rows to sample for quick test")
    parser.add_argument("--n-workers", type=int, default=2,
                        help="Number of Dask workers")
    parser.add_argument("--threads-per-worker", type=int, default=2,
                        help="Threads per Dask worker")
    parser.add_argument("--output", type=str, default="best_params_dask_local.json",
                        help="Output JSON for best params & score")
    args = parser.parse_args()
    print(f"[DEBUG] Starting LocalCluster with "
          f"{args.n_workers} workers × {args.threads_per_worker} threads each", flush=True)

    # 1) spin up local Dask cluster
    cluster = LocalCluster(
        n_workers=args.n_workers,
        threads_per_worker=args.threads_per_worker,
        memory_limit="2GB"
    )
    client = Client(cluster)
    print(f"[DEBUG] Dask dashboard available at {cluster.dashboard_link}", flush=True)

    # 2) load & sample data
    print(f"[INFO] Loading data from {args.parquet}", flush=True)
    df = pd.read_parquet(args.parquet)
    print(f"[INFO] Original DataFrame shape: {df.shape}", flush=True)
    n = min(args.sample_size, len(df))
    df = df.sample(n=n, random_state=42)
    print(f"[INFO] Sampled {n} rows → shape {df.shape}", flush=True)

    # 3) convert to Dask DataFrame
    ddf = dd.from_pandas(df, npartitions=args.n_workers)
    X = ddf.select_dtypes(include=[np.number]) \
           .drop(columns=["TARGET"], errors="ignore")
    y = ddf["TARGET"]
    print(f"[DEBUG] Dask DataFrame partitions: {ddf.npartitions}", flush=True)

    # 4) define model and hyperparameter distributions
    param_dist = {
        "n_estimators":    [50, 100],
        "max_depth":       [3, 5],
        "learning_rate":   [0.1, 0.2]
    }
    print(f"[DEBUG] Param distribution: {param_dist}", flush=True)

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",       # CPU histogram
        use_label_encoder=False,
        random_state=42
    )

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=4,
        scoring="neg_log_loss",
        cv=2,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    print("[DEBUG] Initialized RandomizedSearchCV", flush=True)

    # 5) run hyperparameter search
    print("[INFO] Starting search.fit()", flush=True)
    search.fit(X, y)
    print("[INFO] search.fit() completed", flush=True)

    # 6) record results
    best_params = search.best_params_
    best_score = -search.best_score_
    print(f"[RESULT] Best params: {best_params}", flush=True)
    print(f"[RESULT] Best log-loss: {best_score:.5f}", flush=True)

    # ensure output dir exists
    with open(args.output, "w") as f:
        json.dump({
            "best_params": best_params,
            "best_log_loss": best_score
        }, f, indent=2)
    print(f"[INFO] Results saved to {args.output}", flush=True)

    # 7) cleanup
    client.close()
    cluster.close()
    print("[DEBUG] Dask cluster closed", flush=True)


if __name__ == "__main__":
    main()