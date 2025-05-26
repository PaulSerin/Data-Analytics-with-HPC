#!/usr/bin/env python3
"""
Distributed hyperparameter tuning for XGBoost using Dask, Dask-ML and a SLURM cluster.
Added debug prints to trace progress.
"""
import argparse
import json
from pathlib import Path

import dask.dataframe as dd
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, performance_report
from dask_ml.model_selection import RandomizedSearchCV
import xgboost as xgb

# load_module utility to import your utils.py
from importlib.machinery import SourceFileLoader
def load_utils(utils_path):
    print(f"[DEBUG] Loading utils from {utils_path}", flush=True)
    module = SourceFileLoader("utils", str(utils_path)).load_module()
    print("[DEBUG] Utils loaded successfully", flush=True)
    return module

def main():
    # 0) Parse args
    parser = argparse.ArgumentParser(
        description="Distributed hyperparam tuning for XGBoost with Dask (debug mode)"
    )
    parser.add_argument("--utils-path",  type=Path, required=True,
                        help="Path to your utils.py")
    parser.add_argument("--parquet",     type=Path, required=True,
                        help="Input Parquet file with features")
    parser.add_argument("--cutoff",      type=str, default="2025-01-01",
                        help="Cutoff date for filtering data (YYYY-MM-DD)")
    parser.add_argument("--output",      type=Path, required=True,
                        help="Where to write best_params.json")
    parser.add_argument("--n-iter",      type=int, default=50,
                        help="Number of RandomizedSearchCV iterations")
    parser.add_argument("--n-splits",    type=int, default=4,
                        help="Number of CV splits")
    parser.add_argument("--jobs",        type=int, default=4,
                        help="Number of Dask workers to launch")
    args = parser.parse_args()
    print(f"[DEBUG] Arguments: {args}", flush=True)

    # 1) Load utils module
    utils = load_utils(args.utils_path)

    # 2) Read the dataset as a Dask DataFrame (lazy, parallel)
    print(f"[DEBUG] Reading parquet from {args.parquet}", flush=True)
    ddf = dd.read_parquet(str(args.parquet))
    print(f"[DEBUG] Initial Dask DataFrame partitions: {ddf.npartitions}", flush=True)

    # parse date and filter out future data
    ddf["DATE"] = dd.to_datetime(ddf["TOURNEY_DATE"].astype(str))
    ddf = ddf[ddf["DATE"] < args.cutoff]
    print(f"[DEBUG] Data filtered with cutoff {args.cutoff}", flush=True)

    # extract numeric features & target
    X = ddf.drop(columns=utils.COLS_TO_EXCLUDE, errors="ignore") \
           .select_dtypes(include=["number"])
    y = ddf["TARGET"]
    print(f"[DEBUG] Feature columns count: {len(X.columns)}", flush=True)

    # time‐based train split (up to 2023)
    train = ddf[ddf["DATE"].dt.year <= 2023]
    X_train = train.drop(columns=["TARGET", "TOURNEY_DATE", "DATE"], errors="ignore") \
                   .select_dtypes(include=["number"])
    y_train = train["TARGET"]
    print(f"[DEBUG] Training set partitions: {train.npartitions}", flush=True)

    # 3) Spin up SLURMCluster + Dask client
    print("[DEBUG] Starting SLURMCluster", flush=True)
    cluster = SLURMCluster(
        queue="short",
        account="ulc",
        cores=32,
        processes=1,
        memory="40GB",
        walltime="02:00:00",
        job_extra_directives=["--gres=gpu:a100:1"]
    )
    cluster.scale(jobs=args.jobs)
    print(f"[DEBUG] Launched {args.jobs} workers → dashboard: {cluster.dashboard_link}", flush=True)
    client = Client(cluster)
    print("[DEBUG] Dask client connected", flush=True)

    # persist into cluster memory to avoid repeated I/O
    print("[DEBUG] Persisting training data to cluster memory", flush=True)
    X_train, y_train = client.persist([X_train, y_train])
    client.wait_for_workers(n_workers=args.jobs)
    print("[DEBUG] Data persisted; workers ready", flush=True)

    # 4) Define XGBoost classifier & parameter grid
    print("[DEBUG] Defining XGBClassifier and parameter grid", flush=True)
    base = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="gpu_hist",
        predictor="gpu_predictor",
        random_state=42
    )
    param_dist = {
        "n_estimators":     [100, 200, 300, 500],
        "learning_rate":    [0.01, 0.05, 0.1],
        "max_depth":        [4, 6, 8],
        "subsample":        [0.7, 0.8, 0.9],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_alpha":        [0, 0.1, 0.5, 1.0],    # L1 regularization term
        "reg_lambda":       [1, 5, 10, 20]         # L2 regularization term
    }

    print(f"[DEBUG] Parameter distribution: {param_dist}", flush=True)

    # 5) Set up Dask-ML RandomizedSearchCV
    from sklearn.model_selection import StratifiedKFold
    cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    print(f"[DEBUG] Using StratifiedKFold with {args.n_splits} splits", flush=True)

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=args.n_iter,
        scoring="neg_log_loss",
        cv=cv,
        random_state=42,
        n_jobs=-1,
    )
    print("[DEBUG] RandomizedSearchCV initialized", flush=True)

    # 6) Run the search under a performance report
    print("[DEBUG] Starting search.fit()", flush=True)
    with performance_report(filename="dask-report.html"), client:
        search.fit(X_train, y_train)
    print("[DEBUG] search.fit() completed", flush=True)

    # 7) Write out the best parameters
    print(f"[DEBUG] Saving best parameters to {args.output}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "best_params": search.best_params_,
            "best_score": -search.best_score_
        }, f, indent=2)
    print("[DEBUG] Best parameters saved", flush=True)

    # 8) Clean up
    client.close()
    cluster.close()
    print("[DEBUG] Dask client and cluster closed", flush=True)


if __name__ == "__main__":
    main()
