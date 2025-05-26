#!/usr/bin/env python3
"""
Distributed hyperparameter tuning for XGBoost using Dask, Dask-ML and a SLURM cluster.
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
    return SourceFileLoader("utils", str(utils_path)).load_module()

def main():
    parser = argparse.ArgumentParser(
        description="Distributed hyperparam tuning for XGBoost with Dask"
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

    # 1) Load utils module
    utils = load_utils(args.utils_path)

    # 2) Read the dataset as a Dask DataFrame (lazy, parallel)
    ddf = dd.read_parquet(str(args.parquet))
    # parse date and filter out future data
    ddf["DATE"] = dd.to_datetime(ddf["TOURNEY_DATE"].astype(str))
    ddf = ddf[ddf["DATE"] < args.cutoff]

    # extract numeric features & target
    X = ddf.drop(columns=utils.COLS_TO_EXCLUDE, errors="ignore") \
           .select_dtypes(include=["number"])
    y = ddf["TARGET"]

    # time‐based train split (up to 2023)
    train = ddf[ddf["DATE"].dt.year <= 2023]
    X_train = train.drop(columns=["TARGET", "TOURNEY_DATE", "DATE"], errors="ignore") \
                   .select_dtypes(include=["number"])
    y_train = train["TARGET"]

    # 3) Spin up SLURMCluster + Dask client
    cluster = SLURMCluster(
        queue="short",
        project="curso363",
        cores=32,
        processes=1,
        memory="40GB",
        walltime="02:00:00",
        job_extra=["--gres=gpu:a100:1"]
    )
    # launch `args.jobs` worker jobs
    cluster.scale(jobs=args.jobs)
    client = Client(cluster)

    # persist into cluster memory to avoid repeated I/O
    X_train, y_train = client.persist([X_train, y_train])

    # 4) Define XGBoost classifier & parameter grid
    base = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="gpu_hist",
        predictor="gpu_predictor",
        random_state=42,
        use_label_encoder=False
    )
    param_dist = {
        "n_estimators":     [100, 200, 300, 500],
        "learning_rate":    [0.01, 0.05, 0.1],
        "max_depth":        [4, 6, 8],
        "subsample":        [0.7, 0.8, 0.9],
        "colsample_bytree": [0.6, 0.8, 1.0],
    }

    # 5) Set up Dask-ML RandomizedSearchCV
    cv =  StratifiedKFold = None  # placeholder
    from sklearn.model_selection import StratifiedKFold
    cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=args.n_iter,
        scoring="neg_log_loss",
        cv=cv,
        random_state=42,
        n_jobs=-1,       # use all cores per worker
        verbose=1
    )

    # 6) Run the search under a performance report
    with performance_report(filename="dask-report.html"), client:
        search.fit(X_train, y_train)

    # 7) Write out the best parameters
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(search.best_params_, f, indent=2)
    print("Best parameters saved to", args.output)

    client.close()
    cluster.close()


if __name__ == "__main__":
    main()