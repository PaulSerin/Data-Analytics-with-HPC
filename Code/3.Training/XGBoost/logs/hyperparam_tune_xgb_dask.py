#!/usr/bin/env python3
"""
Distributed hyperparameter tuning for XGBoost using Dask, Dask-ML and a SLURM cluster.
Entraîne sur <=2023, teste sur 2024, et stocke tous les résultats de grille + score test.
Génère aussi un rapport Dask et des diagnostics Bokeh.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import dask.dataframe as dd
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, performance_report
from dask_ml.model_selection import RandomizedSearchCV
from dask.diagnostics import Profiler, ResourceProfiler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

from importlib.machinery import SourceFileLoader

def load_utils(utils_path):
    print(f"[DEBUG] Loading utils from {utils_path}", flush=True)
    module = SourceFileLoader("utils", str(utils_path)).load_module()
    print("[DEBUG] Utils loaded successfully", flush=True)
    return module

def main():
    # 0) Arguments
    parser = argparse.ArgumentParser(
        description="Distributed XGB tuning: train<=2023, test=2024, store CV results + test accuracy"
    )
    parser.add_argument("--utils-path", type=Path, required=True,
                        help="Path to your utils.py")
    parser.add_argument("--parquet",    type=Path, required=True,
                        help="Parquet file with all data")
    parser.add_argument("--cutoff",     type=str, default="2025-01-01",
                        help="Exclude data >= this date (YYYY-MM-DD)")
    parser.add_argument("--output",     type=Path, required=True,
                        help="Directory in which to write outputs")
    parser.add_argument("--n-iter",     type=int, default=50,
                        help="Number of RandomizedSearchCV iterations")
    parser.add_argument("--n-splits",   type=int, default=4,
                        help="Number of CV splits")
    parser.add_argument("--jobs",       type=int, default=4,
                        help="Number of Dask workers to launch")
    args = parser.parse_args()
    print(f"[DEBUG] Arguments: {args}", flush=True)

    # 1) Load utils
    utils = load_utils(args.utils_path)

    # 2) Read and filter data
    print(f"[DEBUG] Reading parquet {args.parquet}", flush=True)
    ddf = dd.read_parquet(str(args.parquet))
    ddf["DATE"] = dd.to_datetime(ddf["TOURNEY_DATE"].astype(str))
    ddf = ddf[ddf["DATE"] < args.cutoff]
    print(f"[DEBUG] After cutoff, partitions={ddf.npartitions}", flush=True)

    # 3) Split train/test by year
    train = ddf[ddf["DATE"].dt.year <= 2023]
    test  = ddf[ddf["DATE"].dt.year == 2024]
    print(f"[DEBUG] train partitions={train.npartitions}, test partitions={test.npartitions}", flush=True)

    X_train = train.drop(columns=utils.COLS_TO_EXCLUDE + ["TOURNEY_DATE","DATE"],
                         errors="ignore").select_dtypes(include=["number"])
    y_train = train["TARGET"]
    X_test  = test .drop(columns=utils.COLS_TO_EXCLUDE + ["TOURNEY_DATE","DATE"],
                         errors="ignore").select_dtypes(include=["number"])
    y_test  = test ["TARGET"]

    # 4) Launch SLURMCluster and Dask client
    cluster = SLURMCluster(
        queue="short",
        account="ulc",
        cores=32,
        processes=1,
        memory="40GB",
        walltime="02:00:00",
        job_extra_directives=["--gres=gpu:a100:1"],
        scheduler_options={"idle_timeout":"120s"}
    )
    cluster.scale(jobs=args.jobs)
    print(f"[DEBUG] Launched {args.jobs} workers -> dashboard {cluster.dashboard_link}", flush=True)
    client = Client(cluster)
    client.wait_for_workers(n_workers=args.jobs, timeout=300)
    print("[DEBUG] Client connected", flush=True)

    # 5) Persist data to cluster
    print("[DEBUG] Persisting train and test datasets", flush=True)
    X_train, y_train, X_test, y_test = client.persist([X_train, y_train, X_test, y_test])
    client.wait_for_workers(n_workers=args.jobs)

    # 6) Set up model, grid and CV
    print("[DEBUG] Setting up model, parameter grid and CV", flush=True)
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device="cuda",
        random_state=42
    )
    param_dist = {
        "n_estimators":     [100,200,300,500],
        "learning_rate":    [0.01,0.05,0.1],
        "max_depth":        [4,6,8],
        "subsample":        [0.7,0.8,0.9],
        "colsample_bytree": [0.6,0.8,1.0],
        "reg_alpha":        [0,0.1,0.5,1.0],
        "reg_lambda":       [1,5,10,20]
    }
    cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=args.n_iter,
        scoring="accuracy",   # maximize accuracy in CV
        cv=cv,
        random_state=42,
        n_jobs=-1,
    )

    # 7) Run search with performance report and profilers
    report_html   = args.output / "dask-report.html"
    profiler_html = args.output / "profiler.html"
    rprofiler_html= args.output / "resource_profiler.html"
    prof  = Profiler()
    rprof = ResourceProfiler()

    print("[DEBUG] Running hyperparameter search", flush=True)
    with performance_report(filename=str(report_html)), client.as_current():
        with prof, rprof:
            search.fit(X_train, y_train)
    print("[DEBUG] search.fit() completed", flush=True)

    # export Bokeh diagnostics (requires bokeh installed)
    prof.visualize(filename=str(profiler_html))
    rprof.visualize(filename=str(rprofiler_html))
    print(f"[DEBUG] Profiler reports: {profiler_html}, {rprofiler_html}", flush=True)

    # 8) Save all CV results
    cv_df = pd.DataFrame(search.cv_results_)
    out_cv = args.output / "cv_results_all.csv"
    args.output.mkdir(parents=True, exist_ok=True)
    cv_df.to_csv(out_cv, index=False)
    print(f"[DEBUG] CV results saved to {out_cv}", flush=True)

    # 9) Evaluate best model on 2024
    print("[DEBUG] Computing test-set accuracy on 2024", flush=True)
    X_test_pd = X_test.compute()
    y_test_pd = y_test.compute()
    y_pred    = search.best_estimator_.predict(X_test_pd)
    test_accuracy = accuracy_score(y_test_pd, y_pred)
    print(f"[DEBUG] Test accuracy (2024): {test_accuracy:.4f}", flush=True)

    # 10) Write summary JSON
    summary = {
        "best_params":       search.best_params_,
        "best_accuracy_cv":  search.best_score_,
        "test_accuracy_2024": test_accuracy
    }
    out_json = args.output / "summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[DEBUG] Summary saved to {out_json}", flush=True)

    # 11) Cleanup
    client.close()
    cluster.close()
    print("[DEBUG] Done", flush=True)


if __name__ == "__main__":
    main()
