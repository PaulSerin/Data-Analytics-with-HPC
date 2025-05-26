#!/usr/bin/env python3
"""
Distributed hyperparameter tuning for XGBoost using Dask, Dask-ML and a SLURM cluster.
Entraîne sur <=2023, teste sur 2024, et stocke tous les résultats de grille + score test.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np
import dask.dataframe as dd
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, performance_report
from dask_ml.model_selection import RandomizedSearchCV
from sklearn.metrics import log_loss
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
        description="Distributed XGB tuning: train<=2023, test=2024, store all CV results + test score"
    )
    parser.add_argument("--utils-path", type=Path, required=True,
                        help="Path to utils.py")
    parser.add_argument("--parquet",    type=Path, required=True,
                        help="Parquet with all data")
    parser.add_argument("--cutoff",     type=str, default="2025-01-01",
                        help="Exclude data >= this date")
    parser.add_argument("--output",     type=Path, required=True,
                        help="Directory where to write outputs")
    parser.add_argument("--n-iter",     type=int, default=50,
                        help="Number of RandomizedSearchCV iterations")
    parser.add_argument("--n-splits",   type=int, default=4,
                        help="Number of CV splits")
    parser.add_argument("--jobs",       type=int, default=4,
                        help="Number of Dask workers to launch")
    args = parser.parse_args()
    print(f"[DEBUG] Arguments: {args}", flush=True)

    # 1) Utils
    utils = load_utils(args.utils_path)

    # 2) Lecture & filtrage Dask-DataFrame
    print(f"[DEBUG] Reading parquet {args.parquet}", flush=True)
    ddf = dd.read_parquet(str(args.parquet))
    ddf["DATE"] = dd.to_datetime(ddf["TOURNEY_DATE"].astype(str))
    ddf = ddf[ddf["DATE"] < args.cutoff]
    print(f"[DEBUG] After cutoff, partitions={ddf.npartitions}", flush=True)

    # 3) Split train / test by year
    train = ddf[ddf["DATE"].dt.year <= 2023]
    test  = ddf[ddf["DATE"].dt.year == 2024]
    print(f"[DEBUG] train parts={train.npartitions}, test parts={test.npartitions}", flush=True)

    X_train = train.drop(columns=utils.COLS_TO_EXCLUDE + ["TOURNEY_DATE", "DATE"],
                         errors="ignore").select_dtypes(include=["number"])
    y_train = train["TARGET"]
    X_test  = test .drop(columns=utils.COLS_TO_EXCLUDE + ["TOURNEY_DATE", "DATE"],
                         errors="ignore").select_dtypes(include=["number"])
    y_test  = test["TARGET"]

    # 4) Cluster SLURM + Client
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
    print(f"[DEBUG] Launched {args.jobs} workers @ {cluster.dashboard_link}", flush=True)
    client = Client(cluster)
    print("[DEBUG] Client connected", flush=True)

    # 5) Persist train & test (force tâches Dask)
    print("[DEBUG] Persisting train+test", flush=True)
    X_train, y_train, X_test, y_test = client.persist([X_train, y_train, X_test, y_test])
    client.wait_for_workers(n_workers=args.jobs)

    # 6) CV + grille
    print("[DEBUG] Setting up CV + param grid", flush=True)
    base = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device="cuda",
        random_state=42
    )
    param_dist = {
        "n_estimators":  [100,200,300,500],
        "learning_rate": [0.01,0.05,0.1],
        "max_depth":     [4,6,8],
        "subsample":     [0.7,0.8,0.9],
        "colsample_bytree":[0.6,0.8,1.0],
        "reg_alpha":     [0,0.1,0.5,1.0],
        "reg_lambda":    [1,5,10,20]
    }
    cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=args.n_iter,
        scoring="neg_log_loss",
        cv=cv,
        random_state=42,
        n_jobs=-1,
    )

    # 7) Lancement & rapport
    report_html = args.output / "dask-report.html"
    print("[DEBUG] Running search.fit()", flush=True)
    with performance_report(filename=str(report_html)), client.as_current():
        search.fit(X_train, y_train)
    print("[DEBUG] search.fit() done", flush=True)

    # 8) Récupérer et stocker tous les CV résultats
    cv_df = pd.DataFrame(search.cv_results_)
    out_cv = args.output / "cv_results_all.csv"
    cv_df.to_csv(out_cv, index=False)
    print(f"[DEBUG] All CV results saved to {out_cv}", flush=True)

    # 9) Évaluer sur 2024
    # passer en pandas pour log_loss
    X_test_pd = X_test.compute()
    y_test_pd = y_test.compute()
    y_proba   = search.best_estimator_.predict_proba(X_test_pd)[:,1]
    test_ll   = log_loss(y_test_pd, y_proba)
    print(f"[DEBUG] Test log-loss (2024): {test_ll:.5f}", flush=True)

    # 10) Sauvegarde finale JSON
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "best_params":    search.best_params_,
        "best_score_cv": -search.best_score_,
        "test_log_loss":  test_ll
    }
    out_json = args.output / "summary.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[DEBUG] Summary saved to {out_json}", flush=True)

    # 11) Cleanup
    client.close()
    cluster.close()
    print("[DEBUG] Done", flush=True)

if __name__ == "__main__":
    main()