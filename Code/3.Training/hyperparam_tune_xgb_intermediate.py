#!/usr/bin/env python3
"""
Quick sanity-check of XGBoost hyperparameter tuning using a small local Dask cluster,
with instrumentation to check distribution, timings and resource usage.
"""
import argparse
import json
import pandas as pd
import numpy as np
import time

import dask.dataframe as dd
from dask.distributed import Client, LocalCluster, performance_report
from dask.distributed import get_worker
from dask.diagnostics import Profiler, ResourceProfiler
from dask.diagnostics import visualize
from dask_ml.model_selection import RandomizedSearchCV
import xgboost as xgb

def main():
    # 0) arguments
    parser = argparse.ArgumentParser(
        description="Local Dask cluster XGB tuning with instrumentation"
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

    # 1) Start local cluster
    print(f"[DEBUG] Starting LocalCluster: {args.n_workers} workers × "
          f"{args.threads_per_worker} threads each", flush=True)
    cluster = LocalCluster(
        n_workers=args.n_workers,
        threads_per_worker=args.threads_per_worker,
        memory_limit="2GB"
    )
    client = Client(cluster)
    print(f"[DEBUG] Dashboard URL: {cluster.dashboard_link}", flush=True)

    # 2) Print cluster & scheduler info
    print(f"[DEBUG] Cluster object: {cluster}", flush=True)
    print(f"[DEBUG] Scheduler info keys: {list(client.scheduler_info().keys())}", flush=True)
    print(f"[DEBUG] Number of cores per worker: {client.ncores()}", flush=True)

    # 3) Inspect each worker
    def _worker_info():
        w = get_worker()
        return {
            "address": w.address,
            "nthreads": w.nthreads,
            "memory_limit": w.memory_limit
        }
    infos = client.run(_worker_info)
    print(f"[DEBUG] Worker info: {infos}", flush=True)

    # 4) Load & sample data
    print(f"[INFO] Loading parquet {args.parquet}", flush=True)
    df = pd.read_parquet(args.parquet)
    print(f"[INFO] Original shape: {df.shape}", flush=True)
    n = min(args.sample_size, len(df))
    df = df.sample(n=n, random_state=42)
    print(f"[INFO] Sampled {n} rows → {df.shape}", flush=True)

    # 5) To Dask-DataFrame
    ddf = dd.from_pandas(df, npartitions=args.n_workers)
    X = ddf.select_dtypes(include=[np.number]).drop(columns=["TARGET"], errors="ignore")
    y = ddf["TARGET"]
    print(f"[DEBUG] Dask partitions: {ddf.npartitions}", flush=True)

    # 6) Define model & grid
    param_dist = {
        "n_estimators":  [50, 100],
        "max_depth":     [3, 5],
        "learning_rate": [0.1, 0.2]
    }
    print(f"[DEBUG] Param dist: {param_dist}", flush=True)
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
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

    # 7) Run with profiling + performance report
    report_file = "dask_local_report.html"
    print("[DEBUG] Starting search.fit() with performance_report and Profiler", flush=True)
    t0 = time.time()
    with performance_report(filename=report_file):
        with Profiler() as prof, ResourceProfiler(aggregate=True) as rprof:
            search.fit(X, y)
    t1 = time.time()
    print(f"[DEBUG] Total fit time: {t1-t0:.2f} s", flush=True)

    # 8) Visualize the profiling results (saves PNGs)
    visualize([prof, rprof], show=False, filename="dask_local_profile.png")
    print("[DEBUG] Profiling visualization saved to dask_local_profile.png", flush=True)
    print(f"[DEBUG] Performance report saved to {report_file}", flush=True)

    # 9) Save best params
    best = {
        "best_params": search.best_params_,
        "best_log_loss": -search.best_score_,
        "fit_time_s": t1-t0
    }
    with open(args.output, "w") as f:
        json.dump(best, f, indent=2)
    print(f"[INFO] Results saved to {args.output}", flush=True)

    # 10) Close cluster
    client.close()
    cluster.close()
    print("[DEBUG] Dask cluster closed", flush=True)


if __name__ == "__main__":
    main()