#!/usr/bin/env python3
"""
Quick sanity-check of XGBoost hyperparameter tuning using a small local Dask cluster,
with instrumentation that forces Dask tasks (persist + fit) and produces a Matplotlib bar chart
of task durations.
"""
import argparse
import json
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

import dask.dataframe as dd
from dask.distributed import Client, LocalCluster, wait
from dask.diagnostics import Profiler
from dask_ml.model_selection import RandomizedSearchCV
import xgboost as xgb

def main():
    # 0) Parse arguments
    parser = argparse.ArgumentParser(
        description="Local Dask cluster XGB tuning with profiling"
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
    parser.add_argument("--plot-file", type=str, default="dask_task_durations.png",
                        help="Filename for the Matplotlib bar chart")
    args = parser.parse_args()

    # 1) Start a local Dask cluster
    print(f"[DEBUG] Starting LocalCluster: {args.n_workers} workers × "
          f"{args.threads_per_worker} threads each", flush=True)
    cluster = LocalCluster(
        n_workers=args.n_workers,
        threads_per_worker=args.threads_per_worker,
        memory_limit="2GB"
    )
    client = Client(cluster)
    print(f"[DEBUG] Dashboard URL: {cluster.dashboard_link}", flush=True)

    # 2) Scheduler & worker info
    info = client.scheduler_info()
    for addr, winfo in info.get("workers", {}).items():
        mem_gb = winfo.get("memory_limit", 0) / 1e9
        print(f"[DEBUG] Worker {addr}: {winfo['nthreads']} threads, "
              f"{mem_gb:.2f} GB memory limit", flush=True)

    # 3) Load & sample data
    print(f"[INFO] Loading parquet {args.parquet}", flush=True)
    df = pd.read_parquet(args.parquet)
    print(f"[INFO] Original shape: {df.shape}", flush=True)
    n = min(args.sample_size, len(df))
    df = df.sample(n=n, random_state=42)
    print(f"[INFO] Sampled {n} rows → {df.shape}", flush=True)

    # 4) Convert to Dask DataFrame with extra partitions to generate tasks
    npart = args.n_workers * 4
    ddf = dd.from_pandas(df, npartitions=npart)
    X = ddf.select_dtypes(include=[np.number]).drop(columns=["TARGET"], errors="ignore")
    y = ddf["TARGET"]
    print(f"[DEBUG] Dask partitions: {ddf.npartitions}", flush=True)

    # 5) Define model & hyperparameter distribution
    param_dist = {
        "n_estimators":  [50, 100],
        "max_depth":     [3, 5],
        "learning_rate": [0.1, 0.2]
    }
    print(f"[DEBUG] Param distribution: {param_dist}", flush=True)
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
        n_jobs=-1
    )

    # 6) Persist and fit under Profiler to capture actual Dask tasks
    prof = Profiler()
    print("[DEBUG] Persisting X and y and fitting under Profiler", flush=True)
    t0 = time.time()
    with prof:
        Xp, yp = client.persist([X, y])
        wait([Xp, yp])
        search.fit(Xp, yp)
    t1 = time.time()
    fit_time = t1 - t0
    print(f"[DEBUG] Total fit time: {fit_time:.2f} s", flush=True)

    # 7) Extract profiling events and aggregate durations
    records = []
    for evt in prof.results:
        key = getattr(evt, "key", None)
        start = getattr(evt, "start", None)
        finish = getattr(evt, "finish", None)
        if key and start and finish:
            records.append({
                "task": key,
                "duration": finish - start
            })

    # 8) Plot if we have any records
    if records:
        df_prof = pd.DataFrame(records)
        agg = df_prof.groupby("task")["duration"].sum().sort_values()
        plt.figure(figsize=(8, max(2, len(agg) * 0.4)))
        agg.plot.barh()
        plt.xlabel("Total duration (s)")
        plt.title("Dask task durations")
        plt.tight_layout()
        plt.savefig(args.plot_file)
        print(f"[DEBUG] Saved task-duration plot to {args.plot_file}", flush=True)
    else:
        print("[WARNING] No profiling data collected; skipping plot", flush=True)

    # 9) Save best params & timing
    result = {
        "best_params": search.best_params_,
        "best_log_loss": -search.best_score_,
        "fit_time_s": fit_time
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[INFO] Results saved to {args.output}", flush=True)

    # 10) Cleanup
    client.close()
    cluster.close()
    print("[DEBUG] Dask cluster closed", flush=True)

if __name__ == "__main__":
    main()
