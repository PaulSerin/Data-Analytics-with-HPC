#!/bin/bash -l
#SBATCH --job-name=xgb-interm          # job name
#SBATCH --partition=short               # your partition
#SBATCH --account=ulc                   # your account
#SBATCH --nodes=1                       # SINGLE node
#SBATCH --ntasks=1                      # SINGLE task
#SBATCH --cpus-per-task=4               # enough cores for your 2 workers×2 threads
#SBATCH --mem=8G                        # 8 GB RAM is plenty
#SBATCH --time=00:15:00                 # 15 min max

#SBATCH --chdir=/mnt/netapp2/Store_uni/home/ulc/cursos/curso363/TFM/Data-Analytics-with-HPC/Code/3.Training
#SBATCH --output=logs/intermediate/xgb-interm-%j.out
#SBATCH --error=logs/intermediate/xgb-interm-%j.err

# ensure the directory exists
mkdir -p logs/intermediate

module load python
source $STORE/mypython/bin/activate

echo "[$(date +"%Y-%m-%d %H:%M:%S")] STARTING intermediate Dask local test"

python hyperparam_tune_xgb_intermediate.py \
  --parquet   ../../Datasets/final_tennis_dataset_symmetric.parquet \
  --sample-size 1000 \
  --n-workers 2 \
  --threads-per-worker 2 \
  --output    logs/intermediate/best_params_dask_intermediate.json \
  > logs/intermediate/dask_intermediate.out 2> logs/intermediate/dask_intermediate.err

echo "[$(date +"%Y-%m-%d %H:%M:%S")] FINISHED intermediate Dask local test"