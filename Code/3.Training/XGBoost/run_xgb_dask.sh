#!/bin/bash -l
#SBATCH --job-name=xgb-dask
#SBATCH --output=logs/dask/xgb-dask-%j.out
#SBATCH --error=logs/dask/xgb-dask-%j.err

#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --chdir=/mnt/netapp2/Store_uni/home/ulc/cursos/curso363/TFM/Data-Analytics-with-HPC/Code/3.Training/XGBoost

module load python
source $STORE/mypython/bin/activate

python hyperparam_tune_xgb_dask.py \
  --utils-path ../../0.Utils/utils.py \
  --parquet   ../../../Datasets/final_tennis_dataset_symmetric.parquet \
  --cutoff    2025-01-01 \
  --output    ./logs/dask/best_xgb_params \
  --n-iter    50 \
  --n-splits  4 \
  --jobs      4