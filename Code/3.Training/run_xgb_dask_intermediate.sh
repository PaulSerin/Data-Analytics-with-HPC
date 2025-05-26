#!/bin/bash -l
#SBATCH --job-name=xgb-dask
#SBATCH --output=logs/intermediate/xgb-dask-%j.out
#SBATCH --error=logs/intermediate/xgb-dask-%j.err

#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --chdir=/mnt/netapp2/Store_uni/home/ulc/cursos/curso363/TFM/Data-Analytics-with-HPC/Code/3.Training

module load python
source $STORE/mypython/bin/activate


python hyperparam_tune_xgb_intermediate.py \
  --parquet ../../Datasets/final_tennis_dataset_symmetric.parquet \
  --sample-size 2000 \
  --n-workers 2 \
  --threads-per-worker 2 \
  --output logs/best_params_dask_local.json \
  > logs/dask_local.out 2> logs/dask_local.err

echo "[$(date +"%Y-%m-%d %H:%M:%S")] FINISHED DASK LOCAL TEST"