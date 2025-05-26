#!/bin/bash -l
#SBATCH --job-name=xgb-simple
#SBATCH --partition=short
#SBATCH --account=ulc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00

#SBATCH --chdir=/mnt/netapp2/Store_uni/home/ulc/cursos/curso363/TFM/Data-Analytics-with-HPC/Code/3.Training
#SBATCH --output=logs/xgb-simple-%j.out
#SBATCH --error=logs/xgb-simple-%j.err

# create logs/ if missing
mkdir -p logs

echo "[$(date +"%Y-%m-%d %H:%M:%S")] JOB $SLURM_JOB_ID START"  

module load python
source $STORE/mypython/bin/activate

python hyperparam_tune_xgb_simple.py \
  --parquet ../../Datasets/final_tennis_dataset_symmetric.parquet \
  --sample-size 1000 \
  --output ./logs/best_params_simple.json

echo "[$(date +"%Y-%m-%d %H:%M:%S")] JOB $SLURM_JOB_ID END"  