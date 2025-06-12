#!/bin/bash -l
#SBATCH --job-name=mc-rg-16gpu
#SBATCH --output=logs/mc-rg-16gpu-%A_%a.out
#SBATCH --error=logs/mc-rg-16gpu-%A_%a.err
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --chdir=/mnt/netapp2/Store_uni/home/ulc/cursos/curso363/TFM/Data-Analytics-with-HPC/Code/4.Prediction/RG_2025


module load python
source $STORE/mypython/bin/activate

python -u monte_carlo_rg.py \
  --utils-path    ../../0.Utils/utils.py \
  --json-draw     ./roland_garros_2025_complete_final.json \
  --parquet       ../../../Datasets/final_tennis_dataset_symmetric.parquet \
  --model         ../../../Models/xgb_model.json \
  --cutoff        2025-01-01 \
  --runs-per-job  312 \
  --job-index     $SLURM_ARRAY_TASK_ID \
  --output-dir    ./mc_rg_results_16gpu

echo "[$(date)] Finished job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID"