#!/bin/bash -l
#SBATCH --job-name=mc-rg-exclusive
#SBATCH --output=logs/mc-rg-exclusive-%A_%a.out
#SBATCH --error=logs/mc-rg-exclusive-%A_%a.err
#SBATCH --array=0-3                
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32         # 32 CPU cores per GPU (cluster policy)
#SBATCH --gres=gpu:a100:1          # 1x A100 per task
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --chdir=/mnt/netapp2/Store_uni/home/ulc/cursos/curso363/TFM/Data-Analytics-with-HPC/Code/4.Prediction/RG_2025
#SBATCH --exclusive

module load python
source $STORE/mypython/bin/activate

RUNS_PER_JOB=250                    # 250 simulations each
SCRIPT=monte_carlo_rg.py

echo "[$(date)] Starting job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID"
python -u $SCRIPT \
    --utils-path    ../../0.Utils/utils.py \
    --json-draw     ./roland_garros_2025_complete_final.json \
    --parquet       ../../../Datasets/final_tennis_dataset_symmetric.parquet \
    --model         ../../../Models/xgb_model.json \
    --cutoff        2025-01-01 \
    --runs-per-job  $RUNS_PER_JOB \
    --job-index     $SLURM_ARRAY_TASK_ID \
    --output-dir    ./mc_rg_results_exclusive

echo "[$(date)] Finished job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID"