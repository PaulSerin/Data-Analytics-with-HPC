#!/bin/bash -l
#SBATCH --job-name=mc-ao-2gpu
#SBATCH --output=logs/2gpu/mc-ao-2gpu-%A_%a.out
#SBATCH --error=logs/2gpu/mc-ao-2gpu-%A_%a.err
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --chdir=/mnt/netapp2/Store_uni/home/ulc/cursos/curso363/TFM/Data-Analytics-with-HPC/Code/3.Prediction

module load python
source $STORE/mypython/bin/activate

RUNS_PER_JOB=2500
SCRIPT=monte_carlo_ao.py

echo "[$(date)] Starting job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID"
python -u $SCRIPT \
    --utils-path    ../../0.Utils/utils.py \
    --json-draw     ../../../Datasets/aus_open_2025_matches_all_ids.json \
    --parquet       ../../../Datasets/final_tennis_dataset_symmetric.parquet \
    --model         ../../../Models/xgb_model.json \
    --cutoff        2025-01-01 \
    --runs-per-job  $RUNS_PER_JOB \
    --job-index     $SLURM_ARRAY_TASK_ID \
    --output-dir    ./mc_ao_results_2gpu

echo "[$(date)] Finished job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID"
