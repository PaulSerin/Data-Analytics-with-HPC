#!/bin/bash -l
#SBATCH --job-name=mc-ao
#SBATCH --output=logs/mc-ao-%A_%a.out
#SBATCH --error=logs/mc-ao-%A_%a.err
#SBATCH --array=0-0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=10G
#SBATCH --time=00:05:00
#SBATCH --chdir=/mnt/netapp2/.../Code/3.Prediction

module load python
source $STORE/mypython/bin/activate  

echo "[$(date)] Starting job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID"
echo "Script exists? " $( [ -f "$SCRIPT" ] && echo yes || echo no )
echo "PWD: $(pwd)"
echo "Listing:"
ls -l .

RUNS_PER_JOB=1
SCRIPT=monte_carlo_ao.py

python -u $SCRIPT \
    --utils-path    ../0.Utils/utils.py \
    --json-draw     ../../Datasets/aus_open_2025_matches_all_ids.json \
    --parquet       ../../Datasets/final_tennis_dataset_symmetric.parquet \
    --model         ../../Models/xgb_model.json \
    --cutoff        2025-01-01 \
    --runs-per-job  $RUNS_PER_JOB \
    --job-index     $SLURM_ARRAY_TASK_ID \
    --output-dir    ./mc_results2

echo "[$(date)] Python finished with exit code $?"
