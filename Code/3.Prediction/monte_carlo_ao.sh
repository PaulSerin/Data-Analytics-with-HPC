#!/bin/bash -l
#SBATCH --job-name=mc-ao
#SBATCH --output=logs/mc-ao-%A_%a.out
#SBATCH --error=logs/mc-ao-%A_%a.err
#SBATCH --array=0-3                  # 4 array tasks
#SBATCH --nodes=1                    # one node each
#SBATCH --ntasks=1                   # one task per job
#SBATCH --cpus-per-task=32           # required by GPU policy
#SBATCH --gres=gpu:a100:1            # one A100 per job
#SBATCH --mem=40G                    # memory per node
#SBATCH --time=00:05:00              # walltime
#SBATCH --chdir=/mnt/netapp2/Store_uni/home/ulc/cursos/curso363/TFM/Data-Analytics-with-HPC/Code/3.Prediction

module load python
source $STORE/mypython/bin/activate  

RUNS_PER_JOB=1
SCRIPT=monte_carlo_ao.py

python $SCRIPT \
    --utils-path    ../0.Utils/utils.py \
    --json-draw     ../../Datasets/aus_open_2025_matches_all_ids.json \
    --parquet       ../../Datasets/final_tennis_dataset_symmetric.parquet \
    --model         ../../Models/xgb_model.json \
    --cutoff        2025-01-01 \
    --runs-per-job  $RUNS_PER_JOB \
    --job-index     $SLURM_ARRAY_TASK_ID \
    --output-dir    ./mc_results