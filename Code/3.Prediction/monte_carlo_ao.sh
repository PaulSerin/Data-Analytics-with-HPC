#!/bin/bash -l
#SBATCH --job-name=mc-ao
#SBATCH --output=logs/mc-ao-%A_%a.out
#SBATCH --error=logs/mc-ao-%A_%a.err

#SBATCH --array=0-3                  # 4 tasks in the array
#SBATCH --nodes=1                    # 1 node per job
#SBATCH --ntasks=1                   # 1 MPI task
#SBATCH --cpus-per-task=32           # MUST be 32 for 1 GPU
#SBATCH --gres=gpu:a100:1            # 1 A100 GPU per job
#SBATCH --mem=40G                    # total memory per node
#SBATCH --time=01:00:00              # walltime HH:MM:SS

module load python/3.9 cuda/11.7     # or whatever modules you need
source $STORE/mypython/bin/activate

RUNS_PER_JOB=250
SCRIPT=monte_carlo_ao.py

srun python $SCRIPT \
    --utils-path    /absolute/path/to/0.Utils/utils.py \
    --json-draw     /absolute/path/to/Datasets/aus_open_2025_matches_all_ids.json \
    --parquet       /absolute/path/to/Datasets/final_tennis_dataset_symmetric.parquet \
    --model         /absolute/path/to/Models/xgb_model.json \
    --cutoff        2025-01-01 \
    --runs-per-job  $RUNS_PER_JOB \
    --job-index     $SLURM_ARRAY_TASK_ID \
    --output-dir    ./mc_results