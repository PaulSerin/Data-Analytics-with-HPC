#!/bin/bash -l
#SBATCH --job-name=xgb-simple            # nom du job
#SBATCH --partition=short                # partition valide (default)
#SBATCH --account=ulc                     # votre compte Slurm
#SBATCH --nodes=1                        # 1 nœud
#SBATCH --ntasks=1                       # 1 tâche
#SBATCH --cpus-per-task=4                # 4 cœurs CPU
#SBATCH --mem=16G                        # 16 Go RAM
#SBATCH --time=00:15:00                  # 15 minutes max runtime

#SBATCH --chdir=/HPC/Code/3.Training      # répertoire de travail
#SBATCH --output=logs/xgb-simple-%j.out  # STDOUT
#SBATCH --error=logs/xgb-simple-%j.err   # STDERR

module load python
source $STORE/mypython/bin/activate

python hyperparam_tune_xgb_simple.py \
  --parquet ../../Datasets/final_tennis_dataset_symmetric.parquet \
  --sample-size 1000