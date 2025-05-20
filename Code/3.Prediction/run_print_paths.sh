#!/bin/bash -l
#SBATCH --job-name=debug-paths
#SBATCH --output=logs/debug-paths-%j.out
#SBATCH --error=logs/debug-paths-%j.err
#SBATCH --time=00:05:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

module load python      # (or whatever your Python module is)
source $STORE/mypython/bin/activate  # if you need your venv

which python
srun python print_path.py

deactivate