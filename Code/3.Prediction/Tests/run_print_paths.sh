#!/bin/bash -l
#SBATCH --job-name=debug-paths
#SBATCH --output=logs/debug-paths-%j.out
#SBATCH --error=logs/debug-paths-%j.err
#SBATCH --time=00:05:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --chdir=/mnt/netapp2/Store_uni/home/ulc/cursos/curso363/TFM/Data-Analytics-with-HPC/Code/3.Prediction


module load python      # (or whatever your Python module is)
source $STORE/mypython/bin/activate  # if you need your venv

which python
python print_path.py

deactivate