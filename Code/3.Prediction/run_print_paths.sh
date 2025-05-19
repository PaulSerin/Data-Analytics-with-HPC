#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --mem=4G

#SBATCH --chdir=/mnt/netapp2/Store_uni/home/ulc/cursos/curso363/TFM/Data-Analytics-with-HPC/Code/3.Prediction

# Optional: load modules or activate your environment
# module load python/3.9

source /path/to/venv/bin/activate

srun python print_paths.py
