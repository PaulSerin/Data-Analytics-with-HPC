#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --mem=4G

# Optional: load modules or activate your environment
# module load python/3.9

source /path/to/venv/bin/activate

module load python/3.9
      # or whatever module you need
python3 print_paths.py