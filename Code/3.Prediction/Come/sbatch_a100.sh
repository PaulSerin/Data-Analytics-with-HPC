#!/bin/bash
#SBATCH --job-name=simple_a100job
#SBATCH --output=output_a100.log
#SBATCH --error=error_a100.log
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH -c 32
#SBATCH --mem=2G
#SBATCH --exclusive

source $STORE/mypython310/bin/activate

python simple_script.py
