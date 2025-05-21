#!/bin/bash
#SBATCH --job-name=simple_t4job
#SBATCH --output=output_t4.log
#SBATCH --error=error_t4.log
#SBATCH --qos=viz
#SBATCH -p viz
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:t4:1
#SBATCH --ntasks=1
#SBATCH -c 32
#SBATCH --mem=2G
#SBATCH --exclusive
source $STORE/mypython310/bin/activate

python simple_script.py
