#!/bin/bash
#SBATCH --job-name=launcher
#SBATCH --output=launcher_output.log
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1

echo "Lancement des 93 jobs..."

for i in {0..92}; do #0 à 92 donc y a 93 blocks
    sbatch --export=JOB_ID=$i worker.sbatch
done

echo "Tous les jobs ont été soumis."
