#!/bin/bash
#SBATCH --job-name=worker
#SBATCH --output=worker_%j.log
#SBATCH --partition=cpu
#SBATCH --time=02:30:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=12

# Charger les modules nécessaires, par exemple pour python
module load python  # Si nécessaire

# Activer l'environnement virtuel
source /mnt/netapp2/Store_uni/home/ulc/cursos/curso366/mypython/bin/activate

# Vérifier si l'environnement est activé
echo "L'environnement virtuel est activé."

# Exécution du job avec le bon JOB_ID
echo "Exécution du job $JOB_ID"
python3 compteurDePatterns.py $JOB_ID 93 # le code en question et le nombre de blocks

# Désactivation de l'environnement virtuel
deactivate
