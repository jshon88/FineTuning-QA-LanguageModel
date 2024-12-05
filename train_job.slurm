#!/bin/bash
#SBATCH --job-name=mtr_training_job        # Job name
#SBATCH --output=logs/training_mtr_%j.out     # Standard output and error log
#SBATCH --error=logs/training_mtr_%j.err
#SBATCH --ntasks=1                        # Number of tasks (usually 1 for single-node jobs)
#SBATCH --cpus-per-task=4                 # Number of CPU cores per task
#SBATCH --mem=32G                         # Total memory per node for HPC3 gpu nodes
##SBATCH --mem=16G                         # Total memory per node for HPC1 gpu nodes
#SBATCH --time=72:00:00                   # Time limit (D-HH:MM:SS)
#SBATCH --partition=gpu                    # Partition name (e.g., gpu, compute)
#SBATCH --exclude=g10                       # Exclude g10 node
#SBATCH --nodelist=g16                            # Specify the node to run the job on  for HPC3

# Activate Conda environment
source activate py310LLM

# Navigate to project directory
cd /data/cmpe258-sp24/017553289/cmpe258/Assignment2/

# Run your training script
python LM_comparison.py