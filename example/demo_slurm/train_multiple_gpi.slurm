#!/bin/bash
#SBATCH -J model_optimazation_one_gpi
#SBATCH -t 00:15:00
#SBATCH -N 3 --ntasks-per-node=1

srun python3 jackknife_train_one.py $SLURM_ARRAY_TASK_ID
