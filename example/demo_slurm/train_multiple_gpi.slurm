#!/bin/bash
#SBATCH -J model_optimazation_one_gpi
#SBATCH -p normal
#SBATCH -t 00:00:20
#SBATCH -N 1 --ntasks-per-node=1

logpath="log/"
mkdir -p $logpath
logfile="$logpath/${SLURM_ARRAY_TASK_ID}.out"

echo "Writing Python log to ${logfile}"
scontrol show -dd job $SLURM_JOB_ID
printenv

WRAP=$1
GPIID=$((WRAP+SLURM_ARRAY_TASK_ID))

echo GPIID: ${GPIID}

srun python3 jackknife_train_one.py $GPIID > ${logfile}
