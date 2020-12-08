#!/bin/bash

# Usage: submit.sh start end [batch]

# Start and en GPI ID
START=$1
END=$2
BATCH=${3:-200}
NEXT=$((START+BATCH<END ? START+BATCH: END))

while (( ${START} <= ${END} ))
do
    echo "Submit ${START} to ${NEXT}."

    sentence=$(sbatch --array=${START}-${NEXT} train_multiple_gpi.slurm)
    stringarray=($sentence)
    jobid=(${stringarray[3]}) # get job ID

    while true
    do
        sentence="$(squeue -j $jobid)" 
        stringarray=($sentence)
        jobstatus=(${stringarray[12]}) # get slurm status
        echo "Active job: $jobid"
        echo "Status: $jobstatus"
        if [ -z "$jobstatus" ] 
        then
            START=$((NEXT+1))
            NEXT=$((NEXT+BATCH<END ? NEXT+BATCH : END))
            break
        else
            sleep 120
        fi
    done
done