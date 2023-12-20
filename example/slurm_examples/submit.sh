#!/bin/bash

# Usage: submit.sh start end [batch]

# Start and en GPI ID
START=$1
END=$2
BATCH=${3:-200}
NEXT=$((START+BATCH<END ? START+BATCH: END))

while (( ${START} <= ${END} ))
do
    START_ARRAY=${START}
    END_ARRAY=${NEXT}
    WRAP=0

    # if start with n*1000, first submit the first one
    if (( ${START_ARRAY}%1000 == 0))
    then
        END_ARRAY=${START_ARRAY}
        NEXT=${END_ARRAY}
    fi

    # for wrapping, do not cross n*1000
    if (( ${START_ARRAY}/1000 != ${END_ARRAY}/1000))
    then
        END_ARRAY=$((END_ARRAY-END_ARRAY%1000))
        NEXT=${END_ARRAY}
    fi
    
    echo "Submit ${START_ARRAY} to ${NEXT}."
    
    # wrap back to 1-1000
    while (( ${START_ARRAY} > 1000 ))
    do
        START_ARRAY=$((START_ARRAY-1000))
        END_ARRAY=$((END_ARRAY-1000))
        WRAP=$((WRAP+1000))  
    done

    echo submit with sbatch using "--array=${START_ARRAY}-${END_ARRAY}" 

    sentence=$(sbatch --array=${START_ARRAY}-${END_ARRAY} train_multiple_gpi.slurm ${WRAP})
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