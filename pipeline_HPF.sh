#!/bin/bash -x

#PBS -N CensusFlow
#PBS -l nodes=1:ppn=1
#PBS -l walltime=00:59:59
#PBS -o /home/edrysdale/qsub/
#PBS -e /home/edrysdale/qsub/
#PBS -l vmem=16g
#PBS -l mem=16g

# Assign the argument
ndays=$(($1))

cd /hpf/largeprojects/agoldenb/edrysdale/ED/CensusFlow
. conda.env
source activate CensusFlow
which python

# (1) Remove any e/o files not associated with this run
echo "JOBID = "$PBS_JOBID
jobid=$(echo $PBS_JOBID | cut -d'[' -f1)
fn_rm=$(ls /home/edrysdale/qsub | grep -v $jobid)
fn_keep=$(ls /home/edrysdale/qsub | grep $jobid)

echo "jobid: "$jobid
echo "remove: "$fn_rm
for fn in $fn_rm; do
        rm /home/edrysdale/qsub/$fn
done

# (2) Calculate the day/lead

i=$(($PBS_ARRAYID))
echo "--- array number: "$i"------"

lead=$(($(expr $i / $ndays + 1)))
day=$(($(expr $i % $ndays + 0)))
echo "i: "$i", lead: "$lead", day: "$day

# Call model
python run_lasso.py --day $day --lead $lead --nlags 10

echo "##### end of script: "$i" ######"

