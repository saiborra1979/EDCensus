#!/bin/bash

#PBS -l walltime=01:00:00
#PBS -o /home/edrysdale/qsub/
#PBS -e /home/edrysdale/qsub/
#PBS -l vmem=16g
#PBS -l mem=16g

# Assign the lead
lead=$(($1))
model=$2
echo "Lead: "$lead", model: "$model

# EXAMPLE OF HOW TO RUN
#qsub -N local_lead4_lasso -F "4 lasso" -t 0-181 -l nodes=1:ppn=1 pipeline_HPF.sh

pwd
cd /hpf/largeprojects/agoldenb/edrysdale/ED/CensusFlow
pwd
. conda.env
source activate CensusFlow
which python

# (1) Remove any e/o files not associated with this run
echo "JOBID = "$PBS_JOBID
#jobid=$(echo $PBS_JOBID | cut -d'[' -f1)
#fn_rm=$(ls /home/edrysdale/qsub | grep -v $jobid)
#fn_keep=$(ls /home/edrysdale/qsub | grep $jobid)
#
#echo "jobid: "$jobid
#echo "remove: "$fn_rm
#for fn in $fn_rm; do
#        rm /home/edrysdale/qsub/$fn
#done

# (2) Calculate the day/lead
day=$(($PBS_ARRAYID))
echo "--- day: "$day"------"

# Call model
python run_mdl.py --day $day --lead $lead --nlags 10 --model $model

echo "##### end of script: "$day" ######"
