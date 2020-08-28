#!/bin/bash

#PBS -l walltime=06:00:00
#PBS -o /home/edrysdale/qsub/
#PBS -e /home/edrysdale/qsub/
#PBS -l vmem=16g
#PBS -l mem=16g
#PBS -l nodes=1:ppn=4
#PBS -t 1-12

# Note that the array is over the lead

# EXAMPLE OF HOW TO RUN
#qsub -N gpy_run pipeline_HPF.sh
#-F "4 lasso"  -t 0-181  -l nodes=1:ppn=1

pwd
cd /hpf/largeprojects/agoldenb/edrysdale/ED/CensusFlow
pwd
. conda.env
source activate CensusFlow
which python

echo "JOBID = "$PBS_JOBID

# Assign the lead
lead=$(($PBS_ARRAYID))
model="gpy"  # should line up with mdls/{}.py
echo "Lead: "$lead", model: "$model

# Call model: 212==July 30
python -u run_gp.py --lead $lead --model gpy --dstart 0 --dend 212

echo "##### end of script ######"
