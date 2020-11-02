#!/bin/bash

#PBS -l walltime=12:00:00
#PBS -o /home/edrysdale/qsub/
#PBS -e /home/edrysdale/qsub/
#PBS -l vmem=32g
#PBS -l mem=32g
#PBS -l nodes=1:ppn=4:gpus=1
#PBS -q gpu

#cpu: PBS -l nodes=1:ppn=4
#custom: PBS -t 5-6

# EXAMPLE OF HOW TO RUN: qsub -N gpy_run -t 1-6 pipeline_HPF.sh
echo "Groups: "$groups
ndays=$(($ndays))
echo "Number of days: "$ndays

cd /hpf/largeprojects/agoldenb/edrysdale/ED/CensusFlow || return
pwd
which python
module load python/3.8.1
fold_env="/hpf/largeprojects/agoldenb/edrysdale/venv/CensusFlow/bin/activate"
. $fold_env
which python
python -u check_cuda.py

echo "JOBID = "$PBS_JOBID

# Assign the lead
lead=$(($PBS_ARRAYID))
model="gpy"  # should line up with mdls/{}.py
echo "Lead: "$lead", model: "$model

if [ "$groups" == "None" ]; then
	echo "Not running a group"
	python -u run_gp.py --lead $lead --model gpy --dtrain $ndays --dval 7 --dstart 60 --dend 273
else
	python -u run_gp.py --lead $lead --model gpy --dtrain $ndays --dval 7 --dstart 60 --dend 273 --groups $groups
fi

echo "##### end of script ######"
