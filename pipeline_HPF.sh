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
cd /hpf/largeprojects/agoldenb/edrysdale/ED/CensusFlow || return
pwd
which python
module load python/3.8.1
fold_env="/hpf/largeprojects/agoldenb/edrysdale/venv/CensusFlow/bin/activate"
. $fold_env
python -u check_cuda.py

echo "JOBID = "$PBS_JOBID

# MULTITASK GP
ndays=$(($PBS_ARRAYID))
echo "Number of days: "$ndays
echo "Groups: "$groups
#python -u run_mgp.py --model mgpy --dtrain $ndays --dval 0 --dstart 60 --groups $groups


# VANILLA GP
# lead=$(($PBS_ARRAYID))
# model="gpy"  # should line up with mdls/{}.py
# echo "Lead: "$lead", model: "$model

# # dstart=60 Corresponds to March 1st, 2020
# if [ "$groups" == "None" ]; then
# 	echo "Not running a group"
# 	python -u run_gp.py --lead $lead --model gpy --dtrain $ndays --dval 0 --dstart 60
# else
# 	python -u run_gp.py --lead $lead --model gpy --dtrain $ndays --dval 0 --dstart 60 --groups $groups
# fi

echo "##### end of script ######"
