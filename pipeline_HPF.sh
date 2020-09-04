#!/bin/bash

#PBS -l walltime=06:00:00
#PBS -o /home/edrysdale/qsub/
#PBS -e /home/edrysdale/qsub/
#PBS -l vmem=16g
#PBS -l mem=16g
#PBS -l nodes=1:ppn=4
#PBS -t 1-24

# EXAMPLE OF HOW TO RUN: qsub -N gpy_run pipeline_HPF.sh

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

python -u run_gp.py --lead lead --model gpy --dtrain 125 --dval 7 --dstart 60 --dend 62

echo "##### end of script ######"
