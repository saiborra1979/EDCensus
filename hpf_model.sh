#!/bin/bash

#PBS -l walltime=23:59:59
#PBS -o /hpf/largeprojects/agoldenb/edrysdale/ED_lvl1/qsub/out
#PBS -e /hpf/largeprojects/agoldenb/edrysdale/ED_lvl1/qsub/err
#PBS -l vmem=64g
#PBS -l mem=64g

# Job information
echo "qsub JOBID = "$PBS_JOBID
echo "qsub arrayID = "$PBS_ARRAYID
# Assign array name to month
month=$PBS_ARRAYID

# Modify the model_args command
echo "model args="$model_args
model_args=$(echo $model_args | sed -r "s/\\-/,/g")

echo "model args="$model_args
echo "model_name="$model_name
echo "dtrain="$dtrain
echo "rtrain="$rtrain

# Load python env and move to CensusFlow folder
cd /hpf/largeprojects/agoldenb/edrysdale/ED_lvl1/CensusFlow || return
source set_env.sh

# Run python script
python -u run_mdl.py --month $month --model_name $model_name --model_args $model_args --ylbl census_max --lead 24 --lag 24 --dtrain $dtrain --h_rtrain $rtrain --write_scores --write_model

echo "End of hpf_xgboost.sh"
