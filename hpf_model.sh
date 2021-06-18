#!/bin/bash

#PBS -l walltime=63:59:59
#PBS -o /hpf/largeprojects/agoldenb/edrysdale/ED_lvl1/qsub
#PBS -e /hpf/largeprojects/agoldenb/edrysdale/ED_lvl1/qsub
#PBS -l vmem=32g
#PBS -l mem=32g

# Modify the model_args command
echo "model args="$model_args
model_args=$(echo $model_args | sed -r "s/\\-/,/g")

echo "model args="$model_args
echo "model_name="$model_name
echo "dtrain="$dtrain
echo "rtrain="$rtrain

# Load python env and move to CensusFlow folder
source set_env.sh

# Run python script
python -u run_mdl.py --model_name $model_name --model_args $model_args --ylbl census_max --lead 24 --lag 24 --dtrain $dtrain --h_retrain $rtrain

echo "End of hpf_xgboost.sh"

