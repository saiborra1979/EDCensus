#!/bin/bash

#PBS -l walltime=23:59:59
#PBS -o /home/edrysdale/qsub/
#PBS -e /home/edrysdale/qsub/
#PBS -l vmem=16g
#PBS -l mem=16g

# Modify the model_args command
echo "model args="$model_args
model_args=$(echo $model_args | sed -r "s/\\-/,/g")

echo "model args="$model_args
echo "model_name="$model_name
echo "dtrain="$dtrain
echo "rtrain="$rtrain

cd /hpf/largeprojects/agoldenb/edrysdale/ED_lvl1/CensusFlow || return
module load python/3.8.1
source /hpf/largeprojects/agoldenb/edrysdale/venv/CensusFlow/bin/activate
which python


# Run python script
python -u run_mdl.py --model_name $model_name --model_args $model_args --ylbl census_max --lead 24 --lag 24 --dtrain $dtrain --h_retrain $rtrain

echo "End of hpf_xgboost.sh"

