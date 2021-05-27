#!/bin/bash

#PBS -l walltime=23:59:59
#PBS -o /home/edrysdale/qsub/
#PBS -e /home/edrysdale/qsub/
#PBS -l vmem=16g
#PBS -l mem=16g
#PBS -l nodes=1:ppn=12

# Script to run the XGBoost model
cd /hpf/largeprojects/agoldenb/edrysdale/ED_lvl1/CensusFlow || return
module load python/3.8.1
fold_env="/hpf/largeprojects/agoldenb/edrysdale/venv/CensusFlow/bin/activate"
. $fold_env
which python

# Run python script
python -u run_mdl.py --model_name xgboost  --ylbl census_max --lead 24 --lag 24 \
    --model_args n_trees=$n_tree,depth=$depth,n_jobs=1 \
    --dtrain $dtrain --h_retrain $rtrain

echo "End of hpf_xgboost.sh"

