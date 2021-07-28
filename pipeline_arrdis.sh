#!/bin/bash

# Example of script to make forecasts for arrivals and discharges

cpu=$(hostname)
if [[ $cpu =~ "qlogin" ]]; then
	echo "We're on HPC"
	cd /hpf/largeprojects/agoldenb/edrysdale/ED_lvl1/CensusFlow || return
	source set_env.sh
else
	echo "We're on a local"
	conda activate CensusFlow
fi
ncores=$(nproc)
ncores=$(($ncores - 1))
# Min should be 1
ncores=$(( $ncores < 1 ? 1 : $ncores ))
echo "# of cores = "$ncores

# Set variables
ylbl="tt_arrived tt_discharged"
xlbl="is_holiday"
lead=24
lag=24
dtrain=90
h_rtrain=168
model_name="xgboost"
model_args="lr=0.1-n_trees=100-depth=3-n_jobs=3"

# Run the model
for month in {1..1..1}; do
  echo "month = "$month
  python -u run_mdl_multi.py --ylbl $ylbl --xlbl $xlbl --dtrain $dtrain --h_rtrain $h_rtrain --model_name $model_name --lead $lead --lag $lag --month $month
done


