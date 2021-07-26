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

# Set variables
ylbl="tt_arrival,tt_discharge"
xlbl="is_holiday"
lead=24
lag=24
model_name="ElasticNet"
model_args=""

# Run the model
for month in {1..16..1}; do
  echo "month = "$month
  #python -u run_mdl_multi --ylbl $ylbl --xlbl $xlbl --model_name $model_name --lead $lead --lag $lag
done


