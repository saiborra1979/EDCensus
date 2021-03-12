#!/bin/bash

# SHELL PIPELINE TO REPLICATE RESULTS
echo "THE STARTING DIRECTORY"
pwd

# If on HPF point to the python path
loc=$(pwd | cut -d'/' -f3)

if [ "$loc" == "c" ]; then
  echo "we are on predator"
  conda activate CensusFlow
elif [ "$loc" == "erik" ]; then
  echo "we are on Snowqueen"
  conda activate CensusFlow
elif [ "$loc" == "largeprojects" ]; then
	echo "WE ARE ON HPF"
	cd /hpf/largeprojects/agoldenb/edrysdale/ED/CensusFlow || return
  which python
  module load python/3.8.1
  fold_env="/hpf/largeprojects/agoldenb/edrysdale/venv/CensusFlow/bin/activate"
  . $fold_env
  which python
else
  echo "where are we?!"
  return
fi

#source transfer_csv.sh sep 2020

echo "(0) Doing exploratory analysis"
python run_exploratory.py

echo "(1) Get the demographic data"
python process_demographics.py

echo "(2) Generate the Xy matrix"
python process_flow.py --bfreq "1 hour" --ylbl "census_max" --nlags 10

echo "(3) Get descriptive statistics"
#python explore_AR.py

echo "(4) Run Gaussian Process in Parallel"
for ll in {1..24..1}; do
  echo "Lead: "$ll
#  python -u run_gp.py --lead $ll --model gpy --dtrain 125 --dval 7 --dstart 60 --dend 243
done

echo "(5) Compare the different approaches"
#python run_eval.py

echo "(6) Compare performance to escalation"
#python run_escalation.py

echo "(7) Do granular model comparisons"
#python run_comp.py

echo "END OF SCRIPT"
