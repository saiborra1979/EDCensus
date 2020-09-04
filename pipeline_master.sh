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
	. conda.env
	source activate CensusFlow
else
  echo "where are we?!"
  return
fi

echo "(1) Get the demographic data"
#python process_demographics.py

echo "(2) Generate the Xy matrix"
#python process_flow.py --bfreq "1 hour" --ylbl "census_max" --nlags 10

echo "(3) Get descriptive statistics"
#python explore_AR.py

echo "(4) Run Gaussian Process in Parallel"
for ll in {1..24..1}; do
  echo "Lead: "$ll
  python -u run_gp.py --lead $ll --model gpy --dtrain 125 --dval 7 --dstart 0 --dend 243
#   > "../lead"$ll".log" &
done

echo "(5) Compare performance to escalation"
#python run_escalation.py

#echo "(4) Fit Lasso model over various days"
#if [ $loc != "mnt" ]; then
#  echo "WE ARE ON HPF"
#	#qsub -N local_lead4 -F "4" pipeline_HPF.sh
#else
#	echo "WE ARE ON LOCAL"
#	for tt in {75..76..1}; do
#    echo "Day: "$tt
#    python run_mdl.py --day $tt --lead 4 --nlags 10 --model local
#  done
#fi

echo "END OF SCRIPT"
