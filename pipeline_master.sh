#!/bin/bash

# SHELL PIPELINE TO REPLICATE RESULTS
echo "THE STARTING DIRECTORY"
pwd

# If on HPF point to the python path
loc=$(pwd | cut -d'/' -f2)
if [ $loc != "mnt" ]; then
	echo "WE ARE ON HPF"
	cd /hpf/largeprojects/agoldenb/edrysdale/ED/CensusFlow
	. conda.env
	source activate CensusFlow
else
	echo "WE ARE ON LOCAL"
	conda activate CensusFlow
fi

echo "(1) Get the demographic data"
#python process_demographics.py

echo "(2) Generate the Xy matrix"
#python process_flow.py --bfreq "1 hour" --ylbl "census_max" --nlags 10

echo "(3) Get descriptive statistics"
#python explore_AR.py

echo "(4) Run Gaussian Process in Parallel"
for ll in {1..7..1}; do
  echo "Lead: "$ll
#  python -u run_gp.py --lead $ll --model gpy > "../lead"$ll".log" &
done

echo "(5) Compare performance to escalation"
python run_escalation.py

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
