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

# (1) Get the demographic data
#python process_demographics.py

# (2) Generate the Xy matrix
#python process_flow.py --bfreq "1 hour" --ylbl "census_max" --nlags 10

# (3) Get descriptive statistics
#python explore_AR.py

# (4) Fit Lasso model over various leads/days
ndays=182
nleads=9
ntot=$(($(expr $ndays \* $nleads - 1)))
echo "There are "$ndays" days and "$nleads" nleads, and a total of "$(($ntot+1))" scripts"

if [[ $(command -v qsub | wc -l) -eq 1 ]]; then
	echo "RUNNING MODEL FITTING ON HPF"
	qsub -t 0-$ntot -F "$ndays" pipeline_HPF.sh
else
	echo "RUN MODEL FITTING LOCALLY LOCALLY"
	python run_lasso.py --day 0 --lead 1 --nlags 10
fi


echo "END OF SCRIPT"

