#!/bin/bash

# SHELL PIPELINE TO REPLICATE RESULTS

conda activate CensusFlow

# (1) Get the demographic data
python process_demographics.py

# (2) Generate the Xy matrix
python process_flow.py --bfreq "1 hour" --ylbl "census_max" --nlags 10

# (3) Get descriptive statistics
#python explore_AR.py

# (4) Fit Lasso model over various leads/days
for lead in {1..10}; do
  for day in {0..181}; do
    if [ $lead -eq 5 ] && [ $day -eq 0 ]; then
      echo "lead: "$lead", day: "$day
      python run_lasso.py --day 0 --lead 1 --nlags 10
    fi
  done
done


