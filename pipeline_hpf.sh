#!/bin/bash

# Load python env and move to CensusFlow folder
source set_env.sh

echo "(1.A) process_demographics: demographic triage data"
python process_demographics.py
# output: demo4flow.csv

echo "(1.B) process_flow: generate the Xy matrix"
python process_flow.py
# output: all_DI.csv, all_labs.csv, hourly_yX.csv

echo "End of processing script"
