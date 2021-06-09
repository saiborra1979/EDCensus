#!/bin/bash

cd /hpf/largeprojects/agoldenb/edrysdale/ED_lvl1/CensusFlow || return
module load python/3.8.1
souce /hpf/largeprojects/agoldenb/edrysdale/venv/CensusFlow/bin/activate
which python

echo "(1.A) process_demographics: demographic triage data"
python process_demographics.py
# output: demo4flow.csv

echo "(1.B) process_flow: generate the Xy matrix"
python process_flow.py
# output: all_DI.csv, all_labs.csv, hourly_yX.csv

echo "End of processing script"