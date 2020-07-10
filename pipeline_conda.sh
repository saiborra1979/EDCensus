#!/bin/bash

cd /hpf/largeprojects/agoldenb/edrysdale/ED/CensusFlow

# Build the environment
module load anaconda3/4.4.0
check=$(($(conda env list | grep CensusFlow | wc -l)))
if [ $check -eq 0 ]; then
        echo "Conda environment does not exist, creating"
        conda env create -f environment.yml
else
        echo "Conda environment already exists"
fi

source activate CensusFlow
which python

