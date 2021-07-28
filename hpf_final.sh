#!/bin/bash

# Set parameters
n_jobs=11  # Needs to line up hpf_model.sh: nodes=1:ppn=n_jobs+1
ppn=$(($n_jobs+1))
n_tree=100
depth=3

# Training days
dtrain=366
# Validation hours
nval=48
# Retraining hours
rtrain=24

perm="model=gp_stacker_dtrain="$dtrain"_rtrain="$rtrain"_hval="$nval
echo $perm
model_args=base=rxgboost-nval=$nval-max_iter=100-lr=0.1-n_trees=$n_tree-depth=$depth-n_jobs=$n_jobs

# Run model
qsub -t 1-16 -N "gpu_"$perm -q gpu -l nodes=1:ppn=$ppn:gpus=1 -v model_name=gp_stacker,dtrain=$dtrain,rtrain=$rtrain,model_args=$model_args hpf_model.sh


echo "End of hpf_hyperparameter.sh"
