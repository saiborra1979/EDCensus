#!/bin/bash

# hyperparametr search for different models
dir_err=/home/edrysdale/qsub


echo "End of hpf_hyperparameter.sh"

# --- (1) R/XGBOOST --- #
tree_seq="100"
depth_seq="3"
dtrain_seq="120"
retrain_seq="72"
model_seq="xgboost rxgboost"

#for dtrain in {60..525..15}; do
#for rtrain in {24..720..24}; do

i=0

for model in $model_seq; do
for n_tree in $tree_seq; do
for depth in $depth_seq; do
for dtrain in $dtrain_seq; do
for rtrain in $retrain_seq; do
	i=$(($i+1))
	echo "Iteration: "$i
	perm="model="$model"_n_trees="$n_tree"_depth="$depth"_dtrain="$dtrain"_rtrain="$rtrain
	echo $perm
	# model_args written with "-", needs to become "," in hpf_model.sh
	model_args=n_trees=$n_tree-depth=$depth-n_jobs=11
	qsub -N $perm -v model_name=$model,dtrain=$dtrain,rtrain=$rtrain,model_args=$model_args hpf_model.sh
#        return
done
done
done
done
done
