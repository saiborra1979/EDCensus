#!/bin/bash

# hyperparametr search for different models
dir_err=/home/edrysdale/qsub

# --- (1) XGBOOST --- #
tree_seq="100"
depth_seq="3"
dtrain_seq="180 360 520"
retrain_seq="24 48 72"

i=0
for n_tree in $tree_seq; do
for depth in $depth_seq; do
for dtrain in $dtrain_seq; do
for rtrain in $retrain_seq; do
	i=$(($i+1))
	echo "Iteration: "$i
	perm="n_trees="$n_tree"_depth="$depth"_dtrain="$dtrain"_rtrain="$rtrain
	echo $perm
	model_args=n_trees=$n_tree,depth=$depth,n_jobs=11
	qsub -N xgboost_$perm -v model_name="xgboost",model_args=$model_args hpf_model.sh
done
done
done
done

echo "End of hpf_hyperparameter.sh"
