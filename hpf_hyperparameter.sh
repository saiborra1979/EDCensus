#!/bin/bash

# hyperparametr search for different models
dir_err=/home/edrysdale/qsub

section_list="gp_stacker"  #xgboost

n_jobs=11  # Needs to line up hpf_model.sh: nodes=1:ppn=n_jobs+1
ppn=$(($n_jobs+1))

##########################
# --- (2) GP STACKER --- #

if [[ "$section_list" =~ "gp_stacker" ]]; then
	echo "(2) GP STACKER"

n_tree=100
depth=3

i=0
for nval in {24..168..24}; do
for dtrain in {15..60..15}; do
for rtrain in {1..7..2}; do
	i=$(($i+1))
	echo "Iteration: "$i
	perm="model=gp_stacker_dtrain="$dtrain"_rtrain="$rtrain
	echo $perm
	# model_args written with "-", needs to become "," in hpf_model.sh
	model_args=base=rxgboost-nval=$nval-max_iter=100-lr=0.1-n_trees=$n_tree-depth=$depth-n_jobs=$n_jobs
	break 3
done; done; done


fi


# --------------------- #


#########################
# --- (1) R/XGBOOST --- #


if [[ "$section_list" =~ "xgboost" ]]; then
	echo "(1) R/XGBOOST"

tree_seq="100"
depth_seq="3"
dtrain_seq="120"  #{60..525..15}
retrain_seq="72"  #{24..720..24}
model_seq="rxgboost"  #xgboost

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
	model_args=n_trees=$n_tree-depth=$depth-n_jobs=$n_jobs
	# PBS for number of jobs needs to be set on the commmand line: -l nodes=1:ppn=12
	#qsub -N $perm -l nodes=1:ppn=$ppn -v model_name=$model,dtrain=$dtrain,rtrain=$rtrain,model_args=$model_args hpf_model.sh
#        return
done; done; done; done; done

fi

# --------------------- #




echo "End of hpf_hyperparameter.sh"