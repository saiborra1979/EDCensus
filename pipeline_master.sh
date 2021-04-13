#!/bin/bash

# SHELL PIPELINE TO REPLICATE RESULTS
echo "THE STARTING DIRECTORY"
pwd

# If on HPF point to the python path
loc=$(pwd | cut -d'/' -f3)

if [ "$loc" == "c" ]; then
  echo "we are on predator"
  conda activate CensusFlow
elif [ "$loc" == "erik" ]; then
  echo "we are on Snowqueen"
  conda activate CensusFlow
elif [ "$loc" == "largeprojects" ]; then
	echo "WE ARE ON HPF"
	cd /hpf/largeprojects/agoldenb/edrysdale/ED/CensusFlow || return
  which python
  module load python/3.8.1
  fold_env="/hpf/largeprojects/agoldenb/edrysdale/venv/CensusFlow/bin/activate"
  . $fold_env
  which python
else
  echo "where are we?!"
  return
fi

#source transfer_csv.sh sep 2020

#############################
# --- (1) PREPROCESSING --- #

echo "(1.A) process_demographics: demographic triage data"
python process_demographics.py
# output: demo4flow.csv

echo "(1.B) process_flow: generate the Xy matrix"
python process_flow.py --bfreq "1 hour" --ylbl "census_max" --nlags 10
# output: all_DI.csv, all_labs.csv, df_lead_lags.csv

###########################
# --- (2) EXPLORATORY --- #

echo "(2.A) explore_data.py"
python explore_data.py
# output: gg_census, gg_r2_hour, gg_hour_scatter

echo "(2.B) explore_beds.py"
python explore_beds.py
# output: None

echo "(2.C) explore_AR.py: summary stats"
python explore_AR.py
# output: gg_err_lead, gg_scat_lead, gg_err_dist
#         gg_err_ord, gg_qreg, gg_qreg_linear, gg_qreg_GBR, gg_np_quant


##########################
# --- (3) MODEL RUNS --- #

echo "(3.A) run_gp.py"
# Call source pipeline_GPU.sh on HPF
#   Adjust --dstart 60 --dend 453 with pipeline_HPF.sh for last day
#  python -u run_gp.py --lead $ll --model gpy --dtrain 125 --dval 7 --dstart 60 --dend 243

##########################
# --- (4) EVALUATION --- #

echo "(4.A) eval_gpy.py"
# Will look for most recent GPy folder
python eval_gpy.py

echo "(4.A) eval_models.py"
# Script will compare GP to Lasso/Weighted, and then find the best "groups" for the GP
python eval_models.py
# output: best_ylbl.csv, best_ypred.csv
#         gg_hour, gg_hour2
#         gg_perf_{metric}, gg_gpy_perf_{metric}
#         gg_grpz_leads, gg_ntrain_leads, gg_grpz_months, gg_leads_agg
#         gg_pr_groups_lead, gg_pr_ntrain_lead

echo "(4.B) eval_escalation.py: Compare performance to escalation"
# uses: best_ylbl.csv, best_ypred.csv
python eval_escalation.py

echo "(4.C) eval_retrain.py: Compare retraining stragegies"
# Uses the output/flow/test/\{iterative,retrain\} folders to make R2 comparison
python eval_retrain.py
# output: gg_iter_comp


echo "END OF SCRIPT"
