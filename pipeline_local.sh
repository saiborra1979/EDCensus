#!/bin/bash

conda activate CensusFlow
 
#############################
# --- (1) PREPROCESSING --- #

echo "(1.A) process_demographics: demographic triage data"
python process_demographics.py
# output: demo4flow.csv

echo "(1.B) process_flow: generate the Xy matrix"
python process_flow.py
# output: all_DI.csv, all_labs.csv, hourly_yX.csv

return

###########################
# --- (2) EXPLORATORY --- #

echo "(2.A) explore_data.py"
python explore_data.py
# output: gg_census, gg_r2_hour, gg_hour_scatter

echo "(2.B) explore_beds.py"
python explore_beds.py
# output: None

# echo "(2.C) explore_AR.py: summary stats"
# python explore_AR.py
# # output: gg_err_lead, gg_scat_lead, gg_err_dist
# #         gg_err_ord, gg_qreg, gg_qreg_linear, gg_qreg_GBR, gg_np_quant


##########################
# --- (3) MODEL RUNS --- #

echo "(3.A) run_bl.py"
python run_bl.py --ylbl census_max

# Call source pipeline_GPU.sh on HPF
#   Adjust --dstart 60 --dend 453 with pipeline_HPF.sh for last day
#  python -u run_gp.py --lead $ll --model gpy --dtrain 125 --dval 7 --dstart 60 --dend 243

##########################
# --- (4) EVALUATION --- #

echo "(4.A) merge_qsub.py"
# Combine all of the qsub output
python merge_qsub.py --model_list gp_stacker

echo "(4.B) eval_hyperparameters.py"
# Will look for most recent GPy folder
python eval_hp_agg.py --model_list gp_stacker


echo "END OF SCRIPT"
