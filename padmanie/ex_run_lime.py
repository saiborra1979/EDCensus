"""
EXAMPLE SCRIPT TO SHOW HOW TO LOAD AND RUN THE MODEL, AND THEN RUN IT THROUGH THE LIME EXPLAINER

Assumed folder structure:
On my HPF, the TOP_LEVEL_FOLDER == see /hpf/largeprojects/agoldenb/edrysdale/ED/

TOP_LEVEL_FOLDER
---CensusFlow
------{all the python scripts, etc}
---output
------flow
---------test
------------{date}
---------------*.csv [result output]
---------------pt
------------------[saved model weights]

For example you can download the most recent output and .pt files here
/hpf/largeprojects/agoldenb/edrysdale/ED/output/flow/test/2021_01_11

python padmanie/ex_run_lime.py --lead 10 --mdl_date 2020-09-07 --num_pred 2 --groups mds arr CTAS

Chunks of this have been copy-pasted from the ex_run_mdl.py file, so I am sure that there 
    are some steps that are not necessarily required for the LIME explainer to work.
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lead', type=int, default=1, help='Which lead of the data to predict?')
parser.add_argument('--mdl_date', type=str, default='2020-10-01', help='Which model date to use?')
parser.add_argument('--num_pred', type=int, default=2, help='How many predictions to explain with LIME?')
parser.add_argument('--groups', nargs='+',
                    help='Which kernel groups to include? (mds, health, demo, language, CTAS, arr, labs, DI)')
args = parser.parse_args()
print(args)
lead, mdl_date, num_pred = args.lead, args.mdl_date, args.num_pred
groups = None
if hasattr(args, 'groups'):
    groups = args.groups


import os

dir_base = os.getcwd()
import sys

sys.path.insert(0, os.path.join(dir_base, '..'))
sys.path.insert(0,os.path.join(dir_base))

from funs_support import find_dir_olu
import numpy as np
import pandas as pd
from time import time
from mdls.gpy import mdl
import torch
import gpytorch

use_cuda = torch.cuda.is_available()
sdev = "cuda" if use_cuda else "cpu"
print('Using device: %s' % sdev)
device = torch.device(sdev)

# Find the top level folder (modify this function to add yours)
dir_olu = find_dir_olu()
dir_output = os.path.join(dir_olu, 'output')
dir_flow = os.path.join(dir_output, 'flow')
dir_test = os.path.join(dir_flow, 'test')
# Find the most recent date
fn_test = pd.Series(os.listdir(dir_test))
fn_test = fn_test[fn_test.str.contains('^[0-9]{4}')].reset_index(None,True)
fn_test = fn_test[fn_test.str.contains('[0-9]{2}$')].reset_index(None, True)
fn_test = pd.to_datetime(fn_test.str.replace('\\_', '-'))
fn_test = fn_test[fn_test.idxmax()]
print('Most recent model rune date is: %s' % fn_test.strftime('%b %d, %Y'))
dir_mdl = os.path.join(dir_test, fn_test.strftime('%Y_%m_%d'))
dir_pt = os.path.join(dir_mdl, 'pt')

print('# --- STEP 1: LOAD DATA --- #')
idx = pd.IndexSlice
df_lead_lags = pd.read_csv(os.path.join(dir_flow, 'df_lead_lags.csv'), header=[0, 1], index_col=[0, 1, 2, 3])
# Create dates
dates = df_lead_lags.index.to_frame().astype(str).assign(
    date=lambda x: pd.to_datetime(x.year + '-' + x.month + '-' + x.day + ' ' + x.hour + ':00:00')).date
# Extract y
yval = df_lead_lags.loc[:, idx[:, 'lead_' + str(lead)]].values.flatten()
# Remove lags (GP handles them automatically in the kernel)
Xmat = df_lead_lags.loc[:, idx[:, 'lag_0']].droplevel(1, 1)
cn = list(Xmat.columns)
Xmat = Xmat.values
# Extract date features (remove year/month)
tmat = dates.index.droplevel([0, 1]).to_frame(False).reset_index().rename(columns={'index': 'trend'})
Xmat = np.hstack([tmat.values, Xmat])
cn = list('date_' + tmat.columns) + cn
assert len(cn) == Xmat.shape[1]

print('# --- STEP 2: LOAD MODEL --- #')
mdl_date = pd.to_datetime(pd.Series(mdl_date))[0]
fn_pt = pd.Series(os.listdir(dir_pt))
fn_pt = fn_pt[fn_pt.str.contains('lead_' + str(lead))].reset_index(None, True)
date_pt = pd.to_datetime(fn_pt.str.split('day_', 1, True).iloc[:, 1].str.replace('.pth', ''), format='%Y%m%d')
idx_pt = date_pt[date_pt == mdl_date].index[0]
assert idx_pt is not None  # Ensure model date exists in that folder
path_pt = os.path.join(dir_pt, fn_pt[idx_pt])

# Initialize model. Valid groups: mds, health, demo, language, CTAS, arr, labs, DI
gp = mdl(model='gpy', lead=lead, cn=cn, device=device, groups=groups)
# Fit the model to the data (to create X,y data to condition on for inference time)
# I'm using the first 72 hours here
gp.fit(X=Xmat[:72], y=yval[:72], ntrain=2, nval=1)
gp.gp.load_state_dict(torch.load(path_pt, map_location=device), strict=True) #PMedit: added "strict=False" after model loading error

print('# --- STEP 3: MAKE PREDICTIONS --- #')

gp.gp.float()
gp.gp.eval()
gp.likelihood.eval()
gp.istrained = True


print('#---STEP 4: RUN LIME ON PREDICTIONS---#')
import lime
import lime.lime_tabular
from sklearn.model_selection import train_test_split

def add_to_arr(pred_num, exp_list):
    for idx,ex_ft in enumerate(exp_list):
        spl = ex_ft[0].split("<=")
        exp_row = [pred_num, idx, spl[0], spl[1],ex_ft[1]]
        exp_arr.append(exp_row)

explainer = lime.lime_tabular.LimeTabularExplainer(Xmat[:72],  
mode='regression',training_labels=yval[:72],feature_names=cn)
exp_arr=[]
for i in range(0,num_pred):
    print(f"Explaining prediction # {i}")
    exp = explainer.explain_instance(Xmat[i,:], gp.predict_arr, num_features=5)
    add_to_arr(i, exp.as_list())

df = pd.DataFrame(exp_arr, columns = ['pred_num', 'ft_rank', 'feature', 'ft_val', 'ft_impact'])
df.to_csv(path_or_buf=dir_output+"\\LIME_results.csv")

print(df.head(10))