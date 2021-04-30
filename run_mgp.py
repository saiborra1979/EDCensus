import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dtrain', type=int, default=125, help='How many training days to use?')
parser.add_argument('--dval', type=int, default=7, help='How many validation days to use?')
parser.add_argument('--model', type=str, default='mgpy', help='Which GP to use?')
parser.add_argument('--dstart', type=int, default=0, help='Day to start in 2020 (0==Jan 1)')
parser.add_argument('--groups', nargs='+',
                    help='Which kernel groups to include? (mds, health, demo, language, CTAS, arr, labs, DI)')
parser.add_argument('--save_pt', help='Should kernels be saved?', action='store_true')
args = parser.parse_args()
print(args)
model, save_pt = args.model, args.save_pt
dtrain, dval, dstart = args.dtrain, args.dval, args.dstart
groups = None
if hasattr(args, 'groups'):
    groups = args.groups

# model, dstart, groups, dtrain, dval, save_pt = 'mgpy', 60, ['mds','arr','CTAS'], 3, 0, False

import os
import sys
import numpy as np
import pandas as pd
from time import time
import torch
from datetime import datetime
from funs_support import makeifnot, find_dir_olu
import gpytorch
# LOAD MULTITASK MODEL
from mdls.mgpy import mdl


dir_base = os.getcwd()
dir_olu = find_dir_olu()
dir_figures = os.path.join(dir_olu, 'figures')
dir_output = os.path.join(dir_olu, 'output')
dir_flow = os.path.join(dir_output, 'flow')
dir_test = os.path.join(dir_flow, 'test')
dir_save = os.path.join(dir_test, datetime.now().strftime('%Y_%m_%d'))
dir_save_pt = os.path.join(dir_save, 'pt')
makeifnot(dir_test)
makeifnot(dir_save)
if save_pt:
    makeifnot(dir_save_pt)

idx = pd.IndexSlice
use_cuda = torch.cuda.is_available()
sdev = "cuda" if use_cuda else "cpu"
print('Using device: %s' % sdev)
device = torch.device(sdev)

gpytorch.settings.max_cg_iterations(10000)

####################################
# --- STEP 1: LOAD/CREATE DATA --- #
print('# --- STEP 1: LOAD/CREATE DATA --- #')

# Get dataframe
df_lead_lags = pd.read_csv(os.path.join(dir_flow, 'df_lead_lags.csv'), header=[0, 1], index_col=[0, 1, 2, 3])
# Create dates
dates = df_lead_lags.index.to_frame().astype(str).assign(
    date=lambda x: pd.to_datetime(x.year + '-' + x.month + '-' + x.day + ' ' + x.hour + ':00:00')).date
# Extract patient volumes (inlcuding t_0)
Ymat = df_lead_lags.loc[:,idx['y']].values
# Remove lags (GP handles them automatically in the kernel)
Xmat = df_lead_lags.loc[:, idx[:, 'lag_0']].droplevel(1, 1)
cn = list(Xmat.columns)
Xmat = Xmat.values
# Extract date features (remove year/month)
tmat = dates.index.droplevel([0, 1]).to_frame(False).reset_index().rename(columns={'index': 'trend'})
Xmat = np.hstack([tmat.values, Xmat])
cn = list('date_' + tmat.columns) + cn
assert len(cn) == Xmat.shape[1]

if groups is None:
    sgroups = str(groups)
else:
    sgroups = '-'.join(groups)
# For saving: day start/end/# training days/groups
dat_suffix = pd.DataFrame({'term':['dstart','dtrain','dval','groups'],'val':[dstart,dtrain,dval,sgroups]})
suffix = dat_suffix.assign(lbl=lambda x: '_'+x.term+'_'+x.val.astype(str)).lbl.str.cat(sep='')

################################################
# --- STEP 2: CREATE DATE-SPLITS AND TRAIN --- #
print('# --- STEP 2: CREATE DATE-SPLITS AND TRAIN --- #')

dfmt = '%Y-%m-%d'
dmax = pd.to_datetime((dates.max()-pd.DateOffset(days=1)).strftime(dfmt))
print('date max of time series: %s' % dmax.strftime(dfmt))
# dstart==0 implies Jan 1, 2020
d_range = pd.date_range('2020-01-01', dmax.strftime(dfmt), freq='1d')
d_pred = d_range[dstart:]
# Subset if pred range is outside of date max
if d_pred.min() > dmax:
    sys.exit('Smallest date range is exceeds largest dates in df_lead_lags')
if d_pred.max() > dmax:
    print('d_pred has dates greater than dmax, subsetting')
    d_pred = d_pred[d_pred <= dmax]

# Initialize model. Valid groups: mds, health, demo, language, CTAS, arr, labs, DI
mgp = mdl(model=model, cn=cn, device=device, groups=groups)

holder = []
ii, btime = 0, time()
for day, s_test in enumerate(d_pred):
    ii += 1
    print('Multitask prediction for testing day: %s\nIteration %i of %i' % (s_test.strftime('%Y-%m-%d'), day + 1, len(d_pred)))
    assert day + 1 <= len(d_pred)
    # Using previous week for validation data
    s_valid = s_test - pd.DateOffset(days=dval)
    s_train = s_valid - pd.DateOffset(days=dtrain)
    e_test = s_test + pd.DateOffset(days=1)
    idx_train = ((dates >= s_train) & (dates < s_valid)).values
    idx_valid = ((dates >= s_valid) & (dates < s_test)).values
    idx_test = ((dates >= s_test) & (dates < e_test)).values
    assert sum(idx_valid) == int(24 * dval)
    assert sum(idx_test) <= 24
    # Split matrices
    Xmat_train, Ymat_train = Xmat[idx_train], Ymat[idx_train]
    Xmat_valid, Ymat_valid = Xmat[idx_valid], Ymat[idx_valid]
    Xmat_test, Ymat_test = Xmat[idx_test], Ymat[idx_test]
    ntrain, nval, ntest = Xmat_train.shape[0], Xmat_valid.shape[0], Xmat_test.shape[0]
    print('Training size: %i, validation size: %i, test size: %i' % (ntrain, nval, ntest))
    # Combine train/validation for GP
    Xmat_tval = np.vstack([Xmat_train, Xmat_valid])
    Ymat_tval = np.vstack([Ymat_train, Ymat_valid])
    mgp.fit(X=Xmat_tval, Y=Ymat_tval)
    if ii > 1:
        mgp.gp.load_state_dict(holder_state)  # Load state dict from previous iteration
    if ii == 2:
        btime = time()
    stime = time()
    mgp.tune(max_iter=1000, lr=0.01, get_train=False)
    torch.cuda.empty_cache()
    holder_state = mgp.gp.state_dict().copy()
    fn_state = mgp.fn.replace('.pkl',suffix+'_day_'+s_test.strftime('%Y%m%d')+'.pth')
    if save_pt:
        torch.save(holder_state, os.path.join(dir_save_pt,fn_state))
    res = mgp.predict(X=Xmat_test, Y=Ymat_test)
    res = res.assign(date=s_test).rename(columns={'idx':'hour'})
    holder.append(res)
    nleft, rate = len(d_pred) - ii, ii / (time() - btime)
    print('Model took %i seconds to tune and predict (ETA=%i sec' % (time() - stime, nleft/rate))
    print(res)

####################################
# --- STEP 3: SAVE PREDICTIONS --- #

fn_res = mgp.fn.replace('mdl_', 'res_')
fn_res = fn_res.replace('.pkl', suffix + '.csv')
df_res = pd.concat(holder).reset_index(None, True).rename(columns={'mu': 'pred'})
df_res = df_res.assign(model=model, groups=sgroups, ntrain=ntrain)
df_res.to_csv(os.path.join(dir_save, fn_res), index=False)

print('End of script')
