import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lead', type=int, default=1, help='Which lead of the data to predict?')
parser.add_argument('--dtrain', type=int, default=125, help='How many training days to use?')
parser.add_argument('--model', type=str, default='gpy', help='Which GP to use?')
parser.add_argument('--dstart', type=int, default=0, help='Day to start in 2020 (0==Jan 1)')
parser.add_argument('--groups', nargs='+',
                    help='Which kernel groups to include? (mds, health, demo, language, CTAS, arr, labs, DI)')
parser.add_argument('--save_pt', help='Should kernels be saved?', action='store_true')
args = parser.parse_args()
print(args)
lead, model = args.lead, args.model,
dtrain, dstart = args.dtrain, args.dstart
save_pt = args.save_pt
groups = None
if hasattr(args, 'groups'):
    groups = args.groups

# lead, model, dstart, groups, dtrain, save_pt = 5, 'gpy', 60, ['mds','arr','CTAS'], 5, False

import os
import sys
import numpy as np
import pandas as pd
from time import time
from mdls.gpy import mdl
import torch
from datetime import datetime
from funs_support import makeifnot, find_dir_olu, get_date_range
import gpytorch

dir_base = os.getcwd()
dir_olu = find_dir_olu()
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
# Extract y (lead_0 is the current value)
yval = df_lead_lags.loc[:, idx[:, 'lead_0']].values.flatten()
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
dat_suffix = pd.DataFrame({'term':['dstart','dtrain','groups'],'val':[dstart,dtrain,sgroups]})
suffix = dat_suffix.assign(lbl=lambda x: '_'+x.term+'_'+x.val.astype(str)).lbl.str.cat(sep='')

################################################
# --- STEP 2: CREATE DATE-SPLITS AND TRAIN --- #
print('# --- STEP 2: CREATE DATE-SPLITS AND TRAIN --- #')

dfmt = '%Y-%m-%d'
dmax = pd.to_datetime((dates.max()-pd.DateOffset(days=1)).strftime(dfmt))
dmin = pd.to_datetime('2020-01-01') + pd.DateOffset(days=dstart)
nhours = int((dmax - dmin).total_seconds()/(60*60))
ndays = int(nhours / 24)
print('day start: %s, day stop: %s (hours=%i, days=%i)' % 
    (dmin.strftime(dfmt),dmax.strftime(dfmt),nhours, ndays))

# Initialize model. Valid groups: mds, health, demo, language, CTAS, arr, labs, DI
gp = mdl(model=model, lead=lead, cn=cn, device=device, groups=groups)
# Pre-specify offsets
offset_lead = pd.DateOffset(hours=lead)
offset_train = pd.DateOffset(days=dtrain)
# Total number of hours
holder_ii, stime = [], time()
for ii in range(nhours):
    # (i) Set up the time ranges
    time_ii = dmin + pd.DateOffset(hours=ii)
    hour = ii % 24
    print('Time: %s, hour: %i' % (time_ii,hour))
    s_train = time_ii - offset_train - offset_lead  # of train days plus lead buffer
    idx_train = ((dates > s_train) & (dates <= time_ii)).values
    idx_test = dates == time_ii  # One row of the data
    assert idx_test.sum() == 1
    dates_train = dates[idx_train].values
    dy_train = pd.DataFrame({'dates':dates_train,'y':yval[idx_train]})
    dy_train = dy_train.assign(y_lead=lambda x: x.y.shift(-lead)).dropna()
    dy_train.y_lead = dy_train.y_lead.astype(int)
    y_train = dy_train.y_lead.values
    X_train = Xmat[idx_train][dy_train.index.values].copy()
    X_test = Xmat[idx_test].copy()
    print('Training range: %s' % (get_date_range(dy_train.dates)))    
    # (ii) Fit the model
    gp.fit(X=X_train, y=y_train)  # Initialize model
    if ii > 0:
        gp.gp.load_state_dict(holder_state)  # Re-load weights
    gp.tune(max_iter=1000, lr=0.01)  # Tune hyperparameters
    torch.cuda.empty_cache()
    holder_state = gp.gp.state_dict().copy()  # Make copy of weights
    res = gp.predict(X=X_test).assign(time=time_ii,hour=hour,lead=lead)
    
    # (iii) Save and run time
    fn_state = gp.fn.replace('.pkl',suffix+'_day_'+time_ii.strftime('%Y%m%d')+'.pth')
    if save_pt:
        torch.save(holder_state, os.path.join(dir_save_pt,fn_state))
    holder_ii.append(res)
    nsec, nleft = time() - stime, nhours - (ii+1)
    rate = (ii+1) / nsec
    eta_min = nleft/rate/60
    print('ETA: %0.1f minutes, for %i remaining' % (eta_min, nleft))

####################################
# --- STEP 3: SAVE PREDICTIONS --- #

fn_res = gp.fn.replace('mdl_', 'res_')
fn_res = fn_res.replace('.pkl', suffix + '.csv')

# Merge with actual labels
df_res = pd.concat(holder_ii).reset_index(None,True)
df_res = df_res.rename(columns={'time':'date_rt','mu': 'pred'}).drop(columns='hour')
df_res = df_res.assign(date_pred=lambda x: x.date_rt+offset_lead)
df_res = df_res.merge(pd.DataFrame({'date_pred':dates.values,'y':yval}),'left','date_pred')
df_res = df_res.assign(model=model, groups=sgroups, dtrain=dtrain)
df_res.to_csv(os.path.join(dir_save, fn_res), index=False)
print('End of script')

# from plotnine import *
# from scipy.stats import pearsonr, spearmanr
# from sklearn.metrics import r2_score
# dir_figures = os.path.join(dir_olu, 'figures', 'census')

# tmp = df_res.assign(doy=lambda x: x.date_rt.dt.strftime('%y-%m-%d'))
# tmp.groupby('doy').apply(lambda x: 
#     pd.Series({'spearman':spearmanr(x.y,x.pred)[0],'r2':r2_score(x.y,x.pred)}))

# gg = (ggplot(df_res,aes(x='date_pred')) + theme_bw() + 
#     geom_point(aes(y='y'),color='black') + 
#     geom_line(aes(y='pred')) + 
#     scale_x_datetime(date_breaks='1 day'))
# gg.save(os.path.join(dir_figures, 'gg.png'),height=4,width=6)