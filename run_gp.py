import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lead', type=int, default=1, help='Which lead of the data to predict?')
parser.add_argument('--dtrain', type=int, default=125, help='How many training days to use?')
parser.add_argument('--dval', type=int, default=7, help='How many validation days to use?')
parser.add_argument('--model', type=str, default='gpy', help='Which GP to use?')
parser.add_argument('--dstart', type=int, default=0, help='Day to start in 2020 (0==Jan 1)')
parser.add_argument('--dend', type=int, default=366, help='Day to end in 2020 (366==Dec 31)')
parser.add_argument('--groups', nargs='+',
                    help='Which kernel groups to include? (mds, health, demo, language, CTAS, arr, labs, DI)')
args = parser.parse_args()
print(args)
lead, model = args.lead, args.model,
dtrain, dval = args.dtrain, args.dval
dstart, dend = args.dstart, args.dend
groups = None
if hasattr(args, 'groups'):
    groups = args.groups

# # Debugging in PyCharm (78==March 19th)
# lead, model, dstart, dend, groups, dtrain, dval = 4, 'gpy', 60, 61, ['CTAS'], 125, 7

import os
import sys
import numpy as np
import pandas as pd
from time import time
from mdls.gpy import mdl
import torch
from datetime import datetime
from funs_support import makeifnot
import gpytorch

# from sklearn.metrics import r2_score

dir_base = os.getcwd()
dir_figures = os.path.join(dir_base, '..', 'figures')
dir_output = os.path.join(dir_base, '..', 'output')
dir_flow = os.path.join(dir_output, 'flow')
dir_test = os.path.join(dir_flow, 'test')
dir_save = os.path.join(dir_test, datetime.now().strftime('%Y_%m_%d'))
makeifnot(dir_test)
makeifnot(dir_save)

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

################################################
# --- STEP 2: CREATE DATE-SPLITS AND TRAIN --- #
print('# --- STEP 2: CREATE DATE-SPLITS AND TRAIN --- #')

# Days of 2020
d_range = pd.date_range('2020-01-01', '2020-12-31', freq='1d')
d_pred = d_range[dstart:dend]
# Subset if pred range is outside of date max
dmax = pd.to_datetime((dates - pd.offsets.Day(1)).max().strftime('%Y-%m-%d'))

if d_pred.min() > dmax:
    sys.exit('Smallest date range is exceeds largest dates in df_lead_lags')
if d_pred.max() > dmax:
    print('d_pred has dates greater than dmax, subsetting')
    d_pred = d_pred[d_pred <= dmax]

print('Test days to be predicted:\n%s\n' % ', '.join(d_pred.astype(str)))

# Initialize model. Valid groups: mds, health, demo, language, CTAS, arr, labs, DI
gp = mdl(model=model, lead=lead, cn=cn, device=device, groups=groups)

holder = []
ii = 0
for day, s_test in enumerate(d_pred):
    ii += 1
    # day = 78; s_test = d_pred[day]
    print('Predicting %i hours ahead for testing day: %s\nIteration %i of %i' %
          (lead, s_test.strftime('%Y-%m-%d'), day + 1, len(d_pred)))
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
    Xmat_train, y_train = Xmat[idx_train], yval[idx_train]
    Xmat_valid, y_valid = Xmat[idx_valid], yval[idx_valid]
    Xmat_test, y_test = Xmat[idx_test], yval[idx_test]
    ntrain, nval, ntest = Xmat_train.shape[0], Xmat_valid.shape[0], Xmat_test.shape[0]
    print('Training size: %i, validation size: %i, test size: %i' % (ntrain, nval, ntest))
    # Combine train/validation
    Xmat_tval, y_tval = np.vstack([Xmat_train, Xmat_valid]), np.append(y_train, y_valid)
    gp.fit(X=Xmat_tval, y=y_tval, ntrain=ntrain, nval=nval)
    if ii > 1:
        # Load state dict from previous iteration
        gp.gp.load_state_dict(holder_state)
    stime = time()
    gp.tune(max_iter=1000, lr=0.01)
    torch.cuda.empty_cache()
    holder_state = gp.gp.state_dict().copy()
    res = gp.predict(X=Xmat_test, y=y_test)
    res = res.assign(date=s_test).rename_axis('hour').reset_index()
    holder.append(res)
    print('Model took %i seconds to tune and predict' % (time() - stime))
    print(res)
    # # Day forecast
    # tmp = pd.concat([gp.res_train.drop(columns=['se','idx']), res.assign(tt='test').drop(columns=['hour','date'])])
    # tmp = tmp.reset_index(None, True).rename_axis('idx').reset_index()
    # from plotnine import *
    # gg_torch = (ggplot(tmp[tmp.tt!='train'], aes(x='idx', y='mu', color='tt')) + theme_bw() + geom_line() +
    #             geom_vline(xintercept=gp.ntrain) + geom_ribbon(aes(ymin='lb', ymax='ub'), alpha=0.5) +
    #             geom_point(aes(x='idx', y='y'), color='black', size=0.5, alpha=0.5))
    # gg_torch.save(os.path.join(dir_figures, 'test.png'),width=12,height=7)

####################################
# --- STEP 3: SAVE PREDICTIONS --- #

fn_res = gp.fn.replace('mdl_', 'res_')
if groups is None:
    sgroups = str(groups)
else:
    sgroups = '-'.join(groups)
fn_res = fn_res.replace('.pkl', '_dstart_' + str(dstart) + '_dend_' + str(dend) + '_groups_' + sgroups + '.csv')
df_res = pd.concat(holder).reset_index(None, True).rename(columns={'mu': 'pred'})
df_res = df_res.assign(lead=lead, model=model, groups=sgroups, ntrain=ntrain)
df_res.to_csv(os.path.join(dir_save, fn_res), index=True)
