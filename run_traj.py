# SCRIPT TO COMPARE HOW TRAJECTORY CHANGES FROM PERTURBATION TO FEATURES #

import os
import copy
import numpy as np
import pandas as pd
from plotnine import *
from mdls.gpy import mdl
import torch
import gpytorch

from funs_support import find_dir_olu, gg_save

dir_olu = find_dir_olu()
dir_output = os.path.join(dir_olu, 'output')
dir_flow = os.path.join(dir_output, 'flow')
dir_test = os.path.join(dir_flow, 'test')
dir_figures = os.path.join(dir_olu, 'figures', 'census')
lst_dir = [dir_output, dir_flow, dir_test, dir_figures]
assert all([os.path.exists(z) for z in lst_dir])

use_cuda = torch.cuda.is_available()
sdev = "cuda" if use_cuda else "cpu"
print('Using device: %s' % sdev)
device = torch.device(sdev)

# Find the most recent date
fn_test = pd.Series(os.listdir(dir_test))
fn_test = fn_test[fn_test.str.contains('^[0-9]{4}\\_[0-9]{2}\\_[0-9]{2}$')].reset_index(None,True)
dates_test = pd.to_datetime(fn_test,format='%Y_%m_%d')
df_fn_test = pd.DataFrame({'fn':fn_test,'dates':dates_test,'has_gpy':False})
# Find out which have GP data
for fn in fn_test:
    fn_fold = pd.Series(os.listdir(os.path.join(dir_test,fn)))
    mdls_fold = list(pd.Series(fn_fold.str.split('\\_',2,True).iloc[:,1].unique()).dropna())
    if 'gpy' in mdls_fold:
        df_fn_test.loc[df_fn_test.fn==fn,'has_gpy'] = True
# Get largest date
assert df_fn_test.has_gpy.any()
fold_recent = df_fn_test.loc[df_fn_test.dates.idxmax()].fn
print('Most recent folder: %s' % fold_recent)
path_recent = os.path.join(dir_test,fold_recent)
fn_recent = pd.Series(os.listdir(path_recent))
fold_pt = os.path.join(path_recent, 'pt')
fn_pt = pd.Series(os.listdir(fold_pt))

# !!!!! SET DATE AND CN TO PERMUTE !!!!! #
dt_permute = pd.to_datetime('2020-09-16')
groups = 'mds-arr-CTAS'  # Separate by hyphen
sgroups = groups.split('-')
dval = 7
dtrain = 3
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #

# Find relevant models
dregex = dt_permute.strftime('%Y%m%d')+'.pth'
fn_pt = fn_pt[fn_pt.str.contains(dregex)]
fn_pt = fn_pt[fn_pt.str.contains(groups)].reset_index(None,True)
print('Found %i models' % len(fn_pt))

df_pt = pd.DataFrame({'fn':fn_pt})
df_pt['lead'] = df_pt.fn.str.split('\\_',4,True).iloc[:,3].astype(int)
lead_seq = list(range(1,25))
assert df_pt.lead.isin(lead_seq).sum()==len(lead_seq)

#######################################
# --- (1) LOAD DATA AND NORMALIZE --- #

idx = pd.IndexSlice

# - (i) CREATE Y/X DATA - #
df_lead_lags = pd.read_csv(os.path.join(dir_flow, 'df_lead_lags.csv'), header=[0, 1], index_col=[0, 1, 2, 3])
# Create dates
dates = df_lead_lags.index.to_frame().astype(str).assign(
    date=lambda x: pd.to_datetime(x.year + '-' + x.month + '-' + x.day + ' ' + x.hour + ':00:00')).date
# Extract y (note it's the only column with "lead", the X has "lags")
Ymat = df_lead_lags.loc[:,idx['y']].values
# Remove lags (GP handles them automatically in the kernel)
Xmat = df_lead_lags.loc[:, idx[:, 'lag_0']].droplevel(1, 1)
cn = list(Xmat.columns)
Xmat = Xmat.values
# Extract date features (remove year/month)
tmat = dates.index.droplevel([0, 1]).to_frame(False).reset_index().rename(columns={'index': 'trend'})
Xmat = np.hstack([tmat.values, Xmat])
cn = pd.Series(list('date_' + tmat.columns) + cn)
assert len(cn) == Xmat.shape[1]

pd.DataFrame({'cn':cn,'mx':Xmat.max(0)}).query('mx>1').head(50)
pd.DataFrame({'cn':cn,'mx':Xmat.max(0)}).query('mx==1')

# Column groups
idx_trend = np.where(cn == 'date_trend')[0]
idx_date = np.setdiff1d(np.where(cn.str.contains('date_'))[0],idx_trend)
idx_flow = np.where(cn.str.contains('census_|tt_'))[0]
idx_mds = np.where(cn.str.contains('avgmd|u_mds'))[0]
idx_arr = np.where(cn.str.contains('arr_method'))[0]
idx_CTAS = np.where(cn.str.contains('CTAS_'))[0]
di_idx = {'trend':idx_trend, 'date':idx_date, 'flow':idx_flow,'mds':idx_mds, 'arr':idx_arr, 'CTAS':idx_CTAS}
# Get the name
dat_cidx = pd.concat([pd.DataFrame({'cidx':v,'cn':cn[v]}) for k,v in di_idx.items()])
dat_cidx = dat_cidx.reset_index(None,True).rename_axis('pidx').reset_index()
# Subset X
Xmat = Xmat[:,dat_cidx.cidx.values]
assert len(dates) == len(Xmat)

# - (ii) Subset to date - #
s_valid = dt_permute - pd.DateOffset(days=dval)
s_train = s_valid - pd.DateOffset(days=dtrain)
e_test = dt_permute + pd.DateOffset(days=1)
idx_train = ((dates >= s_train) & (dates < s_valid)).values
idx_valid = ((dates >= s_valid) & (dates < dt_permute)).values
idx_test = ((dates >= dt_permute) & (dates < e_test)).values
Xmat_train, y_train = Xmat[idx_train], Ymat[idx_train]
Xmat_valid, y_valid = Xmat[idx_valid], Ymat[idx_valid]
Xmat_test, y_test = Xmat[idx_test], Ymat[idx_test]
hours_test = dates[idx_test].values
ntrain, nval, ntest = Xmat_train.shape[0], Xmat_valid.shape[0], Xmat_test.shape[0]
print('Training size: %i, validation size: %i, test size: %i' % (ntrain, nval, ntest))
# Combine train/validation
Xmat_tval, y_tval = np.vstack([Xmat_train, Xmat_valid]), np.vstack([y_train, y_valid])

#################################
# --- (2) INITIALIZE MODELS --- #

di_mdl = {}
for lead in lead_seq:
    print('Lead: %i' % lead)
    path_mdl = os.path.join(fold_pt,df_pt.query('lead==@lead').fn.values[0])
    gp = mdl(model='gpy', lead=lead, cn=list(dat_cidx.cn), device=device, groups=sgroups)
    gp.fit(X=Xmat_tval.copy(), y=y_tval[:,lead].copy(), ntrain=ntrain, nval=nval)
    gp.gp.load_state_dict(torch.load(path_mdl, map_location=device))
    gp.tune(max_iter=1, lr=0.0)
    gp.istrained = True
    for name, param in gp.gp.named_parameters():
        param.requires_grad = False
    di_mdl[lead] = copy.deepcopy(gp)
    del gp, name, param

###################################
# --- (3) ROLLING PREDICTIONS --- #

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
cn_permute = 'date_hour'
val_permute = +10
hour_permute = 3  # 0 is midnight, 23 is 11pm
assert hour_permute >= 0 & hour_permute <= 23
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #

# Check that column is in
assert cn_permute in dat_cidx.cn.to_list()

cidx_permute = np.where(dat_cidx.cn == cn_permute)[0][0]
cn_ord = ['dates','tt','X','y','mu','se']

holder = []
for lead in lead_seq:
    print('Lead: %i' % lead)
    y_lead = y_test[:hour_permute+1,lead].copy()
    X_lead = Xmat_test[:hour_permute+1].copy()
    dates_lead = hours_test[:hour_permute+1]
    mdl_lead1 = copy.deepcopy(di_mdl[lead])
    mdl_lead2 = copy.deepcopy(di_mdl[lead])
    # (i) Unchanged X
    # Unconditional forecast
    tmp_pred_u1 = mdl_lead1.predict(X_lead).assign(y=y_lead,tt='U',X='default',dates=dates_lead)
    # Conditional forecast
    tmp_pred_c1 = mdl_lead1.predict(X_lead, y_lead).assign(tt='C',X='default',dates=dates_lead)
    # (ii) Permuted X
    Xmat_permute = X_lead.copy()
    Xmat_permute[hour_permute,cidx_permute] += val_permute
    tmp_pred_u2 = mdl_lead2.predict(Xmat_permute).assign(y=y_lead,tt='U',X='permute',dates=dates_lead)
    tmp_pred_c2 = mdl_lead2.predict(Xmat_permute, y_lead).assign(tt='C',X='permute',dates=dates_lead)
    # Annotate and save for later
    tmp = pd.concat([tmp_pred_c1.tail(1), tmp_pred_u1.tail(1),
                     tmp_pred_c2.tail(1), tmp_pred_u2.tail(1)])
    tmp = tmp[cn_ord].reset_index(None,True)
    tmp.insert(0,'lead',lead)
    holder.append(tmp)
# Merge and compare
df_permute = pd.concat(holder).reset_index(None,True)
df_permute.rename(columns={'dates':'date_rt'},inplace=True)
df_permute = df_permute.assign(date_fcast=lambda x: x.date_rt + pd.offsets.Hour(1)*x.lead)
df_permute.drop(columns='date_rt', inplace=True)

#######################
# --- (4) PLOT IT --- #

di_tt = {'U':'Unconditional', 'C':'Conditional'}

gtit = 'Permuting "%s" at %i by %0.1f on %s' % (cn_permute,hour_permute,val_permute,dt_permute.strftime('%Y-%m-%d'))
gg_permute = (ggplot(df_permute,aes(x='date_fcast',y='mu',color='X')) + 
    theme_bw() + ggtitle(gtit) + 
    geom_line() + 
    labs(y='Actual/Predicted') + 
    theme(axis_title_x=element_blank(),axis_text_x=element_text(angle=90)) + 
    facet_wrap('~tt',labeller=labeller(tt=di_tt)) + 
    scale_color_discrete(name='X modified',labels=['Default','Permuted']) + 
    scale_x_datetime(date_breaks='2 hours',date_labels='%I%p') + 
    geom_point(aes(y='y'),color='black'))
gg_save('gg_permute.png',dir_figures,gg_permute,10,5)
