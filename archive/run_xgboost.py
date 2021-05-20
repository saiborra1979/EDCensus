# # Calls class from ~/mdls folder
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--model', type=str, default='lasso', help='Model to use from mdls/')
# parser.add_argument('--dtrain', type=int, default=5, help='# of training days')
# parser.add_argument('--h_retrain', type=int, default=24, help='Frequency of retraining')
# parser.add_argument('--n_trees', type=int, default=100, help='Number of trees')
# parser.add_argument('--depth', type=int, default=3, help='Max depth of trees')
# args = parser.parse_args()
# print(args)
# model, dtrain, h_retrain = args.model, args.dtrain, args.h_retrain

model, dtrain, h_retrain, n_trees, depth = 'xgboost', 10, 24, 100, 3

import os
import sys
import numpy as np
import pandas as pd
from time import time
from datetime import datetime
from funs_support import makeifnot, find_dir_olu, get_date_range
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from mdls.funs_encode import yX_process

# Import the "mdl" class from the mdl folder
mdls = __import__('mdls.' + model)
assert hasattr(mdls, model)
assert hasattr(getattr(mdls, model), 'mdl')
xgb_mdl = getattr(getattr(mdls, model), 'mdl')

dir_base = os.getcwd()
dir_olu = find_dir_olu()
dir_output = os.path.join(dir_olu, 'output')
dir_flow = os.path.join(dir_output, 'flow')
dir_test = os.path.join(dir_flow, 'test')

idx = pd.IndexSlice

lead, lag = 24, 24

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


######################################
# --- STEP 2: CREATE DATE-SPLITS --- #
print('# --- STEP 2: CREATE DATE-SPLITS AND TRAIN --- #')

assert isinstance(dtrain,int) & isinstance(h_retrain,int)

dfmt = '%Y-%m-%d'
dmax = pd.to_datetime((dates.max()-pd.DateOffset(days=1)).strftime(dfmt))
dmin = pd.to_datetime('2020-03-01')
nhours = int((dmax - dmin).total_seconds()/(60*60))
ndays = int(nhours / 24)
print('day start: %s, day stop: %s (hours=%i, days=%i)' % 
    (dmin.strftime(dfmt),dmax.strftime(dfmt),nhours, ndays))

########################################
# --- STEP 3: TRAIN BASELINE MODEL --- #

cn_ohe = ['date_hour']
cn_cont = ['census_max','tt_arrived','tt_discharged']

offset_train = pd.DateOffset(days=dtrain)

holder = []
stime = time()
for ii in range(nhours):
    time_ii = dmin + pd.DateOffset(hours=ii)  # Current day/hour
    s_train = time_ii - offset_train  # start time
    idx_train = ((dates >= s_train) & (dates <= time_ii)).values
    dates_train = dates[idx_train].reset_index(None,True)
    ytrain, Xtrain = yval[idx_train].copy(), Xmat[idx_train].copy()
    X_now = Xtrain[-(lag+1):]  # Ensure enough rows to calculate lags
    if ii % h_retrain == 0:
        print('Training range: %s' % (get_date_range(dates_train)))
        print('Current time: %s' % time_ii)
        enc_yX = yX_process(cn=cn, cn_ohe=cn_ohe, cn_cont=cn_cont, lead=lead, lag=lag) #
        enc_yX.fit(X=Xtrain)
        regressor = xgb_mdl(encoder=enc_yX, lead=lead, lag=lag)
        regressor.fit(X=Xtrain, y=ytrain, n_trees=n_trees, depth=depth)
        nleft, nsec = nhours-(ii+1), time() - stime
        rate = (ii + 1) / nsec
        eta = nleft/rate
        print('ETA: %.1f minutes' % (eta/60))
    tmp_res = pd.DataFrame(regressor.predict(X_now)).melt(None,None,'lead','pred')
    tmp_res = tmp_res.assign(lead=lambda x: x.lead+1,date_rt=time_ii)
    holder.append(tmp_res)

df_res = pd.concat(holder).reset_index(None,True)
df_res = df_res.assign(date_pred=lambda x: x.date_rt + x.lead*pd.offsets.Hour(1))    
df_res = df_res.merge(pd.DataFrame({'date_rt':dates,'y_rt':yval}))
df_res = df_res.merge(pd.DataFrame({'date_pred':dates,'y':yval}))
df_res = df_res.sort_values(['date_rt','lead']).reset_index(None,True)

####################################
# --- STEP 3: SAVE PREDICTIONS --- #

fn_res = pd.DataFrame({'v1':['model', 'dtrain', 'h_retrain', 'n_trees', 'depth'], 
                       'v2':[model, dtrain, h_retrain, n_trees, depth]})
fn_res = fn_res.assign(v3=lambda x: x.v1 + '-' + x.v2.astype(str))
fn_res = fn_res.v3.str.cat(sep='_')+'.csv'
# Save predictions for later
df_res.to_csv(os.path.join(dir_test, fn_res), index=False)

# from plotnine import *
# dir_figures = os.path.join(dir_olu, 'figures', 'census')

# tmp = df_res.copy().assign(gg=lambda x: x.date_rt.dt.hour.astype(str))
# # tmp.groupby(['date_rt','gg']).size()
# gg_tmp = (ggplot(tmp.query('lead==24')) + theme_bw() + 
#     geom_point(aes(x='date_rt',y='y_rt'),color='black',alpha=0.5) + 
#     geom_point(aes(x='date_pred',y='y'),color='black',alpha=0.5) +
#     geom_line(aes(x='date_pred',y='pred'),color='red',alpha=0.5) +   #,group='gg',color='gg'
#     guides(color=False) + 
#     theme(axis_title_x=element_blank(),axis_text_x=element_text(angle=90)) + 
#     scale_x_datetime(date_breaks='1 day',date_labels='%d, %b'))
# gg_tmp.save(os.path.join(dir_figures, 'gg_tmp.png'),height=4.5,width=14)

# # Compare performance to baseline
# cn_dates = ['date_rt','date_pred']
# dat_bl = pd.read_csv(os.path.join(dir_test,'bl_hour.csv'),nrows=2000)
# dat_bl[cn_dates] = dat_bl[cn_dates].apply(pd.to_datetime,0)
# dat_bl = dat_bl.query('date_rt.isin(@df_res.date_rt.unique())')
# dat_comp = pd.concat([df_res.assign(tt='xgboost'), dat_bl.assign(tt='bl')])
# q1 = dat_comp.groupby(['lead','tt']).apply(lambda x: spearmanr(x.y, x.pred)[0]).reset_index()
# q1 = q1.pivot('lead','tt',0).assign(gain=lambda x: x.xgboost - x.bl)
# # q1

# q2 = dat_comp.groupby(['date_rt','tt']).apply(lambda x: spearmanr(x.y, x.pred)[0]).reset_index()
# q2 = q2.pivot('date_rt','tt',0).assign(gain=lambda x: x.xgboost - x.bl)
# # q2

# print('Lead gain: %0.2f, daily gain: %0.2f' % (q1.gain.mean(), q2.gain.mean()))


# # Get predictions
# fn_res = mdl.fn.replace('.pkl','.csv').replace('mdl_','res_')
# # Save predictions for later
# df_res = pd.DataFrame({'y':y_test,'pred':eta_test,'dates':dates[idx_test],'lead':lead, 'model':model})
# df_res.to_csv(os.path.join(dir_test, fn_res), index=True)
