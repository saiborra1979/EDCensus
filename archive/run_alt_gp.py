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
from mdls.gpy import mdl
import torch

use_cuda = torch.cuda.is_available()
sdev = "cuda" if use_cuda else "cpu"
print('Using device: %s' % sdev)
device = torch.device(sdev)

dir_base = os.getcwd()
dir_olu = find_dir_olu()
dir_output = os.path.join(dir_olu, 'output')
dir_flow = os.path.join(dir_output, 'flow')
dir_test = os.path.join(dir_flow, 'test')

idx = pd.IndexSlice

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

dtrain = 5
h_retrain = 1
assert isinstance(dtrain,int) & isinstance(h_retrain,int)

dfmt = '%Y-%m-%d'
dmax = pd.to_datetime((dates.max()-pd.DateOffset(days=1)).strftime(dfmt))
dmin = pd.to_datetime('2020-03-01')
nhours = int((dmax - dmin).total_seconds()/(60*60))
ndays = int(nhours / 24)
print('day start: %s, day stop: %s (hours=%i, days=%i)' % 
    (dmin.strftime(dfmt),dmax.strftime(dfmt),nhours, ndays))

offset_train = pd.DateOffset(days=dtrain)

########################################
# --- STEP 3: TRAIN BASELINE MODEL --- #

cn_ohe = ['date_hour']
# cn_bin = ['census_max']
cn_cont = ['census_max','tt_arrived','tt_discharged','date_trend',
            'avgmd_arrived', 'avgmd_discharged']

holder = []
for ii in range(1):  #nhours
    time_ii = dmin + pd.DateOffset(hours=ii)  # Current day/hour
    hour = ii % 24  # Extract hour
    idx_now = (dates == time_ii).values
    if (ii+1) % h_retrain == 0:
        s_train = time_ii - offset_train  # start time
        idx_train = ((dates >= s_train) & (dates <= time_ii)).values
        dates_train = dates[idx_train].reset_index(None,True)
        print('Training range: %s' % (get_date_range(dates_train)))
        print('Current time: %s' % time_ii)
        ytrain, Xtrain = yval[idx_train].copy(), Xmat[idx_train].copy()
        X_now = Xmat[idx_now]
        enc_yX = yX_process(cn=cn, cn_cont=cn_cont, cn_ohe=cn_ohe, n_bins=10)
        enc_yX.fit(X=Xtrain, y=ytrain)
        enc_yX.transform_X(Xtrain,rdrop=1).shape
        enc_yX.transform_y(ytrain,leads=24,rdrop=1).shape
        self = mdl(encoder=enc_yX, device=device,leads=24)
        break
        gp_model.fit(X=Xtrain,y=ytrain,n_steps=250,lr=0.2,tol=1e-8)
        gp_model.predict(X_now)
        break
        pd.DataFrame({'y':ytrain,'hour':Xtrain[:,2]}).groupby('hour').y.std()
        
        gp_model.fit(Xtrain, ytrain, n_steps=100, lr=0.001,tol=1e-5)
    # Make predictions
    Mu, Sigma, idx = gp_model.predict(X_now)
    tmp_df = pd.DataFrame({'date_rt':time_ii,'pred':Mu.flat, 'se':Sigma.flat, 'lead':idx+1})
    tmp_df
    tmp_df = tmp_df.assign(date_pred=lambda x: x.date_rt + x.lead*pd.offsets.Hour(1))    
    holder.append(tmp_df)


####################################
# --- STEP 3: SAVE PREDICTIONS --- #

df_res = pd.concat(holder).reset_index(None, True)
df_res.merge(pd.DataFrame({'date_pred':dates,'y':yval}))

from plotnine import *
dir_figures = os.path.join(dir_olu, 'figures', 'census')
q1 = np.append(np.where(idx_train)[0],np.arange(15323,15323+24))
tmp = pd.DataFrame({'y':yval[q1],'date_pred':dates[q1]})

gg = (ggplot(tmp,aes(x='date_pred',y='y')) + theme_bw() + 
    geom_point() + 
    theme(axis_title_x=element_blank(),axis_text_x=element_text(angle=90)) + 
    scale_x_datetime(date_breaks='1 day',date_labels='%d, %b'))
gg.save(os.path.join(dir_figures, 'gg.png'),height=4,width=6)

