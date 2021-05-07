import os
import sys
import numpy as np
import pandas as pd
from time import time
from datetime import datetime
from mdls.funs_encode import yX_process
from funs_support import makeifnot, find_dir_olu, get_date_range, gg_save
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

dir_base = os.getcwd()
dir_olu = find_dir_olu()
dir_output = os.path.join(dir_olu, 'output')
dir_flow = os.path.join(dir_output, 'flow')
dir_test = os.path.join(dir_flow, 'test')
# dir_save = os.path.join(dir_test, datetime.now().strftime('%Y_%m_%d'))
# makeifnot(dir_save)

idx = pd.IndexSlice
use_cuda = torch.cuda.is_available()
sdev = "cuda" if use_cuda else "cpu"
print('Using device: %s' % sdev)
device = torch.device(sdev)

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

dtrain = 1
h_retrain = 1
assert isinstance(dtrain,int) & isinstance(rtrain,int)


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
cn_num = None#['census_max']

holder = []
for ii in range(nhours):  #nhours
    time_ii = dmin + pd.DateOffset(hours=ii)  # Current day/hour
    hour = ii % 24  # Extract hour
    idx_now = (dates == time_ii).values
    if (ii+1) % h_retrain == 0:
        s_train = time_ii - offset_train  # start time
        idx_train = ((dates >= s_train) & (dates <= time_ii)).values
        dates_train = dates[idx_train].reset_index(None,True)
        if ii % 24 == 0:
            print('Training range: %s' % (get_date_range(dates_train)))
            print('Current time: %s' % time_ii)
        ytrain, Xtrain = yval[idx_train].copy(), Xmat[idx_train].copy()
        # Fit encoder
        processor = yX_process(cn=cn, cn_ohe=cn_ohe, cn_num=cn_num)
        processor.fit(X=Xtrain)
        # Get the X, y data
        Xtil, ytil = processor.transform(Xtrain, ytrain)
        # Calculate inverse gram matrix
        igram = np.linalg.pinv((Xtil.T.dot(Xtil)).toarray())
        ixy = Xtil.T.dot(np.where(np.isnan(ytil),0,ytil))
        ibhat = igram.dot(ixy)
    # Get the current X
    X_curr = processor.transform(Xtrain[[-1]],rdrop=0).toarray()
    y_pred = X_curr.dot(ibhat).flatten()
    df_pred = pd.DataFrame({'date_rt':time_ii,'lead':processor.leads,'pred':y_pred})
    holder.append(df_pred)
# Merge
df_res = pd.concat(holder).reset_index(None,True)
# Get the date_pred and the actual value
df_res = df_res.assign(date_pred=lambda x: x.date_rt+x.lead*pd.offsets.Hour(1))
df_res = df_res.merge(pd.DataFrame({'date_pred':dates,'y':yval}))
df_res = df_res.merge(pd.DataFrame({'date_rt':dates,'y_rt':yval}))
# Save for later
df_res.to_csv(os.path.join(dir_test, 'bl_hour.csv'),index=False)


# # Make a plot
# from plotnine import *
# dir_figures = os.path.join(dir_olu, 'figures', 'census')
# # df_res.query('lead==1').sort_values(['date_rt','date_pred']).reset_index(None,True).drop(columns=['lead','pred']).head(10)
# tmp = df_res.assign(gg=lambda x: x.date_rt.dt.strftime('%d.%H').astype(float))
# gg_tmp = (ggplot(tmp) + theme_bw() + 
#     geom_point(aes(x='date_rt',y='y_rt'),alpha=0.5,size=0.75) + 
#     geom_line(aes(x='date_pred',y='pred',color='gg',groups='gg')) +
#     theme(axis_title_x=element_blank(),axis_text_x=element_text(angle=90)) + 
#     labs(y='Actual/Predicted') + 
#     guides(color=False) + 
#     scale_x_datetime(date_breaks='1 day',date_labels='%d, %m'))
# gg_save('gg_tmp.png',dir_figures,gg_tmp,7,4)
