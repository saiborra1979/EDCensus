import os
import numpy as np
import pandas as pd
from time import time
from mdls.funs_encode import yX_process
from funs_support import find_dir_olu, get_date_range

dir_base = os.getcwd()
dir_olu = find_dir_olu()
dir_output = os.path.join(dir_olu, 'output')
dir_flow = os.path.join(dir_output, 'flow')
dir_test = os.path.join(dir_flow, 'test')

ylbl = 'census_max'

####################################
# --- STEP 1: LOAD/CREATE DATA --- #
print('# --- STEP 1: LOAD/CREATE DATA --- #')

# Get dataframe
df_X = pd.read_csv(os.path.join(dir_flow, 'hourly_yX.csv'))
df_X.date = pd.to_datetime(df_X.date)
# print(df_X.loc[10000])
# Extract y
yval = df_X[ylbl].values
dates = df_X.date.copy()

######################################
# --- STEP 2: CREATE DATE-SPLITS --- #
print('# --- STEP 2: CREATE DATE-SPLITS AND TRAIN --- #')

dtrain = 2
h_retrain = 1
lag = 24
lead = 24
assert isinstance(dtrain,int) & isinstance(h_retrain,int)

dfmt = '%Y-%m-%d'
dmax = pd.to_datetime((dates.max()-pd.DateOffset(days=1)).strftime(dfmt))
dmin = pd.to_datetime('2020-03-01')
nhours = int((dmax - dmin).total_seconds()/(60*60))
ndays = int(nhours / 24)
print('day start: %s, day stop: %s (hours=%i, days=%i)' % 
    (dmin.strftime(dfmt),dmax.strftime(dfmt),nhours, ndays))

offset_train = pd.DateOffset(days=dtrain)
print(offset_train)

########################################
# --- STEP 3: TRAIN BASELINE MODEL --- #

di_cn = {'ohe':['hour'], 'bin':None, 'cont':None}  #['date_trend']
cn_all = list(df_X.columns)

holder = []
for ii in range(nhours):
    time_ii = dmin + pd.DateOffset(hours=ii)  # Current day/hour
    s_train = time_ii - offset_train  # start time
    idx_train = ((dates >= s_train) & (dates <= time_ii)).values
    dates_train = dates[idx_train].reset_index(None,True)
    ytrain, Xtrain = yval[idx_train].copy(), df_X[idx_train].copy()
    X_now = Xtrain[-(lag+1):]  # Ensure enough rows to calculate lags
    if (ii+1) % h_retrain == 0:
        if ii % 24 == 0:
            print('Training range: %s' % (get_date_range(dates_train)))
            print('Current time: %s' % time_ii)
        # Fit encoder
        processor = yX_process(cn=cn_all, lead=lead, lag=lag, 
                cn_ohe=di_cn['ohe'], cn_cont=di_cn['cont'], cn_bin=di_cn['bin'])
        processor.fit(X=Xtrain)
        # Get the X, y data
        Xtil = processor.transform_X(Xtrain)
        ytil = processor.transform_y(ytrain)
        # Calculate inverse gram matrix
        igram = np.linalg.pinv((Xtil.T.dot(Xtil)))
        ixy = Xtil.T.dot(np.where(np.isnan(ytil),0,ytil))
        ibhat = igram.dot(ixy)
    # Get the current X
    X_curr = processor.transform_X(X_now,rdrop=0)
    y_pred = X_curr.dot(ibhat).flatten()
    df_pred = pd.DataFrame({'date_rt':time_ii,
        'lead':np.arange(processor.lead)+1,'pred':y_pred})
    holder.append(df_pred)
# Merge
df_res = pd.concat(holder).reset_index(None,True)
# Get the date_pred and the actual value
df_res = df_res.assign(date_pred=lambda x: x.date_rt+x.lead*pd.offsets.Hour(1))
df_res = df_res.merge(pd.DataFrame({'date_pred':dates,'y':yval}))
df_res = df_res.merge(pd.DataFrame({'date_rt':dates,'y_rt':yval}))
# Save for later
df_res.to_csv(os.path.join(dir_test, 'bl_hour.csv'),index=False)

# from scipy.stats import spearmanr
# from sklearn.metrics import mean_absolute_error as MAE
# df_res.groupby('lead').apply(lambda x: MAE(x.y, x.pred))
# df_res.groupby('lead').apply(lambda x: spearmanr(x.pred, x.y)[0])

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
