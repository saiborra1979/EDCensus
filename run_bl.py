# Use yesterday's hourly information
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--ylbl', type=str, default=None, help='Column from hourly_yX.csv to forecast')
args = parser.parse_args()
print(args)
ylbl = args.ylbl

# # For debugging
# ylbl = 'census_max'

import os
import numpy as np
import pandas as pd
from time import time
from mdls.funs_encode import yX_process
from funs_support import find_dir_olu, get_date_range
from funs_esc import esc_bins, esc_lbls, get_esc_levels
from funs_stats import prec_recall_lbls, get_reg_score

dir_base = os.getcwd()
dir_olu = find_dir_olu()
dir_output = os.path.join(dir_olu, 'output')
dir_flow = os.path.join(dir_output, 'flow')
dir_test = os.path.join(dir_flow, 'test')


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

# Add on the escalation levels
df_res = get_esc_levels(df_res,['y','y_rt','pred'],esc_bins, esc_lbls)
df_res = df_res.assign(y_delta=lambda x: np.sign(x.esc_y - x.esc_y_rt),
                pred_delta = lambda x: np.sign(x.esc_pred - x.esc_y_rt) )


#########################################
# --- STEP 4: GET MODEL PERFORMANCE --- #

cn_reg = ['lead']
cn_regn = cn_reg + ['n']
cn_gg = ['lead', 'metric']
cn_ggn = cn_gg + ['n']
cn_ord = ['y_delta','pred_delta','date_rt','lead']

# (1) Calculate spearman and MAE
perf_reg = df_res.groupby(cn_reg).apply(get_reg_score,add_n=True).reset_index()
perf_reg = perf_reg.melt(cn_regn,None,'metric')
perf_reg['n'] = perf_reg.n.astype(int)

# (2) Calculate the precision/recall
perf_ord = prec_recall_lbls(x=df_res[cn_ord],cn_y='y_delta',cn_pred='pred_delta',cn_idx='date_rt')
perf_ord = perf_ord.query('pred_delta == 1').reset_index(None, True)
perf_ord = perf_ord.drop(columns='pred_delta').rename(columns={'den':'n'})

# Merge regression + ordinal
perf_agg = pd.concat([perf_reg, perf_ord]).reset_index(None, True)

# (3) Do boostrap to get the standard errors
n_bs = 1000
holder_agg = []
stime = time()
for i in range(n_bs):
    if (i + 1) % 5 == 0:
        print(i+1)
        dtime, nleft = time() - stime, n_bs-(i+1)
        rate = (i+i)/dtime
        seta = nleft / rate
        print('bootstrap ETA: %i seconds (%i left)' % (seta, nleft))
    # Stratify bootstrap by lead
    bs_res = df_res.groupby('lead').sample(frac=1,replace=True,random_state=i).reset_index(None,True)
    # Regression
    bs_reg = bs_res.groupby(cn_reg).apply(get_reg_score,add_n=True).reset_index()
    bs_reg = bs_reg.melt(cn_regn,None,'metric').assign(n=lambda x: x.n.astype(int))
    # Classification
    bs_ord = prec_recall_lbls(x=bs_res[cn_ord],cn_y='y_delta',cn_pred='pred_delta',cn_idx='date_rt')
    bs_ord = bs_ord.query('pred_delta == 1').reset_index(None, True)
    bs_ord = bs_ord.drop(columns='pred_delta').rename(columns={'den':'n'})
    # Save
    bs_agg = pd.concat([bs_reg, bs_ord]).assign(bidx=i)
    holder_agg.append(bs_agg)

# Calculate bootstrap standard error
bs_agg = pd.concat(holder_agg)
bs_agg = bs_agg.groupby(cn_gg).value.std(ddof=1).reset_index()
bs_agg.rename(columns={'value':'se'}, inplace=True)
# Merge with existing
perf_agg = perf_agg.merge(bs_agg,'left')


########################
# --- STEP 5: SAVE --- #

# Save for later
df_res.to_csv(os.path.join(dir_test, 'bl_scores.csv'),index=False)
perf_agg.to_csv(os.path.join(dir_test, 'bl_agg.csv'),index=False)
