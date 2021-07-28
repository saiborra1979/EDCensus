# run_mdl script that support multiple y labels

"""
ylbl:           Which column(s) should be used for forecast?
xlbl:           Which column(s) should be used for the features? (ylbl is automatically included as a lagged feature)
lead:           Hours of the forecasting horizon to use (default=24)
lag:            Number of default lags to use for X/y (default=24)
month:          Model is trained for this specific month (1==March 2020)
dtrain:         # of training days
h_rtrain:       # of hours before retraining (i.e. 24==1 nightly)
model_name:     Model to use from ~/mdls
model_args:     Optional arguments for model class (e.g. n_trees=100,depth=3,...)
write_scores:   Should model scores be written (default==False)
write_model:    Should model class be pickled? (default==False)
"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--ylbl', nargs='+', help='Column(s) from hourly_yX.csv to forecast')
parser.add_argument('--xlbl', nargs='+', help='Column(s) from hourly_yX.csv to use as features')
parser.add_argument('--lead', type=int, default=24, help='Number of leads to forecast')
parser.add_argument('--lag', type=int, default=24, help='Number of lags to use in X')
parser.add_argument('--month', type=int, default=None, help='Which month to use since March-2020 onwards')
parser.add_argument('--dtrain', type=int, default=None, help='# of training days')
parser.add_argument('--h_rtrain', type=int, default=None, help='Frequency of retraining')
parser.add_argument('--model_name', type=str, default=None, help='Model to use from ~/mdls')
parser.add_argument('--model_args', type=str, default=None, 
    help='Optional arguments for model class (e.g. n_trees=100,depth=3,...)')
parser.add_argument('--write_scores', default=False, action='store_true')
parser.add_argument('--write_model', default=False, action='store_true')

args = parser.parse_args()
print('args = %s' % args)
ylbl, xlbl = args.ylbl, args.xlbl
lead, lag, month, dtrain, h_rtrain = args.lead, args.lag, args.month, args.dtrain, args.h_rtrain
model_name, model_args = args.model_name, args.model_args
write_scores = args.write_scores
write_model = args.write_model

assert isinstance(ylbl, list) and isinstance(xlbl, list)

# import sys
# sys.exit('end here')

# # For debugging
# ylbl=['tt_arrived', 'tt_discharged']
# xlbl=['is_holiday']
# dtrain=30; h_rtrain=int(24*15); lag=24; lead=24; month=9
# model_args='eta=0.3,n_trees=100,depth=3,n_jobs=3'
# model_name='xgboost';write_scores=False; write_model=False

# Load modules
import os
import numpy as np
import pandas as pd
from time import time
from funs_support import find_dir_olu, get_date_range, makeifnot
from mdls.funs_encode import yX_process
from sklearn.metrics import mean_absolute_error as MAE

# (i) Model class should be in mdls folder
assert model_name in list(pd.Series(os.listdir('mdls')).str.replace('.py','',regex=True))

# (ii) Put optional arguments in dict (if any)
if model_args is not None:
    di_model = {}
    for opt in model_args.split(','):
        opt1, opt2 = opt.split('=')
        di_model[opt1] = opt2
else:
    di_model = None
print('di_model: %s' % di_model)

# (iii) Import the "mdl" class from the mdl folder
model_class = __import__('mdls.' + model_name)
assert hasattr(model_class, model_name)
assert hasattr(getattr(model_class, model_name), 'model')
model = getattr(getattr(model_class, model_name), 'model')

# Models must have four mandatory attributes
attr_mand = ['fit','predict','update_Xy','pickle_me']
assert all([hasattr(model, attr) for attr in attr_mand])

# (iv) Set up folders
dir_base = os.getcwd()
dir_olu = find_dir_olu()
dir_output = os.path.join(dir_olu, 'output')
dir_flow = os.path.join(dir_output, 'flow')
dir_test = os.path.join(dir_flow, 'test')
dir_class = os.path.join(dir_test, model_name)
dir_model = os.path.join(dir_class, 'model')
makeifnot(dir_class)
makeifnot(dir_model)


#############################
# --- STEP 1: LOAD DATA --- #
print('# --- STEP 1: LOAD/CREATE DATA --- #')

# Get dataframe
df_X = pd.read_csv(os.path.join(dir_flow, 'hourly_yX.csv'))
df_X.date = pd.to_datetime(df_X.date)

# Check column names
assert len(np.setdiff1d(ylbl, df_X.columns)) == 0
assert len(np.setdiff1d(xlbl, df_X.columns)) == 0

# Extract y
Yval = df_X[ylbl].values
dates = df_X.date.copy()
di_ylbl = dict(zip(range(len(ylbl)), ylbl))
df_Yval = pd.DataFrame(Yval,columns=ylbl).assign(date_pred=dates)
df_Yval = df_Yval.melt('date_pred',None,'ylbl','y')

######################################
# --- STEP 2: CREATE DATE-SPLITS --- #
print('# --- STEP 2: CREATE DATE-SPLITS AND TRAIN --- #')

assert isinstance(dtrain,int) & isinstance(h_rtrain,int)

dfmt = '%Y-%m-%d'
dmin = pd.to_datetime('2020-03-01')
dmax = dmin + pd.DateOffset(months=month) - pd.DateOffset(seconds=1)
dmin = pd.to_datetime(dmax.strftime('%Y-%m')+'-01')
nhours = int(np.ceil((dmax - dmin).total_seconds()/(60*60)))
ndays = int(np.ceil(nhours / 24))
print('day start: %s, day stop: %s (hours=%i, days=%i)' % 
    (dmin.strftime(dfmt),dmax.strftime(dfmt),nhours, ndays))

offset_train = pd.DateOffset(days=dtrain)
print('Training offset: %s' % offset_train)


###############################
# --- STEP 3: TRAIN MODEL --- #

# All model classes a cn dictionary with ohe, cont, and bin
cn_cont = ['census_max','census_var','tt_arrived','tt_discharged']
cn_ohe = ['month','day','hour','dow','is_holiday']
cn_bin = []
di_cn = {'ohe':cn_ohe, 'bin':cn_bin, 'cont':cn_cont}
cn_all = list(df_X.columns)
cn_use = sum(list(di_cn.values()), [])
# Check that all columns can be bound
assert all([cn in cn_all for cn in cn_use])

holder = []
stime = time()
for ii in range(nhours):
    time_ii = dmin + pd.DateOffset(hours=ii)  # Current day/hour
    s_train = time_ii - offset_train  # start time
    idx_train = ((dates >= s_train) & (dates <= time_ii)).values
    dates_train = dates[idx_train].reset_index(None,True)
    Ytrain = Yval[idx_train].copy()
    Xtrain = df_X.loc[idx_train, cn_use].copy()
    X_now = Xtrain[-(lag+1):]  # Ensure enough rows to calculate lags
    if ii % h_rtrain == 0:
        print('Training range: %s' % (get_date_range(dates_train)))
        print('Current time: %s' % time_ii)
        enc_yX = yX_process(cn=cn_all, lead=lead, lag=lag, 
                cn_ohe=di_cn['ohe'], cn_cont=di_cn['cont'], cn_bin=di_cn['bin'])
        enc_yX.fit(X=Xtrain)
        # break
        regressor = model(encoder=enc_yX, di_model=di_model)
        regressor.fit(X=Xtrain, Y=Ytrain)
        nleft, nsec = nhours-(ii+1), time() - stime
        rate = (ii + 1) / nsec
        eta = nleft/rate
        print('ETA: %.1f minutes' % (eta/60))
    else:
        # Update X/y where relevant
        regressor.update_Xy(Xnew=Xtrain, Ynew=Ytrain)
    # Do inference
    pred_ii = regressor.predict(X_now)
    pred_ii = {di_ylbl[k]: v.flatten() for k,v in pred_ii.items()}
    pred_ii = pd.DataFrame.from_dict(pred_ii)
    pred_ii = pred_ii.assign(date_rt=time_ii, lead=np.arange(lead)+1)
    pred_ii = pred_ii.melt(['date_rt','lead'],None,'ylbl','pred')
    holder.append(pred_ii)


###########################
# --- STEP 4: COMBINE --- #


res = pd.concat(holder).reset_index(None,True)
res = res.assign(date_pred = lambda x: x.date_rt + pd.TimedeltaIndex(x.lead, unit='H'))
# Merge with actual label
res = res.merge(df_Yval,'left')
# Example of performance by horizon/ylbl
print(res.groupby(['ylbl','lead']).apply(lambda x: MAE(x.y, x.pred)))


print('~~~ End of run_md1l.py ~~~')