# Calls class from ~/mdls folder
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--lead', type=int, default=None, help='Number of leads to forecast')
parser.add_argument('--lag', type=int, default=None, help='Number of lags to forecast')
parser.add_argument('--dtrain', type=int, default=5, help='# of training days')
parser.add_argument('--h_retrain', type=int, default=24, help='Frequency of retraining')
parser.add_argument('--ylbl', type=str, default=None, help='Column from hourly_yX.csv to forecast')
parser.add_argument('--model_name', type=str, default=None, help='Model to use from ~/mdls')
parser.add_argument('--model_args', type=str, default=None, 
    help='Optional arguments for model class (e.g. n_trees=100,depth=3,...)')
args = parser.parse_args()
print(args)
lead, lag, dtrain, h_retrain = args.lead, args.lag, args.dtrain, args.h_retrain
ylbl, model_name, model_args = args.ylbl, args.model_name, args.model_args

# # For debugging
# dtrain=30; h_retrain=48; lag=24; lead=24; 
# model_args='n_trees=200,depth=5,n_jobs=6'; model_name='xgboost'; ylbl='census_max'

import os
import numpy as np
import pandas as pd

# (i) Model class should be in mdls folder
assert model_name in list(pd.Series(os.listdir('mdls')).str.replace('.py',''))

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

# (iv) Load remaining packages/folders
from time import time
from datetime import datetime
from funs_support import find_dir_olu, get_date_range, makeifnot
from mdls.funs_encode import yX_process

dir_base = os.getcwd()
dir_olu = find_dir_olu()
dir_output = os.path.join(dir_olu, 'output')
dir_flow = os.path.join(dir_output, 'flow')
dir_test = os.path.join(dir_flow, 'test')
dir_model = os.path.join(dir_test, model_name)
makeifnot(dir_model)

#############################
# --- STEP 1: LOAD DATA --- #
print('# --- STEP 1: LOAD/CREATE DATA --- #')

# Get dataframe
df_X = pd.read_csv(os.path.join(dir_flow, 'hourly_yX.csv'))
df_X.date = pd.to_datetime(df_X.date)
# Extract y
yval = df_X[ylbl].values
dates = df_X.date.copy()

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

offset_train = pd.DateOffset(days=dtrain)
print('Training offset: %s' % offset_train)

########################################
# --- STEP 3: TRAIN BASELINE MODEL --- #

# All model classes a cn dictionary with ohe, cont, and bin
#   this is passed into the funs_encode
cn_cont = ['census_max','census_var','u_mds10h','tt_arrived','tt_discharged']
di_cn = {'ohe':['hour','dow'], 'bin':None, 'cont':cn_cont}
cn_all = list(df_X.columns)

holder = []
stime = time()
for ii in range(nhours):
    time_ii = dmin + pd.DateOffset(hours=ii)  # Current day/hour
    s_train = time_ii - offset_train  # start time
    idx_train = ((dates >= s_train) & (dates <= time_ii)).values
    dates_train = dates[idx_train].reset_index(None,True)
    ytrain, Xtrain = yval[idx_train].copy(), df_X[idx_train].copy()
    X_now = Xtrain[-(lag+1):]  # Ensure enough rows to calculate lags
    if ii % h_retrain == 0:
        print('Training range: %s' % (get_date_range(dates_train)))
        print('Current time: %s' % time_ii)
        enc_yX = yX_process(cn=cn_all, lead=lead, lag=lag, 
                cn_ohe=di_cn['ohe'], cn_cont=di_cn['cont'], cn_bin=di_cn['bin'])
        enc_yX.fit(X=Xtrain)
        regressor = model(encoder=enc_yX, lead=lead, lag=lag, di_model=di_model)
        regressor.fit(X=Xtrain, y=ytrain)
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


# from plotnine import *
# dir_figures = os.path.join(dir_olu, 'figures', 'census')
# from funs_support import gg_save

# tmp = df_res.copy().assign(gg=lambda x: x.date_rt.dt.dayofyear+x.date_rt.dt.hour/100)
# # tmp.groupby(['date_rt','gg']).size()
# gg_tmp = (ggplot(tmp,aes(color='gg.astype(str)')) + theme_bw() + 
#     theme(axis_title_x=element_blank(),axis_text_x=element_text(angle=90)) + 
#     scale_x_datetime(date_breaks='1 day',date_labels='%d, %b') + 
#     geom_line(aes(x='date_pred',y='pred'),alpha=0.5) + 
#     geom_point(aes(x='date_rt',y='y_rt'),color='black',alpha=0.5) + 
#     guides(color=False))
# gg_save('gg_tmp.png', dir_figures, gg_tmp, 14, 4.5)

####################################
# --- STEP 3: SAVE PREDICTIONS --- #

# Add on model args
if di_model is not None:
    fn_di = [k+'-'+v for k, v in di_model.items()]
    fn_di = '_'.join(fn_di)
else:
    fn_di = 'None'
fn_res = pd.DataFrame({'v1':['lead', 'lag', 'dtrain', 'h_retrain', 'ylbl', 'model_name','model_args'], 
                       'v2':[lead, lag, dtrain, h_retrain, ylbl, model_name,fn_di]})
fn_res = fn_res.assign(v3=lambda x: x.v1 + '=' + x.v2.astype(str))
fn_res = fn_res.v3.str.cat(sep='+')+'.csv'
# Save predictions for later
df_res.to_csv(os.path.join(dir_model, fn_res), index=False)
