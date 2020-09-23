"""
SCRIPT USING MODEL OF CHOICE FROM MDLS FOLDER
current goals
1) Focus on lead+3, lead+4 (2-3 hours from now)
2) Check whether threshold agrees at lead+3 lead+4 (i.e. does violation last two hours)
3) Dynamic intercept for structural breaks....
"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--nlags', type=int, default=10, help='Max number of lags to use in model')
parser.add_argument('--lead', type=int, default=1, help='Which lead of the data to predict?')
parser.add_argument('--day', type=int, default=1, help='Which day of 2020 to predict? (max==181)')
parser.add_argument('--model', type=str, default='lasso', help='Model to use from mdls/')
args = parser.parse_args()
print(args)
nlags, lead, day, model = args.nlags, args.lead, args.day, args.model

# # Debugging in PyCharm
# nlags, lead, day, model = 1, 4, 76, 'torch_gpy'

import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

dir_base = os.getcwd()
dir_figures = os.path.join(dir_base, '..', 'figures')
dir_output = os.path.join(dir_base, '..', 'output')
dir_flow = os.path.join(dir_output, 'flow')
dir_test = os.path.join(dir_flow, 'test')
if not os.path.exists(dir_test):
    os.mkdir(dir_test)

idx = pd.IndexSlice

####################################
# --- STEP 1: LOAD/CREATE DATA --- #
print('# --- STEP 1: LOAD/CREATE DATA --- #')

# Get dataframe
df_lead_lags = pd.read_csv(os.path.join(dir_flow, 'df_lead_lags.csv'),header=[0,1], index_col=[0,1,2,3])
# Create dates
dates = df_lead_lags.index.to_frame().astype(str).assign(date=lambda x:pd.to_datetime(x.year+'-'+x.month+'-'+x.day+' '+x.hour+':00:00')).date

# Index names
lags = np.arange(nlags+1)
leads = (np.arange(nlags)+1)

# Get breakdown of feature types
cn_list = pd.Series(df_lead_lags.columns.get_level_values(0).unique()[1:])
# cn_list[cn_list.str.contains('arrived|discharged')].str.split('_',2,True).iloc[:,0].value_counts()
cn_list[cn_list.str.contains('arrived')].str.split('_',2,True).iloc[:,0].value_counts()
cn_list[~cn_list.str.contains('arrived|discharged')].str.split('_',2,True).iloc[:,0].value_counts()


# Convert into X/y matrices
yval = df_lead_lags.loc[:,idx[:,'lead_'+str(lead)]].values.flatten()
Xmat = df_lead_lags.loc[:,idx[:,['lag_'+str(lag) for lag in lags]]].values
cn_Y = df_lead_lags.head(1).loc[:,idx[:,['lead_'+str(lead) for lead in leads]]].columns
cn_X = df_lead_lags.head(1).loc[:,idx[:,['lag_'+str(lag) for lag in lags]]].columns
# Not using dates.dt.weekofyear for now
tmat = pd.concat([pd.get_dummies(dates.dt.dayofweek).add_prefix('dow_'),
                  pd.get_dummies(dates.dt.hour).add_prefix('hour_'),
                  (dates - dates.min()).dt.days],1).rename(columns={'date':'trend'})
tmat.columns = pd.MultiIndex.from_product([tmat.columns, ['lag_0']])
cn_X = cn_X.append(tmat.columns)
Xmat = np.hstack([Xmat, tmat.values])
assert cn_X.shape[0] == Xmat.shape[1]

######################################
# --- STEP 2: CREATE DATE-SPLITS --- #
print('# --- STEP 2: CREATE DATE-SPLITS --- #')

d_pred = pd.date_range('2020-01-01', dates.max(), freq='1d')
d_test = d_pred[day]
print('Predicting %i hours ahead for testing day: %s\nDay %i of %i' %
      (lead, d_test, day+1, len(d_pred)))
assert day+1 <= len(d_pred)
# Using previous week for validation data
d_valid = d_test - pd.DateOffset(weeks=1)

idx_train = (dates < d_valid).values
idx_valid = ((dates >= d_valid) & (dates < d_test)).values
idx_test = ((dates >= d_test) & (dates < d_test + pd.DateOffset(days=1))).values
assert sum(idx_valid) == 24*7
assert sum(idx_test) <= 24

# Split matrices
Xmat_train, y_train = Xmat[idx_train], yval[idx_train]
Xmat_valid, y_valid = Xmat[idx_valid], yval[idx_valid]
Xmat_test, y_test = Xmat[idx_test], yval[idx_test]
print('Training size: %i, validation size: %i, test size: %i' %
      (Xmat_train.shape[0], Xmat_valid.shape[0], Xmat_test.shape[0]))

###########################################
# --- STEP 3: TRAIN MODEL AND PREDICT --- #

mdls = __import__('mdls.' + model)
assert hasattr(mdls, model)
assert hasattr(getattr(mdls, model), 'mdl')
# Load the model
tmp = getattr(getattr(mdls, model), 'mdl')
mdl = tmp(model=model, lead=lead, date=d_test, cn=cn_X)
# Fit, tune, save
mdl.fit(X=Xmat_train, y=y_train)
mdl.tune(X=Xmat_valid, y=y_valid)
# mdl.save(folder=dir_test)
# mdl.load(folder=dir_test)
eta_test = mdl.predict(X=Xmat_test)
r2_test = r2_score(y_test, eta_test)
print('Test R-squared: %0.3f' % r2_test)

####################################
# --- STEP 3: SAVE PREDICTIONS --- #

# Get predictions
fn_res = mdl.fn.replace('.pkl','.csv').replace('mdl_','res_')
# Save predictions for later
df_res = pd.DataFrame({'y':y_test,'pred':eta_test,'dates':dates[idx_test],'lead':lead, 'model':model})
df_res.to_csv(os.path.join(dir_test, fn_res), index=True)
