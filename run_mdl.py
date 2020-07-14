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

# Debugging in PyCharm
nlags, lead, day, model = 10, 3, 75, 'lasso'

import os
import numpy as np
import pandas as pd

dir_base = os.getcwd()
dir_figures = os.path.join(dir_base, '..', 'figures')
dir_output = os.path.join(dir_base, '..', 'output')
dir_flow = os.path.join(dir_output, 'flow')
dir_test = os.path.join(dir_flow, 'test')
if not os.path.exists(dir_test):
    os.mkdir(dir_test)

idx = pd.IndexSlice

#############################
# --- STEP 1: LOAD/CREATE DATA --- #
print('# --- STEP 1: LOAD/CREATE DATA --- #')

# Get dataframe
df_lead_lags = pd.read_csv(os.path.join(dir_flow, 'df_lead_lags.csv'),header=[0,1], index_col=[0,1,2,3])
# Create dates
dates = df_lead_lags.index.to_frame().astype(str).assign(date=lambda x:pd.to_datetime(x.year+'-'+x.month+'-'+x.day+' '+x.hour+':00:00')).date

# Index names
lags = np.arange(nlags+1)
leads = (np.arange(nlags)+1)

# Convert into X/y matrices
Ymat = df_lead_lags.loc[:,idx[:,['lead_'+str(lead) for lead in leads]]].values
Xmat = df_lead_lags.loc[:,idx[:,['lag_'+str(lag) for lag in lags]]].values
cn_Y = df_lead_lags.head(1).loc[:,idx[:,['lead_'+str(lead) for lead in leads]]].columns
cn_X = df_lead_lags.head(1).loc[:,idx[:,['lag_'+str(lag) for lag in lags]]].columns

d_pred = pd.date_range('2020-01-01',dates.max(),freq='1d')
d_test = d_pred[day]
print('Predicting %i hours ahead for testing day: %s\nDay %i of %i' %
      (lead, d_test, day+1, len(d_pred)))
assert day+1 <= len(d_pred)
# Using previous week for validation data
d_valid = d_test - pd.DateOffset(weeks=1)

idx_train = (dates < d_valid).values
idx_valid = ((dates >= d_valid) & (dates < d_test)).values
idx_test = ((dates >= d_test) & (dates < d_test + pd.DateOffset(days=1))).values
assert sum(idx_test) <= 24

# Split matrices
Xmat_train, y_train = Xmat[idx_train], Ymat[idx_train,lead]
Xmat_valid, y_valid = Xmat[idx_valid], Ymat[idx_valid,lead]
Xmat_test, y_test = Xmat[idx_test], Ymat[idx_test, lead]
print('Training size: %i, validation size: %i, test size: %i' %
      (Xmat_train.shape[0], Xmat_valid.shape[0], Xmat_test.shape[0]))

###########################################
# --- STEP 2: TRAIN MODEL AND PREDICT --- #

from sklearn.metrics import r2_score
q1 = pd.DataFrame({'y':y_train}).tail(y_valid.shape[0]).assign(tt='train')
q2 = pd.DataFrame({'y':y_valid}).assign(tt='valid')
q3 = pd.DataFrame({'y':y_test}).assign(tt='test')
dat = pd.concat([q1, q2, q3]).reset_index(None, True).reset_index()
gg = (ggplot(dat, aes(x='index',y='y',color='tt')) +
      geom_point() + geom_line() + theme_bw())
gg.save(os.path.join(dir_figures,'tmp.png'))


mdls = __import__('mdls.' + model)
assert hasattr(mdls, model)
assert hasattr(getattr(mdls, model), 'mdl')
# Load the model
tmp = getattr(getattr(mdls, model), 'mdl')
mdl = tmp(model=model, lead=lead, date=d_test, cn=cn_X)
# Fit, tune, save
mdl.fit(X=Xmat_train, y=y_train)
mdl.tune(X=Xmat_valid, y=y_valid)
mdl.save(folder=dir_test)
# mdl.load(folder=dir_test)

####################################
# --- STEP 3: SAVE PREDICTIONS --- #

# Get predictions
eta_test = mdl.predict(Xmat_test)
fn_res = mdl.fn.replace('.pkl','.csv').replace('mdl_','res_')
# Save predictions for later
df_res = pd.DataFrame({'y':y_test,'pred':eta_test,'dates':dates[idx_test],'lead':lead, 'model':model})
df_res.to_csv(os.path.join(dir_test, fn_res), index=False)
