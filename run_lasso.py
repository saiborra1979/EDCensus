"""
SCRIPT USING FALSE-POSITIVE CONTROL LASSO
"""

import sys

#if sys.stdout.isatty():
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--nlags', type=int, default=10, help='Max number of lags to use in model')
parser.add_argument('--lead', type=int, default=1, help='Which lead of the data to predict?')
parser.add_argument('--day', type=int, default=1, help='Which day of 2020 to predict? (max==181)')
args = parser.parse_args()
nlags, lead, day = args.nlags, args.lead, args.day
print(args)
#else:  # Debugging in PyCharm
#    nlags, lead, day = 10, 5, 181

import os
from time import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from glmnet import ElasticNet

dir_base = os.getcwd()
dir_figures = os.path.join(dir_base, '..', 'figures')
dir_output = os.path.join(dir_base, '..', 'output')
dir_flow = os.path.join(dir_output, 'flow')
dir_test = os.path.join(dir_flow, 'test')
if not os.path.exists(dir_test):
    os.mkdir(dir_test)

idx = pd.IndexSlice

#############################
# --- STEP 1: LOAD DATA --- #
print('# --- STEP 1: LOAD DATA --- #')

# Get dataframe
df_lead_lags = pd.read_csv(os.path.join(dir_flow, 'df_lead_lags.csv'),header=[0,1], index_col=[0,1,2,3])
# Create dates
dates = df_lead_lags.index.to_frame().astype(str).assign(date=lambda x:pd.to_datetime(x.year+'-'+x.month+'-'+x.day+' '+x.hour+':00:00')).date

# Index names
lags = np.arange(nlags)+1
leads = np.arange(nlags+1)[1:]

# Convert into X/y matrices
Ymat = df_lead_lags.loc[:,idx[:,['lead_'+str(lead) for lead in leads]]].values
Xmat = df_lead_lags.loc[:,idx[:,['lag_'+str(lag) for lag in lags]]].values
cn_Y = df_lead_lags.head(1).loc[:,idx[:,['lead_'+str(lead) for lead in leads]]].columns
cn_X = df_lead_lags.head(1).loc[:,idx[:,['lag_'+str(lag) for lag in lags]]].columns

#########################################
# --- STEP 2: PREDICT ONE-DAY-AHEAD --- #

nlam = 50

d_pred = pd.date_range('2020-01-01',dates.max(),freq='1d')
d_test = d_pred[day]
print('Predicting %i hours ahead for testing day: %s' % (lead, d_test))

idx_train = (dates <= d_test).values
idx_test = ((dates >= d_test) & (dates < d_test + pd.DateOffset(days=1))).values
assert sum(idx_test) <= 24
# Split matrices
Xmat_train, y_train = Xmat[idx_train], Ymat[idx_train,lead]
Xmat_test, y_test = Xmat[idx_test], Ymat[idx_test, lead]
# Normalize data
enc = StandardScaler().fit(Xmat_train)
Xmat_train, Xmat_test = enc.transform(Xmat_train), enc.transform(Xmat_test)
ntrain, p = Xmat_train.shape
# Train model
elnet = ElasticNet(alpha=1, standardize=False, fit_intercept=True, n_splits=0, n_lambda=nlam, min_lambda_ratio=0.001)
print('Fitting model!')
tstart = time()
elnet.fit(X=Xmat_train, y=y_train)
print('Model took %i seconds to train' % (time() - tstart))

lams = elnet.lambda_path_
res_mat = np.zeros([ntrain,nlam])
# Residuals over the lambdas
e2_holder = np.zeros(nlam)
for jj, ll in enumerate(lams):
    eta_train = Xmat_train.dot(elnet.coef_path_[:,jj]) + elnet.intercept_path_[jj]
    e2_holder[jj] = np.sqrt(np.sum(np.square(y_train - eta_train)))
# Expected number of false discoveries
lams2 = ntrain*lams/e2_holder
nsupp = np.sum(elnet.coef_path_ != 0,0)
efd = norm.cdf(-lams2)*p
fdr = efd/nsupp
df = pd.DataFrame({'fdr':fdr,'nsel':nsupp,'lam':lams,'idx':range(nlam)})
df = df[df.fdr.notnull()]
jj_star = int(df.iloc[np.argmin((df.fdr-0.1)**2)].idx)
bhat_star = elnet.coef_path_[:,jj_star]
ahat_star = elnet.intercept_path_[jj_star]
eta_test = Xmat_test.dot(bhat_star) + ahat_star
df_res = pd.DataFrame({'y':y_test,'pred':eta_test,'dates':dates[idx_test],'lead':lead})
# Save predictions for later
fn_res = 'lasso_res_'+d_test.strftime('%Y_%m_%d')+'_lead'+str(lead)+'.csv'
df_res.to_csv(os.path.join(dir_test, fn_res))

# Calculate the coefficients
df_bhat = pd.Series(bhat_star,index=cn_X).reset_index()
df_bhat = df_bhat.rename(columns={'level_0':'cn', 'level_1':'lag', 0:'bhat_z'})
df_bhat.lag = df_bhat.lag.str.replace('lag_','').astype(int)
df_bhat = df_bhat.assign(bhat = lambda x: x.bhat_z/enc.scale_,lead=lead,day=d_test)
fn_bhat = 'lasso_bhat_'+d_test.strftime('%Y_%m_%d')+'_lead'+str(lead)+'.csv'
df_bhat.to_csv(os.path.join(dir_test, fn_bhat))
