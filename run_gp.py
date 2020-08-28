import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--lead', type=int, default=1, help='Which lead of the data to predict?')
parser.add_argument('--model', type=str, default='gpy', help='Which GP to use?')
parser.add_argument('--dstart', type=int, default=0, help='Day to start in 2020 (0==Jan 1)')
parser.add_argument('--dend', type=int, default=366, help='Day to end in 2020 (366==Dec 31)')
args = parser.parse_args()
print(args)
lead, model, dstart, dend = args.lead, args.model, args.dstart, args.dend

# # Debugging in PyCharm
# lead, model, dstart, dend = 5, 'gpy', 180, 212

import os
import sys
import numpy as np
import pandas as pd
from time import time
from mdls.gpy import mdl
# from sklearn.metrics import r2_score

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
# Extract y
yval = df_lead_lags.loc[:,idx[:,'lead_'+str(lead)]].values.flatten()
# Extract X
cn = ['census_max','census_var','tt_arrived','tt_discharged']
# Remove lags (GP handles them automatically in the kernel)
Xmat = df_lead_lags.loc[:,idx[cn,'lag_0']].droplevel(1,1).values
# Extract date features (remove year/month)
tmat = dates.index.droplevel([0,1]).to_frame(False).reset_index().rename(columns={'index':'trend'})
Xmat = np.hstack([tmat.values, Xmat])

################################################
# --- STEP 2: CREATE DATE-SPLITS AND TRAIN --- #
print('# --- STEP 2: CREATE DATE-SPLITS AND TRAIN --- #')

# Days of 2020
d_range = pd.date_range('2020-01-01','2020-12-31', freq='1d')
d_pred = d_range[dstart:dend]
# Subset if pred range is outside of date max
dmax = pd.to_datetime((dates - pd.offsets.Day(1)).max().strftime('%Y-%m-%d'))

if d_pred.min() > dmax:
    sys.exit('Smallest date range is exceeds largest dates in df_lead_lags')
if d_pred.max() > dmax:
    print('d_pred has dates greater than dmax, subsetting')
    d_pred = d_pred[d_pred <= dmax]

print('Range that will be predicted:\n%s\n' % ', '.join(d_pred.astype(str)))

# d_test = d_pred[day]

# Initialize model
gp = mdl(model=model, lead=lead, cn=cn)

holder = []
for day, d_test in enumerate(d_pred):
    # day = 78; d_test = d_pred[day]
    print('Predicting %i hours ahead for testing day: %s\nIteration %i of %i' %
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
    ntrain, nval, ntest = Xmat_train.shape[0], Xmat_valid.shape[0], Xmat_test.shape[0]
    print('Training size: %i, validation size: %i, test size: %i' % (ntrain, nval, ntest))
    # Combine train/validation
    Xmat_tval, y_tval = np.vstack([Xmat_train, Xmat_valid]), np.append(y_train, y_valid)
    gp.fit(X=Xmat_tval, y=y_tval, ntrain=1000, nval=168)
    if day > 0:
        # Load state dict from previous iteration
        gp.gp.load_state_dict(holder_state)
    stime = time()
    gp.tune(max_iter=1000)
    holder_state = gp.gp.state_dict().copy()
    res = gp.predict(X=Xmat_test,y=y_test)
    res = res.assign(date=d_test).rename_axis('hour').reset_index()
    holder.append(res)
    print('Model took %i seconds to tune and predict' % (time() - stime))
    # # Day forecast
    # tmp = pd.concat([gp.res_train.drop(columns=['se','idx']), res.assign(tt='test').drop(columns=['hour','date'])])
    # tmp = tmp.reset_index(None, True).rename_axis('idx').reset_index()
    # from plotnine import *
    # gg_torch = (ggplot(tmp[tmp.tt!='train'], aes(x='idx', y='mu', color='tt')) + theme_bw() + geom_line() +
    #             geom_vline(xintercept=gp.ntrain) + geom_ribbon(aes(ymin='lb', ymax='ub'), alpha=0.5) +
    #             geom_point(aes(x='idx', y='y'), color='black', size=0.5, alpha=0.5))
    # gg_torch.save(os.path.join(dir_figures, 'test.png'),width=12,height=7)


####################################
# --- STEP 3: SAVE PREDICTIONS --- #

fn_res = gp.fn.replace('mdl_','res_').replace('.pkl','_'+str(dstart)+'_'+str(dend)+'.csv')
df_res = pd.concat(holder).reset_index(None,True).rename(columns={'mu':'pred'})
df_res = df_res.assign(lead=lead, model=model)

df_res.to_csv(os.path.join(dir_test, fn_res), index=True)
