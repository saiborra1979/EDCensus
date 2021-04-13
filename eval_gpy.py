import os
import pandas as pd
import numpy as np
from plotnine import *
from funs_support import ymdh2date, ymd2date, find_dir_olu
from funs_stats import add_bin_CI, get_CI

import torch

dir_base = os.getcwd()
dir_olu = find_dir_olu()
dir_figures = os.path.join(dir_olu, 'figures', 'census')
dir_output = os.path.join(dir_olu, 'output')
dir_flow = os.path.join(dir_output, 'flow')
dir_test = os.path.join(dir_flow, 'test')
lst_dir = [dir_figures, dir_output, dir_flow, dir_test]
assert all([os.path.exists(z) for z in lst_dir])

cn_ymd = ['year', 'month', 'day']
cn_ymdh = cn_ymd + ['hour']
cn_ymdl = cn_ymd + ['lead']

use_cuda = torch.cuda.is_available()
sdev = "cuda" if use_cuda else "cpu"
print('Using device: %s' % sdev)
device = torch.device(sdev)

#########################
# --- (1) LOAD DATA --- #

# (i) Find the most recent folder with GPY results
fn_test = pd.Series(os.listdir(dir_test))
fn_test = fn_test[fn_test.str.contains('^[0-9]{4}\\_[0-9]{2}\\_[0-9]{2}$')].reset_index(None,True)
dates_test = pd.to_datetime(fn_test,format='%Y_%m_%d')
df_fn_test = pd.DataFrame({'fn':fn_test,'dates':dates_test,'has_gpy':False})
# Find out which have GP data
for fn in fn_test:
    fn_fold = pd.Series(os.listdir(os.path.join(dir_test,fn)))
    mdls_fold = list(pd.Series(fn_fold.str.split('\\_',2,True).iloc[:,1].unique()).dropna())
    if 'gpy' in mdls_fold:
        df_fn_test.loc[df_fn_test.fn==fn,'has_gpy'] = True
# Get largest date
assert df_fn_test.has_gpy.any()
fold_recent = df_fn_test.loc[df_fn_test.dates.idxmax()].fn
print('Most recent folder: %s' % fold_recent)
path_recent = os.path.join(dir_test,fold_recent)
fn_recent = pd.Series(os.listdir(path_recent))

# (ii) Load the predicted/actual
fn_res = fn_recent[fn_recent.str.contains('^res\\_.*\\.csv$')]
holder = []
for fn in fn_res:
    holder.append(pd.read_csv(os.path.join(path_recent,fn)))
dat_recent = pd.concat(holder).reset_index(None,True)
dat_recent.insert(0,'dates',pd.to_datetime(dat_recent.date + ' ' + dat_recent.hour.astype(str)+':00:00'))
assert np.all(dat_recent.model == 'gpy')
assert len(dat_recent.groups.unique()) == 1
assert len(dat_recent.ntrain.unique()) == 1
dat_recent.drop(columns = ['date','hour','model','groups','ntrain'], inplace=True)
dat_recent = dat_recent.sort_values(['lead','dates']).reset_index(None,True)
# dat_recent.query('dates == "2020-03-10 10:00:00"')

# (iii) Load the kernel weights
assert 'pt' in list(fn_recent) # Check that coefficient weights are to be found
fold_pt = os.path.join(path_recent, 'pt')
fn_pt = pd.Series(os.listdir(fold_pt))
n_pt = len(fn_pt)
print('A total of %i model weights were found' % n_pt)
holder = []
for i, fn in enumerate(fn_pt):
    if (i+1) % 50 == 0:
        print('Kernel %i of %i' % (i+1, n_pt))
    tmp_dict = torch.load(os.path.join(fold_pt,fn),map_location=device)
    keys = list(tmp_dict.keys())
    vals = torch.cat([v.flatten() for v in tmp_dict.values()]).detach().numpy()
    day = fn.split('day_')[1].replace('.pth','')
    lead = int(fn.split('lead_')[1].split('_dstart')[0])
    tmp_df = pd.DataFrame({'dates':day, 'lead':lead, 'kernel':keys,'value':vals})
    holder.append(tmp_df)
# Reverse... https://docs.gpytorch.ai/en/v1.1.1/examples/00_Basic_Usage/Hyperparameters.html
# Merge and save
dat_kernel = pd.concat(holder).reset_index(None,True)
dat_kernel['dates'] = pd.to_datetime(dat_kernel.dates)
# Make the mapping
di_kern = {'likelihood.noise_covar.raw_noise':'noise','mean.constant':'mean'}
tmp_di = tmp_df[['kernel']].rename(columns={'kernel':'name'}).copy()
tmp_feature = tmp_di.name.str.split('\\.',1,True).iloc[:,0].str.split('\\_',2,True).iloc[:,2]
tmp_di = tmp_di.assign(feature=tmp_feature.fillna('intercept'))
tmp_di = tmp_di.assign(kernel=tmp_di.name.str.split('\\_',2,True).iloc[:,1].fillna('intercept'))
tmp_di[tmp_di.name.str.contains('2')]

di = {1:[10,20], 2:[3,9]}
df = pd.DataFrame({'a':[1,2]}).assign(b=lambda x: x.a.map(di))
df.b.apply(pd.Series)

#.str.split('\\.',2,True).iloc[:,0:2].apply(lambda x: x.str.cat(sep='.'),1).duplicated()