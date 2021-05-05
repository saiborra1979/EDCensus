"""
SCRIPT TO DECOMPOSE MODEL RESULTS
"""

import os
import shutil
import pandas as pd
import numpy as np
from datetime import datetime
from plotnine import *
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
from funs_support import find_dir_olu, gg_save, makeifnot

dir_base = os.getcwd()
dir_olu = find_dir_olu()
dir_figures = os.path.join(dir_olu, 'figures', 'census')
dir_output = os.path.join(dir_olu, 'output')
dir_flow = os.path.join(dir_output, 'flow')
dir_test = os.path.join(dir_flow, 'test')
dir_dtrain = os.path.join(dir_test, 'dtrain')
assert os.path.exists(dir_dtrain)

##################################
# --- (1) COMPARE DTRAIN 1-7 --- #
# Note: Validation days are zero

# cn_desc = ['mean','std','25%','75%']
# lbl_desc = ['Mean','Std. Dev', 'Q25','Q75']
# di_desc = dict(zip(cn_desc,lbl_desc))

cn_gg = ['model','groups','dtrain','lead']
cn_date = ['date_rt','date_pred']
cn_cat = ['dtrain']

fn_dtrain = pd.Series(os.listdir(dir_dtrain))

holder = []
for fn in fn_dtrain:
    holder.append(pd.read_csv(os.path.join(dir_dtrain, fn)))
df_iter = pd.concat(holder).reset_index(None,True)
df_iter[cn_date] = df_iter[cn_date].apply(pd.to_datetime)
assert df_iter.groupby(['model','groups','dtrain','lead']).size().unique().shape[0] == 1
dtrain = np.sort(df_iter.dtrain.unique())
df_iter = df_iter.assign(woy=lambda x: x.date_rt.dt.weekofyear, year=lambda x: x.date_rt.dt.year)

# Aggregate R2/Correlation
dat_r2_agg = df_iter.groupby(cn_gg).apply(lambda x: 
    pd.Series({'r2':r2_score(x.y,x.pred), 'spearman': spearmanr(x.y,x.pred)[0],
               'pearson':pearsonr(x.y,x.pred)[0]})).reset_index()
dat_r2_agg[cn_cat] = dat_r2_agg[cn_cat].apply(pd.Categorical,0)
dat_r2_agg = dat_r2_agg.melt(cn_gg,None,'msr')

# Variations in weekly performance
cn_gg2 = cn_gg+['year','woy']
dat_r2_weekly = df_iter.groupby(cn_gg2).apply(lambda x: 
    pd.Series({'r2':r2_score(x.y,x.pred), 'spearman': spearmanr(x.y,x.pred)[0],
               'pearson':pearsonr(x.y,x.pred)[0]})).reset_index()
dat_r2_weekly = dat_r2_weekly.melt(cn_gg2,None,'msr')
dat_r2_weekly = dat_r2_weekly.groupby(cn_gg+['msr']).value.apply(lambda x: 
    pd.Series({'mu':x.mean(),'lb':x.quantile(0.25),'ub':x.quantile(0.75)}))
dat_r2_weekly = dat_r2_weekly.reset_index().rename(columns={'level_'+str(len(cn_gg)+1):'metric'})
# dat_r2_weekly = dat_r2_weekly.reset_index().pivot_table('value',cn_gg+['msr'],'level_'+str(len(cn_gg)+1)).reset_index()
dat_r2_weekly[cn_cat] = dat_r2_weekly[cn_cat].apply(pd.Categorical,0)
dat_r2_weekly.loc[0]

####################################
# --- (2) DETERMINE BEST MODEL --- #

dat_best = dat_r2_weekly.query('msr=="r2" & metric=="mu"').reset_index(None,True).drop(columns=['msr','metric'])
# Find the winning number of dtraining days by lead
dat_best = dat_best.sort_values(['lead','value'],ascending=False).groupby('lead').head(1)

today = datetime.now().strftime('%Y_%m_%d')
dir_today = os.path.join(dir_test,today)
makeifnot(dir_today)

# Loop through and make a copy into the current day's folder
for ii, rr in dat_best.iterrows():
    pat = '_dtrain_'+str(rr['dtrain'])+'_'
    tmp_fn = fn_dtrain[fn_dtrain.str.contains(pat)]
    pat = '_lead_'+str(rr['lead'])+'_'
    tmp_fn = list(tmp_fn[tmp_fn.str.contains(pat)])
    assert len(tmp_fn) == 1
    tmp_fn = tmp_fn[0]
    path_from = os.path.join(dir_dtrain,tmp_fn)
    path_to = os.path.join(dir_today, tmp_fn)
    shutil.copy(path_from, path_to)


#######################
# --- (3) PLOT IT --- #

shpz = list('$'+pd.Series(dtrain).astype(str)+'$')
posd = position_dodge(0.5)

# Daily
tmp = dat_r2_agg.query('dtrain.astype("int")>3')
gg_r2_agg = (ggplot(tmp,aes(x='lead',y='value',color='lead',shape='dtrain')) + 
    theme_bw() + 
    geom_point(position=posd,size=2.5) + 
    scale_shape_manual(values=shpz) + guides(shape=False) + 
    ggtitle('Numbers indicate training days') + 
    labs(y='Performance measure',x='Forecasting horizon (hours)') + 
    ggtitle('R2/correlation by training days/leads') + 
    geom_hline(yintercept=0,linetype='--') + 
    facet_wrap('~msr',nrow=1))
gg_save('gg_r2_agg.png',dir_figures,gg_r2_agg,12,4)

# Moments of weekly
tmp = dat_r2_weekly.query('dtrain.astype("int")>3')
gg_r2_desc = (ggplot(tmp,aes(x='lead',y='value',color='lead',shape='dtrain')) + 
    theme_bw() + 
    geom_point(position=posd) + 
    facet_grid('metric~msr') + 
    guides(shape=False) + 
    labs(y='Performance measure',x='Forecasting horizon (hours)') + 
    scale_shape_manual(values=shpz) + 
    geom_hline(yintercept=0,linetype='--') + 
    ggtitle('Weekly performance range (IQR and mean)'))
gg_save('gg_r2_desc.png',dir_figures,gg_r2_desc,12,10)


# ############################################
# # --- (1) Compare 72 hours pretraining --- #

# # NOTE! RETRAIN MEANS THERE IS NO PRETRAINING, WHEREAS ITERATIVE HAS PRETRAINING
# dir_iterative = os.path.join(dir_test,'iterative')
# dir_retrain = os.path.join(dir_test,'retrain')
# fn_iterative = pd.Series(os.listdir(dir_iterative))
# fn_retrain = pd.Series(os.listdir(dir_retrain))
# assert fn_iterative.isin(fn_retrain).all()
# di = {'iterative':fn_iterative, 'retrain':fn_retrain}

# holder = []
# for fold in di:
#     storage = []
#     for fn in di[fold]:
#         storage.append(pd.read_csv(os.path.join(dir_test, fold, fn)))
#     holder.append(pd.concat(storage).assign(tt=fold))
# df_iter = pd.concat(holder).reset_index(None,True)
# df_iter.date = pd.to_datetime(df_iter.date)
# # Lets get monthly performance by lead
# cn_gg = ['tt','ntrain','lead']
# r2_iter = df_iter.assign(month=lambda x: x.date.dt.month).groupby(cn_gg+['date']).apply(lambda x: r2_score(x.y, x.pred)).reset_index().rename(columns={0:'r2'})
# # Average daily performance over the months
# r2_iter = r2_iter.assign(month=lambda x: x.date.dt.month).groupby(cn_gg+['month']).r2.mean().reset_index()
# r2_iter = r2_iter.pivot_table('r2',list(r2_iter.columns.drop(['r2','tt'])),'tt').assign(spread=lambda x: x.iterative - x.retrain).reset_index()
# r2_iter_month = r2_iter.drop(columns='lead').groupby(['ntrain','month']).mean().reset_index()
# # Aggregate over all months
# r2_iter_agg = r2_iter.groupby(['ntrain','lead']).spread.mean().reset_index()

# tmp = r2_iter_agg.query('ntrain==72').reset_index(None, True)
# gg_iter_comp = (ggplot(tmp, aes(x='lead',y='spread')) +
#                 theme_bw() + geom_point(size=2,color='blue') + geom_line() +
#                 labs(x='Forecasting lead',y='R-squared (iterative - no pretraining)') +
#                 ggtitle('Average over one-day-ahead R2') +
#                 scale_color_discrete(name='Training hours') +
#                 scale_x_continuous(breaks=list(range(1,25,2))) +
#                 geom_hline(yintercept=0,linetype='--'))
# gg_iter_comp.save(os.path.join(dir_figures, 'gg_iter_comp.png'), height=3, width=4)
