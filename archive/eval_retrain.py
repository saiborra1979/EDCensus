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
from sklearn.metrics import mean_absolute_error as mean_AE
from sklearn.metrics import median_absolute_error as med_AE
from scipy.stats import pearsonr, spearmanr
from funs_support import find_dir_olu, gg_save, makeifnot
from funs_support import get_reg_score, get_iqr

dir_base = os.getcwd()
dir_olu = find_dir_olu()
dir_figures = os.path.join(dir_olu, 'figures', 'census')
dir_output = os.path.join(dir_olu, 'output')
dir_flow = os.path.join(dir_output, 'flow')
dir_test = os.path.join(dir_flow, 'test')
dir_dtrain = os.path.join(dir_test, 'dtrain')
assert os.path.exists(dir_dtrain)

############################
# --- (1) LOAD IN DATA --- #
# Note: Validation days are zero

# cn_desc = ['mean','std','25%','75%']
# lbl_desc = ['Mean','Std. Dev', 'Q25','Q75']
# di_desc = dict(zip(cn_desc,lbl_desc))

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
df_iter = df_iter.assign(woy=lambda x: x.date_rt.dt.weekofyear, 
    year=lambda x: x.date_rt.dt.year, month=lambda x: x.date_rt.dt.month)
# Note that if it's week 53 Dec 28-Jan 3, should really be 2020
df_iter = df_iter.assign(year = lambda x: np.where((x.woy==53)&(x.month==1), x.year-1, x.year))
# Remove low week of year count
freq_woy = df_iter.groupby(['woy','year']).size().reset_index().rename(columns={0:'n'})
freq_woy = freq_woy.query('n == n.max()').drop(columns='n')

# Mappings of year/woy to date
lookup_woy = df_iter.groupby(['year','woy']).date_rt.min().reset_index()
lookup_woy = lookup_woy.merge(freq_woy,'inner')

#################################################
# --- (2) REGRESSION METRICS AND RANK ORDER --- #

cn_gg = ['model','groups','dtrain','lead']
# (i) Aggregate performance
perf_agg = df_iter.groupby(cn_gg).apply(get_reg_score).reset_index()
perf_agg[cn_cat] = perf_agg[cn_cat].apply(pd.Categorical,0)
perf_agg = perf_agg.melt(cn_gg,None,'msr')

cn_woy = ['year','woy']
cn_gg2 = cn_gg+cn_woy
# (ii) Weekly performance
perf_weekly = df_iter.groupby(cn_gg2).apply(get_reg_score).reset_index()
perf_weekly = perf_weekly.melt(cn_gg2,None,'msr')
perf_weekly[cn_cat] = perf_weekly[cn_cat].apply(pd.Categorical,0)
# Subset to the WOY range
perf_weekly = perf_weekly.merge(lookup_woy,'inner')

# (iii) Range in weekly performance
perf_weekly_range = perf_weekly.groupby(cn_gg+['msr']).value.apply(get_iqr).reset_index()
perf_weekly_range = perf_weekly_range.rename(columns={'level_'+str(len(cn_gg)+1):'metric'})

cn_gg3 = ['msr','model','groups','lead']
# (iv) Rank order performance (note we need to make the MAE negative)
tmp = perf_weekly.assign(value=lambda x: np.where(x.msr=='MAE',-x.value,x.value)).copy()
perf_weekly_rank = tmp.sort_values(cn_gg3+cn_woy+['value'],ascending=False)
del tmp
perf_weekly_rank['ridx'] = perf_weekly_rank.groupby(cn_gg3+cn_woy).cumcount()+1
perf_weekly_rank = perf_weekly_rank.groupby(cn_gg3+['ridx','dtrain']).size().reset_index().rename(columns={0:'n'})
perf_weekly_rank = perf_weekly_rank.query('ridx <= 3').reset_index(None,True)
n_woy = perf_weekly_rank.groupby(cn_gg3+['ridx']).n.sum().unique()
assert len(n_woy) == 1
perf_weekly_rank = perf_weekly_rank.assign(pct=lambda x: x.n/n_woy[0])

################################
# --- (3) SAVE BEST MODEL --- #

# Find the one with the highest rank
cn_gg_rank = ['msr','model','groups']
rank1 = perf_weekly_rank.query('ridx==1').groupby(cn_gg_rank+['dtrain']).pct.mean()
rank1 = rank1.reset_index().sort_values(cn_gg_rank+['pct'],ascending=False)
rank1 = rank1.groupby(cn_gg_rank).head(1).reset_index(None,True)
rank1 = rank1.assign(dtrain=lambda x: x.dtrain.astype(int)).groupby(['model','groups','dtrain']).size()
rank1 = rank1.sort_values(ascending=False).reset_index().drop(columns=[0]).reset_index(None,True)
best_model, best_groups, best_dtrain = rank1['model'][0], rank1['groups'][0], rank1['dtrain'][0]

# Save for later
today = datetime.now().strftime('%Y_%m_%d')
dir_today = os.path.join(dir_test,today)
makeifnot(dir_today)

# Find the files
pat = '_dtrain_'+str(best_dtrain)+'_'
fn_best = fn_dtrain[fn_dtrain.str.contains(pat)].reset_index(None,True)
assert len(fn_best) == 24  # 24 leads
for fn in fn_best:
    path_from = os.path.join(dir_dtrain,fn)
    path_to = os.path.join(dir_today, fn)
    shutil.copy(path_from, path_to)

#######################
# --- (3) PLOT IT --- #

shpz = list('$'+pd.Series(dtrain).astype(str)+'$')
posd = position_dodge(0.5)

# Daily: note that 1-3 are pretty poor
tmp = perf_agg.query('dtrain.astype("int")>3')
gg_r2_agg = (ggplot(tmp,aes(x='lead',y='value',color='lead',shape='dtrain')) + 
    theme_bw() + 
    theme(subplots_adjust={'wspace': 0.15}) + 
    geom_point(position=posd,size=2.5) + 
    scale_shape_manual(values=shpz) + guides(shape=False) + 
    ggtitle('Numbers indicate training days') + 
    labs(y='Performance measure',x='Forecasting horizon (hours)') + 
    ggtitle('R2/correlation by training days/leads') + 
    geom_hline(yintercept=0,linetype='--') + 
    facet_wrap('~msr',nrow=1,scales='free') + 
    scale_x_continuous(breaks=list(range(1,24,2))))
gg_save('gg_perf_agg.png',dir_figures,gg_r2_agg,12,4)

# Weekly performance trend
tmp = perf_weekly.query('dtrain.astype("int")>3 & msr=="spearman"')
tmp.drop(columns=['model','groups','year','woy','msr'],inplace=True)
tmp.lead.value_counts()

gg_perf_weekly = (ggplot(tmp,aes(x='date_rt',y='value',color='dtrain')) + 
    theme_bw() + geom_line() + 
    ggtitle('Spearman correaltion') + 
    facet_wrap('~lead',labeller=label_both) + 
    theme(axis_text_x=element_text(angle=90),axis_title_x=element_blank()) + 
    scale_x_datetime(date_breaks='2 months',date_labels='%b, %y'))
gg_save('gg_perf_weekly.png',dir_figures,gg_perf_weekly,12,8)
    
# Range over weekly performance
tmp = perf_weekly_range.query('dtrain.astype("int")>3')
gg_perf_range = (ggplot(tmp,aes(x='lead',y='value',color='lead',shape='dtrain')) + 
    theme_bw() + geom_point(position=posd) + 
    facet_grid('msr~metric',scales='free_y') + 
    guides(shape=False) + 
    labs(y='Performance measure',x='Forecasting horizon (hours)') + 
    scale_shape_manual(values=shpz) + 
    ggtitle('Weekly performance range (IQR and mean)'))
gg_save('gg_perf_range.png',dir_figures,gg_perf_range,10,7)

# Ranking by dtrain
tmp = perf_weekly_rank.drop(columns=['model','groups']).copy()
gg_perk_rank = (ggplot(tmp,aes(x='dtrain',y='pct',color='lead')) + 
    geom_point(size=1) +
    theme_bw() + 
    labs(x='# of training days',y='Percent in rank') + 
    facet_grid('ridx~msr',labeller=label_both,scales='free_y'))
gg_save('gg_perk_rank.png',dir_figures,gg_perk_rank,10,7)

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
